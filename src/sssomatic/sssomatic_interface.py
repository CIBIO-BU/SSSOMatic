"""
Streamlit app for revision of subject/object term-mapping pairs.

Locally, every accept/reject/edit is written straight to disk as it happens.
Filtering/sorting uses Streamlit's built-in dataframe widget.

NAVIGATION
----------
The app opens on the review interface (empty until data is loaded). A
"Upload files" button in the top-right corner switches to a dedicated
upload page; loading files there returns you to the (now populated)
review page. Deployments configured via secrets/CLI args (see below)
skip the upload page entirely - the button won't appear.

RUN LOCALLY
-----------
    pip install streamlit pandas requests --break-system-packages
    streamlit run mapping_revision_interface.py -- pairs.jsonl \\
        --subject-terms faire_terms.json --object-terms object_terms.json \\
        --decisions-log out/objct/decisions_log.jsonl --sssom-out out/objct/mappings.sssom.tsv

DEPLOY ON STREAMLIT COMMUNITY CLOUD
------------------------------------
Cloud only takes a repo/branch/main-file path - there's no way to pass the
`--` CLI args above. Either bundle the input files in the repo and set an
[app] table in the app's Secrets (Settings -> Secrets in the Cloud dashboard),
or leave [app] unset and use the in-app "Upload files" page instead:

    [app]
    pairs = "pairs.jsonl"
    subject_terms = "faire_terms.json"
    object_terms = "dwc_terms.json"
    decisions_log = "decisions_log.jsonl"
    sssom_out = "mappings.sssom.tsv"

    [github]
    token = "ghp_..."          # PAT with repo contents read/write, on the private repo only
    repo = "your-org/your-repo"
    branch = "main"

The [github] block is optional but strongly recommended: Cloud's local disk
is not durable (redeploys, sleep/wake, and resource restarts can wipe it),
so decisions_log/sssom_out are also committed back to the repo on every save
when it's present. Without it, the app still works, but the audit trail only
survives as long as the container does - and files uploaded through the
in-app upload page only survive as long as the current browser session.

OUTPUTS
-------
- <decisions-log>: JSONL, one line appended per decision (accept/reject/edit),
  with a timestamp. Serves as a full audit trail, including overturned decisions.
- <sssom-out>: SSSOM-formatted TSV, rebuilt from the CURRENT state of
  accepted/edited decisions each time you click "Write SSSOM file". Clicking
  that button also triggers an automatic browser download of both files,
  zipped together, so you always leave with a local copy regardless of
  whether GitHub durability is configured.
"""
import argparse
import base64
import io
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

from generate_html_report import (
    load_records, load_term_lookup, build_report_data, PREDICATE_ORDER,
)

st.set_page_config(page_title="SSSOmatic®", layout="wide")


def _secrets_has(key: str) -> bool:
    """`key in st.secrets` raises StreamlitSecretNotFoundError - rather than
    just being False - when no secrets.toml exists at all, which is the
    normal case for a local run. Treat "no secrets file" as "key absent"."""
    try:
        return key in st.secrets
    except StreamlitSecretNotFoundError:
        return False


@st.cache_resource
def get_static_args():
    """Files supplied via secrets.toml [app] or `streamlit run ... -- ...`
    CLI args. Returns None when neither is configured, in which case the
    app falls back to the in-app upload page."""
    if _secrets_has("app"):
        cfg = st.secrets["app"]
        return argparse.Namespace(
            pairs=Path(cfg["pairs"]),
            subject_terms=Path(cfg["subject_terms"]) if cfg.get("subject_terms") else None,
            object_terms=Path(cfg["object_terms"]) if cfg.get("object_terms") else None,
            decisions_log=Path(cfg.get("decisions_log", "decisions_log.jsonl")),
            sssom_out=Path(cfg.get("sssom_out", "mappings.sssom.tsv")),
        )
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("pairs", type=Path)
        ap.add_argument("--subject-terms", type=Path, default=None)
        ap.add_argument("--object-terms", type=Path, default=None)
        ap.add_argument("--decisions-log", type=Path, default=Path("decisions_log.jsonl"))
        ap.add_argument("--sssom-out", type=Path, default=Path("mappings.sssom.tsv"))
        return ap.parse_args(sys.argv[1:])
    return None


STATIC_ARGS = get_static_args()

# Optional GitHub-backed durability for decisions_log/sssom_out. Only active
# when an [github] secrets table is configured - see DEPLOY note above.

def _github_configured() -> bool:
    return _secrets_has("github")


def _github_headers() -> dict:
    return {
        "Authorization": f"Bearer {st.secrets['github']['token']}",
        "Accept": "application/vnd.github+json",
    }


def github_read_file(path: str) -> tuple[str, str | None]:
    """Returns (content, sha). sha is None if the file doesn't exist in the repo yet."""
    url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{path}"
    r = requests.get(url, headers=_github_headers(), timeout=10)
    if r.status_code == 404:
        return "", None
    r.raise_for_status()
    data = r.json()
    return base64.b64decode(data["content"]).decode(), data["sha"]


def github_write_file(path: str, content: str, message: str) -> None:
    """Creates or updates a file in the configured repo/branch via the Contents API."""
    _, sha = github_read_file(path)
    url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": st.secrets["github"].get("branch", "main"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_github_headers(), json=payload, timeout=10)
    r.raise_for_status()


@st.cache_data
def load_data(pairs_path, subj_path, obj_path):
    records = load_records(pairs_path)
    subject_terms = load_term_lookup(subj_path)
    object_terms = load_term_lookup(obj_path)
    report = build_report_data(records, subject_terms, object_terms)
    return report, object_terms


EMPTY_REPORT = {"fields": [], "tally": {}, "total_fields": 0}


def render_upload_page() -> None:
    """Dedicated upload page. Reachable only via the top-bar button (or
    automatically for a deployment that has neither secrets nor CLI args
    AND has no data uploaded yet in this session)."""
    st.title("Upload input files")

    if STATIC_ARGS is not None:
        st.info("This deployment already has files configured via secrets/CLI args - no upload needed.")
        if st.button("← Back to interface"):
            st.session_state["view"] = "review"
            st.rerun()
        st.stop()

    pairs_file = st.file_uploader("Pairs (JSONL)", type=["jsonl"])
    subj_file = st.file_uploader("Subject terms (JSON)", type=["json"])
    obj_file = st.file_uploader("Object terms (JSON)", type=["json"])

    can_load = bool(pairs_file and subj_file and obj_file)
    col_load, col_cancel = st.columns([1, 1])

    if col_load.button("Load files", type="primary", disabled=not can_load):
        if "_upload_dir" not in st.session_state:
            st.session_state["_upload_dir"] = tempfile.mkdtemp()
        tmp_dir = Path(st.session_state["_upload_dir"])

        def stage(uploaded_file):
            dest = tmp_dir / uploaded_file.name
            dest.write_bytes(uploaded_file.getvalue())
            return dest

        st.session_state["uploaded_args"] = argparse.Namespace(
            pairs=stage(pairs_file),
            subject_terms=stage(subj_file),
            object_terms=stage(obj_file),
            decisions_log=tmp_dir / "decisions_log.jsonl",
            sssom_out=tmp_dir / "mappings.sssom.tsv",
        )
        # A newly uploaded dataset invalidates any previously loaded report
        # and per-field decision state, so both get rebuilt fresh below.
        load_data.clear()
        st.session_state.pop("decisions", None)
        st.session_state["view"] = "review"
        st.rerun()

    if col_cancel.button("Cancel"):
        st.session_state["view"] = "review"
        st.rerun()

    if not can_load:
        st.info("Upload all three files, then click Load files.")
    st.stop()


def trigger_zip_download(filename: str, files: dict[str, bytes]) -> None:
    """Streamlit's st.download_button always needs an explicit second click
    to actually save a file - there's no way to hand bytes to the browser
    from Python alone. This works around that by rendering an invisible
    page fragment with a normal download link and firing its `.click()`
    from JS the moment it loads, so the save happens as a direct
    consequence of the *first* click (the Streamlit button that called
    this). Everything is bundled into one zip rather than one call per
    file, since most browsers block a second silent auto-download that
    isn't itself tied to a fresh user gesture."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    b64 = base64.b64encode(buf.getvalue()).decode()
    components.html(
        f"""
        <a id="auto-dl" href="data:application/zip;base64,{b64}" download="{filename}"></a>
        <script>document.getElementById('auto-dl').click();</script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------

if "view" not in st.session_state:
    st.session_state["view"] = "review"  # always land on the (possibly empty) review page

if st.session_state["view"] == "upload":
    render_upload_page()

args = STATIC_ARGS or st.session_state.get("uploaded_args")

top_l, top_r = st.columns([6, 1])
with top_r:
    if STATIC_ARGS is None and st.button("Upload files", type="primary", use_container_width=True):
        st.session_state["view"] = "upload"
        st.rerun()

if args is not None:
    report, object_terms_lookup = load_data(args.pairs, args.subject_terms, args.object_terms)
else:
    report, object_terms_lookup = EMPTY_REPORT, {}

fields_by_id = {f["subject_id"]: f for f in report["fields"]}


# Decision state is made one entry per (subject field, candidate), kept in
# session_state.
# action: "pending" | "accept" | "reject" | "edit"

def blank_decision():
    return {
        "action": "pending", "object_id": None, "object_label": None,
        "predicate": None, "confidence": None, "comment": None,
    }


if "decisions" not in st.session_state:
    st.session_state.decisions = {
        fid: {c["key"]: blank_decision() for c in f["candidates"]}
        for fid, f in fields_by_id.items()
    }


def log_decision(subject_id: str, candidate_key: str, decision: dict) -> None:
    """Appends one line to the audit-trail JSONL. Every decision is logged, not just accepts."""
    args.decisions_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id, "candidate_key": candidate_key, **decision,
    }
    line = json.dumps(entry) + "\n"
    with args.decisions_log.open("a") as f:
        f.write(line)
    if _github_configured():
        # Local disk on Streamlit Cloud isn't durable across redeploys/sleep,
        # so also append this line to the same file in the repo. One extra
        # GET+PUT per decision - fine at review-tool volumes.
        existing, _ = github_read_file(str(args.decisions_log))
        github_write_file(
            str(args.decisions_log), existing + line,
            f"decision: {decision['action']} {subject_id}/{candidate_key}",
        )


def write_sssom() -> pd.DataFrame:
    """Rebuilds the SSSOM TSV from ALL current accept/edit decisions across
    every field/candidate pair (not append-only)."""
    rows = []
    for fid, candidate_decisions in st.session_state.decisions.items():
        for d in candidate_decisions.values():
            if d["action"] not in ("accept", "edit"):
                continue
            if not d["object_id"] or d["predicate"] == "no_match":
                continue  # no_match isn't a real correspondence
            field = fields_by_id[fid]
            rows.append({
                "subject_id": fid,
                "subject_label": field["subject_label"],
                "predicate_id": d["predicate"],
                "object_id": d["object_id"],
                "object_label": d["object_label"],
                "confidence": d["confidence"],
                "mapping_justification": "semapv:CompositeMatching, semapv:LLMBasedMatching",
                "comment": d["comment"] or "",
            })

    # Namespace prefixes for curie_map are pulled straight from the CURIEs
    # actually being written (fid / object_id "PREFIX:local_name"), rather
    # than hardcoded "Subject"/"Object" placeholders - so if you ever map
    # against a second object source, its prefix shows up here too.
    prefixes = set()
    for row in rows:
        prefixes.add(row["subject_id"].split(":", 1)[0])
        prefixes.add(row["object_id"].split(":", 1)[0])

    header_lines = [
        "# curie_map:",
        '#   skos: "http://www.w3.org/2004/02/skos/core#" #TODO: comfirm and delete this comment',
        '#   semapv: "https://w3id.org/semapv/vocab/" #TODO: comfirm and delete this comment',
    ]
    for prefix in sorted(prefixes):
        header_lines.append(
            f'#   {prefix}: "https://example.org/{prefix.lower()}/" '
            "#TODO: replace with the real namespace IRI"
        )
    header_lines.append("#   TODO: add namespaces as needed")
    df = pd.DataFrame(rows, columns=["subject_id", "subject_label", "predicate_id", "object_id",
                                      "object_label", "confidence", "mapping_justification", "comment"])
    content = "\n".join(header_lines) + "\n" + df.to_csv(sep="\t", index=False)
    with args.sssom_out.open("w") as f:
        f.write(content)
    if _github_configured():
        github_write_file(str(args.sssom_out), content, f"sssom: rebuild ({len(df)} mappings)")
    return df


def candidate_view(field: dict, candidate: dict, decision: dict):
    """(object_label, predicate, confidence) for one candidate row - the
    decision's own override if it was accepted/edited, else the candidate's
    original retrieved values. Edits don't carry a confidence score, so
    those fall back to the candidate's original confidence too."""
    if decision["action"] in ("accept", "edit") and decision["object_id"]:
        confidence = decision["confidence"] if decision["confidence"] is not None else candidate["confidence"]
        return decision["object_label"], decision["predicate"], confidence
    return candidate["object_label"], candidate["predicate"], candidate["confidence"]


def max_confidence(field: dict) -> float:
    return max((c["confidence"] for c in field["candidates"]), default=0.0)


# Sidebar filters

st.sidebar.header("Filters")
search = st.sidebar.text_input("Search field name:").strip().lower()
predicate_filter = st.sidebar.selectbox("Predicate:", ["All"] + PREDICATE_ORDER)
# Default to "pending" - the review workflow is mostly about clearing this queue,
# and defaulting to "All" meant every fresh session opened on a wall of
# already-decided rows.
status_filter = st.sidebar.selectbox("Status:", ["All", "pending", "accept", "reject", "edit"], index=1)
min_conf = st.sidebar.slider("Min. confidence:", 0, 100, 0) / 100
max_conf = st.sidebar.slider("Max. confidence:", 0, 100, 100) / 100

# One row per (field, candidate) - each candidate is its own mapping
# proposal with its own decision, so a table grouped by field was hiding
# that structure. The "select field to review" panel below stays grouped,
# since reviewing is still done one field at a time.
rows = []
for field in report["fields"]:
    fid = field["subject_id"]
    for candidate in field["candidates"]:
        decision = st.session_state.decisions[fid][candidate["key"]]
        obj_label, predicate, confidence = candidate_view(field, candidate, decision)
        if search and search not in (field["subject_label"] + " " + (obj_label or "")).lower():
            continue
        if predicate_filter != "All" and predicate != predicate_filter:
            continue
        if status_filter != "All" and decision["action"] != status_filter:
            continue
        if confidence < min_conf:
            continue
        if confidence > max_conf:
            continue
        rows.append({
            "Field": field["subject_label"],
            "Candidate match": obj_label or "No match found",
            "Predicate": predicate,
            "Confidence": round(confidence * 100),
            "Status": decision["action"],
            "_subject_id": fid,
        })

overview_df = pd.DataFrame(rows, columns=[
    "Field",
    "Candidate match",
    "Predicate",
    "Confidence",
    "Status",
    "_subject_id",
])


# Header: predicate tally + live review progress

t = report["tally"]
st.title("SSSOmatic®")
st.subheader("*Discover matches. Assess relationships. Validate mappings.*")
st.caption(
    f"{report['total_fields']} subject fields &middot; "
    + " &middot; ".join(f"{t.get(p, 0)} {p}" for p in PREDICATE_ORDER),
    unsafe_allow_html=True,
)

# Summed across every field/candidate decision, matching the demuxed table
# above - not one count per field.
all_decisions = [d for cand_map in st.session_state.decisions.values() for d in cand_map.values()]
accepted_n = sum(1 for d in all_decisions if d["action"] == "accept")
edited_n = sum(1 for d in all_decisions if d["action"] == "edit")
rejected_n = sum(1 for d in all_decisions if d["action"] == "reject")
pending_n = sum(1 for d in all_decisions if d["action"] == "pending")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accepted", accepted_n)
m2.metric("Edited", edited_n)
m3.metric("Rejected", rejected_n)
m4.metric("Pending", pending_n)
m5.metric("Showing", len(overview_df))

decided_n = accepted_n + edited_n + rejected_n
total_candidates = len(all_decisions)
st.progress(
    decided_n / total_candidates if total_candidates else 0,
    text=f"{decided_n} / {total_candidates} candidates reviewed",
)

# st.dataframe's column headers are sortable by click
st.dataframe(
    overview_df.drop(columns="_subject_id"),
    width="stretch",
    hide_index=True,
)

if args is not None:
    col_export1, col_export2 = st.columns([1, 3])
    if col_export1.button("Write SSSOM file", type="primary"):
        sssom_df = write_sssom()
        st.success(f"Wrote {len(sssom_df)} accepted mappings to {args.sssom_out}.")
        st.warning("**WARNING**: Please confirm and edit the SSSOM's CURIE map.")
        export_files = {args.sssom_out.name: args.sssom_out.read_bytes()}
        if args.decisions_log.exists():
            export_files[args.decisions_log.name] = args.decisions_log.read_bytes()
        trigger_zip_download("sssomatic_export.zip", export_files)
    col_export2.caption(f"Decisions log: {args.decisions_log} (appended on every accept/reject/edit)")
else:
    st.info("No data loaded yet. Click **Upload files** above to get started.")

st.divider()


# Detail / review panel, grouped by field
# reviewing happens by one field (and its candidates)

options = overview_df["_subject_id"].drop_duplicates().tolist() if not overview_df.empty else []
if not options:
    if args is None:
        st.info("Nothing to review yet. Upload files to load subject/object mapping candidates.")
    else:
        st.info("No fields match the current filters.")
    st.stop()


def first_pending_candidate(subject_id: str) -> str:
    """The first not-yet-decided candidate for a field, in retrieval order,
    or its first candidate at all if every one is already decided."""
    for c in fields_by_id[subject_id]["candidates"]:
        if st.session_state.decisions[subject_id][c["key"]]["action"] == "pending":
            return c["key"]
    return fields_by_id[subject_id]["candidates"][0]["key"]


def advance_selection(current_id: str) -> None:
    """Point the selectbox at the next remaining FIELD after `current_id`'s
    candidates are all decided. Uses the *pre-decision* filtered order:
    since only `current_id`'s status changes, every other item's filter
    membership is unaffected, so walking forward (then backward, for the
    last item in the list) in the list we already have is safe without
    recomputing filters first.

    Streamlit forbids writing to a widget's own state key once that widget
    has already been instantiated in this run (and the buttons that call
    this live below the selectbox), so this stages the target in a plain
    key instead. It's applied to `field_select` on the next run, before the
    selectbox is created - see the seeding block below."""
    idx = options.index(current_id)
    for candidate in options[idx + 1:]:
        st.session_state["_pending_select"] = candidate
        return
    for candidate in reversed(options[:idx]):
        st.session_state["_pending_select"] = candidate
        return
    # current_id was the only item left - nothing to advance to.


def advance_within_field(subject_id: str, current_candidate_key: str) -> None:
    """Move to the next candidate, or to the next field if this is the last one."""

    cand_keys = [
        c["key"]
        for c in fields_by_id[subject_id]["candidates"]
    ]

    current_idx = cand_keys.index(current_candidate_key)

    if current_idx + 1 < len(cand_keys):
        # Move to the next candidate
        st.session_state["_pending_candidate"] = cand_keys[current_idx + 1]
    else:
        # No more candidates in this field → move to next field
        advance_selection(subject_id)

def save_decision(subject_id: str, candidate_key: str, new_decision: dict, label: str) -> None:
    st.session_state.decisions[subject_id][candidate_key] = new_decision
    log_decision(subject_id, candidate_key, new_decision)
    st.session_state["editing_open"] = False
    advance_within_field(subject_id, candidate_key)
    st.toast(f"Saved \u2014 {label}: {fields_by_id[subject_id]['subject_label']}")
    st.rerun()


# Apply any pending navigation (from advance_selection, in a prior run)
# before the widget is created - this is the one point in the script where
# writing to `field_select` is legal. Also re-seed if the previous selection
# fell out of the current filter.
if "_pending_select" in st.session_state:
    st.session_state["field_select"] = st.session_state.pop("_pending_select")
elif st.session_state.get("field_select") not in options:
    st.session_state["field_select"] = options[0]

selected_id = st.selectbox(
    "Select a field to review:",
    options,
    format_func=lambda fid: fields_by_id[fid]["subject_label"],
    key="field_select",
)

# Reset the inline edit form and jump to this field's first pending
# candidate whenever the selected field changes, so a half-filled override
# (or mid-review candidate position) from the previous field never leaks in.
if st.session_state.get("_last_selected_id") != selected_id:
    st.session_state["editing_open"] = False
    st.session_state["_last_selected_id"] = selected_id
    st.session_state["_pending_candidate"] = first_pending_candidate(selected_id)

field = fields_by_id[selected_id]
position = options.index(selected_id) + 1
st.caption(f"Reviewing {position} of {len(options)} subject terms.")

candidates_by_key = {c["key"]: c for c in field["candidates"]}
candidate_keys = [c["key"] for c in field["candidates"]]
candidate_decisions = st.session_state.decisions[selected_id]

# Same pattern as field_select above: stage-then-apply, since this can also
# be set from a button handler below (which runs after this widget in the
# same script pass).
if "_pending_candidate" in st.session_state:
    st.session_state["candidate_select"] = st.session_state.pop("_pending_candidate")
elif st.session_state.get("candidate_select") not in candidate_keys:
    st.session_state["candidate_select"] = first_pending_candidate(selected_id)

chosen_key = st.radio(
    "Prospective candidates:",
    candidate_keys,
    format_func=lambda k: (
        f'**[{candidate_decisions[k]["action"].capitalize()}]** '
        f'{candidates_by_key[k]["object_label"]}  ·  '
        f'{candidates_by_key[k]["predicate"]}  ·  '
        f'{round(candidates_by_key[k]["confidence"] * 100)}%'
    ),
    key="candidate_select",
)
chosen = candidates_by_key[chosen_key]
chosen_decision = candidate_decisions[chosen_key]

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Subject Field")
        st.markdown(f"**Field name:** {field['subject_label']}")
        st.write(f"**Description:** {field['subject_description'] or 'No description available.'}")
        if field["subject_example"]:
            st.caption(f"Example: {field['subject_example']}")
with right:
    with st.container(border=True):
        st.subheader("Candidate Object Field")
        st.markdown(f"**Field name:** {chosen['object_label']}")
        st.write(f"**Description:** {chosen['object_description'] or 'No description available.'}")
        if chosen["object_example"]:
            st.caption(f"Example: {chosen['object_example'] or 'No example available.'}")

with st.container(border=True):
    st.subheader("LLM Adjudication Summary")
    st.markdown(
        f"**Predicate:** `{chosen['predicate']}`  &nbsp;&nbsp; "
        f"**Syntactic compatibility:** `{chosen['syntactic_compatibility']}`  &nbsp;&nbsp; "
        f"**Confidence:** {round(chosen['confidence'] * 100)}%",
        unsafe_allow_html=True,
    )
    st.write(f"**LLM-adjudicated predicate justification:** {chosen['justification']}")
    if chosen["transformation_note"]:
        st.write(f"**Suggested syntax transformation**: {chosen['transformation_note']}")

    st.caption(
    "For guidance on SKOS mapping properties, "
    "[consult the SKOS documentation](https://www.w3.org/TR/skos-reference/#mapping)."
    )


# Decision - one-click accept/reject for the common case; edit expands an
# inline override form only when needed, instead of always showing every
# override field behind a form-submit round trip. Applies to the currently
# selected CANDIDATE only - the field itself advances once every candidate
# has a decision (see advance_within_field above).

st.markdown("#### Decision")
st.warning(
    "**WARNING**: If you agree with a 'no_match' decision, select 'Reject' to reject the mapping."
)
b1, b2, b3, b4 = st.columns(4)

if b1.button("Accept", use_container_width=True):
    save_decision(selected_id, chosen_key, {
        "action": "accept",
        "object_id": chosen["object_id"],
        "object_label": chosen["object_label"],
        "predicate": chosen["predicate"],
        "confidence": chosen["confidence"],
        "comment": chosen["justification"],
    }, "accepted")

if b2.button("Reject", use_container_width=True):
    save_decision(selected_id, chosen_key, {
        **blank_decision(),
        "action": "reject",
    }, "rejected")

if b3.button("Edit", use_container_width=True):
    st.session_state["editing_open"] = True

if b4.button("Skip", use_container_width=True, help="Move on without recording a decision"):
    advance_within_field(selected_id, chosen_key)
    st.rerun()

if st.session_state.get("editing_open"):
    # Keying the form on the candidate too (not just the field) means
    # switching candidates while mid-edit gets a fresh form instead of
    # carrying over the previous candidate's typed-in overrides.
    with st.form(key=f"edit_form_{selected_id}_{chosen_key}"):
        st.caption("Override the predicate, object term and/or comment, then save.")
        edit_predicate = st.selectbox(
            "Predicate override:",
            PREDICATE_ORDER,
            index=PREDICATE_ORDER.index(chosen["predicate"]) if chosen["predicate"] in PREDICATE_ORDER else 0,
        )
        all_object_terms = sorted(object_terms_lookup.keys())
        manual_choice = st.selectbox(
            "Object term override (optional: search the full term list, not just retrieved candidates):",
            ["<keep candidate above>"] + all_object_terms,
        )
        if manual_choice != "<keep candidate above>":
            edit_object_id = f"{chosen['object_source']}:{manual_choice}"
            edit_object_label = manual_choice
        else:
            edit_object_id = chosen["object_id"]
            edit_object_label = chosen["object_label"]
        # Default to this candidate's own justification text - if it was
        # already edited before, prefer the reviewer's own prior comment.
        default_comment = (
            chosen_decision["comment"]
            if chosen_decision["action"] == "edit" and chosen_decision["comment"]
            else chosen["justification"]
        )
        edit_comment = st.text_area("Comment/reason for edit:", value=default_comment)

        col_save, col_cancel = st.columns([1, 1])
        submitted = col_save.form_submit_button("Save edit", type="primary")
        cancelled = col_cancel.form_submit_button("Cancel", help="Cancel edits.")

        if submitted:
            save_decision(selected_id, chosen_key, {
                "action": "edit",
                "object_id": edit_object_id,
                "object_label": edit_object_label,
                "predicate": edit_predicate,
                "confidence": None,
                "comment": edit_comment or None,
            }, "edited")
        elif cancelled:
            st.session_state["editing_open"] = False
            st.rerun()