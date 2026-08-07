"""
Streamlit review app for adjudicated subject/object term-mapping pairs.

Unlike the static HTML report, this runs as a local server: every accept/
reject/edit is written straight to disk as it happens, and filtering/sorting
uses Streamlit's built-in dataframe widget - no browser download/upload
round-trip needed to persist review progress.

Reuses the loading + report-building logic from generate_report.py so the
two tools never disagree about how a "field" or "candidate" is shaped.

RUN
---
    pip install streamlit pandas --break-system-packages
    streamlit run review_app.py -- pairs.jsonl \\
        --subject-terms faire_terms.json --object-terms dwc_terms.json \\
        --decisions-log decisions_log.jsonl --sssom-out mappings.sssom.tsv

OUTPUTS
-------
- <decisions-log>: JSONL, one line appended per decision (accept/reject/edit),
  with a timestamp - a full audit trail, including overturned decisions.
- <sssom-out>: SSSOM-formatted TSV, rebuilt from the CURRENT state of
  accepted/edited decisions each time you click "Write SSSOM file now".
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from generate_report import (
    load_records, load_term_lookup, build_report_data, PREDICATE_ORDER,
)

st.set_page_config(page_title="Term Mapping Review", layout="wide")


# ---------------------------------------------------------------------------
# CLI args, passed after `--` in `streamlit run review_app.py -- ...`
# ---------------------------------------------------------------------------
@st.cache_resource
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--subject-terms", type=Path, default=None)
    ap.add_argument("--object-terms", type=Path, default=None)
    ap.add_argument("--decisions-log", type=Path, default=Path("decisions_log.jsonl"))
    ap.add_argument("--sssom-out", type=Path, default=Path("mappings.sssom.tsv"))
    return ap.parse_args(sys.argv[1:])


args = get_args()


@st.cache_data
def load_data(pairs_path, subj_path, obj_path):
    records = load_records(pairs_path)
    subject_terms = load_term_lookup(subj_path)
    object_terms = load_term_lookup(obj_path)
    report = build_report_data(records, subject_terms, object_terms)
    return report, object_terms


report, object_terms_lookup = load_data(args.pairs, args.subject_terms, args.object_terms)
fields_by_id = {f["subject_id"]: f for f in report["fields"]}


# ---------------------------------------------------------------------------
# Decision state - one entry per subject field, kept in session_state.
# action: "pending" | "accept" | "reject" | "edit"
# ---------------------------------------------------------------------------
def blank_decision():
    return {
        "action": "pending", "object_id": None, "object_label": None,
        "predicate": None, "confidence": None, "comment": None,
        "original_key": None,
    }


if "decisions" not in st.session_state:
    st.session_state.decisions = {fid: blank_decision() for fid in fields_by_id}


def log_decision(subject_id: str, decision: dict) -> None:
    """Appends one line to the audit-trail JSONL. Every decision is logged, not just accepts."""
    args.decisions_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "subject_id": subject_id, **decision}
    with args.decisions_log.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def write_sssom() -> pd.DataFrame:
    """Rebuilds the SSSOM TSV from ALL current accept/edit decisions (not append-only)."""
    rows = []
    for fid, d in st.session_state.decisions.items():
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
            "mapping_justification": "semapv:ManualMappingCuration",
            "comment": d["comment"] or "",
        })

    header_lines = [
        "# curie_map:",
        '#   skos: "http://www.w3.org/2004/02/skos/core#"',
        '#   semapv: "https://w3id.org/semapv/vocab/"',
        '#   DwC: "http://rs.tdwg.org/dwc/terms/"',
        '#   FAIReSheets: "https://example.org/faire/terms/"  # TODO: replace with your real namespace IRI',
        '# mapping_set_id: "https://example.org/mappings/term-mapping-report"  # TODO: replace',
        '# license: "https://creativecommons.org/publicdomain/zero/1.0/"  # TODO: confirm',
    ]
    df = pd.DataFrame(rows, columns=["subject_id", "subject_label", "predicate_id", "object_id",
                                      "object_label", "confidence", "mapping_justification", "comment"])
    with args.sssom_out.open("w") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, sep="\t", index=False)
    return df


def effective_view(field: dict):
    """(object_label, predicate, confidence, status) reflecting any decision already made."""
    d = st.session_state.decisions[field["subject_id"]]
    if d["action"] in ("accept", "edit") and d["object_id"]:
        return d["object_label"], d["predicate"], d["confidence"] or 0.0, d["action"]
    top = field["top"]
    return top["object_label"], top["predicate"], top["confidence"], d["action"]


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search field or match name").strip().lower()
predicate_filter = st.sidebar.selectbox("Predicate", ["All"] + PREDICATE_ORDER)
status_filter = st.sidebar.selectbox("Status", ["All", "pending", "accept", "reject", "edit"])
min_conf = st.sidebar.slider("Min confidence", 0, 100, 0) / 100

rows = []
for field in report["fields"]:
    obj_label, predicate, confidence, status = effective_view(field)
    if search and search not in (field["subject_label"] + " " + (obj_label or "")).lower():
        continue
    if predicate_filter != "All" and predicate != predicate_filter:
        continue
    if status_filter != "All" and status != status_filter:
        continue
    if confidence < min_conf:
        continue
    rows.append({
        "Field": field["subject_label"],
        "Best match": obj_label or "No match found",
        "Predicate": predicate,
        "Confidence": round(confidence * 100),
        "Status": status,
        "_subject_id": field["subject_id"],
    })

overview_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Header: predicate tally + live review progress
# ---------------------------------------------------------------------------
t = report["tally"]
st.title("Term Mapping Review")
st.caption(
    f"{report['total_fields']} fields &middot; "
    + " &middot; ".join(f"{t.get(p, 0)} {p}" for p in PREDICATE_ORDER),
    unsafe_allow_html=True,
)

accepted_n = sum(1 for d in st.session_state.decisions.values() if d["action"] == "accept")
edited_n = sum(1 for d in st.session_state.decisions.values() if d["action"] == "edit")
rejected_n = sum(1 for d in st.session_state.decisions.values() if d["action"] == "reject")
pending_n = sum(1 for d in st.session_state.decisions.values() if d["action"] == "pending")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accepted", accepted_n)
m2.metric("Edited", edited_n)
m3.metric("Rejected", rejected_n)
m4.metric("Pending", pending_n)
m5.metric("Showing", len(overview_df))

# st.dataframe's column headers are sortable by click - no extra code needed.
st.dataframe(overview_df.drop(columns="_subject_id"), use_container_width=True, hide_index=True)

col_export1, col_export2 = st.columns([1, 3])
if col_export1.button("Write SSSOM file now", type="primary"):
    sssom_df = write_sssom()
    st.success(f"Wrote {len(sssom_df)} accepted mappings to {args.sssom_out}")
col_export2.caption(f"Decisions log: {args.decisions_log} (appended on every accept/reject/edit)")

st.divider()

# ---------------------------------------------------------------------------
# Detail / review panel
# ---------------------------------------------------------------------------
options = overview_df["_subject_id"].tolist() if not overview_df.empty else []
if not options:
    st.info("No fields match the current filters.")
    st.stop()

selected_id = st.selectbox(
    "Select a field to review",
    options,
    format_func=lambda fid: fields_by_id[fid]["subject_label"],
)
field = fields_by_id[selected_id]
decision = st.session_state.decisions[selected_id]

candidates_by_key = {c["key"]: c for c in field["candidates"]}
candidate_keys = [c["key"] for c in field["candidates"]]
default_key = decision["original_key"] if decision["original_key"] in candidates_by_key else field["top"]["key"]

chosen_key = st.radio(
    "Candidate",
    candidate_keys,
    index=candidate_keys.index(default_key),
    format_func=lambda k: (
        f'{candidates_by_key[k]["object_label"]}  ·  {candidates_by_key[k]["predicate"]}  ·  '
        f'{round(candidates_by_key[k]["confidence"] * 100)}%'
    ),
)
chosen = candidates_by_key[chosen_key]

left, right = st.columns(2)
with left:
    st.subheader("Your field")
    st.markdown(f"**{field['subject_label']}**")
    st.write(field["subject_description"] or "_No description available._")
    if field["subject_example"]:
        st.caption(f"Example: {field['subject_example']}")
with right:
    st.subheader(f"Best {chosen['object_source']} match")
    st.markdown(f"**{chosen['object_label']}**")
    st.write(chosen["object_description"] or "_No description available._")
    if chosen["object_example"]:
        st.caption(f"Example: {chosen['object_example']}")

st.markdown(
    f"**Predicate:** `{chosen['predicate']}`  &nbsp;&nbsp; "
    f"**Syntactic compatibility:** `{chosen['syntactic_compatibility']}`  &nbsp;&nbsp; "
    f"**Confidence:** {round(chosen['confidence'] * 100)}%",
    unsafe_allow_html=True,
)
st.write(chosen["justification"])
if chosen["transformation_note"]:
    st.caption(f"Transformation: {chosen['transformation_note']}")

st.markdown("#### Decision")
with st.form(key=f"decision_form_{selected_id}"):
    action = st.radio("Action", ["accept", "reject", "edit"], horizontal=True)

    edit_predicate = chosen["predicate"]
    edit_object_id = chosen["object_id"]
    edit_object_label = chosen["object_label"]
    edit_comment = ""

    if action == "edit":
        edit_predicate = st.selectbox(
            "Predicate override",
            PREDICATE_ORDER,
            index=PREDICATE_ORDER.index(chosen["predicate"]) if chosen["predicate"] in PREDICATE_ORDER else 0,
        )
        all_object_terms = sorted(object_terms_lookup.keys())
        manual_choice = st.selectbox(
            "Object term override (optional - search the full term list, not just retrieved candidates)",
            ["<keep candidate above>"] + all_object_terms,
        )
        if manual_choice != "<keep candidate above>":
            edit_object_id = f"{chosen['object_source']}:{manual_choice}"
            edit_object_label = manual_choice
        edit_comment = st.text_area("Comment / reason for edit")

    submitted = st.form_submit_button("Save decision")
    if submitted:
        if action == "reject":
            new_decision = {
                **blank_decision(),
                "action": "reject",
                "original_key": chosen_key,
                "comment": edit_comment or None,
            }
        else:
            new_decision = {
                "action": action,
                "object_id": edit_object_id,
                "object_label": edit_object_label,
                "predicate": edit_predicate,
                "confidence": chosen["confidence"] if action == "accept" else None,
                "comment": edit_comment or (chosen["justification"] if action == "accept" else None),
                "original_key": chosen_key,
            }
        st.session_state.decisions[selected_id] = new_decision
        log_decision(selected_id, new_decision)
        st.success(f"Saved: {action}")
        st.rerun()
