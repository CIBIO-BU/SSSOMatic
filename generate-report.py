"""
Generate an interactive HTML report from adjudicated subject/object term-mapping pairs.

INPUT
-----
A JSON array or JSONL file of records shaped like:

    {
      "key": "2c37636ba0342a8df0cda9477aae1c15",
      "subject_id": "FAIReSheets:recordedByID",
      "object_id": "DwC:identifiedByID",
      "adj": {
        "predicate": "skos:relatedMatch",
        "confidence": 0.8,
        "syntactic_compatibility": "transform",
        "justification": "...",
        "transformation_note": "..."
      }
    }

Optionally, two term-dump JSON files (the same shape used elsewhere in the
pipeline: { "<term_name>": {"value": ..., "description": ...}, ... }) to
enrich each row with a description and example value:

    --subject-terms faire_terms.json --object-terms dwc_terms.json

USAGE
-----
    python generate_report.py pairs.jsonl \\
        --subject-terms faire_terms.json --object-terms dwc_terms.json \\
        --out report.html
"""
import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    # Try a single JSON array first, then fall back to JSONL (one object/line).
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_term_lookup(path: Path | None) -> dict:
    """Loads a {term_name: {value, description}} dump into a plain dict, or {} if no path given."""
    if path is None:
        return {}
    with path.open() as f:
        return json.load(f)


def split_id(term_id: str) -> tuple[str, str]:
    """'FAIReSheets:recordedByID' -> ('FAIReSheets', 'recordedByID')"""
    if ":" in term_id:
        source, name = term_id.split(":", 1)
        return source, name
    return "", term_id


# ---------------------------------------------------------------------------
# Predicate tally (drives the summary line, keyed the same as the SKOS predicate)
# ---------------------------------------------------------------------------

PREDICATE_SHORT = {
    "skos:exactMatch": "exact",
    "skos:closeMatch": "close",
    "skos:broadMatch": "broad",
    "skos:narrowMatch": "narrow",
    "skos:relatedMatch": "related",
    "no_match": "unmatched",
}
# Order used in the summary line and for building the initial tally dict.
PREDICATE_ORDER = ["skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                   "skos:narrowMatch", "skos:relatedMatch", "no_match"]


# ---------------------------------------------------------------------------
# Build report data
# ---------------------------------------------------------------------------

def build_report_data(records: list[dict], subject_terms: dict, object_terms: dict) -> dict:
    by_subject: dict[str, dict] = {}

    for rec in records:
        adj = rec["adj"]
        subj_source, subj_name = split_id(rec["subject_id"])
        obj_source, obj_name = split_id(rec["object_id"])

        subj_info = subject_terms.get(subj_name, {})
        obj_info = object_terms.get(obj_name, {})

        candidate = {
            "key": rec.get("key", ""),
            "object_id": rec["object_id"],
            "object_label": obj_name,
            "object_source": obj_source,
            "object_description": obj_info.get("description", ""),
            "object_example": obj_info.get("value", ""),
            "predicate": adj["predicate"],
            "confidence": adj.get("confidence", 0.0),
            "syntactic_compatibility": adj.get("syntactic_compatibility", ""),
            "justification": adj.get("justification", ""),
            "transformation_note": adj.get("transformation_note"),
        }

        entry = by_subject.setdefault(rec["subject_id"], {
            "subject_id": rec["subject_id"],
            "subject_label": subj_name,
            "subject_source": subj_source,
            "subject_description": subj_info.get("description", ""),
            "subject_example": subj_info.get("value", ""),
            "candidates": [],
        })
        entry["candidates"].append(candidate)

    fields = []
    tally = {p: 0 for p in PREDICATE_ORDER}

    for entry in by_subject.values():
        # Rank candidates: non-"no_match" first, then by confidence desc.
        entry["candidates"].sort(
            key=lambda c: (c["predicate"] != "no_match", c["confidence"]),
            reverse=True,
        )
        top = entry["candidates"][0]
        tally[top["predicate"]] = tally.get(top["predicate"], 0) + 1
        entry["top"] = top
        entry["alternates"] = entry["candidates"][1:]
        fields.append(entry)

    return {
        "total_fields": len(fields),
        "tally": tally,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Term Mapping Report</title>
<style>
  :root {
    --canvas: #ffffff;
    --canvas-subtle: #f6f8fa;
    --border: #d0d7de;
    --fg: #1f2328;
    --fg-muted: #656d76;
    --link: #0969da;
    --success: #1a7f37;
    --attention: #9a6700;
    --danger: #cf222e;
    --radius: 6px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--fg);
    background: var(--canvas);
    padding: 32px 24px 80px;
    font-size: 14px;
  }
  code, .mono {
    font-family: ui-monospace, "SF Mono", "Segoe UI Mono", Consolas, monospace;
  }
  .wrap { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
  .summary {
    font-size: 13px;
    color: var(--fg-muted);
    margin: 0 0 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 10px;
    align-items: baseline;
  }
  .summary strong { color: var(--fg); font-weight: 600; }
  .summary .dot { color: var(--border); }
  .summary .tag-unmatched strong { color: var(--danger); }

  table { width: 100%; border-collapse: collapse; }
  thead th {
    text-align: left;
    font-size: 12px;
    color: var(--fg-muted);
    font-weight: 600;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--canvas-subtle);
  }
  thead th:first-child { border-top-left-radius: var(--radius); }
  thead th:last-child { border-top-right-radius: var(--radius); }
  tbody tr.row {
    cursor: pointer;
    border-bottom: 1px solid var(--border);
  }
  tbody tr.row:hover { background: var(--canvas-subtle); }
  tbody tr.row td { padding: 10px 12px; font-size: 14px; vertical-align: middle; }
  .field-name { font-weight: 600; }
  .match-name { font-weight: 400; }
  .no-match .match-name { color: var(--danger); }
  .predicate-pill {
    display: inline-block;
    font-family: ui-monospace, "SF Mono", "Segoe UI Mono", Consolas, monospace;
    font-size: 12px;
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: 100px;
    color: var(--fg-muted);
  }
  .predicate-pill.no-match { color: var(--danger); border-color: var(--danger); }
  .score-cell { font-weight: 600; text-align: right; padding-right: 16px !important; }
  .score-high { color: var(--success); }
  .score-mid { color: var(--attention); }
  .score-low { color: var(--danger); }
  .chevron {
    display: inline-block;
    margin-left: 8px;
    color: var(--fg-muted);
    transition: transform 0.15s ease;
    font-size: 11px;
  }
  tr.row.open .chevron { transform: rotate(90deg); }

  tr.detail-row { display: none; }
  tr.detail-row.open { display: table-row; }
  tr.detail-row td { padding: 0; border-bottom: 1px solid var(--border); }
  .detail {
    background: var(--canvas-subtle);
    padding: 20px 24px 24px;
  }
  .compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 18px;
  }
  .compare .col .eyebrow {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--fg-muted);
    font-weight: 600;
    margin-bottom: 6px;
  }
  .compare .col .name { font-weight: 600; font-size: 15px; margin-bottom: 6px; }
  .compare .col .desc { font-size: 13.5px; line-height: 1.5; margin-bottom: 8px; }
  .compare .col .example { font-size: 12.5px; color: var(--fg-muted); }
  .compare .col .example b { color: var(--fg); font-weight: 600; }
  .compare .col .empty { font-size: 13px; color: var(--fg-muted); font-style: italic; }

  .reasoning {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    background: var(--canvas);
    margin-bottom: 18px;
  }
  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 20px;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .meta-item { font-size: 13px; }
  .meta-item .meta-label {
    color: var(--fg-muted);
    margin-right: 6px;
  }
  .meta-item .meta-value { font-weight: 600; }
  .meta-item.confidence .meta-value.score-high { color: var(--success); }
  .meta-item.confidence .meta-value.score-mid { color: var(--attention); }
  .meta-item.confidence .meta-value.score-low { color: var(--danger); }
  .meta-item.predicate .meta-value.no-match { color: var(--danger); }
  .just { font-size: 13.5px; line-height: 1.55; margin: 0 0 8px; }
  .transform-note { font-size: 12.5px; color: var(--fg-muted); margin: 0; }

  .alt-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--fg-muted);
    font-weight: 600;
    margin-bottom: 8px;
  }
  .alt-empty { font-size: 13px; color: var(--fg-muted); font-style: italic; }
  .alt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 13.5px;
    border: 1px solid transparent;
  }
  .alt-row:hover { border-color: var(--border); background: var(--canvas); }
  .alt-row.selected { border-color: var(--link); background: var(--canvas); }
  .alt-row .alt-label { font-weight: 600; }
  .alt-row .alt-meta { color: var(--fg-muted); font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Term Mapping Report</h1>
  <p class="summary">__SUMMARY_HTML__</p>

  <table>
    <thead>
      <tr>
        <th style="width:26%">Subject Field</th>
        <th style="width:28%">Object Field: Best match</th>
        <th style="width:26%">Assigned predicate</th>
        <th style="width:20%; text-align:right;">Confidence</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
</div>

<script>
const REPORT = __DATA_JSON__;

function pct(x) { return Math.round(x * 100) + "%"; }

function scoreClass(c) {
  if (c >= 0.8) return "score-high";
  if (c >= 0.5) return "score-mid";
  return "score-low";
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : s;
  return d.innerHTML;
}

function renderCompareCol(eyebrow, name, description, example) {
  return `
    <div class="col">
      <div class="eyebrow">${esc(eyebrow)}</div>
      <div class="name">${esc(name)}</div>
      ${description ? `<div class="desc">${esc(description)}</div>` : `<div class="empty">No description available.</div>`}
      ${example ? `<div class="example"><b>Example:</b> ${esc(example)}</div>` : ""}
    </div>`;
}

function renderReasoning(cand) {
  const isNoMatch = cand.predicate === "no_match";
  return `
    <div class="reasoning">
      <div class="meta-row">
        <span class="meta-item predicate">
          <span class="meta-label">Predicate</span>
          <span class="meta-value ${isNoMatch ? 'no-match' : ''}">${esc(cand.predicate)}</span>
        </span>
        <span class="meta-item">
          <span class="meta-label">Syntactic compatibility</span>
          <span class="meta-value">${esc(cand.syntactic_compatibility)}</span>
        </span>
        <span class="meta-item confidence">
          <span class="meta-label">Confidence</span>
          <span class="meta-value ${scoreClass(cand.confidence)}">${pct(cand.confidence)}</span>
        </span>
      </div>
      <p class="just">${esc(cand.justification)}</p>
      ${cand.transformation_note ? `<p class="transform-note"><b>Syntax transformation:</b> ${esc(cand.transformation_note)}</p>` : ""}
    </div>`;
}

function renderDetail(field, selectedCand) {
  const alternates = field.candidates.filter(c => c.key !== selectedCand.key);
  const altHtml = alternates.length
    ? alternates.map(c => `
        <div class="alt-row" data-key="${esc(c.key)}">
          <span class="alt-label">${esc(c.object_label)} <span class="alt-meta">(${esc(c.object_source)})</span></span>
          <span class="alt-meta">${esc(c.predicate)} &middot; ${pct(c.confidence)}</span>
        </div>`).join("")
    : `<div class="alt-empty">No alternate candidates for this field.</div>`;

  return `
    <div class="detail">
      <div class="compare">
        ${renderCompareCol("Your field", field.subject_label, field.subject_description, field.subject_example)}
        ${renderCompareCol(`Best ${selectedCand.object_source} match`, selectedCand.object_label, selectedCand.object_description, selectedCand.object_example)}
      </div>
      ${renderReasoning(selectedCand)}
      <div class="alt-title">Other candidates for this field (${alternates.length})</div>
      ${altHtml}
    </div>`;
}

function buildRows() {
  const tbody = document.getElementById("rows");
  REPORT.fields.forEach((field, idx) => {
    const top = field.top;
    const isNoMatch = top.predicate === "no_match";
    const rowId = `row-${idx}`;

    const tr = document.createElement("tr");
    tr.className = "row" + (isNoMatch ? " no-match" : "");
    tr.id = rowId;
    tr.innerHTML = `
      <td class="field-name">${esc(field.subject_label)}</td>
      <td class="match-name">${isNoMatch ? "No match found" : esc(top.object_label)}<span class="chevron">&#9656;</span></td>
      <td><span class="predicate-pill ${isNoMatch ? 'no-match' : ''}">${esc(top.predicate)}</span></td>
      <td class="score-cell ${scoreClass(top.confidence)}">${pct(top.confidence)}</td>
    `;

    const detailTr = document.createElement("tr");
    detailTr.className = "detail-row";
    const detailTd = document.createElement("td");
    detailTd.colSpan = 4;
    detailTr.appendChild(detailTd);

    let selected = top;

    function paint() {
      detailTd.innerHTML = renderDetail(field, selected);
      detailTd.querySelectorAll(".alt-row").forEach(el => {
        el.addEventListener("click", (e) => {
          e.stopPropagation();
          const key = el.getAttribute("data-key");
          selected = field.candidates.find(c => c.key === key) || selected;
          paint();
        });
      });
    }
    paint();

    tr.addEventListener("click", () => {
      const isOpen = tr.classList.toggle("open");
      detailTr.classList.toggle("open", isOpen);
    });

    tbody.appendChild(tr);
    tbody.appendChild(detailTr);
  });
}

buildRows();
</script>
</body>
</html>
"""


def render_html(report_data: dict) -> str:
    t = report_data["tally"]
    total = report_data["total_fields"]

    parts = [f'<span><strong>{total}</strong> fields</span>']
    for pred in PREDICATE_ORDER:
        short = PREDICATE_SHORT[pred]
        css_class = "tag-unmatched" if pred == "no_match" else ""
        parts.append('<span class="dot">&middot;</span>')
        parts.append(f'<span class="{css_class}"><strong>{t.get(pred, 0)}</strong> {short}</span>')
    summary_html = "".join(parts)

    html_out = PAGE_TEMPLATE.replace("__SUMMARY_HTML__", summary_html)
    html_out = html_out.replace("__DATA_JSON__", json.dumps(report_data))
    return html_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pairs", type=Path, help="JSON array or JSONL file of adjudicated pairs")
    ap.add_argument("--subject-terms", type=Path, default=None, help="Subject term dump JSON (optional)")
    ap.add_argument("--object-terms", type=Path, default=None, help="Object term dump JSON (optional)")
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    args = ap.parse_args()

    records = load_records(args.pairs)
    subject_terms = load_term_lookup(args.subject_terms)
    object_terms = load_term_lookup(args.object_terms)

    data = build_report_data(records, subject_terms, object_terms)
    args.out.write_text(render_html(data))
    print(f"Wrote {args.out} ({data['total_fields']} fields)")


if __name__ == "__main__":
    main()