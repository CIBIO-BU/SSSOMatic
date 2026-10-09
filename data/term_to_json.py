#!/usr/bin/env python3
"""Convert vocabulary sources into one JSON collection of terms with a common shape:

    "term_name": {
      "value": "<example(s)>",          # source examples ("" if none)
      "description": "<definition>"
    }

Supported sources (auto-detected, or force with --format):
    ggbn      GGBN Data Standard HTML page
    humboldt  Humboldt Extension (eco.tdwg.org/list) HTML page
    nmdc      NMDC LinkML schema YAML (slots section)
    genepio   GenEpiO OWL/RDF-XML (native GENEPIO_ classes)

Usage:
    python terms_to_json.py SOURCE -o terms.json [--format F] [--full]
                                   [--keep-prefix] [--example-sep "; "]
    # format-specific:
    #   humboldt: --skip-iri-terms        drop the ecoiri: duplicates
    #   genepio : --key id                key by GENEPIO_ id instead of label
    #             --include-deprecated    keep owl:deprecated classes

Requires: beautifulsoup4, lxml, pyyaml
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET


def clean(text) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def join(items, sep):
    return sep.join(i for i in (clean(x) for x in items) if i)


def entry(value, description, **extra):
    d = {"value": value, "description": description}
    d.update(extra)
    return d


def add(terms, name, e):
    """Identical repeats are merged silently; conflicting ones keep the first."""
    if name in terms:
        if terms[name] != e:
            print(f"warning: conflicting duplicate {name!r}, keeping first", file=sys.stderr)
        return
    terms[name] = e


# --------------------------------------------------------------------- GGBN
def parse_ggbn(path, a):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "lxml")
    raw, cur = [], None
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 2:
            continue
        label, val = clean(tds[0].get_text(" ")), clean(tds[1].get_text(" "))
        if label == "Concept Name:":
            cur = {"concept": val}
            raw.append(cur)
        elif cur is None:
            continue
        elif label == "Normative URI":
            cur["uri"] = val
        elif label == "Label":
            cur["label"] = val
        elif label == "Definition":
            cur["definition"] = val
        elif label == "Examples":
            cur["examples"] = val
        elif label == "Notes:":
            cur["notes"] = val
        elif label == "":
            m = re.search(r"Required:\s*(\w+).*?Repeatable:\s*(\w+)", val)
            if m:
                cur["required"], cur["repeatable"] = m.groups()
    terms = {}
    for t in raw:
        name = t["concept"] if a.keep_prefix else t["concept"].split(":", 1)[-1]
        extra = {}
        if a.full:
            extra = {k: t.get(k, "") for k in ("label", "uri", "required", "repeatable", "notes")}
        add(terms, name, entry(t.get("examples", ""), t.get("definition", ""), **extra))
    return terms


# ----------------------------------------------------------------- Humboldt
def parse_humboldt(path, a):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "lxml")
    terms = {}
    for table in soup.find_all("table"):
        th = table.find("th")
        if not th or "Term Name" not in th.get_text():
            continue
        m = re.search(r"Term Name\s+(\w+):(\S+)", th.get_text(" "))
        if not m:
            continue
        prefix, local = m.groups()
        if prefix == "ecoiri" and a.skip_iri_terms:
            continue
        rows = {}
        for tr in table.find_all("tr"):
            tds = tr.find_all("td", recursive=False)
            if len(tds) != 2:
                continue
            key = clean(tds[0].get_text(" "))
            if key == "Examples":
                lis = tds[1].find_all("li")
                val = join([li.get_text(" ") for li in lis] if lis else [tds[1].get_text(" ")], a.example_sep)
            else:
                val = clean(tds[1].get_text(" "))
            rows.setdefault(key, val)
        # eco: and ecoiri: share local names, so ecoiri keeps its prefix unless
        # --keep-prefix already keeps everything's.
        name = f"{prefix}:{local}" if (a.keep_prefix or prefix == "ecoiri") else local
        extra = {}
        if a.full:
            extra = {
                "label": rows.get("Label", ""),
                "uri": rows.get("Term IRI", ""),
                "type": rows.get("Type", ""),
                "modified": rows.get("Modified", ""),
                "abcd_equivalence": rows.get("ABCD equivalence", ""),
                "notes": rows.get("Notes", ""),
            }
        add(terms, name, entry(rows.get("Examples", ""), rows.get("Definition", ""), **extra))
    return terms


# --------------------------------------------------------------------- NMDC
def parse_nmdc(path, a):
    import yaml

    schema = yaml.safe_load(open(path, encoding="utf-8"))
    terms = {}
    for name, s in (schema.get("slots") or {}).items():
        s = s or {}
        exs = s.get("examples") or []
        vals = [e.get("value", "") if isinstance(e, dict) else e for e in exs]
        extra = {}
        if a.full:
            extra = {
                "title": s.get("title", ""),
                "range": s.get("range", ""),
                "slot_uri": s.get("slot_uri", ""),
                "required": bool(s.get("required", False)),
                "multivalued": bool(s.get("multivalued", False)),
                "comments": join(s.get("comments") or [], " "),
            }
        add(terms, name, entry(join(vals, a.example_sep), clean(s.get("description", "")), **extra))
    return terms


# ------------------------------------------------------------------ GenEpiO
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OBO = "{http://purl.obolibrary.org/obo/}"


def parse_genepio(path, a):
    rows = []
    for _, e in ET.iterparse(path, events=("end",)):
        if e.tag != OWL + "Class":
            continue
        iri = e.get(RDF + "about") or ""
        gid = iri.rsplit("/", 1)[-1]
        if not gid.startswith("GENEPIO_"):  # skip imported FOODON/CHEBI/... classes
            e.clear()
            continue
        dep = e.find(OWL + "deprecated")
        if dep is not None and (dep.text or "").strip().lower() == "true" and not a.include_deprecated:
            e.clear()
            continue
        txt = lambda tag: [x.text for x in e.findall(tag) if x.text]
        label = txt(RDFS + "label")
        if not label:
            e.clear()
            continue
        rows.append(
            {
                "id": gid,
                "label": clean(label[0]),
                "definition": join(txt(OBO + "IAO_0000115"), " "),
                "examples": txt(OBO + "IAO_0000112"),
                "ui_definition": join(txt(OBO + "GENEPIO_0000162"), " "),
                "iri": iri,
            }
        )
        e.clear()

    # Labels are not unique in GenEpiO: disambiguate collisions with the id.
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    terms = {}
    for r in rows:
        if a.key == "id":
            name = r["id"]
        else:
            name = r["label"] if counts[r["label"]] == 1 else f'{r["label"]} ({r["id"]})'
        extra = {}
        if a.full:
            extra = {"id": r["id"], "iri": r["iri"], "label": r["label"], "user_interface_definition": r["ui_definition"]}
        add(terms, name, entry(join(r["examples"], a.example_sep), r["definition"], **extra))
    return terms


PARSERS = {"ggbn": parse_ggbn, "humboldt": parse_humboldt, "nmdc": parse_nmdc, "genepio": parse_genepio}


def detect(path):
    low = path.lower()
    if low.endswith((".yaml", ".yml")):
        return "nmdc"
    if low.endswith((".owl", ".rdf", ".xml")):
        return "genepio"
    head = open(path, encoding="utf-8", errors="replace").read(400_000)
    if "Humboldt Extension" in head or "rs.tdwg.org/eco" in head:
        return "humboldt"
    if "ggbn" in head.lower():
        return "ggbn"
    sys.exit("could not detect format; pass --format")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("-o", "--output", help="output JSON path (default: stdout)")
    ap.add_argument("--format", choices=PARSERS, help="force source format")
    ap.add_argument("--full", action="store_true", help="add extra source fields to each term")
    ap.add_argument("--keep-prefix", action="store_true", help="keep namespace prefixes on term names (ggbn, humboldt)")
    ap.add_argument("--example-sep", default="; ", help='separator when a term has several examples (default "; ")')
    ap.add_argument("--skip-iri-terms", action="store_true", help="humboldt: drop ecoiri: terms")
    ap.add_argument("--key", choices=["label", "id"], default="label", help="genepio: term key (default label)")
    ap.add_argument("--include-deprecated", action="store_true", help="genepio: keep deprecated classes")
    a = ap.parse_args()

    fmt = a.format or detect(a.source)
    terms = PARSERS[fmt](a.source, a)
    out = json.dumps(terms, indent=2, ensure_ascii=False)
    if a.output:
        open(a.output, "w", encoding="utf-8").write(out + "\n")
        print(f"[{fmt}] wrote {len(terms)} terms to {a.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()