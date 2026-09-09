"""
SSSOMatic.

Steps:
1. Load and normalize subject and object term records
2. Candidate retrieval (lexical + embedding = composite score)
3. LLM adjudication with structured JSON output
"""

import os, json, re, hashlib
from pathlib import Path
from typing import Optional, Literal
import random
import time
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field

DATA_DIR = Path("./data")


# Two JSON term dumps, each shaped as:
# { "<term_name>": {"value": "<example value>", "description": "<definition>"}, ... }
SUBJECT_JSON = DATA_DIR / "FAIRe_standard.json"
# OBJECT_JSON  = DATA_DIR / "mixs_terms_701.json"
OBJECT_JSON  = DATA_DIR / "dwc_term_history_filt.json"

SUBJECT_SOURCE_NAME = "FAIRe"
# OBJECT_SOURCE_NAME  = "MIxS"
OBJECT_SOURCE_NAME  = "DwC"

OUT_DIR = Path("./out/MIxS")
OUT_DIR = Path("./out/DwC")
OUT_DIR.mkdir(exist_ok=True)

@dataclass
class TermRecord:
    """
    Term record loaded directly from a JSON dump:
    { "<term_name>": {"value": "<example value>", "description": "<definition>"}, ... }
    """
    id: str                  # "<source_name>:<term_name>"
    label: str                # term name
    description: str = ""
    value: str = ""           # example value
    source: str = ""          # which JSON file this came from

    def embedding_text(self) -> str:
        return " . ".join(p for p in [self.label, self.description] if p)

def load_json_terms(path: Path, source_name: str) -> list[TermRecord]:
    """
    Loads a term dump shaped as:
    { "<term_name>": {"value": "<example value>", "description": "<definition>"}, ... }
    Builds id as "<source_name>:<term_name>".
    """
    with path.open() as f:
        data = json.load(f)

    records = []
    for term_name, fields in data.items():
        records.append(TermRecord(
            id=f"{source_name}:{term_name}",
            label=term_name,
            description=fields.get("description", ""),
            value=fields.get("value", ""),
            source=source_name,
        ))
    return records

# Load Subject
subject_terms = load_json_terms(SUBJECT_JSON, SUBJECT_SOURCE_NAME)
print(f"Loaded {len(subject_terms)} {SUBJECT_SOURCE_NAME} terms")

# Load Object
object_terms = load_json_terms(OBJECT_JSON, OBJECT_SOURCE_NAME)
print(f"Loaded {len(object_terms)} {OBJECT_SOURCE_NAME} terms")

EMB_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # swap for 'bge-small-en-v1.5' or base if needed
emb_model = SentenceTransformer(EMB_MODEL_NAME)


# Lexical and Embedding Scoring
def normalize_label(s: str) -> str:
    """
    remove special characters, lower case to decrease formatting differences
    """
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lexical_score(a: TermRecord, b: TermRecord) -> float:
    """
    Computes string similarity, using RapidFuzz to compare two strings by breaking them into tokens.
    """
    la, lb = normalize_label(a.label), normalize_label(b.label)
    return fuzz.token_set_ratio(la, lb) / 100.0

print("Computing object embeddings...")
obj_texts = [t.embedding_text() for t in object_terms]
obj_embs = emb_model.encode(obj_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

print("Computing subject embeddings...")
subj_texts = [t.embedding_text() for t in subject_terms]
subj_embs = emb_model.encode(subj_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)


# Candidate Retrieval
TOP_K_EACH = 5   # per signal
MAX_CANDIDATES = 5
COMPOSITE_FLOOR = 0.35

def retrieve_candidates(subj_idx: int) -> list[tuple[int, dict]]:
    subj = subject_terms[subj_idx]
    subj_emb = subj_embs[subj_idx].reshape(1, -1)

    emb_sims = cosine_similarity(subj_emb, obj_embs)[0]
    emb_top = np.argsort(-emb_sims)[:TOP_K_EACH]

    lex_scores = np.array([lexical_score(subj, o) for o in object_terms])
    lex_top = np.argsort(-lex_scores)[:TOP_K_EACH]

    cand_ids: dict[int, dict] = {}
    for i in emb_top:
        cand_ids.setdefault(int(i), {})["embedding"] = float(emb_sims[i])
    for i in lex_top:
        cand_ids.setdefault(int(i), {})["lexical"] = float(lex_scores[i])

    # composite for ranking
    for i, s in cand_ids.items():
        s["composite"] = (
            0.6 * s.get("embedding", 0.0)
            + 0.4 * s.get("lexical", 0.0)
        )

    ranked = sorted(
    cand_ids.items(),
    key=lambda kv: -kv[1]["composite"]
    )

    ranked = [
        (i, s) for i, s in ranked
        if s["composite"] >= COMPOSITE_FLOOR
    ][:MAX_CANDIDATES]

    return ranked

# quick sanity check
print(f"\nSanity check: top candidates for {subject_terms[0].label}")
demo = retrieve_candidates(0)
print(f"\nSubject: {subject_terms[0].label}")
for i, s in demo[:5]:
    print(f"  \u2192 {object_terms[i].source}:{object_terms[i].label}  {s}")

# Check all candidates
candidate_pairs = []

for si, subj in enumerate(subject_terms):
    candidates = retrieve_candidates(si)
    for oi, sig in candidates:
        obj = object_terms[oi]
        candidate_pairs.append({
            "subject_id": subj.id,
            "object_id": obj.id,
            "embedding_score": sig.get("embedding", None),
            "lexical_score": sig.get("lexical", None),
            "composite_score": sig.get("composite", None),
        })

candidate_df = pd.DataFrame(candidate_pairs)
candidate_df.to_csv(OUT_DIR / "candidate_pairs.csv", index=False)

print(f"Total Candidates: {candidate_df.shape[0]}")

# LLM-adjudication
SKOS_PREDICATES = ["skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                   "skos:narrowMatch", "skos:relatedMatch", "no_match"]
SYNTACTIC = ["drop_in", "transform", "incompatible"]

class Adjudication(BaseModel):
    predicate: Literal["skos:exactMatch","skos:closeMatch","skos:broadMatch",
                       "skos:narrowMatch","skos:relatedMatch","no_match"]
    confidence: float = Field(ge=0.0, le=1.0)
    syntactic_compatibility: Literal["drop_in","transform","incompatible"]
    justification: str
    transformation_note: Optional[str] = None

PROMPT_TEMPLATE = """You are aligning terms between metadata standards for eDNA and biodiversity data.

SUBJECT (source: {s_source}):
  ID: {s_id}
  Label: {s_label}
  Description: {s_desc}
  Example value: {s_val}

CANDIDATE (source: {o_source}):
  ID: {o_id}
  Label: {o_label}
  Description: {o_desc}
  Example value: {o_val}

Classify the semantic relationship using ONE SKOS predicate:
  skos:exactMatch   \u2014 same concept, interchangeable
  skos:closeMatch   \u2014 very similar, minor scope/wording differences
  skos:broadMatch   \u2014 CANDIDATE is broader than SUBJECT
  skos:narrowMatch  \u2014 CANDIDATE is narrower than SUBJECT
  skos:relatedMatch \u2014 related but not hierarchical
  no_match          \u2014 unrelated

Also classify syntactic compatibility of values:
  drop_in       \u2014 identical value patterns and units
  transform     \u2014 compatible after conversion (unit, format)
  incompatible  \u2014 value patterns fundamentally differ

Respond with ONLY a JSON object, no prose:
{{"predicate": "...", "confidence": 0.0-1.0, "syntactic_compatibility": "...",
  "justification": "<=2 sentences citing specific fields",
  "transformation_note": null or "brief description"}}
"""

def build_prompt(subj: TermRecord, obj: TermRecord) -> str:
    return PROMPT_TEMPLATE.format(
        s_source=subj.source, s_id=subj.id, s_label=subj.label,
        s_desc=subj.description, s_val=subj.value,
        o_source=obj.source, o_id=obj.id, o_label=obj.label,
        o_desc=obj.description, o_val=obj.value,
    )

# Gemini Model
# GOOGLE AI STUDIO VER
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# LLM_MODEL = "gemini-flash-latest"
LLM_MODEL = "gemma-4-31b-it"

# ===== Rate limiting =====

REQUESTS_PER_MINUTE = 25          # free tier limit
SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE #+ 0.5

_last_request_time = 0.0


def wait_for_slot():
    global _last_request_time

    now = time.monotonic()
    elapsed = now - _last_request_time

    if elapsed < SECONDS_PER_REQUEST:
        time.sleep(SECONDS_PER_REQUEST - elapsed)

    _last_request_time = time.monotonic()

def adjudicate(subj: TermRecord, obj: TermRecord) -> Optional[Adjudication]:

    prompt = build_prompt(subj, obj)

    # print(prompt)

    MAX_RETRIES = 6

    for attempt in range(MAX_RETRIES):

        try:

            wait_for_slot()

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Adjudication,
                },
            )

            return response.parsed

        except Exception as e:

            msg = str(e)

            retryable = any(x in msg for x in [
                "429",
                "RESOURCE_EXHAUSTED",
                "503",
                "500",
                "UNAVAILABLE",
            ])

            if retryable and attempt < MAX_RETRIES - 1:

                # exponential backoff with jitter
                delay = min(
                    60,
                    (2 ** attempt) + random.uniform(0, 1)
                )

                print(
                    f"[retry {attempt+1}/{MAX_RETRIES}] "
                    f"{subj.label} vs {obj.label} "
                    f"sleeping {delay:.1f}s"
                )

                time.sleep(delay)
                continue

            print(f"[adjudication failed] {subj.label} vs {obj.label}: {e}")
            return None

    return None

CACHE_PATH = OUT_DIR / "adjudication_cache.jsonl"

def cache_key(s: TermRecord, o: TermRecord) -> str:
    return hashlib.md5(f"{s.id}||{o.id}||{o.source}".encode()).hexdigest()

def load_cache() -> dict:
    cache = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["adj"]
    return cache

def append_cache(key: str, adj: dict, subj: TermRecord, obj: TermRecord):
    with CACHE_PATH.open("a") as f:
        f.write(json.dumps({
            "key": key,
            "subject_id": subj.id,
            "object_id": obj.id,
            "adj": adj,
        }) + "\n")

cache = load_cache()
all_pairs = []   # list of dicts, one per (subject, candidate) pair with adjudication

for si, subj in enumerate(subject_terms):
    candidates = retrieve_candidates(si)
    for oi, sig in candidates:
        obj = object_terms[oi]
        k = cache_key(subj, obj)
        if k in cache:
            adj_dict = cache[k]
        else:
            adj = adjudicate(subj, obj)
            if adj is None:
                continue
            adj_dict = adj.model_dump()
            append_cache(k, adj_dict, subj, obj)
            cache[k] = adj_dict
        all_pairs.append({
            "subject_id": subj.id,
            "subject_label": subj.label,
            "object_id": obj.id,
            "object_label": obj.label,
            "object_source": obj.source,
            "lexical": sig.get("lexical", 0.0),
            "embedding": sig.get("embedding", 0.0),
            "composite": sig["composite"],
            **adj_dict,
        })
    if len(all_pairs) % 10 == 0:
        print(
            f"{datetime.now():%H:%M:%S} | "
            f"{len(all_pairs)} adjudications completed"
        )

pairs_df = pd.DataFrame(all_pairs)
pairs_df.to_csv(OUT_DIR / "all_adjudicated_pairs_gemma.tsv", sep="\t", index=False)
print(f"Total adjudicated pairs: {len(pairs_df)}")

done = set(zip(pairs_df.subject_id, pairs_df.object_id))
missing = candidate_df[
    ~candidate_df.set_index(["subject_id", "object_id"]).index.isin(done)
]

print(f"Missing pairs: {len(missing)}")