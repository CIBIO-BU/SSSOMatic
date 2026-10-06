#!/usr/bin/env python3
"""
SSSOMatic.

Steps:
1. Load and normalize subject and object term records
2. Candidate retrieval (lexical + embedding = composite score)
3. LLM adjudication with structured JSON output

CLI usage:
    python sssomatic.py \\
        --subject-json data/FAIRe_standard.json --subject-source FAIRe \\
        --object-json data/dwc_term_history_filt.json --object-source DwC \\
        --out-dir out/DwC --google-api-key xxxxxxx

Run `python sssomatic.py --help` for the full list of options.
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
import threading

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")


# Data model

@dataclass
class TermRecord:
    """
    Term record loaded directly from a JSON dump:
    { "<term_name>": {"value": "<example value>", "description": "<definition>"}, ... }
    """
    id: str                   # "<source_name>:<term_name>"
    label: str                 # term name
    description: str = ""
    value: str = ""            # an example value
    source: str = ""           # which JSON file this came from

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


# Lexical and embedding scoring

def normalize_label(s: str) -> str:
    """
    Remove special characters, lower case to decrease formatting differences
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


# Candidate retrieval

def retrieve_candidates(
    subj_idx: int,
    subject_terms: list[TermRecord],
    object_terms: list[TermRecord],
    subj_embs: np.ndarray,
    obj_embs: np.ndarray,
    top_k_each: int,
    max_candidates: int,
    composite_floor: float,
) -> list[tuple[int, dict]]:
    subj = subject_terms[subj_idx]
    subj_emb = subj_embs[subj_idx].reshape(1, -1)

    emb_sims = cosine_similarity(subj_emb, obj_embs)[0]
    emb_top = np.argsort(-emb_sims)[:top_k_each]

    lex_scores = np.array([lexical_score(subj, o) for o in object_terms])
    lex_top = np.argsort(-lex_scores)[:top_k_each]

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
        if s["composite"] >= composite_floor
    ][:max_candidates]

    return ranked


# LLM adjudication

SKOS_PREDICATES = ["skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                    "skos:narrowMatch", "skos:relatedMatch", "no_match"]
SYNTACTIC = ["drop_in", "transform", "incompatible"]


class Adjudication(BaseModel):
    predicate: Literal["skos:exactMatch", "skos:closeMatch", "skos:broadMatch",
                        "skos:narrowMatch", "skos:relatedMatch", "no_match"]
    confidence: float = Field(ge=0.0, le=1.0)
    syntactic_compatibility: Literal["drop_in", "transform", "incompatible"]
    justification: str
    transformation_note: Optional[str] = None


PROMPT_TEMPLATE = """You are aligning terms between metadata standards for biodiversity data.

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
  skos:exactMatch   (same concept, interchangeable)
  skos:closeMatch   (very similar, minor scope/wording differences)
  skos:broadMatch   (CANDIDATE is broader than SUBJECT)
  skos:narrowMatch  (CANDIDATE is narrower than SUBJECT)
  skos:relatedMatch (related but not hierarchical)
  no_match         (unrelated)

Also classify syntactic compatibility of values:
  drop_in      (identical value patterns and units)
  transform     (compatible after conversion (unit, format))
  incompatible  (value patterns fundamentally differ)

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


class RateLimiter:
    """Simple thread-safe fixed-interval rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait_for_slot(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time

            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)

            self._last_request_time = time.monotonic()


def adjudicate(
    subj: TermRecord,
    obj: TermRecord,
    client,
    llm_model: str,
    rate_limiter: RateLimiter,
    max_retries: int,
) -> Optional[Adjudication]:

    prompt = build_prompt(subj, obj)

    for attempt in range(max_retries):

        try:

            rate_limiter.wait_for_slot()


            print(
                f"[request] {subj.label} vs {obj.label} "
                f"at {time.strftime('%H:%M:%S')}"
            )

            response = client.models.generate_content(
                model=llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Adjudication,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
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

            if retryable and attempt < max_retries - 1:

                # exponential backoff with jitter
                delay = min(
                    60,
                    (2 ** attempt) + random.uniform(0, 1)
                )

                print(
                    f"[retry {attempt+1}/{max_retries}] "
                    f"{subj.label} vs {obj.label} "
                    f"sleeping {delay:.1f}s"
                )

                time.sleep(delay)
                continue

            print(f"[adjudication failed] {subj.label} vs {obj.label}: {e}")
            return None

    return None


# Adjudication cache

def cache_key(s: TermRecord, o: TermRecord) -> str:
    return hashlib.md5(f"{s.id}||{o.id}||{o.source}".encode()).hexdigest()


def load_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["adj"]
    return cache


def append_cache(cache_path: Path, key: str, adj: dict, subj: TermRecord, obj: TermRecord):
    with cache_path.open("a") as f:
        f.write(json.dumps({
            "key": key,
            "subject_id": subj.id,
            "object_id": obj.id,
            "adj": adj,
        }) + "\n")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align terms between two metadata standards (lexical + "
                     "embedding candidate retrieval, then LLM adjudication).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input data
    parser.add_argument("--data-dir", type=Path, default=Path("./data"),
                         help="Base directory containing the term JSON dumps.")
    parser.add_argument("--subject-json", type=Path, default=None,
                         help="Path to the subject term JSON dump "
                              "(default: <data-dir>/FAIRe_standard.json).")
    parser.add_argument("--object-json", type=Path, default=None,
                         help="Path to the object term JSON dump "
                              "(default: <data-dir>/dwc_term_history_filt.json).")
    parser.add_argument("--subject-source", type=str, default="FAIRe",
                         help="Name of the subject source (used in term IDs).")
    parser.add_argument("--object-source", type=str, default="DwC",
                         help="Name of the object source (used in term IDs).")

    # Output
    parser.add_argument("--out-dir", type=Path, default=Path("./out/DwC"),
                         help="Directory where candidate pairs, cache, and "
                              "adjudicated pairs are written.")

    # Embedding model
    parser.add_argument("--emb-model", type=str, default="BAAI/bge-large-en-v1.5",
                         help="SentenceTransformer model name for embeddings.")

    # Candidate retrieval
    parser.add_argument("--top-k-each", type=int, default=5,
                         help="Top-K candidates to keep per signal (lexical/embedding).")
    parser.add_argument("--max-candidates", type=int, default=5,
                         help="Max candidates to keep per subject term after ranking.")
    parser.add_argument("--composite-floor", type=float, default=0.35,
                         help="Minimum composite score for a candidate to be kept.")
    parser.add_argument( "--skip-candidate-retrieval", action="store_true",
                        help="Skip candidate retrieval and load existing candidate_pairs.csv from --out-dir.", )

    # LLM adjudication
    parser.add_argument("--llm-model", type=str, default="gemma-4-31b-it",
                         help="Gemini/GenAI model name used for adjudication.")
    parser.add_argument("--requests-per-minute", type=int, default=28,
                         help="Rate limit for LLM adjudication requests (free tier default).")
    parser.add_argument("--max-retries", type=int, default=6,
                         help="Max retries per adjudication call on retryable errors.")
    parser.add_argument("--google-api-key", type=str, default=None,
                         help="Google GenAI API key. Defaults to the GOOGLE_API_KEY "
                              "environment variable.")
    parser.add_argument("--skip-adjudication", action="store_true",
                         help="Only run candidate retrieval; skip the LLM adjudication step.")

    # Misc
    parser.add_argument("--adjudicated-tsv-name", type=str,
                         default="all_adjudicated_pairs_gemma.tsv",
                         help="Filename (within --out-dir) for the adjudicated pairs TSV.")
    parser.add_argument("--quiet", action="store_true",
                         help="Reduce console output (skip sanity-check preview).")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "adjudication_cache.jsonl"

    # 1. Load and normalize subject and object term records
    subject_terms = load_json_terms(args.subject_json, args.subject_source)
    print(f"Loaded {len(subject_terms)} {args.subject_source} terms")

    object_terms = load_json_terms(args.object_json, args.object_source)
    print(f"Loaded {len(object_terms)} {args.object_source} terms")

    # 2. Candidate retrieval

    candidate_path = args.out_dir / "candidate_pairs.csv"

    if args.skip_candidate_retrieval:
        if not candidate_path.exists():
            raise SystemExit(
                f"--skip-candidate-retrieval was specified, but "
                f"{candidate_path} does not exist."
            )

        print(f"Loading existing candidates from {candidate_path}")
        candidate_df = pd.read_csv(candidate_path)

        print(f"Loaded {len(candidate_df)} candidate pairs")

    else:
        print(f"Loading embedding model: {args.emb_model}")
        emb_model = SentenceTransformer(args.emb_model)

        print("Computing object embeddings...")
        obj_texts = [t.embedding_text() for t in object_terms]
        obj_embs = emb_model.encode(obj_texts, batch_size=64, show_progress_bar=True,
                                 normalize_embeddings=True)

        print("Computing subject embeddings...")
        subj_texts = [t.embedding_text() for t in subject_terms]
        subj_embs = emb_model.encode(subj_texts, batch_size=64, show_progress_bar=True,
                                  normalize_embeddings=True)

        def get_candidates(subj_idx: int) -> list[tuple[int, dict]]:
            return retrieve_candidates(
                subj_idx, subject_terms, object_terms, subj_embs, obj_embs,
                top_k_each=args.top_k_each,
                max_candidates=args.max_candidates,
                composite_floor=args.composite_floor,
            )

        if not args.quiet and subject_terms:
            print(f"\nSanity check: top candidates for {subject_terms[0].label}")

            demo = get_candidates(0)

            print(f"\nSubject: {subject_terms[0].label}")

            for i, s in demo[:5]:
                print(f"  \u2192 {object_terms[i].source}:{object_terms[i].label}  {s}")

        candidate_pairs = []

        for si, subj in enumerate(subject_terms):
            candidates = get_candidates(si)

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

        candidate_df.to_csv(args.out_dir / "candidate_pairs.csv", index=False)

        print(f"Total Candidates: {candidate_df.shape[0]}")

    # 3. LLM adjudication with structured JSON output
    if args.skip_adjudication:
        print("Skipping LLM adjudication (--skip-adjudication set).")
        return

    api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Google API key found. Please create on in https://aistudio.google.com/api-keys and pass --google-api-key or set the "
            "GOOGLE_API_KEY environment variable, or use --skip-adjudication."
        )

    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(args.requests_per_minute)

    cache = load_cache(cache_path)
    all_pairs = []  # list of dicts, one per (subject, candidate) pair with adjudication

    subject_by_id = {t.id: t for t in subject_terms}
    object_by_id = {t.id: t for t in object_terms}

    for _, row in candidate_df.iterrows():
        subj = subject_by_id[row["subject_id"]]
        obj = object_by_id[row["object_id"]]

        k = cache_key(subj, obj)

        if k in cache:
            adj_dict = cache[k]
        else:
            adj = adjudicate(
                subj, obj, client, args.llm_model, rate_limiter, args.max_retries
            )

            if adj is None:
                continue

            adj_dict = adj.model_dump()

            append_cache(cache_path, k, adj_dict, subj, obj)

            cache[k] = adj_dict

        all_pairs.append({
            "subject_id": subj.id,
            "subject_label": subj.label,
            "object_id": obj.id,
            "object_label": obj.label,
            "object_source": obj.source,
            "lexical": row.get("lexical_score", 0.0),
            "embedding": row.get("embedding_score", 0.0),
            "composite": row["composite_score"],
            **adj_dict,
        })

        if len(all_pairs) % 10 == 0:
            print(
                f"{datetime.now():%H:%M:%S} | "
                f"{len(all_pairs)} adjudications completed"
            )

    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(args.out_dir / args.adjudicated_tsv_name, sep="\t", index=False)
    print(f"Total adjudicated pairs: {len(pairs_df)}")

    if len(pairs_df):
        done = set(zip(pairs_df.subject_id, pairs_df.object_id))
        missing = candidate_df[
            ~candidate_df.set_index(["subject_id", "object_id"]).index.isin(done)
        ]
        print(f"Missing pairs: {len(missing)}")
    else:
        print(f"Missing pairs: {len(candidate_df)}")


if __name__ == "__main__":
    main()
