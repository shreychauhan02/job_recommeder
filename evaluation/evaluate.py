"""Evaluate TF-IDF vs Embeddings vs Hybrid with Precision@5 and MRR.

Loads evaluation/dataset.csv (built by evaluation/build_labels.py) and
evaluation/resumes/*.txt, runs each method, and computes metrics from the
labels. Embeddings/hybrid require Chroma Cloud and the job index
(python scripts/index_jobs.py). Run: python evaluation/evaluate.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend import embeddings, semantic_matcher, vector_store  # noqa: E402
from backend.matching import JobMatcher  # noqa: E402

K = 10  # depth of ranking considered for MRR
DATASET = Path(__file__).resolve().parent / "dataset.csv"
RESUMES_DIR = Path(__file__).resolve().parent / "resumes"


def make_id(title: str, company: str, link: str) -> str:
    """Same scheme as scripts/index_jobs.py."""
    return hashlib.sha1(f"{title}|{company}|{link}".encode("utf-8")).hexdigest()


def tfidf_ids(matcher: JobMatcher, text: str, top_k: int) -> list[str]:
    results = matcher.match_resume(text, top_n=top_k)
    return [make_id(r["title"], r["company"], r["link"]) for r in results]


def chroma_ids(text: str, top_k: int, hybrid: bool) -> list[str]:
    hits = vector_store.query_jobs(embeddings.embed_one(text), top_k=semantic_matcher.RETRIEVAL_TOP_K)
    ranked = semantic_matcher.composite_results(
        text, hits, semantic_weight=1.0, skill_weight=0.0 if not hybrid else semantic_matcher.SKILL_WEIGHT
    )
    if not hybrid:
        # pure semantic: skill_weight 0 already; scores ordered by semantic
        pass
    return [item["id"] for item in ranked[:top_k]]


def metrics(ranked_ids: list[str], positives: set[str]) -> tuple[float, float]:
    top5 = ranked_ids[:5]
    precision_at_5 = sum(1 for job_id in top5 if job_id in positives) / 5.0
    reciprocal_rank = 0.0
    for rank, job_id in enumerate(ranked_ids[:K], start=1):
        if job_id in positives:
            reciprocal_rank = 1.0 / rank
            break
    return precision_at_5, reciprocal_rank


def main() -> None:
    labels = pd.read_csv(DATASET)
    positives = {
        resume_id: set(group[group["relevant"] == 1]["job_id"])
        for resume_id, group in labels.groupby("resume_id")
    }
    texts = {path.stem: path.read_text(encoding="utf-8") for path in sorted(RESUMES_DIR.glob("*.txt"))}
    print(f"Evaluating {len(texts)} synthetic resumes, {len(labels)} labeled pairs.\n")

    matcher = JobMatcher(str(ROOT / "indeed_data.csv"))

    methods = {
        "TF-IDF": lambda text: tfidf_ids(matcher, text, K),
        "Embeddings": lambda text: chroma_ids(text, K, hybrid=False),
        "Hybrid": lambda text: chroma_ids(text, K, hybrid=True),
    }

    rows = []
    for name, rank_fn in methods.items():
        precs, mrrs = [], []
        for resume_id, text in texts.items():
            p, rr = metrics(rank_fn(text), positives[resume_id])
            precs.append(p)
            mrrs.append(rr)
        rows.append((name, sum(precs) / len(precs), sum(mrrs) / len(mrrs)))

    print("| Method | Precision@5 | MRR |")
    print("|--------|--------------|-----|")
    for name, precision, mrr in rows:
        print(f"| {name} | {precision:.3f} | {mrr:.3f} |")


if __name__ == "__main__":
    main()
