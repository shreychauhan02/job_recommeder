"""Index indeed_data.csv job postings into Chroma Cloud.

Deterministic ids are derived from (title, company, link) so re-running
this script upserts the same documents instead of creating duplicates.

Usage: python scripts/index_jobs.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import embeddings, vector_store  # noqa: E402

CSV_PATH = ROOT / "indeed_data.csv"
# CSV columns detected from the dataset: title, company, salary, description, link
# There is no location column, so location is stored as "".


def job_id(title: str, company: str, link: str) -> str:
    """Stable id so indexing is idempotent."""
    raw = f"{title}|{company}|{link}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    print("Reading jobs...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["title", "description"])
    df = df.drop_duplicates(subset=["title", "company", "link"], keep="first")
    df = df.reset_index(drop=True)
    print(f"Found {len(df)} valid jobs (from {len(pd.read_csv(CSV_PATH))} CSV rows).")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        company = str(row["company"]).strip()
        link = "" if pd.isna(row.get("link")) else str(row["link"]).strip()
        description = str(row["description"])
        ids.append(job_id(title, company, link))
        documents.append(f"{title}. {description}")
        metadatas.append(
            {
                "title": title,
                "company": company,
                "location": "",  # dataset has no location column
                "url": link,
                "salary": "" if pd.isna(row.get("salary")) else str(row["salary"]),
            }
        )

    print("Generating embeddings...")
    vectors = embeddings.embed_texts(documents)

    print("Indexing jobs...")
    vector_store.upsert_jobs(ids, documents, vectors, metadatas)

    count = vector_store.get_jobs_collection().count()
    print(f"Successfully indexed {len(ids)} jobs. Collection now holds {count} documents.")


if __name__ == "__main__":
    main()
