"""Generate evaluation/dataset.csv labels from the CSV + synthetic resumes.

Labels are rule-based: a job is relevant (=1) for a resume when its title
matches the role the synthetic resume was written for. Positives therefore
favor keyword methods by construction — this is documented in the README
as a limitation. Negatives (=0) are sampled from other role families.

job_id uses the same sha1(title|company|link) scheme as scripts/index_jobs.py.
Run: python evaluation/build_labels.py
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "indeed_data.csv"


def job_id(title: str, company: str, link: str) -> str:
    return hashlib.sha1(f"{title}|{company}|{link}".encode("utf-8")).hexdigest()


ROLE_RULES: dict[str, list[str]] = {
    "resume_01": ["data scientist"],
    "resume_02": ["data engineer"],
    "resume_03": ["ui developer", "web developer", "front end", "frontend"],
    "resume_04": ["backend", "back end"],
    "resume_05": ["data analyst"],
    "resume_06": ["ui/ux", "ui / ux"],
    "resume_07": ["devops", "system administrator"],
    "resume_08": ["machine learning"],
    "resume_09": ["full stack"],
    "resume_10": ["mobile"],
}


def main() -> None:
    df = pd.read_csv(CSV_PATH).dropna(subset=["title", "description"])
    df = df.drop_duplicates(subset=["title", "company", "link"], keep="first").reset_index(drop=True)

    rows: list[dict] = []
    rng = random.Random(42)
    for resume, keywords in ROLE_RULES.items():
        title_lower = df["title"].astype(str).str.lower()
        mask = pd.Series(False, index=df.index)
        for keyword in keywords:
            mask |= title_lower.str.contains(keyword, na=False)
        positives = df[mask]
        for _, row in positives.iterrows():
            rows.append(
                {
                    "resume_id": resume,
                    "job_id": job_id(str(row["title"]).strip(), str(row["company"]).strip(), str(row.get("link", "")).strip()),
                    "relevant": 1,
                }
            )
        negatives = df[~mask].sample(n=3, random_state=rng.randint(0, 10_000))
        for _, row in negatives.iterrows():
            rows.append(
                {
                    "resume_id": resume,
                    "job_id": job_id(str(row["title"]).strip(), str(row["company"]).strip(), str(row.get("link", "")).strip()),
                    "relevant": 0,
                }
            )
        print(f"{resume}: {len(positives)} positives, 3 negatives")

    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).resolve().parent / "dataset.csv", index=False)
    print(f"Wrote dataset.csv with {len(out)} labeled pairs.")


if __name__ == "__main__":
    main()
