"""Semantic + skill composite scoring for embedding-based matching.

Chroma returns cosine *distance* in [0, 2]; similarity = 1 - distance / 2.
Final ranking uses the configurable weights below (single source of truth).
"""

from __future__ import annotations

from backend import skills
from backend.vector_store import JobMatch

SEMANTIC_WEIGHT = 0.7
SKILL_WEIGHT = 0.3
RETRIEVAL_TOP_K = 50


def cosine_similarity_from_distance(distance: float) -> float:
    """Convert Chroma cosine distance [0,2] to similarity [-1,1] clamped to [0,1]."""
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def skill_overlap(resume_skills: set[str], job_skills: set[str]) -> tuple[float, list[str], list[str]]:
    """Fraction of job skills the resume also has, plus matched/missing lists."""
    if not job_skills:
        return 0.0, [], []
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    return len(matched) / len(job_skills), sorted(matched), sorted(missing)


def composite_results(
    resume_text: str,
    hits: list[JobMatch],
    semantic_weight: float = SEMANTIC_WEIGHT,
    skill_weight: float = SKILL_WEIGHT,
) -> list[dict]:
    """Rank Chroma hits by weighted semantic similarity + skill overlap."""
    resume_skills = skills.extract_skills(resume_text)
    scored: list[dict] = []
    seen_titles: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.metadata.get("title", ""), hit.metadata.get("company", ""))
        if key in seen_titles:  # no duplicate results
            continue
        seen_titles.add(key)
        semantic = cosine_similarity_from_distance(hit.distance)
        job_skill_set = skills.extract_skills(hit.document)
        overlap, matched, missing = skill_overlap(resume_skills, job_skill_set)
        scored.append(
            {
                "id": hit.id,
                "title": hit.metadata.get("title", ""),
                "company": hit.metadata.get("company", ""),
                "location": hit.metadata.get("location", ""),
                "url": hit.metadata.get("url", ""),
                "salary": hit.metadata.get("salary", ""),
                "score": round(semantic_weight * semantic + skill_weight * overlap, 4),
                "semantic_score": round(semantic, 4),
                "skill_score": round(overlap, 4),
                "matched_skills": matched,
                "missing_skills": missing,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored
