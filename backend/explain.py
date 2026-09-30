"""Optional LLM match explanations via an OpenAI-compatible API.

If LLM_API_KEY is not set the endpoint does not crash: it returns
{"configured": false, ...} and the rest of the app keeps working.
"""

from __future__ import annotations

import os

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_PROMPT_TEMPLATE = (
    "You are a career coach. In 3-4 sentences, explain why this candidate is a "
    "match for the job, which skills overlap, which skills are missing, and what "
    "the candidate could learn to improve. Be concrete and concise.\n\n"
    "Candidate resume summary:\n{resume_summary}\n\n"
    "Job: {title} at {company}\n"
    "Matched skills: {matched}\n"
    "Missing skills: {missing}"
)


def generate(payload) -> dict:
    """Generate an explanation; degrade gracefully when unconfigured."""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        return {"configured": False, "message": "LLM explanation is not configured."}

    import requests

    prompt = _PROMPT_TEMPLATE.format(
        resume_summary=payload.resume_summary[:2000],
        title=payload.title,
        company=payload.company,
        matched=", ".join(payload.matched_skills) or "none detected",
        missing=", ".join(payload.missing_skills[:10]) or "none detected",
    )
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
            },
            timeout=30,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"].strip()
        return {"configured": True, "message": message}
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "message": f"LLM call failed: {exc}"}
