"""FastAPI backend: TF-IDF / embeddings / hybrid resume-job matching."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend import embeddings, explain, semantic_matcher, vector_store  # noqa: E402
from backend.matching import JobMatcher  # noqa: E402

Method = Literal["tfidf", "embeddings", "hybrid"]

app = FastAPI(title="Resume Job Matcher API")

CSV_PATH = Path(__file__).resolve().parent.parent / "indeed_data.csv"
matcher: JobMatcher | None = None

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model() -> None:
    global matcher
    matcher = JobMatcher(str(CSV_PATH))


class JobResult(BaseModel):
    title: str
    company: str
    location: str = ""
    url: str = ""
    score: float
    semantic_score: float | None = None
    skill_score: float | None = None
    matched_skills: list[str] = []
    missing_skills: list[str] = []


class MatchResponse(BaseModel):
    results: list[JobResult]
    method: str
    total_jobs: int


class ExplainRequest(BaseModel):
    resume_summary: str
    title: str
    company: str = ""
    matched_skills: list[str] = []
    missing_skills: list[str] = []


class ExplainResponse(BaseModel):
    configured: bool
    message: str


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
    return text


@app.post("/match", response_model=MatchResponse)
def match_resume(
    file: UploadFile = File(..., description="Resume PDF"),
    method: Method = Form("hybrid"),
    top_n: int = Form(20),
) -> MatchResponse:
    if file.content_type not in ("application/pdf", None) or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are accepted.")
    resume_text = _extract_pdf_text(file.file.read())

    if method == "tfidf":
        raw = matcher.match_resume(resume_text, top_n=top_n)
        results = [
            JobResult(
                title=item["title"],
                company=item["company"],
                location=item["location"],
                url=item["link"],
                score=item["match_percent"] / 100.0,
                skill_score=None,
                semantic_score=None,
                matched_skills=item["matched_skills"],
                missing_skills=item["missing_skills"],
            )
            for item in raw
        ]
        return MatchResponse(results=results, method=method, total_jobs=len(matcher.df))

    # embeddings / hybrid: resume embedding -> Chroma top 50 -> ranking
    try:
        hits = vector_store.query_jobs(
            embeddings.embed_one(resume_text),
            top_k=semantic_matcher.RETRIEVAL_TOP_K,
        )
    except KeyError as exc:
        raise HTTPException(status_code=503, detail="Chroma Cloud credentials are not configured.") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Chroma Cloud query failed: {exc}") from exc

    if method == "embeddings":
        # Pure semantic ranking: skill fields are shown but do not affect score.
        ranked = semantic_matcher.composite_results(
            resume_text, hits, semantic_weight=1.0, skill_weight=0.0
        )
    else:
        ranked = semantic_matcher.composite_results(resume_text, hits)
    results = [JobResult(**item) for item in ranked[:top_n]]
    return MatchResponse(results=results, method=method, total_jobs=len(hits))


@app.post("/explain", response_model=ExplainResponse)
def explain_match(req: ExplainRequest) -> ExplainResponse:
    return ExplainResponse(**explain.generate(req))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "jobs_loaded": len(matcher.df) if matcher else 0}
