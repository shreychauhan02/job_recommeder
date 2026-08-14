import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from backend.matching import JobMatcher

app = FastAPI(title="Resume Job Matcher API")

CSV_PATH = Path(__file__).resolve().parent.parent / "indeed_data.csv"
matcher: JobMatcher | None = None


@app.on_event("startup")
def load_model():
    global matcher
    matcher = JobMatcher(str(CSV_PATH))


class MatchRequest(BaseModel):
    resume_text: str
    top_n: int = 20


class JobResult(BaseModel):
    title: str
    company: str
    salary: str
    match_percent: float
    link: str


class MatchResponse(BaseModel):
    results: list[JobResult]
    total_jobs: int


@app.post("/match", response_model=MatchResponse)
def match_resume(req: MatchRequest):
    results = matcher.match_resume(req.resume_text, top_n=req.top_n)
    return MatchResponse(results=results, total_jobs=len(results))


@app.get("/health")
def health():
    return {"status": "ok", "jobs_loaded": len(matcher.df) if matcher else 0}
