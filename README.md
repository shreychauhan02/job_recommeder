# Resume Job Matcher

Semantic resume-to-job matching. Upload a PDF resume and rank **680 real Indeed job postings** with three methods: classic **TF-IDF**, **semantic embeddings** (FastEmbed + Chroma Cloud), and a **hybrid** score that also explains matched/missing skills.

## The problem

Keyword matching alone misses semantically relevant jobs ("customer churn modeling" vs "predictive analytics"). This app combines TF-IDF, embeddings and skill-extraction so results are both accurate and **explainable**.

## Architecture

```mermaid
flowchart TD
    A[User] --> B[Next.js Frontend - Tailwind]
    B --> C[Next.js API Routes /api/match /api/explain]
    C --> D[FastAPI Backend]
    D --> E[TF-IDF Engine - scikit-learn]
    D --> F[FastEmbed bge-small-en-v1.5]
    F --> G[Chroma Cloud - collection: jobs]
    D --> H[Skill Extraction - 150+ phrase matching]
    D --> I[Optional LLM /explain - OpenAI-compatible]
    J[Legacy Streamlit UI - app/app.py] --> D
```

## Matching process

```text
Resume PDF → pypdf text extraction
  ├─ tfidf:      TF-IDF + cosine similarity over all jobs
  ├─ embeddings: resume embedding → Chroma top-50 → semantic ranking
  └─ hybrid:     0.7 × semantic_similarity + 0.3 × skill_overlap
→ ranked jobs with matched_skills / missing_skills
```

Weights live in `backend/semantic_matcher.py` (`SEMANTIC_WEIGHT`, `SKILL_WEIGHT`).

**Why FastEmbed instead of sentence-transformers:** ONNX-based, no PyTorch, roughly 10× lower RAM — fits a 512 MB free-tier host.

## Environment variables

Backend reads `.env` (see `.env.example`). Frontend reads `frontend/.env.local`.

| Variable          | Required | Purpose                              |
| ----------------- | -------- | ------------------------------------ |
| CHROMA_API_KEY    | Yes      | Chroma Cloud auth (embeddings/hybrid) |
| CHROMA_TENANT     | Yes      | Chroma tenant id                     |
| CHROMA_DATABASE   | Yes      | Chroma database (created via console or admin API) |
| LLM_API_KEY       | No       | `/explain`; app works without it     |
| BACKEND_URL       | Yes      | FastAPI URL, server-side only in Next.js |
| FRONTEND_ORIGIN   | Yes      | CORS allowlist (never `*`)           |

Secrets are never sent to the browser: the UI calls Next.js API routes, which proxy to FastAPI.

## Local setup

```bash
git clone <repo> && cd job_match
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env        # fill in CHROMA_* values

# 1. Index jobs into Chroma Cloud (idempotent, ~575 docs)
python scripts/index_jobs.py

# 2. Backend
uvicorn backend.main:app --port 8000

# 3. Frontend
cd frontend && npm install && npm run dev         # http://localhost:3000

# (optional) legacy Streamlit UI
streamlit run app/app.py
```

## Evaluation

Synthetic resumes in `evaluation/resumes/` (created only for evaluation, not real people) with rule-based relevance labels in `evaluation/dataset.csv` (regenerate with `python evaluation/build_labels.py`). `evaluate.py` computes metrics from those labels:

```text
Evaluating 10 synthetic resumes, 318 labeled pairs.

| Method | Precision@5 | MRR |
|--------|--------------|-----|
| TF-IDF | 0.440 | 0.636 |
| Embeddings | 0.700 | 0.825 |
| Hybrid | 0.580 | 0.867 |
```

Run it yourself: `python evaluation/evaluate.py`.

## API

- `GET /health` — status + loaded job count
- `POST /match` — multipart: `file` (PDF, ≤5 MB), `method` (`tfidf|embeddings|hybrid`, default hybrid), `top_n`
- `POST /explain` — JSON: `resume_summary`, `title`, `company`, `matched_skills`, `missing_skills`. Returns `{"configured": false, ...}` when `LLM_API_KEY` is absent.

## Deployment

- **Backend → Render**: `Procfile` runs `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`; `runtime.txt` pins Python 3.12. Set `CHROMA_*`, `FRONTEND_ORIGIN` (your Vercel URL) in the Render dashboard.
- **Frontend → Vercel**: import `frontend/`, set `BACKEND_URL` as a **plain (server-side) env var** — do not use `NEXT_PUBLIC_BACKEND_URL`.

## Limitations

- Scraped Indeed dataset (~680 rows, Southeast-Asia-heavy) — small and postings go stale.
- No location column in the dataset; `location` is returned empty.
- Evaluation resumes and labels are **synthetic and rule-based** (title-role matching), so Precision@5/MRR are indicative, not production-grade.
- `bge-small-en-v1.5` is a compact model; long job descriptions are truncated by the embedding window.
- Skill vocabulary is curated (~150 terms); niche tools are missed.
- LLM explanations are optional and untested without a key.

## Demo

`python scripts/index_jobs.py` → `uvicorn backend.main:app --port 8000` → `cd frontend && npm run dev` → open http://localhost:3000, drop a PDF, pick **Hybrid**, click **Find matching jobs**, then **Why this match?** on any card.
