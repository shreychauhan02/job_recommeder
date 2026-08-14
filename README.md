# Resume Job Matcher

Upload a PDF resume and get ranked job matches from an Indeed dataset using TF-IDF + cosine similarity.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Start the backend API first, then the Streamlit frontend:

```bash
# Terminal 1
uvicorn backend.main:app --reload --port 8000

# Terminal 2
streamlit run app/app.py
```

Open http://localhost:8501, upload a PDF resume, and click "Find Matching Jobs".

## Architecture

- `backend/matching.py` - TF-IDF vectorizer + cosine similarity matching engine
- `backend/main.py` - FastAPI server with `/match` and `/health` endpoints
- `app/app.py` - Streamlit UI (PDF upload, calls API, displays ranked results)
- `indeed_data.csv` - ~680 scraped Indeed job postings
