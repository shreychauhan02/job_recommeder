import io
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/") + "/match"

st.set_page_config(page_title="Resume Job Matcher", layout="wide")
st.title("Resume Job Matcher")
st.markdown("Upload your resume and get the best-matching jobs from our dataset.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
method = st.radio("Matching method", ["hybrid", "embeddings", "tfidf"], index=0, horizontal=True)

if uploaded_file:
    if st.button("Find Matching Jobs"):
        with st.spinner("Matching..."):
            try:
                resp = requests.post(
                    API_URL,
                    files={"file": (uploaded_file.name, io.BytesIO(uploaded_file.getvalue()), "application/pdf")},
                    data={"method": method, "top_n": 20},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.ConnectionError:
                st.error(f"Cannot connect to API at {API_URL}. Make sure FastAPI is running.")
                st.stop()
            except requests.HTTPError:
                st.error(f"API error: {resp.status_code} — {resp.text[:300]}")
                st.stop()

        results = data["results"]
        if not results:
            st.warning("No matches found. Is the Chroma job index built? (python scripts/index_jobs.py)")
            st.stop()

        df = pd.DataFrame(results)
        df["score_pct"] = df["score"] * 100
        st.subheader(f"Top {len(df)} Matches ({data['method']})")

        show_cols = [
            c
            for c in ["title", "company", "url", "score_pct", "semantic_score", "skill_score", "matched_skills"]
            if c in df.columns
        ]
        st.dataframe(df[show_cols].rename(columns={"score_pct": "Score %", "url": "Link"}), use_container_width=True)

        st.subheader("Match Score Distribution")
        chart_df = df[["title", "company", "score_pct"]].copy()
        chart_df["label"] = chart_df.apply(
            lambda r: f"{r['title'][:35]}{'...' if len(r['title']) > 35 else ''} — {r['company']}", axis=1
        )
        chart_df = chart_df.sort_values("score_pct", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(chart_df) * 0.45)))
        colors = plt.cm.YlGnBu(chart_df["score_pct"] / max(chart_df["score_pct"].max(), 1e-9))
        ax.barh(chart_df["label"], chart_df["score_pct"], color=colors, edgecolor="white", height=0.65)
        ax.set_xlabel("Score %", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("Please upload a PDF resume to get started.")
