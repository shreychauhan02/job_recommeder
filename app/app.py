import streamlit as st
import requests
import pdfplumber
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

API_URL = "http://localhost:8001/match"

st.set_page_config(page_title="Resume Job Matcher", layout="wide")
st.title("Resume Job Matcher")
st.markdown("Upload your resume and get the best-matching jobs from our dataset.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        resume_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if not resume_text.strip():
        st.error("Could not extract text from the PDF. Try a different file.")
        st.stop()

    st.success(f"Extracted {len(resume_text)} characters from resume.")
    with st.expander("View extracted text"):
        st.text_area("Resume text", resume_text, height=200)

    if st.button("Find Matching Jobs"):
        with st.spinner("Matching..."):
            try:
                resp = requests.post(API_URL, json={"resume_text": resume_text, "top_n": 20})
                resp.raise_for_status()
                data = resp.json()
            except requests.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
                st.stop()

            results = data["results"]

            if not results:
                st.warning("No matches found.")
                st.stop()

            st.subheader(f"Top {len(results)} Matches")

            df = pd.DataFrame(results)
            st.dataframe(
                df[["title", "company", "salary", "match_percent", "link"]].rename(
                    columns={
                        "title": "Job Title",
                        "company": "Company",
                        "salary": "Salary",
                        "match_percent": "Match %",
                        "link": "Link",
                    }
                ),
                use_container_width=True,
            )

            st.subheader("Match Score Distribution")

            chart_df = df[["title", "company", "match_percent"]].copy()
            chart_df["label"] = chart_df.apply(
                lambda r: f"{r['title'][:35]}{'...' if len(r['title']) > 35 else ''} — {r['company']}", axis=1
            )
            chart_df = chart_df.sort_values("match_percent", ascending=True)

            fig, ax = plt.subplots(figsize=(10, max(5, len(chart_df) * 0.45)))
            colors = plt.cm.YlGnBu(chart_df["match_percent"] / chart_df["match_percent"].max())
            bars = ax.barh(chart_df["label"], chart_df["match_percent"], color=colors, edgecolor="white", height=0.65)

            for bar, pct in zip(bars, chart_df["match_percent"]):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold")

            ax.set_xlabel("Match %", fontsize=11)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            ax.set_xlim(0, min(chart_df["match_percent"].max() * 1.15, 100))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("Please upload a PDF resume to get started.")
