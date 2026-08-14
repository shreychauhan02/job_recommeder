import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cleaningText(text):
    text = str(text)
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\n')
    text = text.replace('\\r', '\n')
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(' nan ', ' ', text)
    text = re.sub(r'\\x[0-9a-z]{2}', r' ', text)
    text = re.sub(r'[0-9]{2,}', r' ', text)
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'xx+', r' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    return text.strip().lower()


class JobMatcher:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(subset=['description'], inplace=True)
        self.df.drop_duplicates(subset=['title', 'company'], inplace=True)
        self.df['cleaned'] = self.df['description'].apply(cleaningText)
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned'])

    def match_resume(self, resume_text: str, top_n: int = 20):
        cleaned = cleaningText(resume_text)
        resume_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(resume_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'title': row['title'],
                'company': row['company'],
                'salary': str(row.get('salary', 'N/A')),
                'match_percent': round(float(similarities[idx]) * 100, 2),
                'link': row.get('link', ''),
            })
        return results
