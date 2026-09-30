"""Skill vocabulary + simple normalized phrase-matching extraction.

Deliberately avoids heavy NLP libraries: we normalize text once, then
match a curated list of ~150 technical skills as substrings, using word
boundaries where needed to avoid false positives (e.g. 'go' inside
'google').
"""

from __future__ import annotations

import re

# Curated vocabulary of common technical skills (~150).
SKILLS: list[str] = [
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go",
    "Rust", "Ruby", "PHP", "Kotlin", "Swift", "Scala", "R", "MATLAB",
    "Dart", "Shell", "Bash", "SQL", "HTML", "CSS",
    # Backend / frameworks
    "Node.js", "Express", "FastAPI", "Django", "Flask", "Spring Boot",
    "Spring", "Rails", "Laravel", "ASP.NET", ".NET", "Gin", "React",
    "Next.js", "Vue", "Nuxt", "Angular", "Svelte", "Redux",
    # Mobile
    "React Native", "Flutter", "Android", "iOS", "SwiftUI", "Jetpack Compose",
    # Data / ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost",
    "LightGBM", "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
    "Spark", "PySpark", "Hadoop", "Kafka", "Airflow", "dbt", "ETL",
    "Data Analysis", "Data Visualization", "Statistics", "A/B Testing",
    "LLM", "Generative AI", "MLOps", "Feature Engineering", "RAG",
    # Databases
    "SQL Server", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis",
    "Cassandra", "DynamoDB", "Elasticsearch", "Snowflake", "BigQuery",
    "Oracle", "TimescaleDB", "Supabase", "Firebase", "Prisma",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes",
    "Terraform", "Ansible", "Jenkins", "GitHub Actions", "GitLab CI",
    "CI/CD", "Linux", "Nginx", "Serverless", "Lambda", "EC2", "S3",
    "CloudFormation", "Helm", "Prometheus", "Grafana", "Datadog",
    # Tooling / practices
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
    "REST API", "REST", "GraphQL", "gRPC", "WebSocket", "Microservices",
    "OAuth", "JWT", "Caching", "Message Queues", "RabbitMQ",
    "Celery", "Agile", "Scrum", "TDD", "Unit Testing", "Pytest",
    "Jest", "Cypress", "Playwright", "Selenium", "Postman",
    # Web / frontend tooling
    "Tailwind CSS", "Tailwind", "SASS", "Bootstrap", "Webpack", "Vite",
    "Storybook", "Accessibility", "SEO", "PWA",
    # Security / data engineering
    "Authentication", "Authorization", "Encryption", "Penetration Testing",
    "OWASP", "Data Modeling", "Data Warehousing", "Airflow DAGs",
    "Databricks", "Power BI", "Tableau", "Looker", "Excel",
    # Misc technical
    "Regex", "Debugging", "System Design", "Distributed Systems",
    "Algorithms", "Data Structures", "Design Patterns", "Low-Latency",
    "High Availability", "Monitoring", "Logging",
]

# Matching is done on lowercased text with normalized punctuation.
# Skills containing only word characters + '.'/'#'/'+' get a word-boundary
# check so short names ('go', 'c', 'r') don't match inside other words.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = []


def _build_patterns() -> None:
    for skill in SKILLS:
        escaped = re.escape(skill.lower())
        # (\b|(?<=.)) style guard: require non-alphanumeric before/after
        pattern = rf"(?<![a-z0-9+.#]){escaped}(?![a-z0-9+#])"
        _PATTERNS.append((skill, re.compile(pattern)))


_build_patterns()


def extract_skills(text: str) -> set[str]:
    """Return the set of known skills mentioned in the text."""
    normalized = text.lower().replace("\\n", " ").replace("\n", " ")
    found: set[str] = set()
    for skill, pattern in _PATTERNS:
        if pattern.search(normalized):
            found.add(skill)
    return found
