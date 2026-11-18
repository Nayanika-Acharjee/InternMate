"""Text normalisation and skill extraction shared by search, ranking and ingestion."""
import html
import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
    "or", "the", "to", "with", "we", "you", "our", "your", "will", "this", "that", "who", "i",
    "me", "my", "want", "looking", "job", "jobs", "role", "roles", "work",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*")


def strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def tokenize(text: str) -> list[str]:
    tokens = (t.rstrip(".") for t in _TOKEN_RE.findall((text or "").lower()))
    return [t for t in tokens if t and t not in STOPWORDS]


ALIASES = {
    "node.js": "node", "nodejs": "node", "reactjs": "react", "react.js": "react",
    "postgres": "postgresql", "js": "javascript", "ts": "typescript", "ml": "machine learning",
    "golang": "go", "k8s": "kubernetes", "sklearn": "scikit-learn", "genai": "generative ai",
    "gen ai": "generative ai", "llms": "llm", "large language models": "llm",
    "large language model": "llm", "express.js": "express", "expressjs": "express",
    "next.js": "nextjs", "powerbi": "power bi", "rest apis": "rest api", "restful": "rest api",
    "vector db": "vector database", "vector databases": "vector database",
    "spring": "spring boot", "gcp": "gcp", "google cloud": "gcp", "amazon web services": "aws",
    "deep-learning": "deep learning", "retrieval-augmented generation": "rag",
}

# Skills recognised in free text. Ambiguous short words ("go", "c") are only accepted as explicit tags.
SKILL_VOCAB = {
    "python", "java", "javascript", "typescript", "c++", "c#", "rust", "kotlin", "swift", "dart",
    "sql", "react", "nextjs", "node", "express", "fastapi", "django", "flask", "spring boot",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "docker", "kubernetes", "aws",
    "gcp", "azure", "terraform", "linux", "git", "llm", "langchain", "langgraph", "rag",
    "embeddings", "vector database", "nlp", "machine learning", "deep learning", "pytorch",
    "tensorflow", "scikit-learn", "pandas", "numpy", "opencv", "computer vision", "spark",
    "airflow", "kafka", "power bi", "tableau", "excel", "figma", "flutter", "android", "ios",
    "selenium", "solidity", "unity", "autocad", "solidworks", "generative ai", "mlops",
    "rest api", "graphql", "html", "css", "tailwind", "statistics", "prompt engineering",
}
TAG_ONLY_SKILLS = {"go", "c"}

_PHRASES = sorted(SKILL_VOCAB | set(ALIASES), key=len, reverse=True)
_PHRASE_RES = [(p, re.compile(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9+#])")) for p in _PHRASES]


def normalize_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", (skill or "").strip().lower())
    return ALIASES.get(s, s)


def is_known_skill(skill: str) -> bool:
    s = normalize_skill(skill)
    return s in SKILL_VOCAB or s in TAG_ONLY_SKILLS


def extract_skills(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {normalize_skill(p) for p, rx in _PHRASE_RES if rx.search(lowered)}


def job_skills(title: str, description: str, tags: list[str]) -> set[str]:
    from_tags = {normalize_skill(t) for t in tags or [] if is_known_skill(t)}
    return from_tags | extract_skills(f"{title} {description}")
