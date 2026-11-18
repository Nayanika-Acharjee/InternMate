import os
import tempfile

# Must be set before any internmate import: settings and engine are created at import time.
_DB = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB}"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["GOOGLE_API_KEY"] = ""

import pytest  # noqa: E402

from internmate.db import Base, SessionLocal, engine, init_db  # noqa: E402
from internmate.embeddings import HashEmbedder  # noqa: E402
from internmate.ingestion import load_sample, upsert_jobs  # noqa: E402
from internmate.search import invalidate_keyword_index  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    init_db()
    yield
    Base.metadata.drop_all(engine)
    invalidate_keyword_index()


@pytest.fixture
def session():
    with SessionLocal() as s:
        yield s


@pytest.fixture
def embedder():
    return HashEmbedder(768)


@pytest.fixture
def seeded(session, embedder):
    upsert_jobs(session, load_sample(), embedder)
    session.commit()
    return session
