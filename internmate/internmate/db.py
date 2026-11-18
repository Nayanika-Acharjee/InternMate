from datetime import UTC, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from internmate.config import get_settings

settings = get_settings()

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
IS_POSTGRES = engine.dialect.name == "postgresql"
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Naive UTC timestamp (stored consistently across SQLite and Postgres)."""
    return datetime.now(UTC).replace(tzinfo=None)


def init_db() -> None:
    from internmate import models  # noqa: F401  (registers tables)

    if IS_POSTGRES:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    if IS_POSTGRES:
        with engine.begin() as conn:
            # Approximate nearest-neighbour index for cosine similarity search.
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_jobs_embedding_hnsw "
                "ON jobs USING hnsw (embedding vector_cosine_ops)"
            ))
