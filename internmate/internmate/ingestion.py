"""Job ingestion: fetch from sources, normalise, deduplicate, embed, upsert, expire stale listings.

Re-embedding only happens when a listing's content hash changes, so daily refreshes are cheap.
"""
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from internmate.config import get_settings
from internmate.db import utcnow
from internmate.embeddings import Embedder
from internmate.models import IngestionRun, Job
from internmate.text import strip_html

log = logging.getLogger(__name__)

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_jobs.json"


@dataclass
class JobIn:
    source: str
    external_id: str
    title: str
    company: str
    location: str = ""
    remote: bool = False
    salary_min: int | None = None
    salary_max: int | None = None
    salary_text: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""
    url: str = ""
    posted_at: datetime | None = None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def dedupe_key(job: JobIn) -> str:
    """Same title at the same company = same listing, even if posted on several boards."""
    return hashlib.sha256(f"{_norm(job.title)}|{_norm(job.company)}".encode()).hexdigest()


def content_hash(job: JobIn) -> str:
    payload = json.dumps(
        [job.title, job.company, job.location, job.remote, job.salary_min, job.salary_max,
         sorted(job.tags), job.description],
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def embedding_text(job: JobIn) -> str:
    remote = "Remote" if job.remote else ""
    return (f"{job.title} at {job.company}. {job.location} {remote}. "
            f"Skills: {', '.join(job.tags)}. {job.description[:2000]}")


# ---------------------------------------------------------------- sources

def load_sample(path: Path = SAMPLE_PATH) -> list[JobIn]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [JobIn(source="sample", **row) for row in rows]


def fetch_remotive(limit: int = 300) -> list[JobIn]:
    resp = httpx.get("https://remotive.com/api/remote-jobs", params={"limit": limit}, timeout=30)
    resp.raise_for_status()
    jobs = []
    for r in resp.json().get("jobs", []):
        posted = r.get("publication_date")
        jobs.append(JobIn(
            source="remotive",
            external_id=str(r["id"]),
            title=r.get("title", ""),
            company=r.get("company_name", ""),
            location=r.get("candidate_required_location") or "Worldwide",
            remote=True,
            salary_text=r.get("salary") or "",
            tags=list(r.get("tags") or []) + ([r["category"]] if r.get("category") else []),
            description=strip_html(r.get("description", "")),
            url=r.get("url", ""),
            posted_at=datetime.fromisoformat(posted).replace(tzinfo=None) if posted else None,
        ))
    return jobs


def fetch_arbeitnow(pages: int = 3) -> list[JobIn]:
    jobs = []
    with httpx.Client(timeout=30) as client:
        for page in range(1, pages + 1):
            resp = client.get("https://www.arbeitnow.com/api/job-board-api", params={"page": page})
            resp.raise_for_status()
            for r in resp.json().get("data", []):
                created = r.get("created_at")
                jobs.append(JobIn(
                    source="arbeitnow",
                    external_id=r["slug"],
                    title=r.get("title", ""),
                    company=r.get("company_name", ""),
                    location=r.get("location", ""),
                    remote=bool(r.get("remote")),
                    tags=list(r.get("tags") or []),
                    description=strip_html(r.get("description", "")),
                    url=r.get("url", ""),
                    posted_at=(datetime.fromtimestamp(created, UTC).replace(tzinfo=None)
                               if created else None),
                ))
    return jobs


SOURCES = {"sample": load_sample, "remotive": fetch_remotive, "arbeitnow": fetch_arbeitnow}


# ---------------------------------------------------------------- upsert / expire

def _apply(row: Job, job: JobIn) -> None:
    for name in ("source", "external_id", "title", "company", "location", "remote", "salary_min",
                 "salary_max", "salary_text", "tags", "description", "url", "posted_at"):
        setattr(row, name, getattr(job, name))


def upsert_jobs(session: Session, jobs: list[JobIn], embedder: Embedder,
                now: datetime | None = None) -> Counter:
    now = now or utcnow()
    batch: dict[str, JobIn] = {}
    for job in jobs:
        if job.title and job.company:
            batch.setdefault(dedupe_key(job), job)

    existing: dict[str, Job] = {}
    keys = list(batch)
    for i in range(0, len(keys), 500):
        for row in session.scalars(select(Job).where(Job.dedupe_key.in_(keys[i:i + 500]))):
            existing[row.dedupe_key] = row

    stats: Counter = Counter()
    to_embed: list[tuple[Job, JobIn]] = []
    for key, job in batch.items():
        h = content_hash(job)
        row = existing.get(key)
        if row is None:
            row = Job(dedupe_key=key, content_hash=h, first_seen_at=now)
            _apply(row, job)
            session.add(row)
            to_embed.append((row, job))
            stats["inserted"] += 1
        elif row.content_hash != h:
            _apply(row, job)
            row.content_hash = h
            to_embed.append((row, job))
            stats["updated"] += 1
        else:
            stats["unchanged"] += 1
        row.last_seen_at = now
        row.is_active = True

    if to_embed:
        vectors = embedder.embed_documents([embedding_text(j) for _, j in to_embed])
        for (row, _), vec in zip(to_embed, vectors):
            row.embedding = vec
    session.flush()
    return stats


def expire_stale(session: Session, sources: list[str], run_started: datetime,
                 max_age_days: int, now: datetime | None = None) -> int:
    """Deactivate listings that disappeared from a successfully fetched source, or are too old."""
    now = now or utcnow()
    expired = 0
    if sources:
        expired += session.execute(
            update(Job)
            .where(Job.is_active, Job.source.in_(sources), Job.last_seen_at < run_started)
            .values(is_active=False)
        ).rowcount
    expired += session.execute(
        update(Job)
        .where(Job.is_active, Job.posted_at.is_not(None),
               Job.posted_at < now - timedelta(days=max_age_days))
        .values(is_active=False)
    ).rowcount
    return expired


def run_ingestion(session: Session, embedder: Embedder, sources: list[str]) -> IngestionRun:
    run = IngestionRun(sources=sources, started_at=utcnow(), errors=[])
    session.add(run)
    totals: Counter = Counter()
    succeeded: list[str] = []
    for name in sources:
        try:
            jobs = SOURCES[name]()
            totals += upsert_jobs(session, jobs, embedder, now=run.started_at)
            succeeded.append(name)
        except Exception as exc:  # one failing source must not break the whole refresh
            log.exception("source %s failed", name)
            run.errors = [*run.errors, f"{name}: {exc}"]

    # Only expire sources we actually fetched; a failed API call must not wipe its jobs.
    run.expired = expire_stale(session, succeeded, run.started_at, get_settings().job_max_age_days)
    run.inserted, run.updated, run.unchanged = totals["inserted"], totals["updated"], totals["unchanged"]
    run.finished_at = utcnow()
    session.commit()

    from internmate.search import invalidate_keyword_index
    invalidate_keyword_index()
    return run
