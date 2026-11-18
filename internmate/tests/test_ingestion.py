from datetime import timedelta

from sqlalchemy import func, select

from internmate.db import utcnow
from internmate.ingestion import JobIn, expire_stale, load_sample, upsert_jobs
from internmate.models import Job


def _job(**kw):
    base = dict(source="test", external_id="1", title="Python Intern", company="Acme",
                description="Build APIs in Python", tags=["python"])
    return JobIn(**(base | kw))


def test_sample_loads_and_is_idempotent(session, embedder):
    jobs = load_sample()
    first = upsert_jobs(session, jobs, embedder)
    assert first["inserted"] == len(jobs)
    second = upsert_jobs(session, jobs, embedder)
    assert second["unchanged"] == len(jobs) and second["inserted"] == 0
    assert session.scalar(select(func.count(Job.id))) == len(jobs)


def test_cross_source_duplicates_are_merged(session, embedder):
    stats = upsert_jobs(session, [_job(), _job(source="other", external_id="x", title="python intern ")], embedder)
    assert stats["inserted"] == 1


def test_changed_content_is_reembedded(session, embedder):
    upsert_jobs(session, [_job()], embedder)
    before = list(session.scalar(select(Job)).embedding)
    stats = upsert_jobs(session, [_job(description="Build ML models in PyTorch")], embedder)
    assert stats["updated"] == 1
    assert list(session.scalar(select(Job)).embedding) != before


def test_missing_and_old_jobs_expire(session, embedder):
    t0 = utcnow() - timedelta(days=1)
    upsert_jobs(session, [_job(), _job(external_id="2", title="Java Intern")], embedder, now=t0)
    run_start = utcnow()
    upsert_jobs(session, [_job()], embedder, now=run_start)  # Java intern disappeared
    upsert_jobs(session, [_job(source="old", title="Ancient", posted_at=utcnow() - timedelta(days=90))],
                embedder, now=run_start)
    expired = expire_stale(session, ["test"], run_start, max_age_days=30)
    session.flush()
    active = set(session.scalars(select(Job.title).where(Job.is_active)))
    assert expired == 2
    assert active == {"Python Intern"}


def test_failed_source_does_not_expire(session, embedder):
    t0 = utcnow() - timedelta(days=1)
    upsert_jobs(session, [_job()], embedder, now=t0)
    assert expire_stale(session, [], utcnow(), max_age_days=30) == 0
