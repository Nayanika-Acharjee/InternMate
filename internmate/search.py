"""Hybrid retrieval: BM25 keyword search + embedding (vector) search, fused with Reciprocal Rank Fusion.

Keyword search catches exact terms ("Flutter", "Solidity"); vector search catches meaning
("build chatbots" ~ "LLM agents"). RRF combines the two rankings without needing to calibrate scores.
"""
import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from internmate.embeddings import Embedder
from internmate.models import Job
from internmate.text import tokenize


@dataclass
class SearchFilters:
    remote_only: bool = False
    location: str | None = None
    min_salary: int | None = None
    source: str | None = None


def apply_filters(stmt: Select, f: SearchFilters) -> Select:
    stmt = stmt.where(Job.is_active)
    if f.remote_only:
        stmt = stmt.where(Job.remote)
    if f.location:
        # Remote jobs are location-independent, so they always pass a location filter.
        stmt = stmt.where(or_(Job.remote, Job.location.ilike(f"%{f.location}%")))
    if f.min_salary:
        # Many listings omit pay; keep them rather than silently hiding most of the index.
        stmt = stmt.where(or_(Job.salary_max.is_(None), Job.salary_max >= f.min_salary))
    if f.source:
        stmt = stmt.where(Job.source == f.source)
    return stmt


# ---------------------------------------------------------------- keyword (BM25)

class _KeywordIndex:
    def __init__(self, ids: list[int], docs: list[list[str]]):
        self.ids = ids
        self.bm25 = BM25Okapi(docs) if docs else None


_index: _KeywordIndex | None = None
_index_sig: tuple | None = None
_lock = threading.Lock()


def invalidate_keyword_index() -> None:
    global _index, _index_sig
    with _lock:
        _index, _index_sig = None, None


def _get_keyword_index(session: Session) -> _KeywordIndex:
    """In-memory BM25 over active jobs, rebuilt whenever the index changes."""
    global _index, _index_sig
    sig = tuple(session.execute(
        select(func.count(Job.id), func.max(Job.last_seen_at), func.max(Job.id)).where(Job.is_active)
    ).one())
    with _lock:
        if _index is None or sig != _index_sig:
            rows = session.execute(
                select(Job.id, Job.title, Job.company, Job.tags, Job.description).where(Job.is_active)
            ).all()
            docs = [tokenize(f"{r.title} {r.title} {' '.join(r.tags or [])} {r.company} {r.description}")
                    for r in rows]
            _index, _index_sig = _KeywordIndex([r.id for r in rows], docs), sig
        return _index


def keyword_search(session: Session, query: str, filters: SearchFilters, k: int = 50) -> list[tuple[int, float]]:
    index = _get_keyword_index(session)
    tokens = tokenize(query)
    if index.bm25 is None or not tokens:
        return []
    allowed = set(session.scalars(apply_filters(select(Job.id), filters)))
    scores = index.bm25.get_scores(tokens)
    hits = [(jid, float(s)) for jid, s in zip(index.ids, scores) if s > 0 and jid in allowed]
    return sorted(hits, key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------- vector

def vector_search(session: Session, query_vec: list[float], filters: SearchFilters,
                  k: int = 50) -> list[tuple[int, float]]:
    if session.get_bind().dialect.name == "postgresql":
        distance = Job.embedding.cosine_distance(query_vec)
        stmt = apply_filters(select(Job.id, distance.label("d")), filters).where(
            Job.embedding.is_not(None)).order_by(distance).limit(k)
        return [(r.id, 1.0 - float(r.d)) for r in session.execute(stmt)]

    rows = session.execute(
        apply_filters(select(Job.id, Job.embedding), filters).where(Job.embedding.is_not(None))
    ).all()
    if not rows:
        return []
    matrix = np.array([r.embedding for r in rows], dtype=float)
    sims = matrix @ np.asarray(query_vec, dtype=float)  # vectors are L2-normalised -> cosine
    order = np.argsort(-sims)[:k]
    return [(rows[i].id, float(sims[i])) for i in order]


# ---------------------------------------------------------------- fusion

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, jid in enumerate(ranking):
            fused[jid] = fused.get(jid, 0.0) + 1.0 / (k + rank + 1)
    return fused


@dataclass
class Candidate:
    job: Job
    relevance: float  # RRF score
    keyword_score: float
    vector_sim: float


def hybrid_search(session: Session, embedder: Embedder, query: str,
                  filters: SearchFilters | None = None, k: int = 50) -> list[Candidate]:
    filters = filters or SearchFilters()
    kw = keyword_search(session, query, filters, k)
    vec = vector_search(session, embedder.embed_query(query), filters, k)
    fused = reciprocal_rank_fusion([[i for i, _ in kw], [i for i, _ in vec]])
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    if not top:
        return []
    jobs = {j.id: j for j in session.scalars(select(Job).where(Job.id.in_(top)))}
    kw_d, vec_d = dict(kw), dict(vec)
    return [Candidate(jobs[i], fused[i], kw_d.get(i, 0.0), vec_d.get(i, 0.0)) for i in top]
