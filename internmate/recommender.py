"""Retrieval + personalised ranking: the single entry point used by the agent, API and evals."""
from sqlalchemy.orm import Session

from internmate import repo
from internmate.embeddings import Embedder
from internmate.models import User
from internmate.ranking import ScoredJob, rank
from internmate.search import SearchFilters, hybrid_search

CANDIDATE_POOL = 50


def recommend(session: Session, embedder: Embedder, query: str, user_id: str | None = None,
              filters: SearchFilters | None = None, k: int = 5) -> list[ScoredJob]:
    profile = session.get(User, user_id) if user_id else None
    filters = filters or SearchFilters()
    if profile is not None and profile.remote_only:
        filters.remote_only = True
    candidates = hybrid_search(session, embedder, query, filters, k=CANDIDATE_POOL)
    liked, disliked = repo.feedback_embeddings(session, user_id) if user_id else ([], [])
    return rank(candidates, profile, liked, disliked)[:k]
