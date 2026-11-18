from sqlalchemy import select

from internmate import repo
from internmate.evaluation import EvalProfile, mrr, ndcg_at_k, recall_at_k
from internmate.models import Job
from internmate.ranking import rank
from internmate.recommender import recommend
from internmate.search import SearchFilters, hybrid_search, reciprocal_rank_fusion


def _ext(session, job_id):
    return session.get(Job, job_id).external_id


def test_hybrid_search_finds_exact_skill(seeded, embedder):
    top = hybrid_search(seeded, embedder, "flutter dart mobile app", k=5)
    assert _ext(seeded, top[0].job.id) == "flutter"


def test_filters(seeded, embedder):
    results = hybrid_search(seeded, embedder, "internship", SearchFilters(remote_only=True), k=50)
    assert results and all(c.job.remote for c in results)
    results = hybrid_search(seeded, embedder, "internship", SearchFilters(location="Gurugram"), k=50)
    assert all(c.job.remote or "Gurugram" in c.job.location for c in results)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in {1, 2}
    assert fused[2] > fused[3] and fused[1] > fused[4]


def test_personalisation_promotes_matching_skills(seeded, embedder):
    cands = hybrid_search(seeded, embedder, "engineering internship", k=50)
    profile = EvalProfile(skills=["solidworks", "autocad"])
    top = rank(cands, profile)[0]
    assert top.job.external_id == "mech-cad"
    assert "solidworks" in top.matched_skills


def test_feedback_changes_ranking(seeded, embedder):
    flutter_id = seeded.scalar(select(Job.id).where(Job.external_id == "flutter"))
    before = [r.job.id for r in recommend(seeded, embedder, "developer internship", "u1", k=30)]
    repo.set_feedback(seeded, "u1", flutter_id, liked=True)
    seeded.commit()
    after = [r.job.id for r in recommend(seeded, embedder, "developer internship", "u1", k=30)]
    assert flutter_id in after
    assert flutter_id not in before or after.index(flutter_id) <= before.index(flutter_id)


def test_profile_merges_and_normalises(session):
    repo.update_profile(session, "u2", skills=["Node.js", "Python"])
    user = repo.update_profile(session, "u2", skills=["python", "SQL"], remote_only=True)
    assert user.skills == ["node", "python", "sql"]
    assert user.remote_only is True


def test_metrics():
    ranked, relevant = [5, 1, 7, 2], {1, 2}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert mrr(ranked, relevant) == 0.5
    assert 0 < ndcg_at_k(ranked, relevant, 4) < 1
    assert ndcg_at_k([1, 2], relevant, 2) == 1.0
