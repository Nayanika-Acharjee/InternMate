"""Offline evaluation of retrieval and ranking quality.

Compares four retrieval strategies on a labelled query set (evals/queries.json) against the
bundled sample jobs, reporting Recall@5, Recall@10, MRR and nDCG@10. Optionally runs an
LLM-as-judge pass that scores the top-3 personalised results for each query (1-5).
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from internmate.embeddings import Embedder
from internmate.models import Job
from internmate.ranking import rank
from internmate.schemas import JudgeScore
from internmate.search import SearchFilters, hybrid_search, keyword_search, vector_search

QUERIES_PATH = Path(__file__).resolve().parent.parent / "evals" / "queries.json"
METHODS = ("keyword", "vector", "hybrid", "hybrid+personalised")


@dataclass
class EvalProfile:
    skills: list[str] = field(default_factory=list)
    preferred_roles: list[str] = field(default_factory=list)
    preferred_locations: list[str] = field(default_factory=list)
    remote_only: bool = False
    min_salary: int | None = None


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list, relevant: set) -> float:
    return next((1.0 / (i + 1) for i, x in enumerate(ranked) if x in relevant), 0.0)


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def _ranked_ids(session: Session, embedder: Embedder, method: str, query: str,
                profile: EvalProfile | None, filters: SearchFilters) -> list[int]:
    if method == "keyword":
        return [i for i, _ in keyword_search(session, query, filters, k=20)]
    if method == "vector":
        return [i for i, _ in vector_search(session, embedder.embed_query(query), filters, k=20)]
    candidates = hybrid_search(session, embedder, query, filters, k=20)
    if method == "hybrid":
        return [c.job.id for c in candidates]
    return [s.job.id for s in rank(candidates, profile)]


def run_eval(session: Session, embedder: Embedder, queries_path: Path = QUERIES_PATH,
             judge_llm=None) -> dict:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    id_map = dict(session.execute(select(Job.external_id, Job.id).where(Job.source == "sample")).all())
    if not id_map:
        raise RuntimeError("No sample jobs found. Run: python -m internmate.cli ingest --source sample")
    filters = SearchFilters(source="sample")

    totals = {m: {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0} for m in METHODS}
    per_query = []
    for q in queries:
        relevant = {id_map[x] for x in q["relevant"] if x in id_map}
        profile = EvalProfile(**q["profile"]) if q.get("profile") else None
        row = {"id": q["id"], "query": q["query"]}
        for m in METHODS:
            ranked = _ranked_ids(session, embedder, m, q["query"], profile, filters)
            t = totals[m]
            t["recall@5"] += recall_at_k(ranked, relevant, 5)
            t["recall@10"] += recall_at_k(ranked, relevant, 10)
            t["mrr"] += mrr(ranked, relevant)
            t["ndcg@10"] += ndcg_at_k(ranked, relevant, 10)
            row[m] = round(recall_at_k(ranked, relevant, 5), 2)
        per_query.append(row)

    n = len(queries)
    summary = {m: {k: round(v / n, 3) for k, v in t.items()} for m, t in totals.items()}
    result = {"num_queries": n, "summary": summary, "per_query_recall@5": per_query}
    if judge_llm is not None:
        result["llm_judge_avg_top3"] = _judge(session, embedder, queries, filters, judge_llm)
    return result


def _judge(session, embedder, queries, filters, llm) -> float:
    judge = llm.with_structured_output(JudgeScore)
    scores = []
    for q in queries:
        profile = EvalProfile(**q["profile"]) if q.get("profile") else None
        for sj in rank(hybrid_search(session, embedder, q["query"], filters, k=20), profile)[:3]:
            j = sj.job
            verdict = judge.invoke(
                "You are grading a job search engine. Rate how well this job matches the search "
                f"query{' and candidate profile' if profile else ''}.\n\nQuery: {q['query']}\n"
                f"Profile: {json.dumps(q.get('profile'))}\n\nJob: {j.title} at {j.company}\n"
                f"{j.description[:1200]}"
            )
            scores.append(verdict.score)
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def to_markdown(result: dict) -> str:
    lines = ["| Method | Recall@5 | Recall@10 | MRR | nDCG@10 |", "|---|---|---|---|---|"]
    for m, s in result["summary"].items():
        lines.append(f"| {m} | {s['recall@5']} | {s['recall@10']} | {s['mrr']} | {s['ndcg@10']} |")
    if "llm_judge_avg_top3" in result:
        lines.append(f"\nLLM-as-judge (avg score of top-3, 1-5): {result['llm_judge_avg_top3']}")
    return "\n".join(lines)
