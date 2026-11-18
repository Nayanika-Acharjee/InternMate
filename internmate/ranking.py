"""Personalised re-ranking of retrieved candidates.

final_score = 0.55 * relevance      (hybrid search score, min-max normalised)
            + 0.25 * skill_match    (share of the job's skills the candidate has)
            + 0.12 * preference     (role / location / remote / stipend preferences met)
            + 0.08 * feedback       (similarity to jobs the user liked minus jobs they disliked)

Signals with no data (empty profile, no feedback) are neutral (0.5) so they never reorder results.
"""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from internmate.search import Candidate
from internmate.text import job_skills, normalize_skill

WEIGHTS = {"relevance": 0.55, "skills": 0.25, "preferences": 0.12, "feedback": 0.08}
NEUTRAL = 0.5


class ProfileLike(Protocol):
    skills: list[str]
    preferred_roles: list[str]
    preferred_locations: list[str]
    remote_only: bool
    min_salary: int | None


@dataclass
class ScoredJob:
    candidate: Candidate
    score: float
    components: dict[str, float]
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)

    @property
    def job(self):
        return self.candidate.job

    def to_dict(self) -> dict:
        j = self.job
        return {
            "job_id": j.id,
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "remote": j.remote,
            "stipend": j.salary_text or None,
            "url": j.url,
            "score": round(self.score, 3),
            "score_breakdown": {k: round(v, 3) for k, v in self.components.items()},
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "snippet": j.description[:300],
        }


def skill_match(profile: ProfileLike | None, job) -> tuple[float, list[str], list[str]]:
    required = job_skills(job.title, job.description, job.tags)
    have = {normalize_skill(s) for s in (profile.skills if profile else [])}
    matched = sorted(required & have)
    missing = sorted(required - have)[:6]
    if not have or not required:
        return NEUTRAL, matched, missing
    return len(matched) / len(required), matched, missing


def preference_match(profile: ProfileLike | None, job) -> float:
    if profile is None:
        return NEUTRAL
    checks: list[bool] = []
    title = job.title.lower()
    if profile.preferred_roles:
        checks.append(any(all(w in title for w in role.lower().split()) for role in profile.preferred_roles))
    if profile.preferred_locations:
        loc = (job.location or "").lower()
        checks.append(job.remote or any(p.lower() in loc for p in profile.preferred_locations))
    if profile.remote_only:
        checks.append(bool(job.remote))
    if profile.min_salary and job.salary_max is not None:
        checks.append(job.salary_max >= profile.min_salary)
    return sum(checks) / len(checks) if checks else NEUTRAL


def feedback_match(job_vec, liked: list, disliked: list) -> float:
    if job_vec is None or (not liked and not disliked):
        return NEUTRAL
    v = np.asarray(job_vec, dtype=float)
    like = max((float(np.dot(v, np.asarray(x, dtype=float))) for x in liked), default=0.0)
    dislike = max((float(np.dot(v, np.asarray(x, dtype=float))) for x in disliked), default=0.0)
    return float(np.clip(NEUTRAL + 0.5 * (like - dislike), 0.0, 1.0))


def rank(candidates: list[Candidate], profile: ProfileLike | None = None,
         liked: list | None = None, disliked: list | None = None) -> list[ScoredJob]:
    if not candidates:
        return []
    rel = np.array([c.relevance for c in candidates])
    span = rel.max() - rel.min()
    rel_norm = (rel - rel.min()) / span if span else np.ones_like(rel)

    scored = []
    for c, r in zip(candidates, rel_norm):
        skills, matched, missing = skill_match(profile, c.job)
        components = {
            "relevance": float(r),
            "skills": skills,
            "preferences": preference_match(profile, c.job),
            "feedback": feedback_match(c.job.embedding, liked or [], disliked or []),
        }
        score = sum(WEIGHTS[k] * v for k, v in components.items())
        scored.append(ScoredJob(c, score, components, matched, missing))
    return sorted(scored, key=lambda s: s.score, reverse=True)
