"""Data access for user profiles (long-term memory) and feedback."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from internmate.models import Feedback, Job, User
from internmate.text import normalize_skill

LIST_FIELDS = ("skills", "preferred_roles", "preferred_locations")
SCALAR_FIELDS = ("remote_only", "min_salary", "experience_years", "resume_summary")


def get_or_create_user(session: Session, user_id: str) -> User:
    user = session.get(User, user_id)
    if user is None:
        user = User(id=user_id, skills=[], preferred_roles=[], preferred_locations=[])
        session.add(user)
        session.flush()
    return user


def update_profile(session: Session, user_id: str, replace: bool = False, **fields) -> User:
    """Merge new facts into the profile. Lists are unioned unless replace=True."""
    user = get_or_create_user(session, user_id)
    for name in LIST_FIELDS:
        values = fields.get(name)
        if values is None:
            continue
        if name == "skills":
            values = [normalize_skill(v) for v in values]
        values = [v.strip() for v in values if v and v.strip()]
        merged = values if replace else list(dict.fromkeys([*(getattr(user, name) or []), *values]))
        setattr(user, name, merged)
    for name in SCALAR_FIELDS:
        if fields.get(name) is not None:
            setattr(user, name, fields[name])
    session.flush()
    return user


def profile_dict(user: User) -> dict:
    return {
        "user_id": user.id,
        "skills": user.skills,
        "preferred_roles": user.preferred_roles,
        "preferred_locations": user.preferred_locations,
        "remote_only": user.remote_only,
        "min_salary": user.min_salary,
        "experience_years": user.experience_years,
        "resume_summary": user.resume_summary,
    }


def set_feedback(session: Session, user_id: str, job_id: int, liked: bool) -> Feedback:
    get_or_create_user(session, user_id)
    if session.get(Job, job_id) is None:
        raise ValueError(f"job {job_id} not found")
    fb = session.scalar(select(Feedback).where(Feedback.user_id == user_id, Feedback.job_id == job_id))
    if fb is None:
        fb = Feedback(user_id=user_id, job_id=job_id, liked=liked)
        session.add(fb)
    else:
        fb.liked = liked
    session.flush()
    return fb


def feedback_embeddings(session: Session, user_id: str) -> tuple[list, list]:
    rows = session.execute(
        select(Feedback.liked, Job.embedding)
        .join(Job, Job.id == Feedback.job_id)
        .where(Feedback.user_id == user_id, Job.embedding.is_not(None))
    ).all()
    return [r.embedding for r in rows if r.liked], [r.embedding for r in rows if not r.liked]
