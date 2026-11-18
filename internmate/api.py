from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from internmate import repo
from internmate.db import SessionLocal, init_db
from internmate.embeddings import get_embedder
from internmate.ingestion import SOURCES, run_ingestion
from internmate.llm import MissingAPIKey, get_llm
from internmate.recommender import recommend
from internmate.schemas import AgentReply
from internmate.search import SearchFilters

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="InternMate", version="0.1.0", lifespan=lifespan)


def get_session():
    with SessionLocal() as session:
        yield session


@lru_cache
def get_agent():
    from internmate.agent import CareerAgent

    return CareerAgent(get_llm(), get_embedder())


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=20000)


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    job_id: int
    liked: bool


class ProfileUpdate(BaseModel):
    skills: list[str] | None = None
    preferred_roles: list[str] | None = None
    preferred_locations: list[str] | None = None
    remote_only: bool | None = None
    min_salary: int | None = None
    replace: bool = False


class IngestRequest(BaseModel):
    sources: list[str] = ["sample"]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=AgentReply)
def chat(req: ChatRequest):
    try:
        agent = get_agent()
    except MissingAPIKey as exc:
        raise HTTPException(503, str(exc)) from exc
    return agent.chat(req.user_id, req.message)


@app.get("/jobs/search")
def search_jobs(q: str = Query(min_length=1), user_id: str | None = None, remote_only: bool = False,
                location: str | None = None, min_salary: int | None = None,
                k: int = Query(10, ge=1, le=50), session: Session = Depends(get_session)):
    filters = SearchFilters(remote_only=remote_only, location=location, min_salary=min_salary)
    return [r.to_dict() for r in recommend(session, get_embedder(), q, user_id, filters, k)]


@app.post("/feedback")
def feedback(req: FeedbackRequest, session: Session = Depends(get_session)):
    try:
        repo.set_feedback(session, req.user_id, req.job_id, req.liked)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    session.commit()
    return {"status": "saved"}


@app.get("/profile/{user_id}")
def get_profile(user_id: str, session: Session = Depends(get_session)):
    user = repo.get_or_create_user(session, user_id)
    session.commit()
    return repo.profile_dict(user)


@app.put("/profile/{user_id}")
def put_profile(user_id: str, body: ProfileUpdate, session: Session = Depends(get_session)):
    fields = body.model_dump(exclude={"replace"})
    user = repo.update_profile(session, user_id, replace=body.replace, **fields)
    session.commit()
    return repo.profile_dict(user)


@app.post("/ingest")
def ingest(req: IngestRequest, session: Session = Depends(get_session)):
    unknown = set(req.sources) - set(SOURCES)
    if unknown:
        raise HTTPException(400, f"unknown sources: {sorted(unknown)}")
    run = run_ingestion(session, get_embedder(), req.sources)
    return {"inserted": run.inserted, "updated": run.updated, "unchanged": run.unchanged,
            "expired": run.expired, "errors": run.errors}
