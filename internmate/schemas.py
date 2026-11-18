from pydantic import BaseModel, Field


class ResumeProfile(BaseModel):
    """Structured facts extracted from a resume."""
    skills: list[str] = Field(description="Technical skills, tools and frameworks, lowercase")
    preferred_roles: list[str] = Field(default_factory=list, description="Job titles the candidate fits")
    experience_years: float | None = Field(default=None, description="Total professional experience")
    summary: str = Field(description="Two-sentence summary of the candidate")


class JobCard(BaseModel):
    job_id: int = Field(description="job_id exactly as returned by a tool")
    title: str
    company: str
    location: str = ""
    url: str = ""
    match_score: int = Field(ge=0, le=100, description="0-100 fit score")
    why_matched: str = Field(description="One sentence on why this job fits THIS user, citing their skills")
    missing_skills: list[str] = Field(default_factory=list)


class AgentReply(BaseModel):
    message: str
    recommendations: list[JobCard] = Field(default_factory=list)


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = irrelevant, 5 = excellent match for the query")
    reason: str
