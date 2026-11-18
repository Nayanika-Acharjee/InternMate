"""Multi-agent career assistant built with LangGraph.

                     ┌──────────────┐
    START → start_turn → supervisor ──(conditional routing)──┐
                     └──────▲───────┘                        │
                            │            ┌───────────────────┼────────────────────┐
                            │            ▼                   ▼                    ▼
                            ├──── profile_agent ⇄ tools   search_agent ⇄ tools   writer → END
                            └──────────────────────────────────┘                (RAG)

* supervisor     - LLM router (structured output) that decides which specialist acts next.
* profile_agent  - resume analysis + remembering skills / preferences (long-term memory in SQL).
* search_agent   - semantic job search, comparison and feedback; collects retrieved listings.
* writer         - RAG step: writes the final answer *only* from listings retrieved this turn and
                   returns Pydantic-validated JobCards. A grounding check drops any job the LLM
                   mentions that was not retrieved and overwrites facts with database values.

Each specialist runs its own tool loop in a private `scratch` channel and hands a short report
back to the supervisor, so agents never see each other's raw tool calls.
Short-term memory = conversation thread in the LangGraph checkpointer (keyed by user_id).
"""
import json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from internmate import repo
from internmate.db import SessionLocal
from internmate.embeddings import Embedder
from internmate.llm import message_text
from internmate.models import Job, User
from internmate.ranking import rank
from internmate.recommender import recommend
from internmate.schemas import AgentReply, JobCard, ResumeProfile
from internmate.search import Candidate, SearchFilters

MAX_HANDOFFS = 4        # supervisor decisions per user turn
MAX_TOOL_STEPS = 4      # tool rounds per specialist visit
MAX_HISTORY = 20        # conversation messages kept in context
MAX_CONTEXT_JOBS = 10   # retrieved listings given to the writer

PROFILE_AGENT, SEARCH_AGENT, WRITER = "profile_agent", "search_agent", "writer"
GROUNDED_TOOLS = {"search_jobs", "compare_jobs"}

SUPERVISOR_PROMPT = """You are the supervisor of InternMate, a team of agents that helps college \
students find internships. Decide who acts next:
- profile_agent: the user shared a resume, or stated skills / preferences (location, remote, \
stipend, roles) that are not yet saved.
- search_agent: the user wants jobs found, compared, or gave feedback (likes / dislikes) on a job.
- respond: enough work is done for this message (or it is small talk) - hand over to the writer.
Typical flow: profile_agent (if there is profile info) -> search_agent (if jobs are needed) -> respond.
Never send the same agent twice in a row unless its report says the work is unfinished."""

PROFILE_PROMPT = """You are the profile agent. Keep the user's profile accurate.
- If the user pasted a resume, call parse_resume with the full resume text.
- If they stated skills or preferences, call save_preferences with ONLY what they said.
- Then reply with a one-line report of what you saved."""

SEARCH_PROMPT = """You are the search agent. Find and evaluate internships for the user.
- Call search_jobs with a focused query (role + key skills). The tool already personalises results \
using the saved profile, so do not repeat profile filters unless the user asked for them now.
- If results look weak, try ONE broader query.
- Use compare_jobs when asked to compare, record_feedback when the user likes or dislikes a job.
- Then reply with a one-line report (e.g. "found 5 matches, best fit is job 12")."""

WRITER_PROMPT = """You are the writer for InternMate. Answer the user's latest message in a \
friendly, skimmable way.
Rules:
- Recommend ONLY jobs from the RETRIEVED LISTINGS below; never invent jobs, companies, links or stipends.
- For each recommended job, write why_matched citing the user's own skills, and list missing skills.
- match_score = listing score x 100, rounded.
- If there are no retrieved listings, do not recommend jobs; answer conversationally or ask what \
they are looking for.
- If results are weak, say so honestly and suggest how to broaden the search."""


# ---------------------------------------------------------------- tools

def _user_id(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


def build_tools(embedder: Embedder, llm) -> dict[str, list]:
    @tool
    def parse_resume(resume_text: str, config: RunnableConfig) -> str:
        """Extract skills, suitable roles and experience from resume text and save them to the user's profile."""
        extracted: ResumeProfile = llm.with_structured_output(ResumeProfile).invoke(
            "Extract the candidate profile from this resume. Be precise; do not guess.\n\n" + resume_text[:12000]
        )
        with SessionLocal() as s:
            user = repo.update_profile(
                s, _user_id(config), skills=extracted.skills, preferred_roles=extracted.preferred_roles,
                experience_years=extracted.experience_years, resume_summary=extracted.summary,
            )
            s.commit()
            return json.dumps(repo.profile_dict(user))

    @tool
    def save_preferences(config: RunnableConfig, skills: list[str] | None = None,
                         preferred_roles: list[str] | None = None,
                         preferred_locations: list[str] | None = None,
                         remote_only: bool | None = None, min_stipend: int | None = None) -> str:
        """Remember the user's skills and job preferences for future searches. Only pass what the user stated."""
        with SessionLocal() as s:
            user = repo.update_profile(
                s, _user_id(config), skills=skills, preferred_roles=preferred_roles,
                preferred_locations=preferred_locations, remote_only=remote_only, min_salary=min_stipend,
            )
            s.commit()
            return json.dumps(repo.profile_dict(user))

    @tool
    def get_my_profile(config: RunnableConfig) -> str:
        """Return what is remembered about the user (skills, preferences)."""
        with SessionLocal() as s:
            user = repo.get_or_create_user(s, _user_id(config))
            s.commit()
            return json.dumps(repo.profile_dict(user))

    @tool
    def search_jobs(query: str, config: RunnableConfig, remote_only: bool = False,
                    location: str | None = None, min_stipend: int | None = None, limit: int = 5) -> str:
        """Search live job listings, ranked for this user using their saved skills, preferences and feedback.

        Args:
            query: what to look for, e.g. "python backend internship fastapi".
            remote_only: only return remote jobs.
            location: city to filter by, e.g. "Bengaluru" (remote jobs always included).
            min_stipend: minimum monthly stipend in INR.
            limit: number of results (max 10).
        """
        filters = SearchFilters(remote_only=remote_only, location=location, min_salary=min_stipend)
        with SessionLocal() as s:
            results = recommend(s, embedder, query, _user_id(config), filters, k=min(limit, 10))
            return json.dumps([r.to_dict() for r in results])

    @tool
    def compare_jobs(job_ids: list[int], config: RunnableConfig) -> str:
        """Compare specific jobs side by side for this user (skill match, missing skills, location, stipend)."""
        with SessionLocal() as s:
            user = s.get(User, _user_id(config))
            jobs = [j for j in (s.get(Job, i) for i in job_ids[:5]) if j is not None]
            liked, disliked = repo.feedback_embeddings(s, _user_id(config))
            scored = rank([Candidate(j, 1.0, 0.0, 0.0) for j in jobs], user, liked, disliked)
            return json.dumps([sj.to_dict() | {"description": sj.job.description[:800]} for sj in scored])

    @tool
    def record_feedback(job_id: int, liked: bool, config: RunnableConfig) -> str:
        """Record that the user liked (true) or disliked (false) a job; future rankings adapt to it."""
        with SessionLocal() as s:
            try:
                repo.set_feedback(s, _user_id(config), job_id, liked)
            except ValueError as exc:
                return f"error: {exc}"
            s.commit()
            return "saved"

    return {
        PROFILE_AGENT: [parse_resume, save_preferences, get_my_profile],
        SEARCH_AGENT: [search_jobs, compare_jobs, record_feedback],
    }


# ---------------------------------------------------------------- state

def _list_channel(old: list, new: list | None) -> list:
    """Append; returning None from a node resets the channel."""
    return [] if new is None else [*old, *new]


def _dict_channel(old: dict, new: dict | None) -> dict:
    return {} if new is None else {**old, **new}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # user <-> writer conversation (persisted)
    scratch: Annotated[list, _list_channel]   # active specialist's private tool loop
    notes: Annotated[list, _list_channel]     # specialist reports for the current turn
    retrieved: Annotated[dict, _dict_channel]  # job_id -> listing retrieved this turn (RAG context)
    handoffs: int
    next: str
    reply: dict | None


class Route(BaseModel):
    next: Literal["profile_agent", "search_agent", "respond"]
    reason: str = Field(description="One short sentence")


class Recommendations(BaseModel):
    message: str = Field(description="The answer shown to the user")
    recommendations: list[JobCard] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers

def _trim_history(messages: list) -> list:
    """Keep recent history, starting at a user message."""
    if len(messages) <= MAX_HISTORY:
        return messages
    tail = messages[-MAX_HISTORY:]
    for i, m in enumerate(tail):
        if isinstance(m, HumanMessage):
            return tail[i:]
    return messages[-1:]


def _notes_block(notes: list[str]) -> str:
    return ("\n\nWork done so far for this message:\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""


def tool_results(messages: list) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.name in GROUNDED_TOOLS:
            try:
                for item in json.loads(message_text(m)):
                    found[int(item["job_id"])] = item
            except (ValueError, TypeError, KeyError):
                continue
    return found


def ground_recommendations(cards: list[JobCard], retrieved: dict[int, dict]) -> list[JobCard]:
    """Drop cards for jobs that were not retrieved and overwrite facts with retrieved data."""
    grounded, seen = [], set()
    for card in cards:
        src = retrieved.get(card.job_id)
        if src is None or card.job_id in seen:
            continue
        seen.add(card.job_id)
        grounded.append(card.model_copy(update={
            "title": src["title"], "company": src["company"], "location": src.get("location") or "",
            "url": src.get("url") or "", "match_score": round(float(src.get("score", 0)) * 100),
            "missing_skills": src.get("missing_skills", card.missing_skills),
        }))
    return grounded


# ---------------------------------------------------------------- graph

def _make_specialist(name: str, prompt: str, tools: list, llm):
    bound = llm.bind_tools(tools)

    def node(state: AgentState):
        context = [SystemMessage(prompt + _notes_block(state["notes"])),
                   *_trim_history(state["messages"]), *state["scratch"]]
        msg = bound.invoke(context)
        steps = sum(isinstance(m, AIMessage) for m in state["scratch"])
        if msg.tool_calls and steps < MAX_TOOL_STEPS:
            return {"scratch": [msg]}
        report = message_text(msg).strip() or "finished"
        return {"scratch": None, "notes": [f"{name}: {report}"], "retrieved": tool_results(state["scratch"])}

    def route(state: AgentState) -> str:
        scratch = state["scratch"]
        return f"{name}_tools" if scratch and getattr(scratch[-1], "tool_calls", None) else "supervisor"

    return node, route


def build_graph(llm, tools: dict[str, list], checkpointer=None):
    router = llm.with_structured_output(Route)
    writer_llm = llm.with_structured_output(Recommendations)

    def start_turn(_: AgentState):
        return {"scratch": None, "notes": None, "retrieved": None, "handoffs": 0, "reply": None}

    def supervisor(state: AgentState):
        if state["handoffs"] >= MAX_HANDOFFS:
            return {"next": "respond"}
        decision = router.invoke([SystemMessage(SUPERVISOR_PROMPT + _notes_block(state["notes"])),
                                  *_trim_history(state["messages"])])
        return {"next": decision.next, "handoffs": state["handoffs"] + 1}

    def writer(state: AgentState):
        retrieved = state["retrieved"]
        context = sorted(retrieved.values(), key=lambda j: j.get("score", 0), reverse=True)[:MAX_CONTEXT_JOBS]
        system = (WRITER_PROMPT + _notes_block(state["notes"]) +
                  "\n\nRETRIEVED LISTINGS (the only source of truth):\n" +
                  (json.dumps(context) if context else "none"))
        draft = writer_llm.invoke([SystemMessage(system), *_trim_history(state["messages"])])
        cards = ground_recommendations(draft.recommendations, retrieved)
        reply = AgentReply(message=draft.message, recommendations=cards)
        return {"messages": [AIMessage(reply.message)], "reply": reply.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("start_turn", start_turn)
    graph.add_node("supervisor", supervisor)
    graph.add_node(WRITER, writer)
    for name, prompt in ((PROFILE_AGENT, PROFILE_PROMPT), (SEARCH_AGENT, SEARCH_PROMPT)):
        node, route = _make_specialist(name, prompt, tools[name], llm)
        graph.add_node(name, node)
        graph.add_node(f"{name}_tools", ToolNode(tools[name], messages_key="scratch"))
        graph.add_conditional_edges(name, route, {f"{name}_tools": f"{name}_tools", "supervisor": "supervisor"})
        graph.add_edge(f"{name}_tools", name)

    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next"],
                                {PROFILE_AGENT: PROFILE_AGENT, SEARCH_AGENT: SEARCH_AGENT, "respond": WRITER})
    graph.add_edge(WRITER, END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


class CareerAgent:
    def __init__(self, llm, embedder: Embedder):
        self.graph = build_graph(llm, build_tools(embedder, llm))

    def chat(self, user_id: str, message: str) -> AgentReply:
        config = {"configurable": {"thread_id": user_id, "user_id": user_id}, "recursion_limit": 40}
        result = self.graph.invoke({"messages": [HumanMessage(message)]}, config=config)
        return AgentReply(**result["reply"])
