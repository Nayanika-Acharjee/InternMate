"""Multi-agent graph tests with a scripted fake LLM (no API key needed)."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select

from internmate.agent import (MAX_HANDOFFS, CareerAgent, Recommendations, Route, build_tools,
                              ground_recommendations)
from internmate.models import Job, User
from internmate.schemas import JobCard


class ScriptedLLM(BaseChatModel):
    """Free-text / tool-call replies come from `script`; structured outputs from `structured[schema]`."""
    script: list = []
    structured: dict = {}
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append("chat")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(_):
            self.calls.append(schema.__name__)
            queue = self.structured[schema]
            return queue.pop(0) if len(queue) > 1 else queue[0]
        return RunnableLambda(respond)


def _card(job_id, title="LLM says"):
    return JobCard(job_id=job_id, title=title, company="?", match_score=99, why_matched="fits")


def test_tool_schemas_hide_injected_config(embedder):
    tools = {t.name: t for group in build_tools(embedder, llm=None).values() for t in group}
    assert set(tools) == {"parse_resume", "save_preferences", "get_my_profile",
                          "search_jobs", "compare_jobs", "record_feedback"}
    assert "config" not in tools["search_jobs"].args and "query" in tools["search_jobs"].args


def test_grounding_drops_hallucinated_jobs():
    retrieved = {7: {"job_id": 7, "title": "Real Title", "company": "Real Co", "location": "Remote",
                     "url": "u", "score": 0.81, "missing_skills": ["sql"]}}
    cards = ground_recommendations([_card(7), _card(999), _card(7)], retrieved)
    assert [c.job_id for c in cards] == [7]
    assert cards[0].title == "Real Title" and cards[0].match_score == 81


def test_supervisor_routes_profile_then_search_then_writer(seeded, embedder):
    flutter_id = seeded.scalar(select(Job.id).where(Job.external_id == "flutter"))
    llm = ScriptedLLM(
        script=[
            # profile agent: save prefs, then report
            AIMessage("", tool_calls=[{"name": "save_preferences", "id": "a",
                                       "args": {"skills": ["Dart", "Flutter"], "remote_only": True}}]),
            AIMessage("saved skills dart, flutter and remote_only"),
            # search agent: search, then report
            AIMessage("", tool_calls=[{"name": "search_jobs", "id": "b", "args": {"query": "flutter mobile"}}]),
            AIMessage("found matches, best is the Flutter role"),
        ],
        structured={
            Route: [Route(next="profile_agent", reason="prefs"), Route(next="search_agent", reason="find"),
                    Route(next="respond", reason="done")],
            Recommendations: [Recommendations(message="Here is a remote Flutter internship.",
                                              recommendations=[_card(flutter_id), _card(424242)])],
        },
        calls=[],
    )
    reply = CareerAgent(llm, embedder).chat("student1", "I know Flutter and Dart, want remote work")

    assert reply.message.startswith("Here is")
    assert [c.job_id for c in reply.recommendations] == [flutter_id]  # hallucinated 424242 removed
    assert llm.calls == ["Route", "chat", "chat", "Route", "chat", "chat", "Route", "Recommendations"]
    seeded.expire_all()
    user = seeded.get(User, "student1")
    assert user.remote_only and {"dart", "flutter"} <= set(user.skills)


def test_small_talk_goes_straight_to_writer_without_cards(embedder):
    llm = ScriptedLLM(structured={
        Route: [Route(next="respond", reason="greeting")],
        # Writer tries to recommend a job although nothing was retrieved -> must be dropped.
        Recommendations: [Recommendations(message="Hi! What are your skills?", recommendations=[_card(1)])],
    }, calls=[])
    reply = CareerAgent(llm, embedder).chat("u", "hello")
    assert reply.recommendations == [] and "skills" in reply.message
    assert llm.calls == ["Route", "Recommendations"]


def test_handoff_limit_prevents_infinite_loops(seeded, embedder):
    llm = ScriptedLLM(
        script=[AIMessage("nothing to do")] * 10,
        structured={Route: [Route(next="search_agent", reason="again")],
                    Recommendations: [Recommendations(message="done")]},
        calls=[],
    )
    reply = CareerAgent(llm, embedder).chat("u", "loop forever")
    assert reply.message == "done"
    assert llm.calls.count("Route") == MAX_HANDOFFS


def test_retrieved_context_resets_each_turn(seeded, embedder):
    flutter_id = seeded.scalar(select(Job.id).where(Job.external_id == "flutter"))
    llm = ScriptedLLM(
        script=[AIMessage("", tool_calls=[{"name": "search_jobs", "id": "s", "args": {"query": "flutter"}}]),
                AIMessage("found")],
        structured={
            Route: [Route(next="search_agent", reason="find"), Route(next="respond", reason="done"),
                    Route(next="respond", reason="chat")],
            Recommendations: [Recommendations(message="one", recommendations=[_card(flutter_id)]),
                              Recommendations(message="two", recommendations=[_card(flutter_id)])],
        },
        calls=[],
    )
    agent = CareerAgent(llm, embedder)
    assert [c.job_id for c in agent.chat("same", "find flutter jobs").recommendations] == [flutter_id]
    assert agent.chat("same", "thanks!").recommendations == []  # no search this turn -> no cards

    state = agent.graph.get_state({"configurable": {"thread_id": "same"}})
    humans = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["find flutter jobs", "thanks!"]  # conversation memory persists
