from fastapi.testclient import TestClient

from internmate.api import app


def test_api_flow():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

        r = client.post("/ingest", json={"sources": ["sample"]})
        assert r.status_code == 200 and r.json()["inserted"] == 30

        assert client.post("/ingest", json={"sources": ["nope"]}).status_code == 400

        r = client.put("/profile/u1", json={"skills": ["Python", "LLM"], "preferred_locations": ["Gurugram"]})
        assert r.json()["skills"] == ["python", "llm"]

        results = client.get("/jobs/search", params={"q": "chatbot llm", "user_id": "u1", "k": 3}).json()
        assert len(results) == 3 and "score_breakdown" in results[0]

        r = client.post("/feedback", json={"user_id": "u1", "job_id": results[0]["job_id"], "liked": True})
        assert r.status_code == 200
        assert client.post("/feedback", json={"user_id": "u1", "job_id": 99999, "liked": True}).status_code == 404

        # No API key in tests -> agent unavailable, but the API degrades cleanly.
        assert client.post("/chat", json={"user_id": "u1", "message": "hi"}).status_code == 503
