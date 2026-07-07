from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.models.state import InterviewState
from backend.models.understanding import (
    JobDescriptionUnderstanding,
    ResumeUnderstanding,
    SkillGapAnalysis,
    UnderstandingResult,
)
from backend.models.state import InterviewStageName


class DummyNode:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, state):
        self.calls.append(state["candidate_answer"])
        return state


class DummyGraph:
    def __init__(self) -> None:
        self.nodes = {name: DummyNode() for name in ["planner", "intro", "retriever", "interview", "record_answer", "evaluation", "memory_update", "closing", "report"]}


def test_voice_demo_endpoint_uses_voice_pipeline(monkeypatch):
    captured = {}

    def fake_run_voice_demo(payload):
        captured["payload"] = payload
        return {
            "candidate_name": payload.candidate_name or "Demo",
            "overall_score": 4.5,
            "hiring_recommendation": "Hire",
            "executive_summary": "voice ok",
            "strong_areas": ["voice"],
            "improvement_areas": ["latency"],
            "turns": 3,
            "dimension_summaries": [],
        }

    monkeypatch.setattr(app_module, "run_voice_demo", fake_run_voice_demo)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/run-voice-demo",
        json={"candidate_name": "Ava", "resume_text": "resume", "jd_text": "jd"},
    )

    assert response.status_code == 200
    assert response.json()["candidate_name"] == "Ava"
    assert captured["payload"].candidate_name == "Ava"


def test_voice_demo_uses_groq_backed_agents_when_available(monkeypatch):
    monkeypatch.setattr(app_module, "settings", SimpleNamespace(groq_api_key="demo-key", log_level="INFO"))

    calls = []

    def fake_build_initial_state(resume_file=None, jd_file=None):
        return InterviewState(candidate_name="Demo", current_stage=InterviewStageName.PLANNING)

    def fake_build_retriever(resume_file=None, jd_file=None):
        return object()

    def fake_build_planner(local):
        calls.append(("planner", local))
        return object()

    def fake_build_interview_agent(local):
        calls.append(("interview", local))
        return object()

    def fake_build_evaluation_agent(local):
        calls.append(("evaluation", local))
        return object()

    def fake_build_graph(**kwargs):
        return type("Graph", (), {"nodes": {}})()

    monkeypatch.setattr(app_module, "build_initial_state", fake_build_initial_state)
    monkeypatch.setattr(app_module, "build_retriever", fake_build_retriever)
    monkeypatch.setattr(app_module, "build_planner", fake_build_planner)
    monkeypatch.setattr(app_module, "build_interview_agent", fake_build_interview_agent)
    monkeypatch.setattr(app_module, "build_evaluation_agent", fake_build_evaluation_agent)
    monkeypatch.setattr(app_module, "build_graph", fake_build_graph)
    monkeypatch.setattr(app_module, "run_voice_mode", lambda **kwargs: None)
    monkeypatch.setattr(app_module, "LocalReportAgent", lambda: type("ReportAgent", (), {"create_report": lambda self, state: type("Report", (), {"candidate_name": "Demo", "overall_score": 4.0, "hiring_recommendation": "Hire", "executive_summary": "ok", "strong_areas": [], "improvement_areas": [], "dimension_summaries": []})()})())
    monkeypatch.setattr(app_module, "store", type("Store", (), {"save": lambda self, payload: None, "list": lambda self: []})())

    app_module.run_voice_demo(app_module.DemoPayload(candidate_name="Ava", resume_text="resume", jd_text="jd"))

    assert ("planner", False) in calls
    assert ("interview", False) in calls
    assert ("evaluation", False) in calls


def test_run_demo_uses_payload_text(monkeypatch):
    state = InterviewState(
        candidate_name="Demo",
        understanding=UnderstandingResult(
            resume=ResumeUnderstanding(candidate_name="Demo", summary=""),
            job_description=JobDescriptionUnderstanding(role_title="GenAI Engineer", summary=""),
            skill_gap_analysis=SkillGapAnalysis(),
        ),
        current_stage=InterviewStageName.PLANNING,
    )

    graph = DummyGraph()

    monkeypatch.setattr(app_module, "build_initial_state", lambda resume_file=None, jd_file=None: state)
    monkeypatch.setattr(app_module, "build_retriever", lambda resume_file=None, jd_file=None: object())
    monkeypatch.setattr(app_module, "build_planner", lambda local=True: object())
    monkeypatch.setattr(app_module, "build_interview_agent", lambda local=True: object())
    monkeypatch.setattr(app_module, "build_evaluation_agent", lambda local=True: object())
    monkeypatch.setattr(app_module, "build_graph", lambda **kwargs: graph)

    class DummyReportAgent:
        def create_report(self, interview_state):
            return type(
                "Report",
                (),
                {
                    "candidate_name": interview_state.candidate_name,
                    "overall_score": 4.0,
                    "hiring_recommendation": "Hire",
                    "executive_summary": "ok",
                    "strong_areas": ["backend design"],
                    "improvement_areas": ["deployment"],
                    "dimension_summaries": [],
                },
            )()

    monkeypatch.setattr(app_module, "LocalReportAgent", DummyReportAgent)
    monkeypatch.setattr(app_module, "store", type("Store", (), {"save": lambda self, payload: payload, "list": lambda self: []})())

    result = app_module.run_demo(
        app_module.DemoPayload(
            candidate_name="Ava",
            resume_text="My resume text",
            jd_text="My job description text",
        )
    )

    combined = " ".join(graph.nodes["record_answer"].calls)
    assert "My resume text" in combined
    assert "My job description text" in combined
    assert result["candidate_name"] == "Ava"
    assert "strong_areas" in result
    assert "improvement_areas" in result
