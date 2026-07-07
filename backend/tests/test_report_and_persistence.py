from pathlib import Path

from backend.models.state import AnswerScore, InterviewState, RubricScore
from backend.services.report import LocalReportAgent
from backend.services.persistence import InterviewStore


def test_local_report_agent_builds_structured_report(tmp_path: Path) -> None:
    state = InterviewState(
        candidate_name="Ava Lee",
        current_stage="complete",
        current_scores=[
            AnswerScore(
                question="Tell me about a project.",
                answer="I built a FastAPI service with Pydantic validation and LangGraph orchestration.",
                rubric_scores=[
                    RubricScore(dimension="technical_accuracy", score=4.5, reasoning="Concrete architecture details."),
                    RubricScore(dimension="communication_clarity", score=4.0, reasoning="Clear explanation."),
                ],
                overall_score=4.3,
                strengths=["backend design", "structured thinking"],
                weaknesses=["deployment depth"],
                reasoning="Good answer.",
            )
        ],
        strong_areas=["backend design", "structured thinking"],
        weak_areas=["deployment depth"],
        topics_covered=["technical", "behavioral"],
        previous_questions=["Tell me about a project."],
        previous_answers=["I built a FastAPI service with Pydantic validation and LangGraph orchestration."],
    )

    report = LocalReportAgent().create_report(state)

    assert report.candidate_name == "Ava Lee"
    assert report.overall_score >= 4.0
    assert report.hiring_recommendation in {"Hire", "Strong Hire"}
    assert report.strong_areas
    assert report.improvement_areas


def test_store_round_trip(tmp_path: Path) -> None:
    store = InterviewStore(storage_path=tmp_path / "interviews.json")
    payload = {
        "candidate_name": "Nina",
        "overall_score": 4.2,
        "summary": "Strong technical interview.",
    }

    store.save(payload)
    records = store.list()

    assert len(records) == 1
    assert records[0]["candidate_name"] == "Nina"
