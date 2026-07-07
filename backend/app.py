from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.graph.graph import build_graph
from backend.models.state import InterviewState
from backend.retrieval.chunking import chunk_many
from backend.retrieval.embeddings import HashingEmbeddingModel
from backend.retrieval.faiss_store import FaissRetriever
from backend.scripts.run_interview_graph import (
    build_evaluation_agent,
    build_initial_state,
    build_interview_agent,
    build_planner,
    build_retriever,
    run_voice_mode,
)
from backend.services.persistence import InterviewStore
from backend.services.report import LocalReportAgent

app = FastAPI(title="HirePilot AI")
settings = get_settings()
configure_logging(settings.log_level)
store = InterviewStore()


class DemoPayload(BaseModel):
    candidate_name: str | None = None
    resume_text: str | None = None
    jd_text: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")


@app.post("/api/run-demo")
def run_demo(payload: DemoPayload) -> dict[str, Any]:
    state = build_initial_state(resume_file=None, jd_file=None)
    if payload.candidate_name:
        state.candidate_name = payload.candidate_name
    if payload.resume_text:
        state.understanding.resume.summary = payload.resume_text
    if payload.jd_text:
        state.understanding.job_description.summary = payload.jd_text

    retriever = build_retriever(resume_file=None, jd_file=None)
    planner = build_planner(local=True)
    interview_agent = build_interview_agent(local=True)
    evaluation_agent = build_evaluation_agent(local=True)
    graph = build_graph(
        planner=planner,
        retriever=retriever,
        interview_agent=interview_agent,
        evaluation_agent=evaluation_agent,
    )

    intro_answer = payload.resume_text or "I built an AI interview assistant with Python and FastAPI."
    followup_answer = payload.jd_text or "I focused on secure API design, retrieval quality, and structured outputs."

    graph_state = {"interview": state, "candidate_answer": intro_answer}
    for node_name in ["planner", "intro", "record_answer", "evaluation", "memory_update"]:
        graph_state = graph.nodes[node_name].invoke(graph_state)

    for _ in range(1, 4):
        graph_state["candidate_answer"] = followup_answer
        for node_name in ["retriever", "interview", "record_answer", "evaluation", "memory_update"]:
            graph_state = graph.nodes[node_name].invoke(graph_state)

    graph_state = graph.nodes["closing"].invoke(graph_state)
    graph_state["candidate_answer"] = "I would be excited to contribute to this role."
    for node_name in ["record_answer", "evaluation", "memory_update"]:
        graph_state = graph.nodes[node_name].invoke(graph_state)
    graph_state = graph.nodes["report"].invoke(graph_state)

    report = LocalReportAgent().create_report(graph_state["interview"])
    payload_to_store = {
        "candidate_name": report.candidate_name,
        "overall_score": report.overall_score,
        "hiring_recommendation": report.hiring_recommendation,
        "executive_summary": report.executive_summary,
        "turns": len(graph_state["interview"].previous_questions),
        "strong_areas": report.strong_areas,
        "improvement_areas": report.improvement_areas,
        "dimension_summaries": [item.model_dump() for item in report.dimension_summaries],
    }
    store.save(payload_to_store)
    return payload_to_store


def run_voice_demo(payload: DemoPayload) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        resume_path = temp_path / "resume.txt"
        jd_path = temp_path / "job_description.txt"
        resume_path.write_text(payload.resume_text or "", encoding="utf-8")
        jd_path.write_text(payload.jd_text or "", encoding="utf-8")

        state = build_initial_state(resume_file=str(resume_path), jd_file=str(jd_path))
        if payload.candidate_name:
            state.candidate_name = payload.candidate_name

        retriever = build_retriever(resume_file=str(resume_path), jd_file=str(jd_path))
        use_local_agents = not bool(getattr(settings, "groq_api_key", ""))
        planner = build_planner(local=use_local_agents)
        interview_agent = build_interview_agent(local=use_local_agents)
        evaluation_agent = build_evaluation_agent(local=use_local_agents)
        graph = build_graph(
            planner=planner,
            retriever=retriever,
            interview_agent=interview_agent,
            evaluation_agent=evaluation_agent,
        )

        run_voice_mode(graph=graph, state=state, turns=6, settings=settings)

    report = LocalReportAgent().create_report(state)
    payload_to_store = {
        "candidate_name": report.candidate_name,
        "overall_score": report.overall_score,
        "hiring_recommendation": report.hiring_recommendation,
        "executive_summary": report.executive_summary,
        "turns": len(state.previous_questions),
        "strong_areas": report.strong_areas,
        "improvement_areas": report.improvement_areas,
        "dimension_summaries": [item.model_dump() for item in report.dimension_summaries],
    }
    store.save(payload_to_store)
    return payload_to_store


@app.post("/api/run-voice-demo")
def run_voice_demo_endpoint(payload: DemoPayload) -> dict[str, Any]:
    return run_voice_demo(payload)


@app.get("/api/reports")
def list_reports() -> list[dict[str, Any]]:
    return store.list()
