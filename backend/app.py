from __future__ import annotations

import json
import logging
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
from backend.services.retriever_node import RetrieverNode
from backend.graph.nodes import (
    make_planner_node,
    intro_node,
    record_answer_node,
    make_evaluation_node,
    memory_update_node,
    should_continue,
    make_retriever_node,
    make_interview_node,
    closing_node,
    report_node,
)

logger = logging.getLogger(__name__)

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


def log_turn_debug(state: InterviewState, step_name: str, decision: str, question: str, answer: str) -> None:
    turns = len(state.previous_questions)
    history_len = len(state.conversation_history)
    retrieval_used = "Yes" if decision == "retriever" else "No"
    
    logger.info(
        "\n=== DEBUG TURN LOG ===\n"
        f"Session ID (Candidate): {state.candidate_name or 'Unknown'}\n"
        f"Turn Number: {turns}\n"
        f"Interview Stage: {state.interview_stage}\n"
        f"Previous AI Question: {state.previous_questions[-2] if len(state.previous_questions) > 1 else 'None'}\n"
        f"Candidate Transcript: {answer}\n"
        f"Decision Taken: {decision}\n"
        f"Retrieval Used: {retrieval_used}\n"
        f"History Length: {history_len}\n"
        f"InterviewState ID: {id(state)}\n"
        f"Generated Question: {question}\n"
        "======================="
    )


def speak_and_listen_server(question: str) -> str:
    from backend.voice.audio_io import LocalAudioIO
    from backend.voice.stt import WhisperSpeechToText
    from backend.voice.tts import EdgeTextToSpeech

    audio_io = LocalAudioIO(sample_rate=settings.voice_sample_rate)
    stt = WhisperSpeechToText(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
    )
    tts = EdgeTextToSpeech(voice=settings.voice_tts_voice)

    repeat_commands = ["repeat", "repeat please", "say that again", "can you repeat", "i didnt hear"]

    while True:
        spoken_path = tts.synthesize_to_file(question)
        audio_io.play_audio_file(spoken_path)
        spoken_path.unlink(missing_ok=True)
        
        answer_path = audio_io.record_until_silence(
            max_duration_seconds=settings.voice_max_record_seconds,
            silence_seconds=settings.voice_silence_seconds,
            vad_threshold=settings.voice_vad_threshold,
        )
        answer = stt.transcribe(answer_path) or "No answer provided."
        answer_path.unlink(missing_ok=True)

        # Normalize answer to check for repeat request
        normalized = "".join(c for c in answer.lower() if c.isalnum() or c.isspace()).strip()
        if normalized in repeat_commands:
            logger.info("[repeat] Command detected: '%s'. Replaying question.", answer)
            continue
        
        return answer


@app.post("/api/run-demo")
def run_demo(payload: DemoPayload) -> dict[str, Any]:
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

        intro_answer = f"My resume details: {payload.resume_text or 'Backend Engineer'}"
        followup_answer = f"To address that, here is my relevant experience: {payload.jd_text or 'FastAPI design'}"

        graph_state = {"interview": state, "candidate_answer": intro_answer}
        # Run Turn 0
        for node_name in ["planner", "intro", "record_answer", "evaluation", "memory_update"]:
            graph_state = graph.nodes[node_name].invoke(graph_state)

        # Dynamic loop for simulation
        for _ in range(6):
            next_step = should_continue(graph_state)
            if next_step == "retriever":
                graph_state["candidate_answer"] = ""
                for node_name in ["retriever", "interview"]:
                    graph_state = graph.nodes[node_name].invoke(graph_state)
                graph_state["candidate_answer"] = followup_answer
                for node_name in ["record_answer", "evaluation", "memory_update"]:
                    graph_state = graph.nodes[node_name].invoke(graph_state)
            elif next_step == "closing":
                graph_state["candidate_answer"] = ""
                graph_state = graph.nodes["closing"].invoke(graph_state)
                graph_state["candidate_answer"] = "I would be excited to contribute to this role."
                for node_name in ["record_answer", "evaluation", "memory_update"]:
                    graph_state = graph.nodes[node_name].invoke(graph_state)
            elif next_step == "report":
                break

        graph_state = report_node(graph_state)

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


@app.post("/api/start-voice-interview")
def start_voice_interview(payload: DemoPayload) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        resume_path = temp_path / "resume.txt"
        jd_path = temp_path / "job_description.txt"
        resume_path.write_text(payload.resume_text or "", encoding="utf-8")
        jd_path.write_text(payload.jd_text or "", encoding="utf-8")

        state = build_initial_state(resume_file=str(resume_path), jd_file=str(jd_path))
        if payload.candidate_name:
            state.candidate_name = payload.candidate_name

    # Preserve raw text in summaries for retrieving on subsequent turns
    if payload.resume_text:
        state.understanding.resume.summary = payload.resume_text
    if payload.jd_text:
        state.understanding.job_description.summary = payload.jd_text

    use_local_agents = not bool(getattr(settings, "groq_api_key", ""))
    planner = build_planner(local=use_local_agents)

    # Run Turn 0: planner and intro
    graph_state = {"interview": state, "candidate_answer": ""}
    graph_state = make_planner_node(planner)(graph_state)
    graph_state = intro_node(graph_state)

    # Speak and listen on the server
    question_text = graph_state["interview"].current_question or ""
    answer_text = speak_and_listen_server(question_text)

    # Process candidate answer
    graph_state["candidate_answer"] = answer_text
    graph_state = record_answer_node(graph_state)
    evaluation_agent = build_evaluation_agent(local=use_local_agents)
    graph_state = make_evaluation_node(evaluation_agent)(graph_state)
    graph_state = memory_update_node(graph_state)

    next_step = should_continue(graph_state)

    # Log turn details
    log_turn_debug(graph_state["interview"], "start_voice_interview", next_step, question_text, answer_text)

    return {
        "state": graph_state["interview"].model_dump(),
        "next_step": next_step
    }


@app.post("/api/next-voice-turn")
def next_voice_turn(state: InterviewState) -> dict[str, Any]:
    use_local_agents = not bool(getattr(settings, "groq_api_key", ""))
    interview_agent = build_interview_agent(local=use_local_agents)
    evaluation_agent = build_evaluation_agent(local=use_local_agents)

    resume_text = state.understanding.resume.summary if state.understanding else ""
    jd_text = state.understanding.job_description.summary if state.understanding else ""

    # Build retriever dynamically
    base_dir = Path(__file__).resolve().parents[0]
    sources = [
        ("resume", resume_text),
        ("job_description", jd_text),
        ("guidelines", (base_dir / "guidelines" / "interview_guidelines.md").read_text(encoding="utf-8")),
    ]
    retriever = FaissRetriever(chunks=chunk_many(sources), embedding_model=HashingEmbeddingModel())
    retriever_node_inst = RetrieverNode(retriever)

    # Retrieve next step
    graph_state = {"interview": state, "candidate_answer": ""}
    next_step = should_continue(graph_state)

    question_text = ""
    report_payload = None
    answer_text = ""

    if next_step == "retriever":
        graph_state = make_retriever_node(retriever_node_inst)(graph_state)
        graph_state = make_interview_node(interview_agent)(graph_state)
        question_text = graph_state["interview"].current_question or ""

        # Speak and listen on the server
        answer_text = speak_and_listen_server(question_text)

        # Process candidate answer
        graph_state["candidate_answer"] = answer_text
        graph_state = record_answer_node(graph_state)
        graph_state = make_evaluation_node(evaluation_agent)(graph_state)
        graph_state = memory_update_node(graph_state)

        next_step = should_continue(graph_state)

        # Log turn details
        log_turn_debug(graph_state["interview"], "next_voice_turn", next_step, question_text, answer_text)

    elif next_step == "closing":
        graph_state = closing_node(graph_state)
        question_text = graph_state["interview"].current_question or ""

        # Speak and listen on the server
        answer_text = speak_and_listen_server(question_text)

        # Process candidate answer
        graph_state["candidate_answer"] = answer_text
        graph_state = record_answer_node(graph_state)
        graph_state = make_evaluation_node(evaluation_agent)(graph_state)
        graph_state = memory_update_node(graph_state)

        next_step = should_continue(graph_state)

        # Log turn details
        log_turn_debug(graph_state["interview"], "next_voice_turn", next_step, question_text, answer_text)

    elif next_step == "report":
        graph_state = report_node(graph_state)
        report = LocalReportAgent().create_report(graph_state["interview"])
        report_payload = {
            "candidate_name": report.candidate_name,
            "overall_score": report.overall_score,
            "hiring_recommendation": report.hiring_recommendation,
            "executive_summary": report.executive_summary,
            "turns": len(graph_state["interview"].previous_questions),
            "strong_areas": report.strong_areas,
            "improvement_areas": report.improvement_areas,
            "dimension_summaries": [item.model_dump() for item in report.dimension_summaries],
        }
        store.save(report_payload)

    return {
        "state": graph_state["interview"].model_dump(),
        "next_step": next_step,
        "report": report_payload
    }


@app.get("/api/reports")
def list_reports() -> list[dict[str, Any]]:
    return store.list()
