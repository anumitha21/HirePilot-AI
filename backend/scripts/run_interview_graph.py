"""M6: Run a full end-to-end interview session through the LangGraph graph.

Local text mode (no mic, no Groq key):
    python -m backend.scripts.run_interview_graph --sample --local --text

Live voice mode (requires GROQ_API_KEY):
    python -m backend.scripts.run_interview_graph --sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.ai.groq_client import GroqStructuredClient
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.prompts import PromptLoader
from backend.graph.graph import build_graph
from backend.graph.nodes import GraphState
from backend.models.state import InterviewStageName, InterviewState
from backend.retrieval.chunking import chunk_many
from backend.retrieval.embeddings import HashingEmbeddingModel
from backend.retrieval.faiss_store import FaissRetriever, Retriever
from backend.services.evaluation import EvaluationAgent, GroqEvaluationAgent, LocalSampleEvaluationAgent
from backend.services.interview import GroqInterviewAgent, InterviewAgent, LocalAdaptiveInterviewAgent
from backend.services.planner import GroqPlannerAgent, LocalSamplePlannerAgent, PlannerAgent
from backend.services.understanding import GroqUnderstandingExtractor, LocalSampleUnderstandingExtractor

logger = logging.getLogger(__name__)

TEXT_ANSWERS = [
    "I built an AI interview assistant with Python and FastAPI. The API accepted resume and job description text and returned structured JSON.",
    "For FastAPI, I separated routes by workflow, used Pydantic schemas for validation, and handled errors with consistent response models.",
    "I also worked on a RAG notes search using LangChain and FAISS, but Docker is still an area where I need more production practice.",
    "For LLM evaluation I used rubric-based scoring with Pydantic structured output so every answer got a consistent score object.",
    "I would approach deployment by containerising the service with Docker and using a managed cloud run service to keep ops overhead low.",
]


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    state = build_initial_state(resume_file=args.resume_file, jd_file=args.jd_file)
    retriever = build_retriever(resume_file=args.resume_file, jd_file=args.jd_file)
    planner = build_planner(local=args.local)
    interview_agent = build_interview_agent(local=args.local)
    evaluation_agent = build_evaluation_agent(local=args.local)

    graph = build_graph(
        planner=planner,
        retriever=retriever,
        interview_agent=interview_agent,
        evaluation_agent=evaluation_agent,
    )

    if args.text:
        run_text_mode(graph=graph, state=state, turns=args.turns)
    else:
        run_voice_mode(graph=graph, state=state, turns=args.turns, settings=settings)

    final_state: InterviewState = state
    import sys
    sys.stdout.buffer.write(final_state.model_dump_json(indent=2).encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    logger.info("M6 graph run complete.")


# ---------------------------------------------------------------------------
# Text mode: feed pre-written answers turn by turn
# ---------------------------------------------------------------------------

def run_text_mode(graph, state: InterviewState, turns: int) -> None:
    graph_state: GraphState = {"interview": state, "candidate_answer": ""}

    # Turn 0: planner + intro greeting (candidate answers with self-introduction)
    graph_state["candidate_answer"] = TEXT_ANSWERS[0]
    for node_name in ["planner", "intro", "record_answer", "evaluation", "memory_update"]:
        graph_state = graph.nodes[node_name].invoke(graph_state)

    interview = graph_state["interview"]
    print("\n--- Introduction ---")
    print(f"Interviewer: {interview.previous_questions[-1]}")
    print(f"Candidate:   {interview.previous_answers[-1]}")

    # Subsequent turns: retriever → interview → record_answer → evaluation → memory_update
    for turn_index in range(1, turns + 1):
        answer = TEXT_ANSWERS[min(turn_index, len(TEXT_ANSWERS) - 1)]
        graph_state["candidate_answer"] = answer

        for node_name in ["retriever", "interview", "record_answer", "evaluation", "memory_update"]:
            graph_state = graph.nodes[node_name].invoke(graph_state)

        interview = graph_state["interview"]
        print(f"\n--- Turn {turn_index} [{interview.interview_stage}/{interview.current_difficulty}] ---")
        print(f"Interviewer: {interview.previous_questions[-1]}")
        print(f"Candidate:   {interview.previous_answers[-1]}")
        if interview.current_scores:
            print(f"Score: {interview.current_scores[-1].overall_score:.2f}")

        if interview.current_stage == InterviewStageName.COMPLETE:
            break

    # closing speech
    graph_state = graph.nodes["closing"].invoke(graph_state)
    graph_state["candidate_answer"] = TEXT_ANSWERS[-1]
    for node_name in ["record_answer", "evaluation", "memory_update"]:
        graph_state = graph.nodes[node_name].invoke(graph_state)
    interview = graph_state["interview"]
    print(f"\n--- Closing ---")
    print(f"Interviewer: {interview.previous_questions[-1]}")
    print(f"Candidate:   {interview.previous_answers[-1]}")

    from backend.graph.nodes import report_node
    graph_state = report_node(graph_state)

    state.__dict__.update(graph_state["interview"].__dict__)


def _run_one_turn(graph, graph_state: GraphState, entry_node: str) -> GraphState:
    """Used by voice mode for fine-grained turn control."""
    if entry_node == "planner":
        nodes_in_order = ["planner", "intro", "record_answer", "evaluation", "memory_update"]
    else:
        nodes_in_order = ["retriever", "interview", "record_answer", "evaluation", "memory_update"]
    for node_name in nodes_in_order:
        graph_state = graph.nodes[node_name].invoke(graph_state)
    return graph_state


# ---------------------------------------------------------------------------
# Voice mode
# ---------------------------------------------------------------------------

def run_voice_mode(graph, state: InterviewState, turns: int, settings) -> None:
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
    graph_state: GraphState = {"interview": state, "candidate_answer": ""}

    def speak_and_listen(question: str) -> str:
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
        return answer

    # --- Intro turn: planner + greeting ---
    graph_state = graph.nodes["planner"].invoke(graph_state)
    graph_state = graph.nodes["intro"].invoke(graph_state)
    greeting = graph_state["interview"].current_question or ""
    answer = speak_and_listen(greeting)
    graph_state["candidate_answer"] = answer
    for node_name in ["record_answer", "evaluation", "memory_update"]:
        graph_state = graph.nodes[node_name].invoke(graph_state)

    # --- Main interview turns ---
    for _ in range(turns):
        graph_state["candidate_answer"] = ""
        for node_name in ["retriever", "interview"]:
            graph_state = graph.nodes[node_name].invoke(graph_state)

        question = graph_state["interview"].current_question or ""
        answer = speak_and_listen(question)
        graph_state["candidate_answer"] = answer

        for node_name in ["record_answer", "evaluation", "memory_update"]:
            graph_state = graph.nodes[node_name].invoke(graph_state)

        if graph_state["interview"].current_stage == InterviewStageName.COMPLETE:
            break

    # closing speech
    if graph_state["interview"].current_stage != InterviewStageName.COMPLETE:
        graph_state = graph.nodes["closing"].invoke(graph_state)
        closing_q = graph_state["interview"].current_question or ""
        answer = speak_and_listen(closing_q)
        graph_state["candidate_answer"] = answer
        for node_name in ["record_answer", "evaluation", "memory_update"]:
            graph_state = graph.nodes[node_name].invoke(graph_state)

    from backend.graph.nodes import report_node
    graph_state = report_node(graph_state)

    state.__dict__.update(graph_state["interview"].__dict__)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_initial_state(resume_file: str | None = None, jd_file: str | None = None) -> InterviewState:
    base_dir = Path(__file__).resolve().parents[1]
    resume_text = Path(resume_file).read_text(encoding="utf-8") if resume_file else (base_dir / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
    jd_text = Path(jd_file).read_text(encoding="utf-8") if jd_file else (base_dir / "samples" / "sample_jd.txt").read_text(encoding="utf-8")
    settings = get_settings()
    if settings.groq_api_key:
        extractor = GroqUnderstandingExtractor(
            client=GroqStructuredClient(api_key=settings.groq_api_key, model=settings.groq_extraction_model),
            prompt_loader=PromptLoader(),
        )
    else:
        extractor = LocalSampleUnderstandingExtractor()
    understanding = extractor.extract(resume_text=resume_text, jd_text=jd_text)
    return InterviewState(
        candidate_name=understanding.resume.candidate_name,
        understanding=understanding,
        current_stage=InterviewStageName.PLANNING,
    )


def build_retriever(resume_file: str | None = None, jd_file: str | None = None) -> Retriever:
    base_dir = Path(__file__).resolve().parents[1]
    resume_text = Path(resume_file).read_text(encoding="utf-8") if resume_file else (base_dir / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
    jd_text = Path(jd_file).read_text(encoding="utf-8") if jd_file else (base_dir / "samples" / "sample_jd.txt").read_text(encoding="utf-8")
    sources = [
        ("resume", resume_text),
        ("job_description", jd_text),
        ("guidelines", (base_dir / "guidelines" / "interview_guidelines.md").read_text(encoding="utf-8")),
    ]
    return FaissRetriever(chunks=chunk_many(sources), embedding_model=HashingEmbeddingModel())


def build_planner(local: bool) -> PlannerAgent:
    if local:
        return LocalSamplePlannerAgent()
    settings = get_settings()
    return GroqPlannerAgent(
        client=GroqStructuredClient(api_key=settings.groq_api_key, model=settings.groq_planner_model),
        prompt_loader=PromptLoader(),
    )


def build_interview_agent(local: bool) -> InterviewAgent:
    if local:
        return LocalAdaptiveInterviewAgent()
    settings = get_settings()
    return GroqInterviewAgent(
        client=GroqStructuredClient(api_key=settings.groq_api_key, model=settings.groq_interview_model),
        prompt_loader=PromptLoader(),
    )


def build_evaluation_agent(local: bool) -> EvaluationAgent:
    if local:
        return LocalSampleEvaluationAgent()
    settings = get_settings()
    return GroqEvaluationAgent(
        client=GroqStructuredClient(api_key=settings.groq_api_key, model=settings.groq_evaluation_model),
        prompt_loader=PromptLoader(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M6 LangGraph interview session.")
    parser.add_argument("--sample", action="store_true", help="Use bundled sample data.")
    parser.add_argument("--local", action="store_true", help="Use local deterministic agents (no Groq key needed).")
    parser.add_argument("--text", action="store_true", help="Use pre-written answers instead of microphone.")
    parser.add_argument("--turns", type=int, default=12, help="Max interview turns across all stages.")
    parser.add_argument("--resume-file", default=None, help="Path to your resume .txt file.")
    parser.add_argument("--jd-file", default=None, help="Path to your job description .txt file.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
