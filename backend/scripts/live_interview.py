import argparse
import logging
from pathlib import Path

from backend.ai.groq_client import GroqStructuredClient
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.prompts import PromptLoader
from backend.models.state import InterviewStageName, InterviewState
from backend.retrieval.chunking import chunk_many
from backend.retrieval.embeddings import HashingEmbeddingModel
from backend.retrieval.faiss_store import FaissRetriever, Retriever
from backend.services.interview import GroqInterviewAgent, InterviewAgent, LocalAdaptiveInterviewAgent
from backend.services.planner import LocalSamplePlannerAgent
from backend.services.retriever_node import RetrieverNode
from backend.services.understanding import LocalSampleUnderstandingExtractor
from backend.voice.audio_io import AudioIOError, LocalAudioIO
from backend.voice.stt import SpeechToTextError, WhisperSpeechToText
from backend.voice.tts import EdgeTextToSpeech, TextToSpeechError

logger = logging.getLogger(__name__)

TEXT_MODE_SAMPLE_ANSWERS = [
    "I built an AI interview assistant with Python and FastAPI. The API accepted resume and job description text and returned structured JSON.",
    "For FastAPI, I separated routes by workflow, used Pydantic schemas for validation, and handled errors with consistent response models.",
    "I also worked on a RAG notes search using LangChain and FAISS, but Docker is still an area where I need more production practice.",
]


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    state = build_sample_state()
    retriever = build_sample_retriever()
    interview_agent = build_interview_agent(local=args.local)

    if args.text:
        run_text_interview(state=state, retriever=retriever, interview_agent=interview_agent, turns=args.turns)
    else:
        run_voice_interview(
            state=state,
            retriever=retriever,
            interview_agent=interview_agent,
            turns=args.turns,
            settings=settings,
        )

    print(state.model_dump_json(indent=2))
    logger.info("M4 interview loop completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M4 adaptive interview loop.")
    parser.add_argument("--sample", action="store_true", help="Use bundled sample resume/JD/guidelines.")
    parser.add_argument("--local", action="store_true", help="Use deterministic local question generation.")
    parser.add_argument("--text", action="store_true", help="Use sample typed answers instead of microphone/TTS.")
    parser.add_argument("--turns", type=int, default=3, help="Number of interview turns to run.")
    return parser.parse_args()


def run_text_interview(
    state: InterviewState,
    retriever: Retriever,
    interview_agent: InterviewAgent,
    turns: int,
) -> None:
    for turn_index in range(turns):
        refresh_context(state=state, retriever=retriever)
        question = interview_agent.next_question(state)
        state.remember_question(question.question)
        state.topics_covered.append(question.topic)

        answer = TEXT_MODE_SAMPLE_ANSWERS[min(turn_index, len(TEXT_MODE_SAMPLE_ANSWERS) - 1)]
        state.remember_answer(answer)


def run_voice_interview(
    state: InterviewState,
    retriever: Retriever,
    interview_agent: InterviewAgent,
    turns: int,
    settings,
) -> None:
    audio_io = LocalAudioIO(sample_rate=settings.voice_sample_rate)
    speech_to_text = WhisperSpeechToText(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
    )
    text_to_speech = EdgeTextToSpeech(voice=settings.voice_tts_voice)

    for _ in range(turns):
        refresh_context(state=state, retriever=retriever)
        question = interview_agent.next_question(state)
        state.remember_question(question.question)
        state.topics_covered.append(question.topic)
        logger.info("Interviewer question: %s", question.question)
        print(f"\nInterviewer: {question.question}")

        try:
            spoken_question_path = text_to_speech.synthesize_to_file(question.question)
            audio_io.play_audio_file(spoken_question_path)
            spoken_question_path.unlink(missing_ok=True)
        except (TextToSpeechError, AudioIOError) as exc:
            logger.warning("Voice playback unavailable, using text only: %s", exc)
            print("[Voice playback unavailable. Continuing in text mode.]")

        try:
            logger.info("Recording answer until silence, up to %s seconds.", settings.voice_max_record_seconds)
            answer_path = audio_io.record_until_silence(
                max_duration_seconds=settings.voice_max_record_seconds,
                silence_seconds=settings.voice_silence_seconds,
                vad_threshold=settings.voice_vad_threshold,
            )
            answer = speech_to_text.transcribe(answer_path)
            answer_path.unlink(missing_ok=True)
        except (AudioIOError, SpeechToTextError) as exc:
            logger.warning("Voice capture unavailable, using fallback answer: %s", exc)
            answer = "I could not answer clearly."

        if not answer:
            answer = "I could not answer clearly."
        print(f"Candidate: {answer}\n")
        state.remember_answer(answer)


def refresh_context(state: InterviewState, retriever: Retriever) -> None:
    query = state.current_question or state.interview_plan.opening_question if state.interview_plan else ""
    if state.previous_answers:
        query = state.previous_answers[-1]
    RetrieverNode(retriever).run(state=state, query_text=query, top_k=4)


def build_sample_state() -> InterviewState:
    base_dir = Path(__file__).resolve().parents[1]
    resume_text = (base_dir / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
    jd_text = (base_dir / "samples" / "sample_jd.txt").read_text(encoding="utf-8")
    understanding = LocalSampleUnderstandingExtractor().extract(resume_text=resume_text, jd_text=jd_text)
    plan = LocalSamplePlannerAgent().create_plan(understanding)

    return InterviewState(
        candidate_name=understanding.resume.candidate_name,
        understanding=understanding,
        interview_plan=plan,
        strong_areas=plan.strong_areas_to_validate,
        weak_areas=plan.weak_or_unclear_areas_to_probe,
        remaining_topics=[category.name for category in plan.question_categories],
        current_stage=InterviewStageName.INTERVIEWING,
    )


def build_sample_retriever() -> Retriever:
    base_dir = Path(__file__).resolve().parents[1]
    sources = [
        ("resume", (base_dir / "samples" / "sample_resume.txt").read_text(encoding="utf-8")),
        ("job_description", (base_dir / "samples" / "sample_jd.txt").read_text(encoding="utf-8")),
        ("guidelines", (base_dir / "guidelines" / "interview_guidelines.md").read_text(encoding="utf-8")),
    ]
    return FaissRetriever(chunks=chunk_many(sources), embedding_model=HashingEmbeddingModel())


def build_interview_agent(local: bool) -> InterviewAgent:
    if local:
        return LocalAdaptiveInterviewAgent()

    settings = get_settings()
    client = GroqStructuredClient(
        api_key=settings.groq_api_key,
        model=settings.groq_interview_model,
    )
    return GroqInterviewAgent(client=client, prompt_loader=PromptLoader())


if __name__ == "__main__":
    main()
