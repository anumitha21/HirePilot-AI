import argparse
import logging

from backend.ai.groq_client import GroqStructuredClient
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.prompts import PromptLoader
from backend.models.state import InterviewStageName, InterviewState
from backend.services.evaluation import (
    EvaluationAgent,
    GroqEvaluationAgent,
    LocalSampleEvaluationAgent,
)
from backend.services.planner import LocalSamplePlannerAgent
from backend.services.understanding import LocalSampleUnderstandingExtractor

logger = logging.getLogger(__name__)

SAMPLE_QUESTION = (
    "You mentioned FastAPI earlier. How did you structure the routes, validation, "
    "and error handling in that API?"
)
SAMPLE_ANSWER = (
    "I separated FastAPI routes by workflow, used Pydantic schemas for request validation, "
    "and returned consistent error response models so clients could handle failures predictably."
)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    state = build_sample_state()
    evaluator = build_evaluator(local=args.local)
    score = evaluator.evaluate_answer(
        state=state,
        question=args.question,
        answer=args.answer,
    )

    if score not in state.current_scores:
        state.current_scores.append(score)

    print(state.model_dump_json(indent=2))
    logger.info("M5 evaluation completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a candidate answer.")
    parser.add_argument("--sample", action="store_true", help="Use the bundled sample question and answer.")
    parser.add_argument("--local", action="store_true", help="Use deterministic local evaluation.")
    parser.add_argument("--question", default=SAMPLE_QUESTION, help="Question to evaluate.")
    parser.add_argument("--answer", default=SAMPLE_ANSWER, help="Candidate answer to evaluate.")
    return parser.parse_args()


def build_sample_state() -> InterviewState:
    resume_text = "Sample resume with Python, FastAPI, LangChain, RAG, and SQL project work."
    jd_text = "GenAI Engineer role requiring Python, FastAPI, structured outputs, and evaluation."
    understanding = LocalSampleUnderstandingExtractor().extract(resume_text=resume_text, jd_text=jd_text)
    plan = LocalSamplePlannerAgent().create_plan(understanding)

    return InterviewState(
        candidate_name=understanding.resume.candidate_name,
        understanding=understanding,
        interview_plan=plan,
        current_question=SAMPLE_QUESTION,
        previous_questions=[SAMPLE_QUESTION],
        previous_answers=[SAMPLE_ANSWER],
        strong_areas=plan.strong_areas_to_validate,
        weak_areas=plan.weak_or_unclear_areas_to_probe,
        current_stage=InterviewStageName.EVALUATING,
    )


def build_evaluator(local: bool) -> EvaluationAgent:
    if local:
        return LocalSampleEvaluationAgent()

    settings = get_settings()
    client = GroqStructuredClient(
        api_key=settings.groq_api_key,
        model=settings.groq_evaluation_model,
    )
    return GroqEvaluationAgent(client=client, prompt_loader=PromptLoader())


if __name__ == "__main__":
    main()

