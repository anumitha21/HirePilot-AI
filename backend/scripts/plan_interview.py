import argparse
import logging
from pathlib import Path

from backend.ai.groq_client import GroqStructuredClient
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.prompts import PromptLoader
from backend.models.understanding import UnderstandingResult
from backend.services.planner import GroqPlannerAgent, LocalSamplePlannerAgent, PlannerAgent
from backend.services.understanding import LocalSampleUnderstandingExtractor

logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    understanding = load_understanding(args)
    planner = build_planner(args.local)
    plan = planner.create_plan(understanding)

    print(plan.model_dump_json(indent=2))
    logger.info("M2 planner completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a structured interview plan.")
    parser.add_argument("--understanding-file", type=Path, help="Path to M1 understanding JSON.")
    parser.add_argument("--sample", action="store_true", help="Use bundled M1 sample inputs.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use deterministic local planning for verification without Groq.",
    )
    return parser.parse_args()


def load_understanding(args: argparse.Namespace) -> UnderstandingResult:
    if args.sample:
        sample_dir = Path(__file__).resolve().parents[1] / "samples"
        resume_text = (sample_dir / "sample_resume.txt").read_text(encoding="utf-8")
        jd_text = (sample_dir / "sample_jd.txt").read_text(encoding="utf-8")
        return LocalSampleUnderstandingExtractor().extract(resume_text=resume_text, jd_text=jd_text)

    if not args.understanding_file:
        raise ValueError("Provide --sample or --understanding-file.")

    return UnderstandingResult.model_validate_json(
        args.understanding_file.read_text(encoding="utf-8")
    )


def build_planner(local: bool) -> PlannerAgent:
    if local:
        return LocalSamplePlannerAgent()

    settings = get_settings()
    client = GroqStructuredClient(
        api_key=settings.groq_api_key,
        model=settings.groq_planner_model,
    )
    return GroqPlannerAgent(client=client, prompt_loader=PromptLoader())


if __name__ == "__main__":
    main()

