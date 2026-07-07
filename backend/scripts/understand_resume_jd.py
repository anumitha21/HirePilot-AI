import argparse
import json
import logging
from pathlib import Path

from backend.ai.groq_client import GroqStructuredClient
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.core.prompts import PromptLoader
from backend.services.understanding import (
    GroqUnderstandingExtractor,
    LocalSampleUnderstandingExtractor,
    UnderstandingExtractor,
)

logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    resume_text, jd_text = load_inputs(args)
    extractor = build_extractor(args.local)
    result = extractor.extract(resume_text=resume_text, jd_text=jd_text)

    print(result.model_dump_json(indent=2))
    logger.info("M1 understanding extraction completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured resume and JD understanding.")
    parser.add_argument("--resume-file", type=Path, help="Path to a plain-text resume file.")
    parser.add_argument("--jd-file", type=Path, help="Path to a plain-text job-description file.")
    parser.add_argument("--sample", action="store_true", help="Use bundled sample resume and JD files.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use deterministic local extraction for verification without Groq.",
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[str, str]:
    if args.sample:
        sample_dir = Path(__file__).resolve().parents[1] / "samples"
        return (
            (sample_dir / "sample_resume.txt").read_text(encoding="utf-8"),
            (sample_dir / "sample_jd.txt").read_text(encoding="utf-8"),
        )

    if not args.resume_file or not args.jd_file:
        raise ValueError("Provide --sample or both --resume-file and --jd-file.")

    return (
        args.resume_file.read_text(encoding="utf-8"),
        args.jd_file.read_text(encoding="utf-8"),
    )


def build_extractor(local: bool) -> UnderstandingExtractor:
    if local:
        return LocalSampleUnderstandingExtractor()

    settings = get_settings()
    client = GroqStructuredClient(
        api_key=settings.groq_api_key,
        model=settings.groq_extraction_model,
    )
    return GroqUnderstandingExtractor(client=client, prompt_loader=PromptLoader())


if __name__ == "__main__":
    main()

