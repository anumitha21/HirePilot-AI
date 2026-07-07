import argparse
import logging
from pathlib import Path

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.models.state import InterviewState
from backend.retrieval.chunking import chunk_many
from backend.retrieval.embeddings import HashingEmbeddingModel
from backend.retrieval.faiss_store import FaissRetriever
from backend.services.retriever_node import RetrieverNode
from backend.services.understanding import LocalSampleUnderstandingExtractor

logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    chunks = chunk_many(load_sources(args))
    retriever = FaissRetriever(chunks=chunks, embedding_model=HashingEmbeddingModel())
    state = build_state(args)
    updated_state = RetrieverNode(retriever).run(state=state, query_text=args.query, top_k=args.top_k)

    print(updated_state.model_dump_json(indent=2))
    logger.info("M3 retriever query completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the M3 FAISS retriever.")
    parser.add_argument("--query", required=True, help="Topic or question to retrieve context for.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to return.")
    parser.add_argument("--sample", action="store_true", help="Use bundled sample resume/JD/guidelines.")
    parser.add_argument("--resume-file", type=Path, help="Plain-text resume file.")
    parser.add_argument("--jd-file", type=Path, help="Plain-text job-description file.")
    parser.add_argument("--guidelines-file", type=Path, help="Plain-text interview guidelines file.")
    return parser.parse_args()


def load_sources(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.sample:
        base_dir = Path(__file__).resolve().parents[1]
        return [
            ("resume", (base_dir / "samples" / "sample_resume.txt").read_text(encoding="utf-8")),
            ("job_description", (base_dir / "samples" / "sample_jd.txt").read_text(encoding="utf-8")),
            ("guidelines", (base_dir / "guidelines" / "interview_guidelines.md").read_text(encoding="utf-8")),
        ]

    if not args.resume_file or not args.jd_file or not args.guidelines_file:
        raise ValueError("Provide --sample or resume, JD, and guidelines files.")

    return [
        ("resume", args.resume_file.read_text(encoding="utf-8")),
        ("job_description", args.jd_file.read_text(encoding="utf-8")),
        ("guidelines", args.guidelines_file.read_text(encoding="utf-8")),
    ]


def build_state(args: argparse.Namespace) -> InterviewState:
    if not args.sample:
        return InterviewState()

    base_dir = Path(__file__).resolve().parents[1]
    resume_text = (base_dir / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
    jd_text = (base_dir / "samples" / "sample_jd.txt").read_text(encoding="utf-8")
    understanding = LocalSampleUnderstandingExtractor().extract(resume_text=resume_text, jd_text=jd_text)

    return InterviewState(
        candidate_name=understanding.resume.candidate_name,
        understanding=understanding,
        strong_areas=understanding.skill_gap_analysis.strong_signals,
        weak_areas=understanding.skill_gap_analysis.missing_or_unclear_skills,
        remaining_topics=understanding.skill_gap_analysis.interview_focus_areas,
    )


if __name__ == "__main__":
    main()

