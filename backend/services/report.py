from __future__ import annotations

from typing import Protocol

from backend.models.report import DimensionSummary, InterviewReport
from backend.models.state import InterviewState


class ReportAgent(Protocol):
    def create_report(self, state: InterviewState) -> InterviewReport:
        """Create a recruiter-ready report for a completed interview."""


class LocalReportAgent:
    def create_report(self, state: InterviewState) -> InterviewReport:
        scores = state.current_scores
        if not scores:
            overall_score = 0.0
        else:
            overall_score = round(sum(item.overall_score for item in scores) / len(scores), 2)

        recommendation = self._recommendation(overall_score)
        role_title = state.interview_plan.role_title if state.interview_plan else "Technical Role"
        strong_areas = state.strong_areas[:3] or ["Clear communication", "Structured thinking"]
        improvement_areas = state.weak_areas[:3] or ["More deployment depth"]

        dimension_map: dict[str, list[float]] = {}
        for score in scores:
            for rubric in score.rubric_scores:
                dimension_map.setdefault(rubric.dimension, []).append(rubric.score)

        dimension_summaries = [
            DimensionSummary(
                dimension=dimension,
                average_score=round(sum(values) / len(values), 2),
                comment=self._dimension_comment(dimension, sum(values) / len(values)),
            )
            for dimension, values in sorted(dimension_map.items())
        ]

        return InterviewReport(
            candidate_name=state.candidate_name or "Candidate",
            role_title=role_title,
            overall_score=overall_score,
            hiring_recommendation=recommendation,
            executive_summary=(
                f"{state.candidate_name or 'The candidate'} demonstrated strong technical grounding "
                f"for the {role_title} role, with clear examples around {', '.join(strong_areas[:2])}. "
                f"The interview showed good communication and structured reasoning, with the main "
                f"growth area being {improvement_areas[0].lower()}."
            ),
            dimension_summaries=dimension_summaries,
            strong_areas=strong_areas,
            improvement_areas=improvement_areas,
            standout_moments=[
                "Provided concrete examples tied to implementation decisions.",
                "Connected technical choices to the role's expectations.",
            ],
            suggested_next_steps=[
                "Advance to a technical deep-dive round.",
                "Give a short take-home task around production deployment.",
            ],
        )

    def _recommendation(self, overall_score: float) -> str:
        if overall_score >= 4.2:
            return "Strong Hire"
        if overall_score >= 3.5:
            return "Hire"
        if overall_score >= 2.5:
            return "Hold"
        return "No Hire"

    def _dimension_comment(self, dimension: str, value: float) -> str:
        if value >= 4.2:
            return f"{dimension.replace('_', ' ')} was a clear strength."
        if value >= 3.0:
            return f"{dimension.replace('_', ' ')} was solid and consistent."
        return f"{dimension.replace('_', ' ')} needs more concrete examples."
