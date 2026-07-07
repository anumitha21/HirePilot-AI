import json
from typing import Protocol

from backend.ai.groq_client import GroqStructuredClient
from backend.core.prompts import PromptLoader
from backend.models.planner import InterviewPlan, InterviewStage, QuestionCategory
from backend.models.understanding import UnderstandingResult


class PlannerAgent(Protocol):
    def create_plan(self, understanding: UnderstandingResult) -> InterviewPlan:
        """Create an interview plan from structured resume/JD understanding."""


class GroqPlannerAgent:
    def __init__(self, client: GroqStructuredClient, prompt_loader: PromptLoader) -> None:
        self.client = client
        self.prompt_loader = prompt_loader

    def create_plan(self, understanding: UnderstandingResult) -> InterviewPlan:
        system_prompt = self.prompt_loader.load("planner.md")
        schema_json = json.dumps(InterviewPlan.model_json_schema(), indent=2)
        resume = understanding.resume
        jd = understanding.job_description
        gaps = understanding.skill_gap_analysis
        user_prompt = (
            f"Required JSON schema:\n{schema_json}\n\n"
            f"Role: {jd.role_title}\n"
            f"Candidate: {resume.candidate_name}\n"
            f"Resume skills: {resume.skills}\n"
            f"JD required skills: {jd.required_skills}\n"
            f"Matching skills: {gaps.matching_skills}\n"
            f"Missing skills: {gaps.missing_or_unclear_skills}\n"
            f"Strong signals: {gaps.strong_signals}\n"
            f"Focus areas: {gaps.interview_focus_areas}"
        )
        result = self.client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=InterviewPlan,
        )
        return InterviewPlan.model_validate(result.model_dump())


class LocalSamplePlannerAgent:
    """Deterministic planner used only for local M2 verification."""

    def create_plan(self, understanding: UnderstandingResult) -> InterviewPlan:
        resume = understanding.resume
        jd = understanding.job_description
        gaps = understanding.skill_gap_analysis
        focus_areas = gaps.interview_focus_areas or gaps.missing_or_unclear_skills
        strong_areas = gaps.strong_signals or resume.skills[:3]
        matching_skills = gaps.matching_skills or resume.skills[:3]

        return InterviewPlan(
            role_title=jd.role_title,
            candidate_name=resume.candidate_name,
            interview_goal=(
                f"Assess fit for {jd.role_title} by validating practical project depth, "
                "role-critical skills, and communication clarity."
            ),
            opening_question=(
                "Thanks for joining. Could you walk me through the project on your resume "
                "that best matches this GenAI engineer role?"
            ),
            stages=[
                InterviewStage(
                    name="Opening and resume grounding",
                    purpose="Establish context and let the candidate anchor the conversation.",
                    target_minutes=5,
                    sample_questions=[
                        "Which project best represents your current backend and GenAI skills?"
                    ],
                ),
                InterviewStage(
                    name="Technical depth",
                    purpose="Probe implementation details for the strongest resume signals.",
                    target_minutes=15,
                    sample_questions=[
                        "You mentioned FastAPI; how did you structure the API and validation layer?"
                    ],
                ),
                InterviewStage(
                    name="Gap and scenario probing",
                    purpose="Explore unclear or missing requirements from the job description.",
                    target_minutes=10,
                    sample_questions=[
                        f"How would you approach {focus_areas[0]} in a production AI workflow?"
                    ],
                ),
                InterviewStage(
                    name="Wrap-up",
                    purpose="Clarify remaining risks and summarize candidate strengths.",
                    target_minutes=5,
                    sample_questions=["What part of this role would be your steepest learning curve?"],
                ),
            ],
            question_categories=[
                QuestionCategory(
                    name="Backend API design",
                    priority=5,
                    skills_to_probe=[skill for skill in matching_skills if skill in {"python", "fastapi", "sql"}],
                    rationale="The role needs reliable API services around AI workflows.",
                ),
                QuestionCategory(
                    name="LLM application engineering",
                    priority=5,
                    skills_to_probe=[skill for skill in matching_skills if skill in {"llm", "langchain", "rag"}],
                    rationale="The candidate should explain retrieval, prompts, and structured output choices.",
                ),
                QuestionCategory(
                    name="Skill gaps",
                    priority=4,
                    skills_to_probe=focus_areas,
                    rationale="Unclear requirements should become adaptive follow-up topics.",
                ),
            ],
            strong_areas_to_validate=strong_areas,
            weak_or_unclear_areas_to_probe=focus_areas,
            skill_gap_questions=[
                f"Your resume is less explicit about {area}. What have you built or learned there?"
                for area in focus_areas
            ],
            communication_signals_to_watch=[
                "Can explain trade-offs without overclaiming.",
                "Gives concrete examples from projects.",
                "Clarifies assumptions before proposing solutions.",
            ],
            exit_criteria=[
                "Core matching skills have been validated with project-specific evidence.",
                "Missing or unclear skills have been probed at least once.",
                "Candidate has answered a practical scenario question.",
            ],
        )

