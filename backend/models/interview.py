from enum import StrEnum

from pydantic import BaseModel, Field


class InterviewStageFlow(StrEnum):
    INTRODUCTION = "introduction"
    RESUME_VALIDATION = "resume_validation"
    PROJECT_DEEP_DIVE = "project_deep_dive"
    TECHNICAL_SKILLS = "technical_skills"
    PROBLEM_SOLVING = "problem_solving"
    BEHAVIORAL = "behavioral"
    JOB_FIT = "job_fit"
    CLOSING = "closing"


class DifficultyLevel(StrEnum):
    EASY = "easy"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ARCHITECTURE = "architecture"
    SYSTEM_DESIGN = "system_design"


class NextQuestion(BaseModel):
    question: str
    topic: str
    rationale: str
    stage: str = "introduction"
    difficulty: str = "easy"
    uses_retrieved_context: bool = False
    suggested_followups: list[str] = Field(default_factory=list)

