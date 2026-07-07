from pydantic import BaseModel, Field


class InterviewStage(BaseModel):
    name: str
    purpose: str
    target_minutes: int = Field(ge=1)
    sample_questions: list[str] = Field(default_factory=list)


class QuestionCategory(BaseModel):
    name: str
    priority: int = Field(ge=1, le=5)
    skills_to_probe: list[str] = Field(default_factory=list)
    rationale: str


class InterviewPlan(BaseModel):
    role_title: str
    candidate_name: str | None = None
    interview_goal: str
    opening_question: str
    stages: list[InterviewStage]
    question_categories: list[QuestionCategory]
    strong_areas_to_validate: list[str] = Field(default_factory=list)
    weak_or_unclear_areas_to_probe: list[str] = Field(default_factory=list)
    skill_gap_questions: list[str] = Field(default_factory=list)
    communication_signals_to_watch: list[str] = Field(default_factory=list)
    exit_criteria: list[str] = Field(default_factory=list)

