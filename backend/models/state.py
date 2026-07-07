from enum import StrEnum

from pydantic import BaseModel, Field

from backend.models.planner import InterviewPlan
from backend.models.understanding import UnderstandingResult


class InterviewStageName(StrEnum):
    PLANNING = "planning"
    INTERVIEWING = "interviewing"
    EVALUATING = "evaluating"
    REPORTING = "reporting"
    COMPLETE = "complete"


class ConversationTurn(BaseModel):
    speaker: str
    text: str


class RubricScore(BaseModel):
    dimension: str
    score: float = Field(ge=0, le=5)
    reasoning: str


class AnswerScore(BaseModel):
    question: str
    answer: str
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0, le=5)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    reasoning: str


class RetrievedContext(BaseModel):
    source: str
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class InterviewState(BaseModel):
    candidate_name: str | None = None
    understanding: UnderstandingResult | None = None
    interview_plan: InterviewPlan | None = None
    current_question: str | None = None
    previous_questions: list[str] = Field(default_factory=list)
    previous_answers: list[str] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    current_scores: list[AnswerScore] = Field(default_factory=list)
    strong_areas: list[str] = Field(default_factory=list)
    weak_areas: list[str] = Field(default_factory=list)
    topics_covered: list[str] = Field(default_factory=list)
    remaining_topics: list[str] = Field(default_factory=list)
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    current_stage: InterviewStageName = InterviewStageName.PLANNING
    final_report: dict[str, object] | None = None

    def remember_question(self, question: str) -> None:
        self.current_question = question
        self.previous_questions.append(question)
        self.conversation_history.append(ConversationTurn(speaker="interviewer", text=question))

    def remember_answer(self, answer: str) -> None:
        self.previous_answers.append(answer)
        self.conversation_history.append(ConversationTurn(speaker="candidate", text=answer))
