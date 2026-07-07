from pydantic import BaseModel, Field


class NextQuestion(BaseModel):
    question: str
    topic: str
    rationale: str
    uses_retrieved_context: bool = False
    suggested_followups: list[str] = Field(default_factory=list)

