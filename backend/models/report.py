from pydantic import BaseModel, Field


class DimensionSummary(BaseModel):
    dimension: str
    average_score: float = Field(ge=0, le=5)
    comment: str


class InterviewReport(BaseModel):
    candidate_name: str
    role_title: str
    overall_score: float = Field(ge=0, le=5)
    hiring_recommendation: str          # "Strong Hire" | "Hire" | "Hold" | "No Hire"
    executive_summary: str
    dimension_summaries: list[DimensionSummary] = Field(default_factory=list)
    strong_areas: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)
    standout_moments: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
