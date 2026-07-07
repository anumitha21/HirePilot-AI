from pydantic import BaseModel, Field


class CandidateProject(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    impact: list[str] = Field(default_factory=list)


class CandidateExperience(BaseModel):
    title: str = "Unknown"
    company: str | None = None
    duration: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str
    credential: str = "Unknown"
    field: str | None = None
    year: str | None = None


class ResumeUnderstanding(BaseModel):
    model_config = {"extra": "ignore"}

    candidate_name: str | None = None
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    projects: list[CandidateProject] = Field(default_factory=list)
    experience: list[CandidateExperience] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    years_of_experience: float | None = None


class JobDescriptionUnderstanding(BaseModel):
    model_config = {"extra": "ignore"}

    role_title: str
    company: str | None = None
    summary: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    experience_level: str | None = None


class SkillGapAnalysis(BaseModel):
    matching_skills: list[str] = Field(default_factory=list)
    missing_or_unclear_skills: list[str] = Field(default_factory=list)
    strong_signals: list[str] = Field(default_factory=list)
    interview_focus_areas: list[str] = Field(default_factory=list)


class UnderstandingResult(BaseModel):
    resume: ResumeUnderstanding
    job_description: JobDescriptionUnderstanding
    skill_gap_analysis: SkillGapAnalysis

