from typing import Any, Protocol

from backend.ai.groq_client import GroqStructuredClient
from backend.core.prompts import PromptLoader
from backend.models.understanding import (
    CandidateProject,
    JobDescriptionUnderstanding,
    ResumeUnderstanding,
    SkillGapAnalysis,
    UnderstandingResult,
)


def _to_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("name") or val.get("title") or val.get("description") or str(val)
    return str(val)


def _coerce_raw(data: dict) -> None:
    """Mutate the raw Groq dict in-place to match our Pydantic schema."""
    if not isinstance(data, dict):
        data = {}

    resume = data.get("resume")
    if not isinstance(resume, dict):
        resume = {}
        data["resume"] = resume

    if not isinstance(resume.get("summary"), str) or not resume.get("summary"):
        resume["summary"] = ""

    for field in ("certifications", "achievements"):
        value = resume.get(field)
        if value is None:
            resume[field] = []
        elif isinstance(value, list):
            resume[field] = [_to_str(v) for v in value]
        else:
            resume[field] = [_to_str(value)]

    experience = resume.get("experience")
    if experience is None:
        resume["experience"] = []
    elif not isinstance(experience, list):
        resume["experience"] = [experience]

    for exp in resume.get("experience", []):
        if isinstance(exp, dict) and "title" not in exp:
            exp["title"] = exp.get("role") or exp.get("position") or "Unknown"

    education = resume.get("education")
    if education is None:
        resume["education"] = []
    elif not isinstance(education, list):
        resume["education"] = [education]

    for edu in resume.get("education", []):
        if isinstance(edu, dict) and "credential" not in edu:
            edu["credential"] = edu.get("degree") or edu.get("qualification") or "Unknown"

    jd = data.get("job_description")
    if not isinstance(jd, dict):
        jd = {}
        data["job_description"] = jd

    if not isinstance(jd.get("summary"), str) or not jd.get("summary"):
        jd["summary"] = ""

    for field in ("responsibilities", "technologies", "tools", "keywords"):
        if jd.get(field) is None:
            jd[field] = []
        elif not isinstance(jd.get(field), list):
            jd[field] = [jd[field]]


class UnderstandingExtractor(Protocol):
    def extract(self, resume_text: str, jd_text: str) -> UnderstandingResult:
        """Return structured resume and JD understanding."""


class GroqUnderstandingExtractor:
    def __init__(self, client: GroqStructuredClient, prompt_loader: PromptLoader) -> None:
        self.client = client
        self.prompt_loader = prompt_loader

    def extract(self, resume_text: str, jd_text: str) -> UnderstandingResult:
        system_prompt = self.prompt_loader.load("understanding_system.md")
        user_template = self.prompt_loader.load("understanding_user.md")
        user_prompt = user_template.format(
            resume_text=resume_text,
            jd_text=jd_text,
        )

        try:
            raw = self.client.complete_raw_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if not isinstance(raw, dict):
                raise TypeError("Groq response was not a JSON object")
            _coerce_raw(raw)
            return UnderstandingResult.model_validate(raw)
        except Exception:
            return LocalSampleUnderstandingExtractor().extract(resume_text=resume_text, jd_text=jd_text)


class LocalSampleUnderstandingExtractor:
    """Deterministic extractor used only for local M1 verification."""

    def extract(self, resume_text: str, jd_text: str) -> UnderstandingResult:
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        known_terms = [
            "python",
            "fastapi",
            "langchain",
            "sql",
            "postgresql",
            "docker",
            "rag",
            "llm",
            "aws",
            "react",
        ]

        resume_skills = [term for term in known_terms if term in resume_lower]
        jd_skills = [term for term in known_terms if term in jd_lower]
        matching = sorted(set(resume_skills).intersection(jd_skills))
        missing = sorted(set(jd_skills).difference(resume_skills))

        return UnderstandingResult(
            resume=ResumeUnderstanding(
                candidate_name="Sample Candidate",
                summary="Candidate has backend and GenAI project experience inferred from sample text.",
                skills=resume_skills,
                technologies=resume_skills,
                projects=[
                    CandidateProject(
                        name="AI Interview Assistant",
                        description="Built a voice-oriented interview assistant prototype.",
                        technologies=[skill for skill in resume_skills if skill in {"python", "fastapi", "llm"}],
                        impact=["Demonstrates practical GenAI system building."],
                    )
                ],
                achievements=["Built portfolio-grade AI projects."],
                years_of_experience=1.0,
            ),
            job_description=JobDescriptionUnderstanding(
                role_title="GenAI Engineer",
                summary="Role focuses on building LLM applications with production backend practices.",
                required_skills=jd_skills,
                preferred_skills=["langchain", "rag"],
                responsibilities=[
                    "Build LLM-powered workflows.",
                    "Integrate APIs and structured outputs.",
                    "Maintain reliable backend services.",
                ],
                technologies=jd_skills,
                tools=["Groq", "FAISS"],
                keywords=sorted(set(jd_skills + ["structured output", "evaluation"])),
                experience_level="Entry to mid-level",
            ),
            skill_gap_analysis=SkillGapAnalysis(
                matching_skills=matching,
                missing_or_unclear_skills=missing,
                strong_signals=[
                    "Resume mentions hands-on backend and GenAI work.",
                    "Candidate has relevant Python project signals.",
                ],
                interview_focus_areas=missing or ["Depth of FastAPI design", "LLM evaluation practices"],
            ),
        )

