import json
from typing import Protocol

from backend.ai.groq_client import GroqStructuredClient
from backend.core.prompts import PromptLoader
from backend.models.state import AnswerScore, InterviewState, RubricScore


class EvaluationAgent(Protocol):
    def evaluate_answer(self, state: InterviewState, question: str, answer: str) -> AnswerScore:
        """Score a candidate answer and return structured reasoning."""


class GroqEvaluationAgent:
    def __init__(self, client: GroqStructuredClient, prompt_loader: PromptLoader) -> None:
        self.client = client
        self.prompt_loader = prompt_loader

    def evaluate_answer(self, state: InterviewState, question: str, answer: str) -> AnswerScore:
        system_prompt = self.prompt_loader.load("evaluator.md")
        schema_json = json.dumps(AnswerScore.model_json_schema(), indent=2)
        role = state.interview_plan.role_title if state.interview_plan else "Unknown"
        skills = ", ".join(state.understanding.resume.skills[:10]) if state.understanding else ""
        user_prompt = (
            f"Role: {role}\nCandidate skills: {skills}\n\n"
            f"Required JSON schema:\n{schema_json}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}"
        )
        result = self.client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AnswerScore,
        )
        return AnswerScore.model_validate(result.model_dump())


class LocalSampleEvaluationAgent:
    """Deterministic evaluator used for local M5 verification."""

    def evaluate_answer(self, state: InterviewState, question: str, answer: str) -> AnswerScore:
        answer_lower = answer.lower()
        concrete_terms = ["fastapi", "pydantic", "validation", "routes", "schemas", "errors"]
        technical_hits = sum(1 for term in concrete_terms if term in answer_lower)
        technical_score = min(5.0, 2.0 + technical_hits * 0.6)
        communication_score = 4.0 if len(answer.split()) >= 12 else 2.5
        relevance_score = 4.5 if any(term in answer_lower for term in ["fastapi", "api", "validation"]) else 2.5
        depth_score = min(5.0, 2.5 + technical_hits * 0.5)
        overall = round((technical_score + communication_score + relevance_score + depth_score) / 4, 2)

        score = AnswerScore(
            question=question,
            answer=answer,
            rubric_scores=[
                RubricScore(
                    dimension="technical_accuracy",
                    score=round(technical_score, 2),
                    reasoning="Credits concrete backend/API implementation details in the answer.",
                ),
                RubricScore(
                    dimension="communication_clarity",
                    score=communication_score,
                    reasoning="Answer is understandable and gives a compact explanation.",
                ),
                RubricScore(
                    dimension="role_relevance",
                    score=relevance_score,
                    reasoning="Answer maps to backend API work expected for the GenAI role.",
                ),
                RubricScore(
                    dimension="depth",
                    score=round(depth_score, 2),
                    reasoning="Depth increases when the answer names specific design choices.",
                ),
            ],
            overall_score=overall,
            strengths=[
                "Mentions concrete implementation components.",
                "Connects project work to API design.",
            ],
            weaknesses=[
                "Could include a specific failure example or trade-off.",
            ],
            reasoning="The answer is relevant and technically grounded, with room for more detailed examples.",
        )
        state.current_scores.append(score)
        return score

