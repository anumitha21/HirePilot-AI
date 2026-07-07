import json
from typing import Protocol

from backend.ai.groq_client import GroqStructuredClient
from backend.core.prompts import PromptLoader
from backend.models.interview import NextQuestion
from backend.models.state import InterviewState


class InterviewAgent(Protocol):
    def next_question(self, state: InterviewState) -> NextQuestion:
        """Generate the next adaptive interview question."""


class GroqInterviewAgent:
    def __init__(self, client: GroqStructuredClient, prompt_loader: PromptLoader) -> None:
        self.client = client
        self.prompt_loader = prompt_loader

    def next_question(self, state: InterviewState) -> NextQuestion:
        system_prompt = self.prompt_loader.load("interviewer.md")
        schema_json = json.dumps(NextQuestion.model_json_schema(), indent=2)
        context_texts = " | ".join(c.text for c in state.retrieved_context[:3])
        recent_history = [
            f"{t.speaker}: {t.text}"
            for t in state.conversation_history[-8:]
        ]
        user_prompt = (
            f"Required JSON schema:\n{schema_json}\n\n"
            f"=== INTERVIEW CONTEXT ===\n"
            f"Role: {state.interview_plan.role_title if state.interview_plan else 'Unknown'}\n"
            f"Current stage: {state.interview_stage}\n"
            f"Current difficulty: {state.current_difficulty}\n"
            f"Topics already covered: {state.topics_covered}\n"
            f"Weak areas to probe: {state.weak_areas[:5]}\n"
            f"Strong areas validated: {state.strong_areas[:5]}\n"
            f"Remaining topics: {state.remaining_topics[:5]}\n\n"
            f"=== RECENT CONVERSATION ===\n"
            + "\n".join(recent_history) + "\n\n"
            f"=== RETRIEVED CONTEXT ===\n{context_texts[:500]}\n\n"
            f"Generate the next question. It must follow naturally from the last candidate answer."
        )
        result = self.client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=NextQuestion,
        )
        return NextQuestion.model_validate(result.model_dump())


class LocalAdaptiveInterviewAgent:
    """Deterministic adaptive interviewer used for local M4 verification."""

    def next_question(self, state: InterviewState) -> NextQuestion:
        if not state.previous_questions and state.interview_plan:
            return NextQuestion(
                question=state.interview_plan.opening_question,
                topic="opening",
                rationale="Start from the planner's opening question.",
                uses_retrieved_context=False,
                suggested_followups=["Ask for project architecture details."],
            )

        last_answer = state.previous_answers[-1].lower() if state.previous_answers else ""
        context_text = " ".join(context.text.lower() for context in state.retrieved_context)
        asked_text = " ".join(state.previous_questions).lower()

        if "validation" in last_answer or "pydantic" in last_answer:
            return NextQuestion(
                question=(
                    "You mentioned Pydantic validation and consistent error responses. "
                    "How did you test those schemas and make sure API failures stayed predictable?"
                ),
                topic="api validation testing",
                rationale="The latest answer described validation choices, so the follow-up probes testing depth.",
                uses_retrieved_context=True,
                suggested_followups=[
                    "Ask for an example validation failure.",
                    "Ask how schema changes were handled.",
                ],
            )

        if ("fastapi" in last_answer or "fastapi" in context_text) and "how did you structure the routes" not in asked_text:
            return NextQuestion(
                question=(
                    "You mentioned FastAPI earlier. How did you structure the routes, "
                    "validation, and error handling in that API?"
                ),
                topic="fastapi architecture",
                rationale="The candidate referenced FastAPI, so the next question probes implementation depth.",
                uses_retrieved_context=True,
                suggested_followups=[
                    "Ask about dependency injection.",
                    "Ask how request/response schemas were tested.",
                ],
            )

        if "rag" in last_answer or "langchain" in last_answer or "retrieval" in context_text:
            return NextQuestion(
                question=(
                    "You brought up retrieval work. How did you decide the chunking strategy, "
                    "and how did you check whether retrieved context was actually useful?"
                ),
                topic="rag quality",
                rationale="The answer points toward retrieval, so the follow-up targets RAG quality decisions.",
                uses_retrieved_context=True,
                suggested_followups=["Ask about hallucination controls."],
            )

        weak_area = state.weak_areas[0] if state.weak_areas else "deployment"
        return NextQuestion(
            question=(
                f"Your resume is less explicit about {weak_area}. What exposure do you have there, "
                "and how would you approach it in this role?"
            ),
            topic=weak_area,
            rationale="The planner marked this as an unclear area to probe.",
            uses_retrieved_context=bool(state.retrieved_context),
            suggested_followups=["Ask for a practical scenario if the answer is broad."],
        )
