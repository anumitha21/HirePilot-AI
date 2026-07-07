"""LangGraph node functions for the HirePilot interview graph.

Each node receives the full GraphState dict, mutates the embedded
InterviewState, and returns the updated GraphState.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from backend.models.interview import DifficultyLevel, InterviewStageFlow
from backend.models.state import InterviewStageName, InterviewState

if TYPE_CHECKING:
    from backend.retrieval.faiss_store import Retriever
    from backend.services.evaluation import EvaluationAgent
    from backend.services.interview import InterviewAgent
    from backend.services.planner import PlannerAgent
    from backend.services.retriever_node import RetrieverNode

logger = logging.getLogger(__name__)

MAX_TURNS = 12  # enough to cover all 8 stages

STAGE_ORDER = [
    "introduction",
    "resume_validation",
    "project_deep_dive",
    "technical_skills",
    "problem_solving",
    "behavioral",
    "job_fit",
    "closing",
]

# minimum turns to spend in each stage before advancing
STAGE_MIN_TURNS = {
    "introduction": 1,
    "resume_validation": 1,
    "project_deep_dive": 2,
    "technical_skills": 2,
    "problem_solving": 1,
    "behavioral": 1,
    "job_fit": 1,
    "closing": 1,
}


class GraphState(TypedDict):
    interview: InterviewState
    candidate_answer: str  # latest spoken/typed answer fed in from outside


# ---------------------------------------------------------------------------
# Node: intro (runs once — interviewer greeting + "tell me about yourself")
# ---------------------------------------------------------------------------

def intro_node(state: GraphState) -> GraphState:
    interview = state["interview"]
    role = interview.interview_plan.role_title if interview.interview_plan else "this role"
    company_hint = ""
    if interview.understanding and interview.understanding.job_description.company:
        company_hint = f" at {interview.understanding.job_description.company}"

    greeting = (
        f"Hi {interview.candidate_name or 'there'}, I'm Alex, your interviewer today. "
        f"We're here to discuss your application for the {role}{company_hint} position. "
        f"This will be a conversational interview — just speak naturally. "
        f"Let's start simple: could you tell me a bit about yourself and what brought you to apply for this role?"
    )
    interview.remember_question(greeting)
    interview.interview_stage = "introduction"
    interview.topics_covered.append("introduction")
    logger.info("[intro] %s", greeting)
    return {"interview": interview, "candidate_answer": state["candidate_answer"]}


# ---------------------------------------------------------------------------
# Node: planner
# ---------------------------------------------------------------------------

def make_planner_node(planner: PlannerAgent):
    def planner_node(state: GraphState) -> GraphState:
        interview = state["interview"]
        if interview.interview_plan is None:
            logger.info("[planner] Creating interview plan.")
            interview.interview_plan = planner.create_plan(interview.understanding)
            interview.remaining_topics = [
                cat.name for cat in interview.interview_plan.question_categories
            ]
            interview.strong_areas = interview.interview_plan.strong_areas_to_validate
            interview.weak_areas = interview.interview_plan.weak_or_unclear_areas_to_probe
        interview.current_stage = InterviewStageName.INTERVIEWING
        return {"interview": interview, "candidate_answer": state["candidate_answer"]}

    return planner_node


# ---------------------------------------------------------------------------
# Node: retriever
# ---------------------------------------------------------------------------

def make_retriever_node(retriever_node: RetrieverNode):
    def retriever(state: GraphState) -> GraphState:
        interview = state["interview"]
        query = (
            state["candidate_answer"]
            or interview.current_question
            or (interview.interview_plan.opening_question if interview.interview_plan else "")
        )
        logger.info("[retriever] Querying with: %s", query[:80])
        retriever_node.run(state=interview, query_text=query, top_k=4)
        return {"interview": interview, "candidate_answer": state["candidate_answer"]}

    return retriever


# ---------------------------------------------------------------------------
# Node: interview (question generation)
# ---------------------------------------------------------------------------

def make_interview_node(interview_agent: InterviewAgent):
    def interview_node(state: GraphState) -> GraphState:
        interview = state["interview"]
        next_q = interview_agent.next_question(interview)
        interview.remember_question(next_q.question)
        interview.topics_covered.append(next_q.topic)
        if next_q.topic in interview.remaining_topics:
            interview.remaining_topics.remove(next_q.topic)
        # update stage and difficulty from agent's decision
        if next_q.stage:
            interview.interview_stage = next_q.stage
        # track stage turns for decision node
        interview.topics_covered.append(str(interview.interview_stage))
        logger.info("[interview] [%s/%s] Q: %s", next_q.stage, next_q.difficulty, next_q.question)
        return {"interview": interview, "candidate_answer": state["candidate_answer"]}

    return interview_node


# ---------------------------------------------------------------------------
# Node: record_answer  (injects the candidate_answer into InterviewState)
# ---------------------------------------------------------------------------

def record_answer_node(state: GraphState) -> GraphState:
    interview = state["interview"]
    answer = state["candidate_answer"] or "No answer provided."
    interview.remember_answer(answer)
    logger.info("[record_answer] Answer recorded (%d chars).", len(answer))
    return {"interview": interview, "candidate_answer": ""}


# ---------------------------------------------------------------------------
# Node: evaluation
# ---------------------------------------------------------------------------

def make_evaluation_node(evaluation_agent: EvaluationAgent):
    def evaluation_node(state: GraphState) -> GraphState:
        interview = state["interview"]
        if not interview.previous_questions or not interview.previous_answers:
            return state
        question = interview.previous_questions[-1]
        answer = interview.previous_answers[-1]
        interview.current_stage = InterviewStageName.EVALUATING
        score = evaluation_agent.evaluate_answer(
            state=interview, question=question, answer=answer
        )
        if score not in interview.current_scores:
            interview.current_scores.append(score)
        # update strong/weak areas from latest score
        if score.overall_score >= 3.5:
            for s in score.strengths:
                if s not in interview.strong_areas:
                    interview.strong_areas.append(s)
        else:
            for w in score.weaknesses:
                if w not in interview.weak_areas:
                    interview.weak_areas.append(w)
        logger.info("[evaluation] Score: %.2f", score.overall_score)
        return {"interview": interview, "candidate_answer": state["candidate_answer"]}

    return evaluation_node


# ---------------------------------------------------------------------------
# Node: memory_update
# ---------------------------------------------------------------------------

_DIFFICULTY_UP = {
    "easy": "intermediate",
    "intermediate": "advanced",
    "advanced": "architecture",
    "architecture": "system_design",
    "system_design": "system_design",
}
_DIFFICULTY_DOWN = {
    "easy": "easy",
    "intermediate": "easy",
    "advanced": "intermediate",
    "architecture": "advanced",
    "system_design": "architecture",
}


def memory_update_node(state: GraphState) -> GraphState:
    interview = state["interview"]
    interview.current_stage = InterviewStageName.INTERVIEWING

    if interview.current_scores:
        last_score = interview.current_scores[-1].overall_score
        d = str(interview.current_difficulty)
        if last_score >= 4.0:
            interview.current_difficulty = DifficultyLevel(_DIFFICULTY_UP.get(d, d))
        elif last_score <= 2.5:
            interview.current_difficulty = DifficultyLevel(_DIFFICULTY_DOWN.get(d, d))

    logger.info(
        "[memory] turns=%d  stage=%s  difficulty=%s",
        len(interview.previous_questions),
        interview.interview_stage,
        interview.current_difficulty,
    )
    return {"interview": interview, "candidate_answer": state["candidate_answer"]}


# ---------------------------------------------------------------------------
# Node: closing (interviewer wraps up before report)
# ---------------------------------------------------------------------------

def closing_node(state: GraphState) -> GraphState:
    interview = state["interview"]
    scores = interview.current_scores
    overall = round(sum(s.overall_score for s in scores) / len(scores), 2) if scores else 0.0
    tone = "really strong" if overall >= 4.0 else "solid" if overall >= 3.0 else "a good start"

    closing = (
        f"That brings us to the end of our interview, {interview.candidate_name or 'thank you'}. "
        f"You've covered some {tone} ground today — especially around "
        f"{', '.join(interview.strong_areas[:2]) if interview.strong_areas else 'your projects'}. "
        f"Is there anything you'd like to add that we haven't covered, "
        f"or any questions you have for us?"
    )
    interview.remember_question(closing)
    interview.interview_stage = InterviewStageFlow.CLOSING
    logger.info("[closing] %s", closing)
    return {"interview": interview, "candidate_answer": state["candidate_answer"]}


# ---------------------------------------------------------------------------
# Node: report (stub — full Report Agent is M7)
# ---------------------------------------------------------------------------

def report_node(state: GraphState) -> GraphState:
    interview = state["interview"]
    interview.current_stage = InterviewStageName.REPORTING
    scores = interview.current_scores
    overall = round(sum(s.overall_score for s in scores) / len(scores), 2) if scores else 0.0
    interview.final_report = {
        "candidate_name": interview.candidate_name,
        "turns": len(interview.previous_questions),
        "overall_score": overall,
        "strong_areas": interview.strong_areas,
        "weak_areas": interview.weak_areas,
        "topics_covered": interview.topics_covered,
        "transcript": [t.model_dump() for t in interview.conversation_history],
        "note": "Full report generated in M7.",
    }
    interview.current_stage = InterviewStageName.COMPLETE
    logger.info("[report] Interview complete. Overall score: %.2f", overall)
    return {"interview": interview, "candidate_answer": state["candidate_answer"]}


# ---------------------------------------------------------------------------
# Decision node — owns interview progression (per spec)
# Decides: clarify / follow-up / increase difficulty / decrease difficulty /
#          continue topic / next topic / next stage / finish
# ---------------------------------------------------------------------------

def should_continue(state: GraphState) -> str:
    interview = state["interview"]
    turns = len(interview.previous_questions)
    last_score = interview.current_scores[-1].overall_score if interview.current_scores else 3.0
    current_stage = str(interview.interview_stage)

    # Hard stop → go to closing first
    if turns >= MAX_TURNS or current_stage == "closing":
        return "report"

    # Count turns spent in current stage
    stage_turns = sum(1 for t in interview.topics_covered if t == current_stage)

    # Very weak answer → stay for clarification
    if last_score < 2.0 and stage_turns < 3:
        return "retriever"

    # Weak answer → stay in stage one more turn
    if last_score <= 2.5 and stage_turns < STAGE_MIN_TURNS.get(current_stage, 1) + 1:
        return "retriever"

    # Haven't met minimum turns for this stage → stay
    if stage_turns < STAGE_MIN_TURNS.get(current_stage, 1):
        return "retriever"

    # Advance to next stage
    try:
        next_stage_index = STAGE_ORDER.index(current_stage) + 1
        if next_stage_index < len(STAGE_ORDER):
            next_stage = STAGE_ORDER[next_stage_index]
            interview.interview_stage = InterviewStageFlow(next_stage)
            logger.info("[decision] Advancing stage: %s → %s", current_stage, next_stage)
            if next_stage == "closing":
                return "closing"
            return "retriever"
    except ValueError:
        pass

    return "closing"
