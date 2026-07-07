"""LangGraph node functions for the HirePilot interview graph.

Each node receives the full GraphState dict, mutates the embedded
InterviewState, and returns the updated GraphState.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from backend.models.state import InterviewStageName, InterviewState

if TYPE_CHECKING:
    from backend.retrieval.faiss_store import Retriever
    from backend.services.evaluation import EvaluationAgent
    from backend.services.interview import InterviewAgent
    from backend.services.planner import PlannerAgent
    from backend.services.retriever_node import RetrieverNode

logger = logging.getLogger(__name__)

MAX_TURNS = 5  # default exit after this many Q&A turns


class GraphState(TypedDict):
    interview: InterviewState
    candidate_answer: str  # latest spoken/typed answer fed in from outside


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
        logger.info("[interview] Question: %s", next_q.question)
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
# Node: memory_update  (lightweight — state already mutated in-place above)
# ---------------------------------------------------------------------------

def memory_update_node(state: GraphState) -> GraphState:
    interview = state["interview"]
    interview.current_stage = InterviewStageName.INTERVIEWING
    logger.info(
        "[memory] turns=%d  topics_covered=%s",
        len(interview.previous_questions),
        interview.topics_covered,
    )
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
# Decision node: should we continue?
# ---------------------------------------------------------------------------

def should_continue(state: GraphState) -> str:
    interview = state["interview"]
    turns = len(interview.previous_questions)
    exit_reached = turns >= MAX_TURNS or not interview.remaining_topics
    return "report" if exit_reached else "retriever"
