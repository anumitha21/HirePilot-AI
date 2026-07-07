"""Builds and returns the compiled HirePilot LangGraph interview graph.

Graph flow:
    planner → retriever → interview → record_answer
           → evaluation → memory_update → decision
               ├── continue → retriever  (loop)
               └── end     → report
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.graph.nodes import (
    GraphState,
    make_evaluation_node,
    make_interview_node,
    make_planner_node,
    make_retriever_node,
    memory_update_node,
    record_answer_node,
    report_node,
    should_continue,
)
from backend.retrieval.faiss_store import Retriever
from backend.services.evaluation import EvaluationAgent
from backend.services.interview import InterviewAgent
from backend.services.planner import PlannerAgent
from backend.services.retriever_node import RetrieverNode


def build_graph(
    planner: PlannerAgent,
    retriever: Retriever,
    interview_agent: InterviewAgent,
    evaluation_agent: EvaluationAgent,
):
    """Return a compiled LangGraph graph ready to invoke."""
    retriever_node = RetrieverNode(retriever)

    builder = StateGraph(GraphState)

    builder.add_node("planner", make_planner_node(planner))
    builder.add_node("retriever", make_retriever_node(retriever_node))
    builder.add_node("interview", make_interview_node(interview_agent))
    builder.add_node("record_answer", record_answer_node)
    builder.add_node("evaluation", make_evaluation_node(evaluation_agent))
    builder.add_node("memory_update", memory_update_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "interview")
    # After interview generates a question, record_answer injects the answer.
    # In the automated/text mode the answer is already in GraphState.
    # In voice mode the caller updates candidate_answer before resuming.
    builder.add_edge("interview", "record_answer")
    builder.add_edge("record_answer", "evaluation")
    builder.add_edge("evaluation", "memory_update")
    builder.add_conditional_edges(
        "memory_update",
        should_continue,
        {"retriever": "retriever", "report": "report"},
    )
    builder.add_edge("report", END)

    return builder.compile()
