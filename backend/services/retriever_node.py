from backend.models.state import InterviewState
from backend.retrieval.faiss_store import Retriever


class RetrieverNode:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def run(self, state: InterviewState, query_text: str, top_k: int = 4) -> InterviewState:
        state.retrieved_context = self.retriever.query(query_text=query_text, top_k=top_k)
        return state

