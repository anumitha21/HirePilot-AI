from typing import Protocol

from backend.models.state import RetrievedContext
from backend.retrieval.chunking import TextChunk
from backend.retrieval.embeddings import HashingEmbeddingModel


class Retriever(Protocol):
    def query(self, query_text: str, top_k: int = 4) -> list[RetrievedContext]:
        """Return relevant context for a query."""


class FaissRetriever:
    def __init__(self, chunks: list[TextChunk], embedding_model: HashingEmbeddingModel) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required to build the FAISS index.")

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required for M3 retrieval. Install dependencies with `python -m pip install -e .`."
            ) from exc

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatIP(embedding_model.dimensions)
        self.index.add(embedding_model.embed([chunk.text for chunk in chunks]))

    def query(self, query_text: str, top_k: int = 4) -> list[RetrievedContext]:
        query_vector = self.embedding_model.embed([query_text])
        scores, indexes = self.index.search(query_vector, min(top_k, len(self.chunks)))

        results: list[RetrievedContext] = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]
            results.append(
                RetrievedContext(
                    source=chunk.source,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )

        return results

