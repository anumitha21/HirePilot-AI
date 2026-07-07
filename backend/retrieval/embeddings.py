import hashlib
import re

import numpy as np


class HashingEmbeddingModel:
    """Small deterministic embedding model for local retrieval verification."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype="float32")
        for row_index, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
                column = int.from_bytes(digest[:4], byteorder="little") % self.dimensions
                vectors[row_index, column] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

