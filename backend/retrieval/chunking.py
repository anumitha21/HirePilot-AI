from collections.abc import Iterable

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    source: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


def chunk_text(source: str, text: str, max_words: int = 80) -> list[TextChunk]:
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    chunks: list[TextChunk] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        for offset in range(0, len(words), max_words):
            chunk_words = words[offset : offset + max_words]
            chunks.append(
                TextChunk(
                    source=source,
                    text=" ".join(chunk_words),
                    metadata={"paragraph": str(paragraph_index), "word_offset": str(offset)},
                )
            )

    return chunks


def chunk_many(items: Iterable[tuple[str, str]]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for source, text in items:
        chunks.extend(chunk_text(source=source, text=text))
    return chunks

