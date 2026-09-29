"""SentenceTransformer-compatible adapter for the Modal-hosted embedder.

The hosted model reproduces the local `Qwen/Qwen3-Embedding-0.6B` vectors
(cosine similarity 0.9998 on document text, 0.999999 on instructions qualified
queries), so it is exposed through the same interface the local model has:
`encode()`, `prompts` and `get_embedding_dimension()`.

Keeping that interface means the caller's existing batch-size handling, dimension
checks, finiteness checks and embedding-cache identity all continue to apply
unchanged, and the local model remains a drop-in replacement.
"""

from __future__ import annotations

from typing import Any

from qubettera.modal_client import CLOUD_MAX_BATCH_SIZE, embed_texts

# Retrieval instruction published by the model in its
# `config_sentence_transformers.json` as `prompts.query`. The hosted service
# embeds documents as-is, so the prefix is applied here for query encoding only:
# without it query parity drops from 0.999999 to 0.9085.
QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)

_DOCUMENT_INSTRUCTION = ""


class CloudEmbeddingModel:
    """Embed texts through the Modal-hosted Qwen3-Embedding service."""

    # Mirrors the local model's prompt table. `_encode_query` inspects this to
    # decide whether to pass `prompt_name="query"`.
    prompts: dict[str, str] = {
        "query": QUERY_INSTRUCTION,
        "document": _DOCUMENT_INSTRUCTION,
    }

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Any, **kwargs: Any) -> list[list[float]]:
        """Embed one text or a list of texts, returning a list of vectors.

        Accepts and ignores the local model's encode kwargs (`batch_size`,
        `show_progress_bar`, `normalize_embeddings`) so the two providers are
        interchangeable. `prompt_name` selects the retrieval instruction.
        """
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return []

        if kwargs.get("prompt_name") == "query":
            instruction = self.prompts["query"]
            batch = [
                text if text.startswith(instruction) else f"{instruction}{text}"
                for text in batch
            ]

        # The hosted service rejects large document batches, so requests are
        # split to the documented limit rather than failing the whole run.
        vectors: list[list[float]] = []
        for start in range(0, len(batch), CLOUD_MAX_BATCH_SIZE):
            vectors.extend(embed_texts(batch[start:start + CLOUD_MAX_BATCH_SIZE]))
        return vectors
