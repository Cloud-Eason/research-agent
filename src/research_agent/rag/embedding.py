"""Dense embedding via sentence-transformers, plus a tokenizer adapter for chunking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Embedder(Protocol):
    dim: int
    max_tokens: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def count_tokens(self, text: str) -> int: ...


class HFTokenizerAdapter:
    """Exposes a HuggingFace tokenizer through the chunker's encode/decode protocol."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tok.decode(tokens, skip_special_tokens=True))


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        query_instruction: str = "",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._query_instruction = query_instruction
        self._batch_size = batch_size
        # Content budget excludes the [CLS]/[SEP] special tokens.
        self.max_tokens = int(getattr(self._model, "max_seq_length", 512) or 512) - 2
        self.dim = len(self.embed_query("dimension probe"))

    @property
    def tokenizer(self) -> HFTokenizerAdapter:
        return HFTokenizerAdapter(self._model.tokenizer)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,  # cosine similarity == inner product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._query_instruction + text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
