"""Embedding providers.

GeminiEmbedder  - real semantic embeddings (requires GOOGLE_API_KEY)
HashEmbedder    - deterministic offline fallback (hashed unigrams + bigrams). It only captures
                  word overlap, NOT meaning; it exists so tests and demos run without an API key.
"""
import hashlib
from functools import lru_cache
from typing import Protocol

import numpy as np

from internmate.config import get_settings
from internmate.text import tokenize


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: np.ndarray) -> list[float]:
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).astype(float).tolist()


class HashEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dim)
        for feat in features:
            h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbedder:
    BATCH_SIZE = 100

    def __init__(self, api_key: str, model: str, dim: int):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[i:i + self.BATCH_SIZE],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            )
            # Truncated Gemini embeddings must be re-normalised for cosine similarity.
            out.extend(_normalize(np.array(e.values)) for e in resp.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "gemini":
        if not s.google_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=gemini requires GOOGLE_API_KEY in .env")
        return GeminiEmbedder(s.google_api_key, s.embedding_model, s.embedding_dim)
    return HashEmbedder(s.embedding_dim)
