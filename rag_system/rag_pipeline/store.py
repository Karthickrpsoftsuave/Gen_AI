"""Dense embedding + vector search over chunks (in-memory, cosine).

Embedding model: BAAI/bge-small-en-v1.5 (dim 384) via fastembed (ONNX, no
torch dependency). One model, two stores — the chunking comparison therefore
changes exactly one thing at a time (Set B, section 7: never change the
chunker AND the embedding model in the same run).

The store keeps the chunk + its vector, and supports top-K cosine search with
optional metadata filtering on dietary_tags (requirement 4). Filtering is
applied BEFORE ranking: excluded chunks never enter the top-K.
"""

from __future__ import annotations

import os

# Keep the ONNX model in a project-local cache so the demo survives a Temp wipe.
os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(os.path.dirname(__file__), "..", ".model_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from fastembed import TextEmbedding

from rag_pipeline.chunker import Chunk

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class VectorStore:
    def __init__(self, strategy: str, model_name: str = MODEL_NAME):
        self.strategy = strategy
        self.embedder = TextEmbedding(model_name)
        self.chunks: list[Chunk] = []
        self._mat: np.ndarray | None = None

    # ------------------------------------------------------------------ ingest
    def add_all(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = np.vstack([v for v in self.embedder.embed(texts)]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._mat = vectors / norms
        self.chunks = list(chunks)

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 5, tag_filter: str | None = None) -> list[tuple[Chunk, float]]:
        """Top-K cosine retrieval. tag_filter keeps only chunks whose
        dietary_tags contain the tag (case-insensitive, comma-split)."""
        q = list(self.embedder.embed([query]))[0].astype(np.float32)
        q = q / np.linalg.norm(q)

        if tag_filter:
            wanted = {t.strip().lower() for t in tag_filter.split(",")}
            keep = [i for i, c in enumerate(self.chunks)
                    if wanted <= {t.strip().lower() for t in c.dietary_tags.split(",")}]
            if not keep:
                return []
            scores = self._mat[keep] @ q
            order = np.argsort(-scores)
            out = []
            for pos in order[:top_k]:
                i = keep[int(pos)]
                out.append((self.chunks[i], float(scores[int(pos)])))
            return out

        scores = self._mat @ q
        order = np.argsort(-scores)
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in order[:top_k]]
