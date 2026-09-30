"""Gemini embeddings, retrieval, filtering, grounded generation, and refusal."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from rag_app.chunking import Chunk, build_chunks, load_cards

EMBEDDING_MODEL = "gemini-embedding-001"
RRF_K = 60
GENERATION_PARAMETERS = {"temperature": 0, "max_output_tokens": 500}
FORCED_REFUSAL = "I couldn't find that in the recipe documents."


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class BM25:
    """Minimal lexical scorer used only for the Week 4 BM25 + RRF change."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [_tokens(document) for document in documents]
        self.average_length = sum(map(len, self.documents)) / len(self.documents)
        self.document_frequency: dict[str, int] = {}
        for document in self.documents:
            for token in set(document):
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def scores(self, query: str) -> list[float]:
        query_tokens = _tokens(query)
        values: list[float] = []
        for document in self.documents:
            frequencies = {token: document.count(token) for token in set(document)}
            score = 0.0
            for token in query_tokens:
                if token not in frequencies:
                    continue
                document_frequency = self.document_frequency[token]
                inverse_frequency = math.log(
                    1 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequencies[token] + self.k1 * (
                    1 - self.b + self.b * len(document) / self.average_length
                )
                score += inverse_frequency * frequencies[token] * (self.k1 + 1) / denominator
            values.append(score)
        return values


class GeminiRAG:
    """One indexed chunking strategy plus dense or hybrid retrieval."""

    def __init__(self, api_key: str, index_path: Path, model: str, chunking_strategy: str = "section"):
        self.client = genai.Client(api_key=api_key)
        self.index_path = index_path
        self.model = model
        self.chunking_strategy = chunking_strategy
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []
        self.bm25: BM25 | None = None

    def index_documents(self, documents_dir: Path) -> int:
        """Index only the supplied cards under this instance's chunking strategy."""
        cards = load_cards(documents_dir)
        chunks = build_chunks(cards, self.chunking_strategy)
        fingerprint = _fingerprint(chunks, self.chunking_strategy)
        cached = self._load_cache(fingerprint)
        if cached is not None:
            self.chunks, self.embeddings = cached
        else:
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[chunk.text for chunk in chunks],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            self.chunks = chunks
            self.embeddings = [embedding.values for embedding in response.embeddings]
            self._save_cache(fingerprint)
        self.bm25 = BM25([chunk.text for chunk in self.chunks])
        return len(self.chunks)

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        strategy: str = "hybrid",
        dietary_tag: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve with dense-only or BM25 + RRF, optionally filtering metadata first."""
        if not self.chunks:
            raise RuntimeError("Index documents before asking a question.")
        if strategy not in {"dense", "hybrid"}:
            raise ValueError("strategy must be 'dense' or 'hybrid'")
        candidate_indices = self._candidate_indices(dietary_tag)
        if not candidate_indices:
            return []
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_embedding = response.embeddings[0].values
        dense_scores = [_cosine_similarity(query_embedding, embedding) for embedding in self.embeddings]
        if strategy == "dense":
            return _rank_subset(self.chunks, candidate_indices, dense_scores, top_k)
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not ready.")
        bm25_scores = self.bm25.scores(question)
        local_dense = [dense_scores[index] for index in candidate_indices]
        local_bm25 = [bm25_scores[index] for index in candidate_indices]
        fused_scores = _rrf_scores(local_dense, local_bm25)
        return _rank_subset(self.chunks, candidate_indices, fused_scores, top_k, scores_are_local=True)

    def answer(self, question: str, top_k: int = 3, strategy: str = "hybrid") -> tuple[str, list[SearchResult], str]:
        """Run hybrid RAG and force refusal when evidence is insufficient."""
        sources = self.retrieve(question, top_k=top_k, strategy=strategy)
        if not _has_enough_evidence(question, sources):
            return FORCED_REFUSAL, sources, ""
        answer, prompt = self.answer_from_sources(question, sources)
        return answer, sources, prompt

    def answer_from_sources(self, question: str, sources: list[SearchResult]) -> tuple[str, str]:
        """Generate only from supplied chunks; each factual claim must cite a chunk ID."""
        context = "\n\n".join(
            f"[Source: {result.chunk.id} | {result.chunk.recipe_id} | {result.chunk.source_file}]\n{result.chunk.text}"
            for result in sources
        )
        prompt = f"""Answer the user's question using only the retrieved recipe context below.
If the answer is absent from this context, say exactly: {FORCED_REFUSAL}
Do not use outside knowledge. Keep the answer concise. Cite every factual claim using the source ID in square brackets.

Retrieved recipe context:
{context}

User question: {question}"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**GENERATION_PARAMETERS),
        )
        return response.text.strip(), prompt

    def _candidate_indices(self, dietary_tag: str | None) -> list[int]:
        if not dietary_tag:
            return list(range(len(self.chunks)))
        wanted = dietary_tag.strip().lower()
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if wanted in {tag.strip().lower() for tag in chunk.dietary_tags.split(",")}
        ]

    def _load_cache(self, fingerprint: str) -> tuple[list[Chunk], list[list[float]]] | None:
        if not self.index_path.exists():
            return None
        try:
            cached = json.loads(self.index_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") != fingerprint or cached.get("embedding_model") != EMBEDDING_MODEL:
                return None
            return [Chunk(**chunk) for chunk in cached["chunks"]], cached["embeddings"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _save_cache(self, fingerprint: str) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "fingerprint": fingerprint,
            "embedding_model": EMBEDDING_MODEL,
            "strategy": self.chunking_strategy,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "embeddings": self.embeddings,
        }
        self.index_path.write_text(json.dumps(data), encoding="utf-8")


def read_recipe_chunks(documents_dir: Path) -> list[Chunk]:
    """Return section-strategy chunks for every card in documents_dir.

    Used by unit tests and any script that needs the chunk list without
    spinning up a full GeminiRAG instance.
    IDs are ``{file_stem}-{section_index}`` e.g. ``R001_country_sourdough-1``.
    """
    cards = load_cards(documents_dir)
    return build_chunks(cards, "section")


def _fingerprint(chunks: list[Chunk], strategy: str) -> str:
    content = "\n".join(f"{chunk.id}\n{chunk.text}" for chunk in chunks)
    return hashlib.sha256(f"{strategy}\n{content}".encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    return dot_product / (first_length * second_length) if first_length and second_length else 0.0


def _rank_subset(
    chunks: list[Chunk],
    candidate_indices: list[int],
    scores: list[float],
    top_k: int,
    scores_are_local: bool = False,
) -> list[SearchResult]:
    if scores_are_local:
        ranked_local = sorted(range(len(candidate_indices)), key=lambda index: scores[index], reverse=True)
        return [
            SearchResult(chunks[candidate_indices[index]], float(scores[index]))
            for index in ranked_local[:top_k]
        ]
    ranked_indices = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)
    return [SearchResult(chunks[index], float(scores[index])) for index in ranked_indices[:top_k]]


def _rrf_scores(dense_scores: list[float], bm25_scores: list[float]) -> list[float]:
    dense_order = sorted(range(len(dense_scores)), key=lambda index: dense_scores[index], reverse=True)
    bm25_order = sorted(range(len(bm25_scores)), key=lambda index: bm25_scores[index], reverse=True)
    dense_ranks = {index: rank for rank, index in enumerate(dense_order, start=1)}
    bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}
    return [
        1 / (RRF_K + dense_ranks[index]) + 1 / (RRF_K + bm25_ranks[index])
        for index in range(len(dense_scores))
    ]


def _has_enough_evidence(question: str, sources: list[SearchResult]) -> bool:
    """Force refusal unless at least two meaningful question terms occur in context."""
    stop_words = {"a", "an", "the", "and", "at", "for", "how", "in", "is", "of", "or", "the", "to", "what", "which", "with"}
    question_terms = {term for term in _tokens(question) if term not in stop_words}
    context_terms = set(_tokens(" ".join(source.chunk.text for source in sources)))
    return len(question_terms & context_terms) >= 2
