"""Interactive command-line interface for the RAG application."""

from __future__ import annotations

import sys

from rag_app.config import load_settings
from rag_app.rag import GeminiRAG
from rag_app.tracing import save_trace


def main() -> None:
    settings = load_settings()
    rag = GeminiRAG(settings.api_key, settings.index_path, settings.llm_model)
    chunk_count = rag.index_documents(settings.documents_dir)
    print(f"Ready: {chunk_count} recipe sections indexed with Gemini embeddings.")

    question = " ".join(sys.argv[1:]).strip()
    while True:
        if not question:
            question = input("\nAsk a recipe question (or type exit): ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if question:
            answer, sources, prompt = rag.answer(question, strategy="hybrid")
            trace_id = save_trace(
                settings.trace_path,
                question,
                answer,
                prompt,
                sources,
                settings.llm_model,
                "hybrid",
                top_k=3,
            )
            print(f"\nAnswer: {answer}\n")
            print("Retrieved sources:")
            for source in sources:
                print(f"- [{source.chunk.id}] {source.chunk.source_file} (score: {source.score:.3f})")
            print(f"Trace ID: {trace_id}")
        if len(sys.argv) > 1:
            return
        question = ""
