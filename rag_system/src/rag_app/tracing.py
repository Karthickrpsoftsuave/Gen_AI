"""Append-only JSONL trace writer for recipe RAG conversations.

Every question asked through the CLI appends one JSON record here.
Week 5 replay uses these traces directly; do not delete the file.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_app.rag import SearchResult


def save_trace(
    trace_path: Path,
    question: str,
    answer: str,
    prompt: str,
    sources: "list[SearchResult]",
    model: str,
    retrieval_strategy: str,
    top_k: int,
) -> str:
    """Append one trace record and return its unique trace_id."""
    trace_id = str(uuid.uuid4())
    record = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "retrieval_strategy": retrieval_strategy,
        "top_k": top_k,
        "question": question,
        "answer": answer,
        "prompt": prompt,
        "sources": [
            {
                "chunk_id": result.chunk.id,
                "recipe_id": result.chunk.recipe_id,
                "source_file": result.chunk.source_file,
                "section": result.chunk.section,
                "score": result.score,
                "text": result.chunk.text,
            }
            for result in sources
        ],
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return trace_id
