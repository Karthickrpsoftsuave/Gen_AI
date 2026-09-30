"""FastAPI REST API and Swagger UI server for the Gemini Recipe Chatbot."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from rag_app.config import load_settings
from rag_app.rag import GeminiRAG
from rag_app.tracing import save_trace

app = FastAPI(
    title="Gemini Recipe Chatbot API",
    description=(
        "Retrieval-Augmented Generation (RAG) recipe chatbot powered by Google Gemini API. "
        "Answers questions grounded strictly in indexed recipe cards, with citations and forced refusal."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global lazy-loaded RAG instance
_rag_instance: GeminiRAG | None = None
_settings = None


def get_rag() -> GeminiRAG:
    global _rag_instance, _settings
    if _rag_instance is None:
        _settings = load_settings()
        _rag_instance = GeminiRAG(_settings.api_key, _settings.index_path, _settings.llm_model)
        _rag_instance.index_documents(_settings.documents_dir)
    return _rag_instance


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        description="The recipe question to ask",
        example="How much fine sea salt does the 2 kg country sourdough loaf need?",
    )
    top_k: int = Field(
        default=3,
        description="Number of recipe context chunks to retrieve",
        ge=1,
        le=10,
    )
    dietary_tag: str | None = Field(
        default=None,
        description="Optional metadata filter e.g. 'vegan', 'gluten-free', 'vegetarian'",
        example="vegan",
    )


class SourceChunk(BaseModel):
    chunk_id: str
    recipe_id: str
    source_file: str
    section: str
    score: float
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    trace_id: str


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root endpoint directly to Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Status"])
def health_check():
    """Check API server health and index status."""
    try:
        rag = get_rag()
        return {
            "status": "online",
            "indexed_chunks": len(rag.chunks),
            "model": rag.model,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/recipes", tags=["Recipes"])
def list_recipes():
    """List all recipe cards currently indexed in the RAG system."""
    try:
        rag = get_rag()
        recipes = {}
        for chunk in rag.chunks:
            if chunk.recipe_id not in recipes:
                recipes[chunk.recipe_id] = {
                    "recipe_id": chunk.recipe_id,
                    "source_file": chunk.source_file,
                    "cuisine": chunk.cuisine,
                    "dietary_tags": chunk.dietary_tags,
                }
        return {"total_recipes": len(recipes), "recipes": list(recipes.values())}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/ask", response_model=AnswerResponse, tags=["Chatbot"])
def ask_question(request: QuestionRequest):
    """Ask a recipe question and receive a cited Gemini answer or forced refusal."""
    try:
        rag = get_rag()
        # Retrieve sources (with optional dietary tag filter)
        if request.dietary_tag:
            sources = rag.retrieve(
                request.question,
                top_k=request.top_k,
                strategy="hybrid",
                dietary_tag=request.dietary_tag,
            )
            # Check evidence and generate answer
            from rag_app.rag import FORCED_REFUSAL, _has_enough_evidence
            if not sources or not _has_enough_evidence(request.question, sources):
                answer, prompt = FORCED_REFUSAL, ""
            else:
                answer, prompt = rag.answer_from_sources(request.question, sources)
        else:
            answer, sources, prompt = rag.answer(request.question, top_k=request.top_k, strategy="hybrid")

        # Save trace record
        global _settings
        trace_id = save_trace(
            _settings.trace_path,
            request.question,
            answer,
            prompt,
            sources,
            _settings.llm_model,
            "hybrid",
            top_k=request.top_k,
        )

        formatted_sources = [
            SourceChunk(
                chunk_id=src.chunk.id,
                recipe_id=src.chunk.recipe_id,
                source_file=src.chunk.source_file,
                section=src.chunk.section,
                score=round(src.score, 4),
                text=src.chunk.text,
            )
            for src in sources
        ]

        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=formatted_sources,
            trace_id=trace_id,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


def start():
    """Start uvicorn server programmatically."""
    uvicorn.run("rag_app.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start()
