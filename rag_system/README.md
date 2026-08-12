# Week 3 · M2 — Set B: "Ask my documents" RAG app (Recipes & Food)

Mini RAG pipeline for the fermentation-chapter cookbook drop. It ingests the 6
new recipe cards, measures two chunking strategies on 8 known-answer questions,
supports a `dietary_tags` metadata filter, and answers with citations — or
refuses, when the documents don't contain the answer.

## Quick start

```
python -m venv .venv
.venv\Scripts\pip install numpy fastembed
.venv\Scripts\python evaluate.py
```

First run downloads the embedding model (BAAI/bge-small-en-v1.5, ~130 MB) into
`.model_cache/`. `evaluate.py` then regenerates `results.md` and the search
dumps in `output/`.

## Layout

```
data/cards/            the 6 supplied recipe cards (R001–R006)
questions/questions.json   the 8 benchmark questions + demo/bonus questions
rag_pipeline/chunker.py    two chunkers + the source_file ingest gate
rag_pipeline/store.py      dense embeddings + top-K cosine search + tag filter
rag_pipeline/generate.py   grounded extractive generator + forced refusal
evaluate.py                the measurement run -> results.md
output/                    search-only dumps for both strategies
```

## Key numbers (run on 2026-08-11)

| Metric | Naive (200-char window) | Structure-aware |
|---|---|---|
| Hit-in-top-5 (8 known-answer questions) | 6/8 | 8/8 |
| MRR | 0.588 | 1.000 |

Same embedding model for both — only the chunker changed. Full per-question
record, filter demo, cited answers, refusals and the chunking defence are in
[`results.md`](results.md).

## Notes

- Extractive grounding: no LLM API key is used; every claim is a sentence
  pulled verbatim from a retrieved chunk, so a citation always resolves to a
  chunk that contains the claim. The refusal gate is hard — there is no
  "use your best judgement" fallback.
- Scope: only the 6 new cards were indexed (no full-corpus re-index).
