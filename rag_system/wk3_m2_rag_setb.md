---
name: wk3_m2_rag_setb
description: Week 3 M2 Set B RAG assignment state: built app, results, deadlines
type: project
created_at: 2026-08-11T14:56:50Z
updated_at: 2026-08-11T14:56:50Z
source: agent
---

Week 3 M2 (Retrieval & RAG) Set B assignment — "ask my documents" recipes app. Built 2026-08-11 at D:\python_saturday\rag_system. Deliverable: results.md + code, committed locally (git, no remote).

- **Why:** evaluated Week 4 Monday 2026-08-17 (mentor check Friday 2026-08-14); user may need to re-run or extend for review.
- **How to apply:** stack = Python 3.14 venv (.venv) + numpy + fastembed (BAAI/bge-small-en-v1.5 ONNX, no torch; model cached in .model_cache/). Extractive grounded generator (no LLM API key) with hard refusal gate at evidence 0.42. Key numbers: hit-in-top-5 naive 6/8 vs structure-aware 8/8, MRR 0.588 vs 1.000. Data: 6 invented fermentation cards R001–R006 in data/cards/. Re-run with `.venv\Scripts\python evaluate.py` (regenerates results.md + output/ dumps). Chunking must stay deterministic — section order is a tuple, not a set (set iteration was randomized per process).
