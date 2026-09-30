# Gemini Recipe Q&A — RAG Project

A small, professional Python RAG application built with the Gemini API. It answers questions from the markdown recipe cards in `data/cards/` and provides the required Week 4 retrieval evaluation and Week 5 trace-analysis tooling.

## Architecture

```text
Recipe markdown files
        ↓
Gemini embeddings + local JSON index
        ↓
Dense retrieval + BM25 / RRF fusion
        ↓
Gemini grounded answer with citations
        ↓
Replayable JSONL trace
```

The application uses no Docker, database, LangChain, or external vector database. The local `output/index.json` cache is rebuilt automatically when a recipe changes.

## Project layout

```text
src/rag_app/       Application package: configuration, RAG service, tracing, CLI
scripts/           Week 4 evaluation and Week 5 replay/analysis implementations
tests/             Offline unit tests
data/cards/        Source recipe documents
questions/         Week 4 golden set
output/            Generated local cache and traces (not committed)
app.py             Compatibility launcher for the CLI
```

## Requirements

You need only:

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Internet access when Gemini embeds documents or answers questions

You do not need Docker, a database, or a paid third-party vector-store account.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

Open `.env` and set your private key:

```env
GEMINI_API_KEY=your_real_key_here
```

Never commit `.env` or share the key. It is already listed in `.gitignore`.

## Run the application

Interactive mode:

```powershell
recipe-rag
```

Ask one question directly:

```powershell
recipe-rag "How much salt is in the country sourdough loaf?"
```

`python app.py` and double-clicking `run.bat` also work after setup.

Every question saves a complete trace in `output/traces.jsonl`, including the prompt, model, retrieval scores, retrieved text, and raw Gemini output. Keep these real traces for Week 5.

## Week 4: Retrieval evaluation

The evaluator compares the dense-only baseline with exactly one retrieval change: BM25 + Reciprocal Rank Fusion (RRF, `k=60`). It uses the 12-question golden set in `questions/week4_golden.json`.

```powershell
python evaluate_week4.py
```

It writes `week4_results.md` with hit-rate@3, p50 latency, R/G/Not-In-Corpus inspection, fixed/unfixed status, and a shipping decision. This command uses Gemini API quota.

## Week 5: Trace analysis

1. Ask at least 20 genuine questions through `recipe-rag`.
2. Create the seeded sample:

   ```powershell
   python analyze_week5.py --seed 20260820
   ```

3. Read all 20 traces and write one honest observation per trace in `notes.md`.
4. Group the observations into 4–7 failure modes in `taxonomy.md`.
5. Replay the selected trace using the command written in `notes.md`, or run:

   ```powershell
   python replay_trace.py <trace_id>
   ```

6. Add a dated, falsifiable prediction in `taxonomy.md` and commit it before you change the system.

The scripts prepare the reproducible sample and evidence. The observations and taxonomy must be your analysis of real traces; they should not be fabricated.

## Test

Run offline tests (no API key or Gemini quota required):

```powershell
python -m unittest discover -s tests
```
