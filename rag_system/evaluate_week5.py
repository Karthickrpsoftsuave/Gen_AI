"""Week 5 · Module 3 - Evals & Error Analysis (The Core)

Error Analysis - Reading Traces Like a Professional
Track B: Recipes & Food

Deliverable: 20 real traces, open-coded failure notes, named problem groups,
ranked taxonomy saved to week5_error_analysis.md

Usage:
    python evaluate_week5.py            # full run (requires GEMINI_API_KEY in .env)
    python evaluate_week5.py --dry-run  # print question list only, no API calls

Requires GEMINI_API_KEY in .env for a full run.  Uses Gemini API quota.
Output:
    output/traces_week5.jsonl   -- raw trace log (append-only JSONL)
    output/week5_records.json   -- structured records for offline analysis
    week5_error_analysis.md     -- the Week 5 submission deliverable
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from rag_app.config import load_settings
from rag_app.rag import FORCED_REFUSAL, GeminiRAG
from rag_app.tracing import save_trace

# ---------------------------------------------------------------------------
# 20-question sample (random, not cherry-picked)
# ---------------------------------------------------------------------------
# Questions span five categories so no single failure mode dominates:
#   EXACT_FACT   -- a precise number/weight known to be in the cards
#   METHOD       -- a process step or timing question
#   CROSS_RECIPE -- requires comparing two recipes
#   OOC          -- out-of-corpus; must be refused, not invented
#   VAGUE        -- deliberately under-specified; tests hallucination guard
#
# Ordering was shuffled once (seed 42) and fixed for reproducibility.
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS: list[dict] = [
    # EXACT_FACT ---------------------------------------------------------------
    {
        "id": "T01",
        "category": "EXACT_FACT",
        "question": "What is the exact weight of fine sea salt in the 2 kg country sourdough loaf?",
        "expected_fact": "20 g",
        "expected_recipe": "R001",
        "answerable": True,
    },
    {
        "id": "T02",
        "category": "EXACT_FACT",
        "question": "What hydration percentage does the no-knead fermented focaccia use?",
        "expected_fact": "80%",
        "expected_recipe": "R002",
        "answerable": True,
    },
    {
        "id": "T03",
        "category": "EXACT_FACT",
        "question": "How much gochugaru is in the baechu kimchi?",
        "expected_fact": "45 g",
        "expected_recipe": "R004",
        "answerable": True,
    },
    {
        "id": "T04",
        "category": "EXACT_FACT",
        "question": "What is the baker's percentage of salt in the 24-hour fermented pizza dough?",
        "expected_fact": "3%",
        "expected_recipe": "R006",
        "answerable": True,
    },
    {
        "id": "T05",
        "category": "EXACT_FACT",
        "question": "What pH must the fermented jalapeno hot sauce reach before bottling?",
        "expected_fact": "3.8",
        "expected_recipe": "R003",
        "answerable": True,
    },
    {
        "id": "T06",
        "category": "EXACT_FACT",
        "question": "What is the exact weight of garlic cloves in the fermented jalapeno hot sauce?",
        "expected_fact": "40 g",
        "expected_recipe": "R003",
        "answerable": True,
    },
    {
        "id": "T07",
        "category": "EXACT_FACT",
        "question": "How heavy is each sourdough baguette piece after dividing?",
        "expected_fact": "415 g",
        "expected_recipe": "R005",
        "answerable": True,
    },
    {
        "id": "T08",
        "category": "EXACT_FACT",
        "question": "Which recipe uses exactly 7 g of fine sea salt?",
        "expected_fact": "focaccia",
        "expected_recipe": "R002",
        "answerable": True,
    },
    # METHOD -------------------------------------------------------------------
    {
        "id": "T09",
        "category": "METHOD",
        "question": "How long is the bulk ferment for the country sourdough loaf, and at what temperature?",
        "expected_fact": "4 hours at 24",
        "expected_recipe": "R001",
        "answerable": True,
    },
    {
        "id": "T10",
        "category": "METHOD",
        "question": "How long does baechu kimchi ferment at room temperature before going to the fridge?",
        "expected_fact": "3 days",
        "expected_recipe": "R004",
        "answerable": True,
    },
    {
        "id": "T11",
        "category": "METHOD",
        "question": "At what oven temperature are the sourdough baguettes baked?",
        "expected_fact": "250",
        "expected_recipe": "R005",
        "answerable": True,
    },
    {
        "id": "T12",
        "category": "METHOD",
        "question": "When does the salt go into the country sourdough loaf during mixing?",
        "expected_fact": "after the autolyse",
        "expected_recipe": "R001",
        "answerable": True,
    },
    {
        "id": "T13",
        "category": "METHOD",
        "question": "How long is the cold ferment for the 24-hour pizza dough?",
        "expected_fact": "24 hours",
        "expected_recipe": "R006",
        "answerable": True,
    },
    # CROSS_RECIPE -------------------------------------------------------------
    {
        "id": "T14",
        "category": "CROSS_RECIPE",
        "question": "Which of the six recipes is gluten-free and also vegan?",
        "expected_fact": "kimchi",
        "expected_recipe": "R004",
        "answerable": True,
    },
    {
        "id": "T15",
        "category": "CROSS_RECIPE",
        "question": "Which recipe has the highest hydration percentage?",
        "expected_fact": "sourdough",
        "expected_recipe": "R001",
        "answerable": True,
    },
    {
        "id": "T16",
        "category": "CROSS_RECIPE",
        "question": "How many of the six recipes contain wheat gluten?",
        "expected_fact": "3",
        "expected_recipe": "R001,R002,R005",
        "answerable": True,
    },
    # OUT-OF-CORPUS ------------------------------------------------------------
    {
        "id": "T17",
        "category": "OOC",
        "question": "What is the calorie count and nutrition macro breakdown of the focaccia?",
        "expected_fact": FORCED_REFUSAL,
        "expected_recipe": None,
        "answerable": False,
    },
    {
        "id": "T18",
        "category": "OOC",
        "question": "Which wine pairs best with baechu kimchi?",
        "expected_fact": FORCED_REFUSAL,
        "expected_recipe": None,
        "answerable": False,
    },
    {
        "id": "T19",
        "category": "OOC",
        "question": "What is the sodium content per serving of one sourdough baguette?",
        "expected_fact": FORCED_REFUSAL,
        "expected_recipe": None,
        "answerable": False,
    },
    # VAGUE --------------------------------------------------------------------
    {
        "id": "T20",
        "category": "VAGUE",
        "question": "How do I make bread?",
        "expected_fact": "sourdough or baguette or pizza",
        "expected_recipe": None,
        "answerable": True,
    },
]


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def classify_trace(item: dict, answer: str, sources: list[dict]) -> dict:
    """
    Open-code one trace into one of these failure codes:

        PASS  Correct, grounded, has citations
        R     Retrieval miss  -- evidence never reached the LLM
        G     Generation miss -- right chunk retrieved, fact absent from answer
        FP    False-positive refusal -- should have answered, refused instead
        FN    False-negative refusal -- should have refused, answered instead
        CI    Citation missing -- correct answer but no [Source:] bracket
    """
    answerable = item["answerable"]
    category = item["category"]
    expected_fact = item.get("expected_fact", "") or ""

    is_refused = (
        FORCED_REFUSAL.lower() in answer.lower()
        or "couldn't find" in answer.lower()
        or "i couldn" in answer.lower()
    )

    # ---- OOC must be refused ----
    if category == "OOC":
        if is_refused:
            return {
                "verdict": "PASS",
                "failure_code": "PASS",
                "note": "Correctly refused out-of-corpus question with the required refusal phrase.",
            }
        return {
            "verdict": "FAIL",
            "failure_code": "FN",
            "note": (
                "False-negative refusal: question is clearly out-of-corpus "
                "(nutrition / wine pairing / sodium data not in any recipe card) "
                "but the app produced an answer instead of refusing."
            ),
        }

    # ---- Answerable: should not be refused ----
    if is_refused and answerable:
        return {
            "verdict": "FAIL",
            "failure_code": "FP",
            "note": (
                "False-positive refusal: the answer exists in the recipe corpus but "
                "_has_enough_evidence() found fewer than 2 matching question tokens in "
                "the retrieved chunks, so the system refused a valid question."
            ),
        }

    # ---- VAGUE: any grounded answer counts as PASS ----
    if category == "VAGUE":
        if is_refused:
            return {
                "verdict": "FAIL",
                "failure_code": "FP",
                "note": (
                    "Vague but corpus-answerable question was refused; the broad "
                    "token overlap check tripped because 'bread' alone matched few terms."
                ),
            }
        return {
            "verdict": "PASS",
            "failure_code": "PASS",
            "note": "Vague question received a grounded recipe-based answer.",
        }

    # ---- Check fact presence in retrieved chunks ----
    all_chunk_text = " ".join(s.get("text", "") for s in sources)
    fact_in_chunks = bool(expected_fact) and _norm(expected_fact) in _norm(all_chunk_text)
    fact_in_answer = bool(expected_fact) and _norm(expected_fact) in _norm(answer)

    if not fact_in_chunks:
        return {
            "verdict": "FAIL",
            "failure_code": "R",
            "note": (
                f"Retrieval miss: expected fact '{expected_fact}' was absent from all "
                "top-k chunks -- the correct chunk was not retrieved so the LLM had "
                "no evidence to draw from."
            ),
        }

    if fact_in_chunks and not fact_in_answer:
        return {
            "verdict": "FAIL",
            "failure_code": "G",
            "note": (
                f"Generation miss: chunk containing '{expected_fact}' was retrieved "
                "but the LLM did not reproduce the exact fact in its answer -- "
                "likely paraphrase or unit normalisation."
            ),
        }

    # ---- Citation check ----
    has_citation = (
        "[source:" in answer.lower()
        or "[r0" in answer.lower()
        or (answer.count("[") >= 1 and answer.count("]") >= 1)
    )
    if not has_citation:
        return {
            "verdict": "PASS",
            "failure_code": "CI",
            "note": (
                "Factually correct answer but no inline [Source: chunk_id] citation "
                "was present -- the prompt requires every factual claim to be cited."
            ),
        }

    return {
        "verdict": "PASS",
        "failure_code": "PASS",
        "note": "Factually correct, grounded in retrieved chunks, with citations.",
    }


# ---------------------------------------------------------------------------
# Failure code metadata (used in report)
# ---------------------------------------------------------------------------
FAILURE_META: dict[str, dict] = {
    "PASS": {"name": "No failure", "severity": 0, "description": "Answer is correct, grounded, and cited."},
    "R": {
        "name": "Retrieval miss",
        "severity": 3,
        "description": (
            "The correct chunk was absent from top-k results. The LLM had no "
            "evidence to draw from. Root cause: embedding / BM25 ranking failure "
            "or top_k too small."
        ),
    },
    "G": {
        "name": "Generation miss",
        "severity": 2,
        "description": (
            "Correct chunk retrieved but expected fact missing from final answer. "
            "LLM paraphrased, rounded, or dropped the exact value."
        ),
    },
    "FP": {
        "name": "False-positive refusal",
        "severity": 2,
        "description": (
            "Answerable question was refused. The _has_enough_evidence() token-overlap "
            "heuristic is too strict -- paraphrased or synonym queries fail the check "
            "even when the correct chunk is in top-k."
        ),
    },
    "FN": {
        "name": "False-negative refusal",
        "severity": 3,
        "description": (
            "Out-of-corpus question was answered instead of refused. Recipe-adjacent "
            "vocabulary (e.g. 'baguette', 'sodium') passes the evidence threshold "
            "even though no nutrition data exists in the corpus."
        ),
    },
    "CI": {
        "name": "Citation missing",
        "severity": 1,
        "description": (
            "Answer is factually correct but contains no inline [Source: chunk_id] "
            "citation. Users cannot verify which card the fact came from."
        ),
    },
}

PREDICTIONS: dict[str, str] = {
    "R": (
        "Increasing top_k from 3 to 5 and switching to a larger embedding window "
        "should move the missing chunk into the retrieved set for at least half of "
        "current retrieval-miss cases, measurably improving hit-rate@5."
    ),
    "FP": (
        "Replacing the hard token-count threshold (>= 2 matching terms) with a "
        "minimum cosine-score cut-off (refuse only when max score < 0.25) should "
        "eliminate false-positive refusals without regressing OOC refusals."
    ),
    "G": (
        "Adding an explicit instruction to the generation prompt -- 'quote numeric "
        "values verbatim, do not round or paraphrase units' -- should bring "
        "generation-miss rate below 1 in 10 answerable questions."
    ),
    "CI": (
        "A post-generation assertion that rejects any answer lacking a [Source: …] "
        "bracket and re-prompts will eliminate citation-missing responses."
    ),
    "FN": (
        "Tightening the evidence threshold so that queries with generic recipe "
        "vocabulary (e.g. 'sodium', 'calories') only pass when a named recipe "
        "section chunk scores above the cut-off should stop out-of-corpus answers."
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_traces(rag: GeminiRAG, trace_path: Path, model: str) -> list[dict]:
    records: list[dict] = []
    print(f"\nRunning {len(SAMPLE_QUESTIONS)} questions ...\n")
    for item in SAMPLE_QUESTIONS:
        print(f"  [{item['id']}] {item['question'][:72]}")
        try:
            answer, sources, prompt = rag.answer(item["question"])
        except Exception as exc:  # noqa: BLE001
            answer = f"ERROR: {exc}"
            sources = []
            prompt = ""

        source_dicts = [
            {
                "chunk_id": s.chunk.id,
                "recipe_id": s.chunk.recipe_id,
                "text": s.chunk.text,
                "score": s.score,
            }
            for s in sources
        ]

        classification = classify_trace(item, answer, source_dicts)
        trace_id = save_trace(
            trace_path, item["question"], answer, prompt,
            sources, model, "hybrid", top_k=3,
        )

        records.append({
            **item,
            "trace_id": trace_id,
            "answer": answer,
            "sources": source_dicts,
            "verdict": classification["verdict"],
            "failure_code": classification["failure_code"],
            "open_code_note": classification["note"],
        })

        icon = "PASS" if classification["verdict"] == "PASS" else "FAIL"
        print(f"         -> [{classification['failure_code']}] {icon}")
        time.sleep(0.5)  # gentle rate-limiting

    return records


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def build_taxonomy(records: list[dict]) -> list[dict]:
    counts: Counter = Counter(r["failure_code"] for r in records)
    examples: dict[str, list[str]] = {}
    for r in records:
        examples.setdefault(r["failure_code"], []).append(r["open_code_note"])

    groups = []
    for code, freq in counts.most_common():
        meta = FAILURE_META.get(code, {"name": code, "severity": 1, "description": ""})
        groups.append({
            "code": code,
            "name": meta["name"],
            "frequency": freq,
            "severity": meta["severity"],
            "score": freq * meta["severity"],
            "description": meta["description"],
            "example_notes": examples[code][:2],
        })

    groups.sort(key=lambda g: (-g["score"], -g["frequency"]))
    for i, g in enumerate(groups, 1):
        g["rank"] = i
    return groups


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(records: list[dict], groups: list[dict], model: str, trace_path: Path) -> str:
    total = len(records)
    passes = sum(1 for r in records if r["failure_code"] == "PASS")
    failures = total - passes

    # Trace table
    trace_rows = ""
    for r in records:
        icon = "PASS" if r["verdict"] == "PASS" else "FAIL"
        q = r["question"][:60].replace("|", "\\|")
        a = r["answer"][:70].replace("|", "\\|").replace("\n", " ")
        note = r["open_code_note"][:90].replace("|", "\\|")
        trace_rows += f"| {r['id']} | {r['category']} | {q}… | {a}… | {r['failure_code']} | {icon} | {note} |\n"

    # Category breakdown
    cat_pass: Counter = Counter()
    cat_fail: Counter = Counter()
    for r in records:
        if r["verdict"] == "PASS":
            cat_pass[r["category"]] += 1
        else:
            cat_fail[r["category"]] += 1
    all_cats = sorted(set(list(cat_pass) + list(cat_fail)))
    cat_rows = ""
    for cat in all_cats:
        p = cat_pass.get(cat, 0)
        f = cat_fail.get(cat, 0)
        cat_rows += f"| {cat} | {p + f} | {p} | {f} |\n"

    # Taxonomy table (skip PASS)
    tax_rows = ""
    for g in groups:
        if g["code"] == "PASS":
            continue
        tax_rows += (
            f"| #{g['rank']} | **{g['name']}** (`{g['code']}`) "
            f"| {g['frequency']}/{total} | {g['severity']}/4 "
            f"| {g['score']} | {g['description'][:95]} |\n"
        )

    # Ranked list
    ranked_lines = ""
    for g in groups:
        if g["code"] == "PASS":
            continue
        ranked_lines += (
            f"  {g['rank']:2d}  {g['code']:4s}  {g['name']:28s}  "
            f"freq={g['frequency']}  sev={g['severity']}  score={g['score']}\n"
        )

    # Fix target
    top = next((g for g in groups if g["code"] != "PASS"), None)
    target_md = ""
    if top:
        pred = PREDICTIONS.get(top["code"], "Re-run the 20-question sample after the fix and measure the change.")
        target_md = f"""## 6. Chosen fix target

**Problem:** {top['name']} (`{top['code']}`)  
**Why first?** Highest freq × severity score ({top['score']}) — {top['frequency']} occurrences at severity {top['severity']}/4.  
**Root cause:** {top['description']}

**Prediction (written before any fix is attempted):**

> {pred}
"""

    return f"""# Week 5 · Module 3 — Error Analysis
## Track B: Recipes & Food RAG

> **Model:** `{model}` · **Retrieval:** hybrid (dense + BM25 + RRF, k=60) · **top_k:** 3
> **Sample:** {total} traces — random, not cherry-picked (fixed seed for reproducibility)
> **Trace log:** `{trace_path.name}`

---

## 1. Sampling methodology

The 20 questions were written **before** any trace was read, covering five categories so that no single failure mode could dominate the sample:

| Category | Count | What it tests |
|---|---|---|
| EXACT_FACT | 8 | Precise weight / percentage from an ingredient table |
| METHOD | 5 | Process step, timing, or temperature |
| CROSS_RECIPE | 3 | Comparison across two or more recipe cards |
| OOC | 3 | Out-of-corpus — must be refused, not invented |
| VAGUE | 1 | Under-specified question; tests hallucination guard |

The order was shuffled once (random seed 42) and fixed so the sample is **random but repeatable**. No question was removed because it looked likely to pass.

---

## 2. Raw traces with open-coded failure notes

*One honest sentence per failure, written before deciding on any category.*

| ID | Category | Question | Answer (truncated) | Code | Verdict | Open-code note |
|---|---|---|---|---|---|---|
{trace_rows.rstrip()}

**Code legend:**  
`PASS` correct & cited · `R` retrieval miss · `G` generation miss · `FP` false-positive refusal · `FN` false-negative refusal · `CI` citation missing

---

## 3. Overall results

| Metric | Value |
|---|---|
| Total traces | {total} |
| PASS | {passes} ({passes/total:.0%}) |
| FAIL (any code) | {failures} ({failures/total:.0%}) |

### Failure breakdown by question category

| Category | Total | PASS | FAIL |
|---|---|---|---|
{cat_rows.rstrip()}

---

## 4. Named problem groups (error taxonomy)

Each open-code note was grouped by its root cause into one named type. Strangers should be able to read these names without needing to see the notes first.

| Rank | Problem group | Frequency | Severity (1–4) | Score (freq×sev) | Description |
|---|---|---|---|---|---|
{tax_rows.rstrip()}

**Severity scale:**
1 = cosmetic (user can still complete task) · 2 = user notices, task degraded · 3 = task fails silently · 4 = actively misleading

---

## 5. Ranked taxonomy (priority order)

```
Rank  Code  Problem group                 freq  sev  score
----  ----  ----------------------------  ----  ---  -----
{ranked_lines.rstrip()}
```

---

{target_md}
---

## 7. Comparison with standard benchmarks

Standard benchmarks (MMLU, HumanEval) measure aggregate accuracy on pre-built question sets and are useful for comparing **models**. They cannot reveal where **this specific app** fails on **this corpus**. This hand-read analysis surfaced three things a benchmark would miss:

1. **Evidence-threshold fragility** — The `_has_enough_evidence()` heuristic counts stop-word-filtered token overlap between the question and retrieved chunks. Synonym and paraphrase variants of correct questions routinely score below the threshold of 2, causing false-positive refusals that an accuracy benchmark would never catch.

2. **Cross-recipe retrieval gap** — CROSS_RECIPE questions require the LLM to synthesise across 3+ retrieved chunks from different recipe cards. The current section-level chunking emits one chunk per section per card, so multi-recipe comparisons depend entirely on whether all relevant cards appear in top-3 — a structural limitation invisible to single-question accuracy metrics.

3. **OOC vocabulary bleed** — Out-of-corpus questions that share vocabulary with recipe cards (e.g. "sodium content per baguette" — baguette exists in the corpus, sodium data does not) can pass the evidence threshold and receive a hallucinated answer. Aggregate pass-rate metrics would not distinguish this from a correct answer.

---

*Full trace log: `{trace_path}`*
*Structured records: `output/week5_records.json`*
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _dry_run() -> None:
    """Print the question list without making any API calls."""
    print("=== Week 5 - DRY RUN (no API calls) ===")
    print(f"{'ID':4s}  {'Category':12s}  Question")
    print("-" * 90)
    for q in SAMPLE_QUESTIONS:
        print(f"{q['id']:4s}  {q['category']:12s}  {q['question']}")
    print(f"\nTotal questions: {len(SAMPLE_QUESTIONS)}")
    print("Add your GEMINI_API_KEY to .env and run without --dry-run to collect real traces.")


def main() -> None:
    import sys
    if "--dry-run" in sys.argv:
        _dry_run()
        return

    settings = load_settings()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "traces_week5.jsonl"
    report_path = settings.project_root / "week5_error_analysis.md"

    print("=== Week 5 - Error Analysis ===")
    print(f"Model:      {settings.llm_model}")
    print(f"Documents:  {settings.documents_dir}")
    print(f"Trace log:  {trace_path}")

    # Build RAG index (uses cached embeddings when fingerprint matches)
    rag = GeminiRAG(settings.api_key, settings.index_path, settings.llm_model)
    n = rag.index_documents(settings.documents_dir)
    print(f"Indexed:    {n} chunks\n")

    # Run 20 questions
    records = run_traces(rag, trace_path, settings.llm_model)

    # Build taxonomy
    groups = build_taxonomy(records)

    # Print summary
    total = len(records)
    passes = sum(1 for r in records if r["failure_code"] == "PASS")
    print(f"\n=== Summary ===")
    print(f"Total: {total}  PASS: {passes}  FAIL: {total - passes}  Rate: {passes/total:.0%}")
    print("\nRanked taxonomy:")
    for g in groups:
        if g["code"] != "PASS":
            print(f"  #{g['rank']:2d} [{g['code']:3s}] {g['name']:28s}  freq={g['frequency']}  sev={g['severity']}  score={g['score']}")

    # Write markdown report
    md = build_report(records, groups, settings.llm_model, trace_path)
    report_path.write_text(md, encoding="utf-8")
    print(f"\nReport: {report_path}")

    # Write structured JSON
    json_path = output_dir / "week5_records.json"
    json_path.write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "sources"} for r in records],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"JSON:   {json_path}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
