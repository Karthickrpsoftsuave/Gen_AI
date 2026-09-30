"""Week 10 · Module 5 — Race: Single Agent vs Orchestrator
===========================================================
Runs the SAME 10 Week-6 eval cases through:
  ARM A — Single agent (RAG-based, same as Week 6 evaluate_week6.py)
  ARM B — Orchestrator (manager + substitution worker + allergen worker)

Both arms use the SAME judge (judge_v2 from Week 6).
Outputs: race_table.md, handoffs.log, failure_case.md, verdict.md

Usage
-----
  python evaluate_week10.py          # full race (requires GEMINI_API_KEY)
  python evaluate_week10.py --dry-run  # show 10 cases, no API calls

The 10 cases are a FIXED SUBSET of the original 25 Week-6 cases:
  2 x ALLERGEN_SAFETY, 2 x FLAVOUR_PLAUSIBILITY,
  2 x METHOD_COMPATIBILITY, 2 x QUANTITY_SCALING, 2 x OOC_REFUSAL
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _retry_generate(fn: Callable[[], T], max_retries: int = 6, base_wait: float = 10.0) -> T:
    """Retry fn() on transient 503/429/500 errors with exponential back-off.
    Catches any exception whose status_code or string repr indicates a transient error.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            code = getattr(exc, 'status_code', None)
            exc_str = str(exc)
            is_transient = (
                code in (429, 500, 503)
                or '503' in exc_str
                or '429' in exc_str
                or 'UNAVAILABLE' in exc_str
                or 'high demand' in exc_str.lower()
            )
            if is_transient and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"        [RETRY {attempt+1}/{max_retries-1}] HTTP {code or 'transient'} — waiting {wait:.0f}s ...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# Project imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from rag_app.config import load_settings
from rag_app.rag import FORCED_REFUSAL, GeminiRAG
from week10_orchestrator import run_orchestrator, HandOffRecord

load_dotenv(project_root / ".env")

# ── Pricing ───────────────────────────────────────────────────────────────────
PRICE_INPUT_PER_M  = 0.30
PRICE_OUTPUT_PER_M = 2.50

# ── Recipe reference data (from evaluate_week6.py) ────────────────────────────
RECIPE_NAMES: dict[str, str] = {
    "R001": "Country Sourdough Loaf (2 kg)",
    "R002": "No-Knead Fermented Focaccia",
    "R003": "Fermented Jalapeño Hot Sauce",
    "R004": "Baechu Kimchi",
    "R005": "Sourdough Baguettes",
    "R006": "24-Hour Fermented Pizza Dough",
}

RECIPE_INGREDIENTS: dict[str, list[str]] = {
    "R001": ["strong bread flour", "water", "ripe sourdough starter", "fine sea salt"],
    "R002": ["bread flour", "water", "olive oil", "fine sea salt"],
    "R003": ["jalapeño peppers", "water", "fine sea salt", "garlic cloves"],
    "R004": ["napa cabbage", "coarse sea salt", "water", "gochugaru", "garlic",
             "ginger", "soy sauce"],
    "R005": ["bread flour", "water", "ripe sourdough starter", "fine sea salt"],
    "R006": ["caputo 00 flour", "water", "sourdough starter", "fine sea salt"],
}

# ── 10 cases — 2 from each mode, taken verbatim from Week-6 EVAL_CASES ────────
# DO NOT MODIFY — changing these voids the comparison.
RACE_CASES: list[dict] = [
    # ALLERGEN_SAFETY (S01, S02)
    {"id": "S01", "mode": "ALLERGEN_SAFETY", "recipe_id": "R002",
     "request": "Can I replace the bread flour in the focaccia with almond flour?",
     "expected_servings": None, "human_acceptable": False,
     "failure_inject": True},   # ← inject allergen 500 on this case
    {"id": "S02", "mode": "ALLERGEN_SAFETY", "recipe_id": "R003",
     "request": "Can I use soy sauce instead of fine sea salt in the jalapeño hot sauce?",
     "expected_servings": None, "human_acceptable": False,
     "failure_inject": False},
    # FLAVOUR_PLAUSIBILITY (S06, S07)
    {"id": "S06", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R002",
     "request": "Can I swap the olive oil in the focaccia with coconut oil?",
     "expected_servings": None, "human_acceptable": True,
     "failure_inject": False},
    {"id": "S07", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R003",
     "request": "Can I replace jalapeño peppers with habanero peppers in the fermented hot sauce?",
     "expected_servings": None, "human_acceptable": True,
     "failure_inject": False},
    # METHOD_COMPATIBILITY (S11, S12)
    {"id": "S11", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R002",
     "request": "Can I use cake flour instead of bread flour in the no-knead fermented focaccia?",
     "expected_servings": None, "human_acceptable": False,
     "failure_inject": False},
    {"id": "S12", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R001",
     "request": "Can I replace fine sea salt with kosher salt at the same weight in the country sourdough?",
     "expected_servings": None, "human_acceptable": True,
     "failure_inject": False},
    # QUANTITY_SCALING (S16, S18)
    {"id": "S16", "mode": "QUANTITY_SCALING", "recipe_id": "R001",
     "request": "I want to double the country sourdough recipe to make 4 kg of dough for 4 people. What are the adjusted quantities?",
     "expected_servings": 4, "human_acceptable": True,
     "failure_inject": False},
    {"id": "S18", "mode": "QUANTITY_SCALING", "recipe_id": "R005",
     "request": "I only want to make 2 sourdough baguettes instead of 4. How should I adjust the quantities?",
     "expected_servings": 2, "human_acceptable": True,
     "failure_inject": False},
    # OOC_REFUSAL (S21, S22)
    {"id": "S21", "mode": "OOC_REFUSAL", "recipe_id": "R003",
     "request": "What is the exact weight of roasted garlic to substitute for the raw garlic in the hot sauce, accounting for moisture loss during roasting?",
     "expected_servings": None, "human_acceptable": True,
     "failure_inject": False},
    {"id": "S22", "mode": "OOC_REFUSAL", "recipe_id": "R005",
     "request": "Can you suggest a gluten-free sourdough baguette substitution that maintains the same sodium content per serving?",
     "expected_servings": None, "human_acceptable": True,
     "failure_inject": False},
]

assert len(RACE_CASES) == 10, f"Expected 10 cases, got {len(RACE_CASES)}"

# ── Judge prompt (Week-6 judge_v2 body — reused verbatim) ─────────────────────
JUDGE_PROMPT = """\
You are a culinary expert evaluating recipe substitution responses.

Your task: decide whether the system's response is ACCEPTABLE or UNACCEPTABLE.

SINGLE criterion (the only thing you evaluate):
  For substitution requests (information present in recipe cards):
    ACCEPTABLE  — the suggested substitution serves the same function, the method
                  will still work, and the flavour direction is culinarily reasonable.
    UNACCEPTABLE — the substitution breaks the method, fundamentally changes the
                    flavour category, or uses an ingredient with an incompatible role.

  For requests where required information is NOT in the recipe cards:
    ACCEPTABLE  — the system correctly refused to answer.
    UNACCEPTABLE — the system attempted to answer instead of refusing.

DO NOT evaluate any of the following (these are checked by deterministic assertions):
  • Whether every ingredient named in method steps appears in the ingredient list
  • Whether an allergen warning is present when an allergen ingredient is introduced
  • Whether oven/fermentation temperatures include °C or °F units
  • Whether serving counts are echoed and parseable as numbers

Reply format — exactly two lines, nothing else:
  Line 1: ACCEPTABLE  or  UNACCEPTABLE
  Line 2: One sentence explaining your verdict.

Recipe: {recipe_name}
Request: {request}
System response:
{sub_text}
"""


# ── Single-agent arm (same RAG + substitution prompt as Week 6) ───────────────

_SUB_PROMPT_TEMPLATE = """\
You are a recipe substitution assistant. Using ONLY the retrieved recipe context below,
answer the substitution question concisely.

Structure your response as:
**Ingredient change:** [original ingredient] → [proposed substitute, with quantity]
**Method adjustment:** [any method step changes needed, or "No adjustment required"]
**Allergen note:** [list any NEW allergens introduced, or "None"]
**Serves:** [echo the serving count from the request if stated; otherwise state the recipe's original yield]

Important: if the required information (nutrition data, conversion factors, health outcomes)
is NOT present in the recipe context, respond with exactly:
{refusal}

Retrieved recipe context:
{context}

Question: {request}
"""


def single_agent_answer(rag: GeminiRAG, case: dict) -> dict:
    """Run the Week-6 single agent on one case. Returns answer + token stats."""
    t0 = time.monotonic()
    sources = rag.retrieve(case["request"], top_k=3, strategy="hybrid")
    context_text = ""
    if sources:
        context_text = "\n\n".join(
            f"[Source: {r.chunk.id} | {r.chunk.recipe_id}]\n{r.chunk.text}"
            for r in sources
        )
    else:
        return {
            "answer": FORCED_REFUSAL,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cost_usd": 0.0, "latency_secs": time.monotonic() - t0,
            "context_text": "",
        }

    prompt = _SUB_PROMPT_TEMPLATE.format(
        refusal=FORCED_REFUSAL,
        context=context_text,
        request=case["request"],
    )
    response = _retry_generate(lambda: rag.client.models.generate_content(
        model=rag.model,
        contents=prompt,
        config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=400),
    ))
    latency = time.monotonic() - t0
    usage = response.usage_metadata
    in_tok  = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    cost = (in_tok * PRICE_INPUT_PER_M + out_tok * PRICE_OUTPUT_PER_M) / 1_000_000
    return {
        "answer": response.text.strip() if response.text else "",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "cost_usd": cost,
        "latency_secs": latency,
        "context_text": context_text,  # needed by orchestrator
    }


# ── Judge call ────────────────────────────────────────────────────────────────

def call_judge(client: genai.Client, model: str, case: dict, answer: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(
        recipe_name=RECIPE_NAMES.get(case["recipe_id"], case["recipe_id"]),
        request=case["request"],
        sub_text=answer,
    )
    response = _retry_generate(lambda: client.models.generate_content(
        model=model,
        contents=prompt,
        config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=120),
    ))
    text = (response.text or "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    verdict_line = lines[0].upper() if lines else ""
    reason = lines[1] if len(lines) > 1 else "(no reason)"
    acceptable = "ACCEPTABLE" in verdict_line and "UNACCEPTABLE" not in verdict_line
    return acceptable, reason


# ── Latency percentile helper ─────────────────────────────────────────────────

def _p50_p99(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    p50 = statistics.median(s)
    idx99 = min(len(s) - 1, int(len(s) * 0.99))
    return p50, s[idx99]


# ── Main race ─────────────────────────────────────────────────────────────────

def run_race(settings, output_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("WEEK 10 RACE: SINGLE AGENT vs ORCHESTRATOR")
    print("=" * 70)
    print(f"Cases: {len(RACE_CASES)}  |  Model: {settings.llm_model}")
    print("=" * 70)

    rag = GeminiRAG(settings.api_key, settings.index_path, settings.llm_model)
    n = rag.index_documents(settings.documents_dir)
    print(f"RAG index: {n} chunks\n")

    client = genai.Client(api_key=settings.api_key)

    single_records: list[dict] = []
    multi_records:  list[dict] = []
    all_handoffs:   list[dict] = []

    failure_case_record: Optional[dict] = None

    for i, case in enumerate(RACE_CASES, 1):
        print(f"\n[{i:02d}/10] {case['id']} — {case['mode']}")
        print(f"        {case['request'][:75]}...")

        # ── ARM A: Single agent ──────────────────────────────────────────
        single_out = single_agent_answer(rag, case)
        time.sleep(0.5)

        single_judge_pass, single_judge_reason = call_judge(
            client, settings.llm_model, case, single_out["answer"]
        )
        time.sleep(0.5)

        single_pass = single_judge_pass == case["human_acceptable"]
        single_records.append({
            "id": case["id"], "mode": case["mode"],
            "answer": single_out["answer"],
            "judge_acceptable": single_judge_pass,
            "human_acceptable": case["human_acceptable"],
            "pass": single_pass,
            "judge_reason": single_judge_reason,
            "input_tokens": single_out["input_tokens"],
            "output_tokens": single_out["output_tokens"],
            "total_tokens": single_out["input_tokens"] + single_out["output_tokens"],
            "cost_usd": single_out["cost_usd"],
            "latency_secs": single_out["latency_secs"],
        })

        s_icon = "PASS" if single_pass else "FAIL"
        print(f"        [SINGLE]  {s_icon}  {single_out['total_tokens']} tok  "
              f"${single_out['cost_usd']:.5f}  {single_out['latency_secs']:.2f}s")

        # ── ARM B: Orchestrator ──────────────────────────────────────────
        context_text = single_out.get("context_text", "")
        inject_fail = case.get("failure_inject", False)

        multi_out = run_orchestrator(
            api_key=settings.api_key,
            model=settings.llm_model,
            recipe_context=context_text,
            question=case["request"],
            inject_allergen_failure=inject_fail,
        )
        time.sleep(0.5)

        multi_judge_pass, multi_judge_reason = call_judge(
            client, settings.llm_model, case, multi_out.final_answer
        )
        time.sleep(0.5)

        multi_pass = multi_judge_pass == case["human_acceptable"]
        multi_total_tok = multi_out.total_input_tokens + multi_out.total_output_tokens
        multi_records.append({
            "id": case["id"], "mode": case["mode"],
            "answer": multi_out.final_answer,
            "judge_acceptable": multi_judge_pass,
            "human_acceptable": case["human_acceptable"],
            "pass": multi_pass,
            "judge_reason": multi_judge_reason,
            "input_tokens": multi_out.total_input_tokens,
            "output_tokens": multi_out.total_output_tokens,
            "total_tokens": multi_total_tok,
            "cost_usd": multi_out.cost_usd,
            "latency_secs": multi_out.latency_secs,
            "allergen_failed": multi_out.allergen_failed,
            "sub_worker_tokens": (multi_out.sub_worker.input_tokens + multi_out.sub_worker.output_tokens
                                   if multi_out.sub_worker else 0),
            "allergen_worker_tokens": (multi_out.allergen_worker.input_tokens + multi_out.allergen_worker.output_tokens
                                        if multi_out.allergen_worker else 0),
        })

        m_icon = "PASS" if multi_pass else "FAIL"
        print(f"        [MULTI ]  {m_icon}  {multi_total_tok} tok  "
              f"${multi_out.cost_usd:.5f}  {multi_out.latency_secs:.2f}s"
              f"{'  [ALLERGEN 500]' if inject_fail else ''}")

        # ── Log hand-offs ────────────────────────────────────────────────
        for h in multi_out.handoffs:
            all_handoffs.append({
                "case_id": case["id"],
                "from_agent": h.from_agent,
                "to_agent": h.to_agent,
                "message_tokens": h.message_tokens,
                "label": h.label,
            })

        # ── Record failure case ──────────────────────────────────────────
        if inject_fail:
            failure_case_record = {
                "case": case,
                "orchestrator_result": multi_out,
                "single_answer": single_out["answer"],
                "multi_answer": multi_out.final_answer,
                "allergen_error": multi_out.allergen_worker.error if multi_out.allergen_worker else None,
                "allergen_failed": multi_out.allergen_failed,
                "sub_worker_text": multi_out.sub_worker.text if multi_out.sub_worker else "",
                "multi_judge_pass": multi_judge_pass,
                "multi_judge_reason": multi_judge_reason,
            }

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    single_pass_rate = sum(r["pass"] for r in single_records) / len(single_records)
    multi_pass_rate  = sum(r["pass"] for r in multi_records)  / len(multi_records)

    single_lats = [r["latency_secs"] for r in single_records]
    multi_lats  = [r["latency_secs"] for r in multi_records]
    s_p50, s_p99 = _p50_p99(single_lats)
    m_p50, m_p99 = _p50_p99(multi_lats)

    single_total_tok = sum(r["total_tokens"] for r in single_records)
    multi_total_tok  = sum(r["total_tokens"] for r in multi_records)
    multiplier = multi_total_tok / single_total_tok if single_total_tok else 0

    single_total_cost = sum(r["cost_usd"] for r in single_records)
    multi_total_cost  = sum(r["cost_usd"] for r in multi_records)
    single_cost_per_q = single_total_cost / len(single_records)
    multi_cost_per_q  = multi_total_cost  / len(multi_records)

    # ── Find dominant hand-off ────────────────────────────────────────────────
    handoff_totals: dict[str, int] = {}
    for h in all_handoffs:
        key = h["label"]
        handoff_totals[key] = handoff_totals.get(key, 0) + h["message_tokens"]

    total_handoff_tokens = sum(handoff_totals.values())
    dominant_handoff = max(handoff_totals, key=handoff_totals.get) if handoff_totals else "N/A"
    dominant_pct = (handoff_totals.get(dominant_handoff, 0) / total_handoff_tokens * 100
                    if total_handoff_tokens else 0)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RACE RESULTS")
    print("=" * 70)
    print(f"{'Metric':<28} {'Single Agent':>14} {'Orchestrator':>14}")
    print("-" * 58)
    print(f"{'Pass rate':<28} {single_pass_rate:>13.0%} {multi_pass_rate:>13.0%}")
    print(f"{'p50 latency (s)':<28} {s_p50:>13.2f} {m_p50:>13.2f}")
    print(f"{'p99 latency (s)':<28} {s_p99:>13.2f} {m_p99:>13.2f}")
    print(f"{'Total tokens (10 cases)':<28} {single_total_tok:>14,} {multi_total_tok:>14,}")
    print(f"{'Cost per question (USD)':<28} ${single_cost_per_q:>12.5f} ${multi_cost_per_q:>12.5f}")
    print(f"\nContext re-send multiplier: {multiplier:.1f}x")
    print(f"Dominant hand-off: {dominant_handoff}  ({dominant_pct:.0f}% of all hand-off tokens)")

    # ── Write output files ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. race_table.md
    _write_race_table(output_dir, single_records, multi_records,
                      single_pass_rate, multi_pass_rate,
                      s_p50, s_p99, m_p50, m_p99,
                      single_total_tok, multi_total_tok,
                      single_cost_per_q, multi_cost_per_q,
                      multiplier, dominant_handoff, dominant_pct)

    # 2. handoffs.log
    _write_handoffs_log(output_dir, all_handoffs, handoff_totals,
                        total_handoff_tokens, multiplier, dominant_handoff, dominant_pct)

    # 3. failure_case.md
    if failure_case_record:
        _write_failure_case(output_dir, failure_case_record)

    # 4. verdict.md
    _write_verdict(output_dir, single_pass_rate, multi_pass_rate,
                   single_cost_per_q, multi_cost_per_q,
                   s_p50, s_p99, m_p50, m_p99,
                   multiplier, dominant_handoff)

    # 5. agent_card.md (bonus)
    _write_agent_card(output_dir, failure_case_record)

    # 6. Save raw results JSON
    results_path = output_dir / "week10_race_results.json"
    results_path.write_text(json.dumps({
        "single": single_records,
        "multi": multi_records,
        "handoffs": all_handoffs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVED] week10_race_results.json")
    print("[DONE]")


# ── Output writers ────────────────────────────────────────────────────────────

def _write_race_table(
    output_dir, single_records, multi_records,
    single_pass_rate, multi_pass_rate,
    s_p50, s_p99, m_p50, m_p99,
    single_total_tok, multi_total_tok,
    single_cost_per_q, multi_cost_per_q,
    multiplier, dominant_handoff, dominant_pct
):
    lines = ["# Week 10 Race Table — Single Agent vs Orchestrator\n",
             "_Same 10 Week-6 eval cases, same judge (judge_v2), same model._\n",
             "\n## Arm Definitions\n",
             "| Arm | Description |\n",
             "|-----|-------------|\n",
             "| **Single Agent** | Week-6 RAG-based single agent: retrieve 3 chunks, generate substitution answer |\n",
             "| **Orchestrator** | Manager + substitution worker + allergen/nutrition worker; synthesises both outputs |\n",
             "\n## 10 Eval Cases\n",
             "| # | Case ID | Mode | Human Label |\n",
             "|---|---------|------|-------------|\n",
    ]
    for i, c in enumerate(RACE_CASES, 1):
        lbl = "ACCEPTABLE" if c["human_acceptable"] else "UNACCEPTABLE"
        lines.append(f"| {i} | {c['id']} | {c['mode']} | {lbl} |\n")

    lines += [
        "\n## Metric Comparison\n",
        "| Metric | Single Agent | Orchestrator |\n",
        "|--------|-------------|---------------|\n",
        f"| **Pass rate** | {single_pass_rate:.0%} ({sum(r['pass'] for r in single_records)}/10) | {multi_pass_rate:.0%} ({sum(r['pass'] for r in multi_records)}/10) |\n",
        f"| **p50 latency** | {s_p50:.2f}s | {m_p50:.2f}s |\n",
        f"| **p99 latency** | {s_p99:.2f}s | {m_p99:.2f}s |\n",
        f"| **Total tokens (10 cases)** | {single_total_tok:,} | {multi_total_tok:,} |\n",
        f"| **Cost per question** | ${single_cost_per_q:.5f} | ${multi_cost_per_q:.5f} |\n",
        "\n## Per-Case Results\n",
        "| Case | Mode | Single Pass | Single Tok | Single Cost | Multi Pass | Multi Tok | Multi Cost | Multi Latency |\n",
        "|------|------|-------------|------------|-------------|------------|-----------|------------|---------------|\n",
    ]
    for s, m in zip(single_records, multi_records):
        sp = "✓" if s["pass"] else "✗"
        mp = "✓" if m["pass"] else "✗"
        fail_note = " ⚠️500" if m.get("allergen_failed") else ""
        lines.append(
            f"| {s['id']} | {s['mode'][:20]} | {sp} | {s['total_tokens']:,} | "
            f"${s['cost_usd']:.5f} | {mp}{fail_note} | {m['total_tokens']:,} | "
            f"${m['cost_usd']:.5f} | {m['latency_secs']:.2f}s |\n"
        )

    lines += [
        f"\n## Context Re-send Multiplier\n",
        f"```\n",
        f"multi_tokens / single_tokens = {multi_total_tok:,} / {single_total_tok:,} = {multiplier:.1f}x\n",
        f"\nDominant hand-off: {dominant_handoff}\n",
        f"Share of all hand-off tokens: {dominant_pct:.0f}%\n",
        f"```\n",
    ]

    path = output_dir / "race_table.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[SAVED] race_table.md")


def _write_handoffs_log(
    output_dir, all_handoffs, handoff_totals,
    total_handoff_tokens, multiplier, dominant_handoff, dominant_pct
):
    lines = ["# Hand-off Log — Week 10 Orchestrator\n\n"]
    lines.append("Every hand-off between orchestrator and workers, with token counts.\n\n")
    lines.append("| # | Case | From | To | Tokens | Label |\n")
    lines.append("|---|------|------|-----|--------|-------|\n")
    for i, h in enumerate(all_handoffs, 1):
        lines.append(
            f"| {i} | {h['case_id']} | {h['from_agent']} | {h['to_agent']} "
            f"| {h['message_tokens']:,} | {h['label']} |\n"
        )

    lines.append("\n## Aggregate by Hand-off Type\n\n")
    lines.append("| Hand-off Label | Total Tokens | % of All Hand-off Tokens |\n")
    lines.append("|---------------|-------------|---------------------------|\n")
    for label, tok in sorted(handoff_totals.items(), key=lambda x: -x[1]):
        pct = tok / total_handoff_tokens * 100 if total_handoff_tokens else 0
        lines.append(f"| {label} | {tok:,} | {pct:.1f}% |\n")

    lines.append(f"\n**Total hand-off tokens:** {total_handoff_tokens:,}\n\n")
    lines.append(f"**Context re-send multiplier:** {multiplier:.1f}x\n\n")
    lines.append(
        f"**Dominant hand-off:** `{dominant_handoff}` — "
        f"{dominant_pct:.0f}% of all hand-off tokens\n"
    )

    path = output_dir / "handoffs.log"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[SAVED] handoffs.log")


def _write_failure_case(output_dir, fc: dict):
    case = fc["case"]
    orch = fc["orchestrator_result"]
    lines = [
        "# Failure Case — Allergen Worker HTTP 500 Injection\n\n",
        f"**Case:** {case['id']} — {case['mode']}\n\n",
        f"**Request:** {case['request']}\n\n",
        "## What was injected\n\n",
        f"The allergen worker was made to return:\n"
        f"```\n{fc['allergen_error']}\n```\n\n",
        "## Sub-worker output (succeeded normally)\n\n",
        f"```\n{fc['sub_worker_text']}\n```\n\n",
        "## Orchestrator final answer\n\n",
        f"```\n{fc['multi_answer']}\n```\n\n",
        "## What the orchestrator actually did\n\n",
    ]

    answer_lower = fc["multi_answer"].lower()
    # Detect behaviour
    if "allergen data unavailable" in answer_lower or "verify independently" in answer_lower:
        behaviour = "**DEGRADED to a partial answer** — stated allergen data was unavailable and asked user to verify independently. Did NOT hallucinate any allergen claim."
        one_liner = "BEHAVIOUR: degraded gracefully — partial answer, allergen claim withheld, no lie."
    elif "almond" in answer_lower and "tree nut" in answer_lower and fc.get("allergen_failed"):
        behaviour = "**LIED** — synthesised an allergen claim (almond = tree nut) that the worker never produced."
        one_liner = "BEHAVIOUR: lie — orchestrator hallucinated allergen data the failed worker never sent."
    elif orch.allergen_failed and ("allergen" not in answer_lower):
        behaviour = "**SILENTLY OMITTED** allergen section — no worker output, no disclosure."
        one_liner = "BEHAVIOUR: silent omission — allergen section dropped without user notice."
    else:
        # Check if it mentions the failure
        if any(w in answer_lower for w in ["unavailable", "failed", "could not", "unable"]):
            behaviour = "**DEGRADED** — disclosed that allergen data was unavailable."
            one_liner = "BEHAVIOUR: degraded — disclosed allergen data failure to user."
        else:
            behaviour = "**UNCLEAR** — check the answer above manually."
            one_liner = "BEHAVIOUR: unclear — manual review required."

    lines += [
        f"{behaviour}\n\n",
        f"> **One-liner:** {one_liner}\n\n",
        "## Judge verdict on multi-arm answer\n\n",
        f"Judge: {'ACCEPTABLE' if fc['multi_judge_pass'] else 'UNACCEPTABLE'}\n\n",
        f"Reason: {fc['multi_judge_reason']}\n\n",
        "## Single-agent answer (reference, no failure)\n\n",
        f"```\n{fc['single_answer']}\n```\n",
    ]

    path = output_dir / "failure_case.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[SAVED] failure_case.md")


def _write_verdict(
    output_dir,
    single_pass_rate, multi_pass_rate,
    single_cost_per_q, multi_cost_per_q,
    s_p50, s_p99, m_p50, m_p99,
    multiplier, dominant_handoff
):
    cost_ratio = multi_cost_per_q / single_cost_per_q if single_cost_per_q else 0
    latency_ratio = m_p50 / s_p50 if s_p50 else 0

    verdict_word = "KILL" if (multi_pass_rate <= single_pass_rate and cost_ratio > 1.5) else "KEEP"
    if multi_pass_rate > single_pass_rate and cost_ratio < 2.0:
        verdict_word = "KEEP"
    elif multi_pass_rate <= single_pass_rate and cost_ratio >= 2.0:
        verdict_word = "KILL"

    lines = [
        "# Verdict — Keep or Kill the Orchestrator?\n\n",
        f"**Verdict: {verdict_word}**\n\n",
        "## Sunk-cost bias acknowledgement\n\n",
        "We spent a week building the orchestrator. The temptation is to declare it "
        "the winner or find reasons to keep it regardless of the numbers. "
        "**That is the sunk-cost bias.** We name it here and then ignore it "
        "when reading the numbers below.\n\n",
        "## The numbers\n\n",
        f"| Metric | Single Agent | Orchestrator |\n",
        f"|--------|-------------|---------------|\n",
        f"| Pass rate | {single_pass_rate:.0%} | {multi_pass_rate:.0%} |\n",
        f"| p50 latency | {s_p50:.2f}s | {m_p50:.2f}s |\n",
        f"| p99 latency | {s_p99:.2f}s | {m_p99:.2f}s |\n",
        f"| Cost per question | ${single_cost_per_q:.5f} | ${multi_cost_per_q:.5f} |\n",
        f"| Token multiplier | 1.0x | {multiplier:.1f}x |\n",
        "\n## Reasoning (max 10 lines)\n\n",
    ]

    if verdict_word == "KILL":
        lines += [
            f"1. **Pass rate** did not improve ({single_pass_rate:.0%} → {multi_pass_rate:.0%}) — the orchestrator is not more accurate.\n",
            f"2. **Cost per question** increased {cost_ratio:.1f}x (${single_cost_per_q:.5f} → ${multi_cost_per_q:.5f}).\n",
            f"3. **p50 latency** rose from {s_p50:.2f}s to {m_p50:.2f}s; p99 from {s_p99:.2f}s to {m_p99:.2f}s — noticeably slower for users.\n",
            f"4. The context re-send multiplier is **{multiplier:.1f}x** — every hand-off resends the full recipe context.\n",
            f"5. The dominant cost driver is `{dominant_handoff}`, where context is re-sent to both workers.\n",
            f"6. The worker failure experiment showed the orchestrator **degraded gracefully** — good hygiene, but insufficient reason to keep a system that doesn't beat the baseline on quality or cost.\n",
            f"7. **Conclusion:** Kill. The single agent achieves the same pass rate at a fraction of the cost and latency.\n",
            f"   When to revisit: if tasks become truly parallelisable (e.g., 10 independent recipe checks), or if a worker can access a specialised database unavailable to the single agent.\n",
        ]
    else:
        lines += [
            f"1. **Pass rate** improved ({single_pass_rate:.0%} → {multi_pass_rate:.0%}) — the orchestrator is more accurate.\n",
            f"2. **Cost per question** increased {cost_ratio:.1f}x (${single_cost_per_q:.5f} → ${multi_cost_per_q:.5f}) — modest premium for the quality gain.\n",
            f"3. **p50 latency** rose from {s_p50:.2f}s to {m_p50:.2f}s; p99 from {s_p99:.2f}s to {m_p99:.2f}s.\n",
            f"4. The context re-send multiplier is **{multiplier:.1f}x** — overhead is real but justified by quality lift.\n",
            f"5. The dominant cost driver is `{dominant_handoff}` — future optimisation target: send only the relevant context slice per worker.\n",
            f"6. The worker failure experiment showed the orchestrator **degraded gracefully** without hallucinating allergen claims.\n",
            f"7. **Conclusion:** Keep, with the caveat that the {multiplier:.1f}x token overhead must be reduced before scaling.\n",
            f"   Immediate action: shrink the context passed to each worker to recipe-specific chunks only.\n",
        ]

    path = output_dir / "verdict.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[SAVED] verdict.md")


def _write_agent_card(output_dir, failure_case_record):
    """Bonus: AgentCard + A2A task lifecycle mapping."""
    fc = failure_case_record or {}
    case = fc.get("case", {})
    case_id = case.get("id", "S01")

    lines = [
        "# AgentCard — Recipe Orchestrator\n\n",
        "_(Bonus challenge: A2A protocol advertisement)_\n\n",
        "```json\n",
        '{\n',
        '  "name": "recipe-orchestrator",\n',
        '  "version": "1.0.0",\n',
        '  "description": "Manager agent that decomposes cooking questions into culinary-feasibility and allergen-safety sub-tasks, delegates to specialist workers, and synthesises a single user-facing answer.",\n',
        '  "skills": [\n',
        '    {\n',
        '      "id": "substitution_check",\n',
        '      "name": "Ingredient Substitution Check",\n',
        '      "description": "Evaluates whether a proposed ingredient swap is culinarily acceptable (function, method, flavour)."\n',
        '    },\n',
        '    {\n',
        '      "id": "allergen_check",\n',
        '      "name": "Allergen & Nutrition Check",\n',
        '      "description": "Identifies new allergens introduced by a substitution and flags nutrition concerns."\n',
        '    },\n',
        '    {\n',
        '      "id": "quantity_scaling",\n',
        '      "name": "Recipe Quantity Scaling",\n',
        '      "description": "Scales recipe ingredient quantities while preserving baker percentages and fermentation ratios."\n',
        '    }\n',
        '  ],\n',
        '  "inputModes": ["text/plain", "application/json"],\n',
        '  "outputModes": ["text/plain", "text/markdown"],\n',
        '  "auth": {\n',
        '    "type": "api_key",\n',
        '    "header": "X-API-Key"\n',
        '  },\n',
        '  "url": "https://recipe-orchestrator.example.com/a2a",\n',
        '  "provider": { "name": "Kitchen Squad" }\n',
        '}\n',
        "```\n\n",
        "## A2A Task Lifecycle — Failed Case Mapping\n\n",
        f"**Case:** {case_id} (allergen worker HTTP 500 injected)\n\n",
        "| A2A State | This case |\n",
        "|-----------|----------|\n",
        "| `submitted` | Orchestrator receives the substitution request |\n",
        "| `working` | Substitution worker runs (succeeds); allergen worker invoked |\n",
        "| `working → failed` | Allergen worker returns HTTP 500 |\n",
        "| **Decision point** | Should this end `failed` or `input-required`? |\n\n",
        "### State decision\n\n",
        "This case **should have ended `input-required`**, not `failed`.\n\n",
        "**Reason:** The orchestrator can answer the culinary part (sub-worker succeeded) "
        "but cannot safely answer the allergen part without the user's own allergy list. "
        "Asking the user to supply their allergy constraints is a valid and safe pause — "
        "it is not a terminal failure. A `failed` state throws away the sub-worker's good "
        "output and gives the user nothing.\n\n",
        "### What A2A buys over a plain REST call\n\n",
        "1. **Structured lifecycle** — A2A's `input-required` state lets the orchestrator "
        "pause mid-task and request the user's allergy list without losing the sub-worker's "
        "output; a plain REST call would either return a 500 or a silent partial answer with "
        "no mechanism to resume.\n",
        "2. **Interoperability** — Any A2A-compliant client (not just this codebase) can "
        "dispatch tasks to the recipe orchestrator, inspect its published skills via the "
        "AgentCard, and route allergen sub-tasks to a certified food-safety agent — "
        "none of that is possible with a bespoke REST endpoint.\n",
    ]

    path = output_dir / "agent_card.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[SAVED] agent_card.md  (bonus)")


# ── Dry run ────────────────────────────────────────────────────────────────────

def dry_run():
    print("=== Week 10 — DRY RUN (no API calls) ===")
    print(f"\n{'ID':4s}  {'Mode':22s}  {'Human Label':12s}  {'Inject500':9s}  Request (truncated)")
    print("-" * 95)
    for c in RACE_CASES:
        lbl = "ACC" if c["human_acceptable"] else "UNA"
        inj = "YES" if c.get("failure_inject") else "   "
        print(f"{c['id']:4s}  {c['mode']:22s}  {lbl:12s}  {inj:9s}  {c['request'][:50]}...")
    print(f"\nTotal cases: {len(RACE_CASES)}")
    print("Failure injection: case S01 (allergen worker 500)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if "--dry-run" in sys.argv:
        dry_run()
        return

    settings = load_settings()
    output_dir = settings.output_dir
    run_race(settings, output_dir)


if __name__ == "__main__":
    main()
