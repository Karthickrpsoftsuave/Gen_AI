"""Week 10 · Module 5 — Multi-Agent Orchestrator
=================================================
Architecture:
  ORCHESTRATOR (manager) — decomposes the cooking question
    → SUBSTITUTION WORKER  — culinary feasibility / method compatibility only
    → ALLERGEN WORKER      — allergen flags + nutrition only

Each worker is a focused single-shot LLM call with a NARROW system prompt.
The orchestrator synthesises both worker outputs into a final answer.

Hand-off token accounting
-------------------------
Every inter-agent message is logged with token counts so the context
re-send multiplier can be computed precisely in evaluate_week10.py.

Failure injection
-----------------
Pass  inject_allergen_failure=True  (or set env ALLERGEN_WORKER_FAIL=1)
to make the allergen worker return a 500-like error on one call.
The orchestrator must degrade gracefully — partial answer, no hallucination.

Usage (standalone test)
-----------------------
  python week10_orchestrator.py
  python week10_orchestrator.py --inject-failure
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

load_dotenv(Path(__file__).parent / ".env")

T = TypeVar("T")


def _retry_generate(fn: Callable[[], T], max_retries: int = 6, base_wait: float = 10.0) -> T:
    """Retry fn() on transient 503/429/500 with exponential back-off.
    Catches any exception whose status_code or string repr indicates transient error.
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

# ── Pricing (gemini-2.5-flash) ────────────────────────────────────────────────
PRICE_INPUT_PER_M  = 0.30   # USD per 1 M input tokens
PRICE_OUTPUT_PER_M = 2.50   # USD per 1 M output tokens

# ── Worker system prompts (NARROW — each worker has fewer tools/context) ───────

_SUBSTITUTION_SYSTEM = (
    "You are a culinary substitution specialist. "
    "Your ONLY job is to evaluate whether a proposed ingredient substitution "
    "is culinarily acceptable given the recipe context provided. "
    "Focus on: ingredient function, method compatibility, and flavour direction. "
    "Do NOT comment on allergens or nutrition — that is a separate specialist's job. "
    "Be concise. If the information is not in the provided context, say so clearly. "
    "Format: one short paragraph, no more than 80 words."
)

_ALLERGEN_SYSTEM = (
    "You are a food-safety and allergen specialist. "
    "Your ONLY job is to identify allergens introduced by the proposed substitution "
    "and flag any nutrition concerns (sodium, gluten, shellfish, nuts, dairy, soy, eggs). "
    "Do NOT comment on culinary merit or method — that is handled elsewhere. "
    "Be precise. If no allergens are introduced, say 'No new allergens'. "
    "Format: one short paragraph listing allergens, or 'No new allergens'. Max 60 words."
)

_ORCHESTRATOR_SYSTEM = (
    "You are a recipe adaptation manager. "
    "You receive outputs from two specialist workers and synthesise them into "
    "a single, user-facing recipe substitution answer. "
    "Structure your final answer as: "
    "**Ingredient change:** [original -> substitute] "
    "**Method adjustment:** [changes or 'No adjustment required'] "
    "**Allergen note:** [from allergen worker — do NOT invent allergen claims if worker failed] "
    "**Serves:** [echo serving count if stated] "
    "IMPORTANT: if the allergen worker reported a failure, say "
    "'Allergen data unavailable — please verify independently' — never invent allergen claims. "
    "If required information is not in the recipe context, respond with exactly: "
    "I couldn't find that in the recipe documents."
)


# ── Token / timing dataclasses ────────────────────────────────────────────────

@dataclass
class HandOffRecord:
    """One logged inter-agent message with token counts."""
    from_agent: str
    to_agent: str
    message_tokens: int          # tokens in the message being handed off
    label: str = ""              # descriptive label for the hand-off


@dataclass
class WorkerResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_secs: float = 0.0
    error: Optional[str] = None  # set if worker returned a simulated 500


@dataclass
class OrchestratorResult:
    final_answer: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost_usd: float = 0.0
    latency_secs: float = 0.0
    handoffs: list[HandOffRecord] = field(default_factory=list)
    sub_worker: Optional[WorkerResult] = None
    allergen_worker: Optional[WorkerResult] = None
    allergen_failed: bool = False


# ── Token counting helper ─────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (good enough for accounting)."""
    return max(1, len(text) // 4)


# ── Individual worker calls ───────────────────────────────────────────────────

def _call_substitution_worker(
    client: genai.Client,
    model: str,
    recipe_context: str,
    question: str,
) -> WorkerResult:
    """Narrow substitution worker — culinary feasibility only."""
    user_message = (
        f"Recipe context:\n{recipe_context}\n\n"
        f"Substitution question: {question}"
    )
    t0 = time.monotonic()
    try:
        response = _retry_generate(lambda: client.models.generate_content(
            model=model,
            contents=user_message,
            config=gtypes.GenerateContentConfig(
                system_instruction=_SUBSTITUTION_SYSTEM,
                temperature=0,
                max_output_tokens=250,
            ),
        ))
        latency = time.monotonic() - t0
        usage = response.usage_metadata
        in_tok  = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        return WorkerResult(
            text=response.text.strip() if response.text else "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_secs=latency,
        )
    except Exception as exc:
        return WorkerResult(
            text="",
            latency_secs=time.monotonic() - t0,
            error=str(exc),
        )


def _call_allergen_worker(
    client: genai.Client,
    model: str,
    recipe_context: str,
    question: str,
    inject_failure: bool = False,
) -> WorkerResult:
    """Narrow allergen/nutrition worker.

    If inject_failure=True, simulates a 500 error WITHOUT calling the LLM,
    so we can record what the orchestrator does when this worker fails.
    """
    if inject_failure:
        # Simulated 500 — log what happens but do NOT call the API
        return WorkerResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_secs=0.0,
            error="HTTP 500 Internal Server Error (injected for failure test)",
        )

    user_message = (
        f"Recipe context:\n{recipe_context}\n\n"
        f"Substitution question (allergen focus only): {question}"
    )
    t0 = time.monotonic()
    try:
        response = _retry_generate(lambda: client.models.generate_content(
            model=model,
            contents=user_message,
            config=gtypes.GenerateContentConfig(
                system_instruction=_ALLERGEN_SYSTEM,
                temperature=0,
                max_output_tokens=150,
            ),
        ))
        latency = time.monotonic() - t0
        usage = response.usage_metadata
        in_tok  = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        return WorkerResult(
            text=response.text.strip() if response.text else "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_secs=latency,
        )
    except Exception as exc:
        return WorkerResult(
            text="",
            latency_secs=time.monotonic() - t0,
            error=str(exc),
        )


# ── Orchestrator (manager) ────────────────────────────────────────────────────

def run_orchestrator(
    api_key: str,
    model: str,
    recipe_context: str,
    question: str,
    inject_allergen_failure: bool = False,
) -> OrchestratorResult:
    """
    Full orchestrator run:
      1. Dispatch substitution worker
      2. Dispatch allergen worker (possibly failing)
      3. Synthesise via orchestrator LLM
    Returns OrchestratorResult with full token accounting and hand-off log.
    """
    wall_start = time.monotonic()
    client = genai.Client(api_key=api_key)
    handoffs: list[HandOffRecord] = []
    total_in = total_out = 0

    # ── Hand-off 0: orchestrator -> substitution worker (context resend) ────
    sub_msg_tokens = _approx_tokens(recipe_context) + _approx_tokens(question)
    handoffs.append(HandOffRecord(
        from_agent="orchestrator",
        to_agent="substitution_worker",
        message_tokens=sub_msg_tokens,
        label="orchestrator->substitution_worker resend",
    ))

    sub_result = _call_substitution_worker(client, model, recipe_context, question)
    total_in  += sub_result.input_tokens
    total_out += sub_result.output_tokens

    # ── Hand-off 1: orchestrator -> allergen worker (context resend) ────────
    allergen_msg_tokens = _approx_tokens(recipe_context) + _approx_tokens(question)
    handoffs.append(HandOffRecord(
        from_agent="orchestrator",
        to_agent="allergen_worker",
        message_tokens=allergen_msg_tokens,
        label="orchestrator->allergen_worker resend",
    ))

    allergen_result = _call_allergen_worker(
        client, model, recipe_context, question,
        inject_failure=inject_allergen_failure,
    )
    total_in  += allergen_result.input_tokens
    total_out += allergen_result.output_tokens

    allergen_failed = allergen_result.error is not None

    # ── Hand-off 2: workers -> orchestrator (synthesis) ─────────────────────
    sub_out_tokens   = _approx_tokens(sub_result.text)
    alg_out_tokens   = _approx_tokens(allergen_result.text) if not allergen_failed else 0
    synthesis_in_tokens = sub_msg_tokens + sub_out_tokens + allergen_msg_tokens + alg_out_tokens
    handoffs.append(HandOffRecord(
        from_agent="substitution_worker+allergen_worker",
        to_agent="orchestrator",
        message_tokens=synthesis_in_tokens,
        label="workers->orchestrator synthesis resend",
    ))

    # ── Build synthesis prompt ──────────────────────────────────────────────
    allergen_section = (
        f"ALLERGEN WORKER OUTPUT:\n{allergen_result.text}"
        if not allergen_failed
        else "ALLERGEN WORKER OUTPUT:\n[FAILED — HTTP 500 — data unavailable]"
    )

    synthesis_prompt = (
        f"Original question: {question}\n\n"
        f"SUBSTITUTION WORKER OUTPUT:\n{sub_result.text}\n\n"
        f"{allergen_section}\n\n"
        f"Recipe context (for serving count):\n{recipe_context[:600]}"
    )

    t0 = time.monotonic()
    try:
        synth_response = _retry_generate(lambda: client.models.generate_content(
            model=model,
            contents=synthesis_prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=_ORCHESTRATOR_SYSTEM,
                temperature=0,
                max_output_tokens=400,
            ),
        ))
        usage = synth_response.usage_metadata
        s_in  = getattr(usage, "prompt_token_count", 0) or 0
        s_out = getattr(usage, "candidates_token_count", 0) or 0
        total_in  += s_in
        total_out += s_out
        final_answer = synth_response.text.strip() if synth_response.text else ""
    except Exception as exc:
        final_answer = "Orchestrator synthesis failed: " + str(exc)

    total_cost = (total_in * PRICE_INPUT_PER_M + total_out * PRICE_OUTPUT_PER_M) / 1_000_000

    return OrchestratorResult(
        final_answer=final_answer,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        cost_usd=total_cost,
        latency_secs=time.monotonic() - wall_start,
        handoffs=handoffs,
        sub_worker=sub_result,
        allergen_worker=allergen_result,
        allergen_failed=allergen_failed,
    )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-failure", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise SystemExit("Missing GEMINI_API_KEY in .env")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Minimal demo context
    ctx = (
        "Recipe: No-Knead Fermented Focaccia\n"
        "Ingredients: bread flour 500g, water 400g, olive oil 50g, fine sea salt 10g\n"
        "Method: Mix flour, water, salt, oil. Ferment 18h at room temp. Dimple and bake 220 degrees C.\n"
        "Serves: 8"
    )
    q = "Can I replace the bread flour in the focaccia with almond flour?"

    result = run_orchestrator(api_key, model, ctx, q,
                              inject_allergen_failure=args.inject_failure)

    print("=" * 60)
    print("ORCHESTRATOR DEMO")
    print("=" * 60)
    print(f"Question: {q}")
    print(f"\nFINAL ANSWER:\n{result.final_answer}")
    print(f"\nAllergen worker failed: {result.allergen_failed}")
    print(f"Total tokens: {result.total_input_tokens + result.total_output_tokens}")
    print(f"Cost: ${result.cost_usd:.6f}")
    print(f"Latency: {result.latency_secs:.2f}s")
    print("\nHand-offs:")
    for h in result.handoffs:
        print(f"  [{h.from_agent} -> {h.to_agent}]  {h.message_tokens} tokens  ({h.label})")
