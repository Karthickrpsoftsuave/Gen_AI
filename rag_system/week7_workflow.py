"""Week 7 Fixed Workflow (no loop).

Implements identical recipe adaptation as week7_agent.py but as hard-coded
three steps with NO loop and NO branching on what the tools return.

Steps (always in this order, regardless of input):
  Step 1: search_recipes  -> recipe card context        (no LLM)
  Step 2: scale_recipe    -> scaled quantities           (no LLM)
  Step 3: get_allergen_profile -> substitute             (no LLM, one call only)
  Final : single LLM call synthesising all three results

Same tools, same model, same output contract as the agent. No loop inside.

Usage:
    python week7_workflow.py
    python week7_workflow.py --demo
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from google import genai
from google.genai import types as gtypes

from rag_app.config import load_settings
from week7_agent import (
    PRICE_INPUT_PER_M,
    PRICE_OUTPUT_PER_M,
    RunMetrics,
    _tool_get_allergen_profile,
    _tool_scale_recipe,
    _tool_search_recipes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("week7_workflow")


@dataclass
class WorkflowRequest:
    """Structured parameters the workflow needs — caller resolves these from the request."""
    recipe_name: str
    recipe_id: str
    query: str
    target_servings: int
    allergen: str
    diet_constraint: str


WORKFLOW_SYSTEM = (
    "You are a recipe adaptation assistant. "
    "You have been given the results of three tool calls: "
    "the recipe card, the scaled quantities, and the allergen substitute. "
    "Write a concise adapted recipe incorporating all three results. "
    "Include: recipe name, target servings, scaled ingredient list in grams, "
    "the allergen swap with the safe substitute, and the method steps. "
    "Do not add information not present in the tool results."
)


def run_workflow(
    req: WorkflowRequest,
    api_key: str,
    model: str,
    verbose: bool = True,
) -> RunMetrics:
    """Execute the fixed three-step workflow — no loop, no branching.

    The workflow always makes exactly three tool calls (steps 1-3) then
    one LLM synthesis call. It cannot handle cascade allergen cases where
    the substitute returned by step 3 is itself an allergen; it will
    return that unsafe substitute and pass will depend on whether the
    evaluator accepts it.
    """
    client = genai.Client(api_key=api_key)
    metrics = RunMetrics()
    start_wall = time.monotonic()

    if verbose:
        log.info("=== WORKFLOW START ===")
        log.info("recipe=%s scale_to=%d allergen=%s diet=%s",
                 req.recipe_name, req.target_servings, req.allergen, req.diet_constraint)

    # Step 1: Retrieve recipe card (deterministic lookup, no LLM)
    if verbose:
        log.info("[step 1] search_recipes -> %s", req.recipe_name)
    step1 = _tool_search_recipes(req.recipe_name, req.query)
    if verbose:
        log.info("[step 1] result preview: %s", step1[:160])

    # Step 2: Scale quantities (deterministic arithmetic, no LLM)
    if verbose:
        log.info("[step 2] scale_recipe -> %s to %d servings", req.recipe_id, req.target_servings)
    step2 = _tool_scale_recipe(req.recipe_id, req.target_servings)
    if verbose:
        log.info("[step 2] result preview: %s", step2[:160])

    # Step 3: Allergen lookup (deterministic lookup, no LLM)
    # LIMITATION: always exactly one call. substitute_itself_allergen is NOT checked.
    # Cascade cases (e.g. almond->nuts, coconut aminos->coconut) will silently pass
    # an unsafe substitute to the synthesis call.
    if verbose:
        log.info("[step 3] get_allergen_profile -> %s / %s / %s",
                 req.recipe_id, req.allergen, req.diet_constraint)
    step3 = _tool_get_allergen_profile(req.recipe_id, req.allergen, req.diet_constraint)
    if verbose:
        log.info("[step 3] result preview: %s", step3[:160])

    # Final: single LLM call that synthesises the three results
    prompt = "\n\n".join([
        "Recipe context (from search_recipes):\n" + step1,
        "Scaled quantities (from scale_recipe):\n" + step2,
        "Allergen substitute (from get_allergen_profile):\n" + step3,
        "Write the adapted recipe using ONLY the information above.",
    ])

    if verbose:
        log.info("[final] LLM synthesis call")

    response = client.models.generate_content(
        model=model,
        contents=[gtypes.Content(role="user", parts=[gtypes.Part(text=prompt)])],
        config=gtypes.GenerateContentConfig(
            system_instruction=WORKFLOW_SYSTEM,
            temperature=0,
            max_output_tokens=800,
        ),
    )

    metrics.iterations = 1  # one model call (synthesis)
    metrics.add_usage(response.usage_metadata)
    metrics.cost_usd = (
        metrics.input_tokens * PRICE_INPUT_PER_M +
        metrics.output_tokens * PRICE_OUTPUT_PER_M
    ) / 1_000_000
    metrics.tool_calls_log = [
        {"step": 1, "tool": "search_recipes",       "args": {"recipe_name": req.recipe_name}},
        {"step": 2, "tool": "scale_recipe",          "args": {"recipe_id": req.recipe_id, "target_servings": req.target_servings}},
        {"step": 3, "tool": "get_allergen_profile",  "args": {"recipe_id": req.recipe_id, "allergen": req.allergen, "diet_constraint": req.diet_constraint}},
    ]
    metrics.answer = response.text.strip() if response.text else "[no answer]"
    metrics.latency_secs = time.monotonic() - start_wall

    if verbose:
        log.info("=== WORKFLOW DONE === tok=%d cost=%.5f latency=%.2fs",
                 metrics.total_tokens, metrics.cost_usd, metrics.latency_secs)
        log.info("Answer (first 200): %s", metrics.answer[:200])

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 Fixed Workflow (no loop)")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    settings = load_settings()

    if args.demo:
        req = WorkflowRequest(
            recipe_name="kimchi",
            recipe_id="R004",
            query="Adapt kimchi for a soy-free vegan household",
            target_servings=4,
            allergen="soy",
            diet_constraint="vegan",
        )
    else:
        req = WorkflowRequest(
            recipe_name="focaccia",
            recipe_id="R002",
            query="Make focaccia gluten-free for 8 servings",
            target_servings=8,
            allergen="gluten",
            diet_constraint="gluten_free",
        )

    print("\n" + "=" * 60)
    print("FIXED WORKFLOW (no loop)")
    print("=" * 60)
    m = run_workflow(req, settings.api_key, settings.llm_model, verbose=True)
    print("\n" + "-" * 60)
    print("FINAL ANSWER:")
    print("-" * 60)
    print(m.answer)
    print("-" * 60)
    print(f"tok={m.total_tokens} cost=${m.cost_usd:.5f} latency={m.latency_secs:.2f}s")


if __name__ == "__main__":
    main()
