"""Week 7 · Module 4 — Agent Loops and When Not to Use Them
Recipe adaptation agent: find recipe, scale, swap allergen.

Tools
-----
1. search_recipes       — retrieve recipe card from the index for a named recipe
2. scale_recipe         — scale ingredient quantities to a target serving count
3. get_allergen_profile — return allergen profile + safe substitute for ONE named
                          allergen in a recipe (single job, enum parameters)

Budgets enforced every iteration (all four, not just one)
----------------------------------------------------------
MAX_ITERATIONS  : 8   (hard cap on tool-call laps)
MAX_TOKENS      : 6000 (cumulative input+output tokens, all laps summed)
MAX_COST_USD    : 0.05 (cumulative cost per request in USD)
MAX_WALL_SECS   : 30.0 (wall-clock seconds per request)

Usage
-----
    python week7_agent.py
    python week7_agent.py --demo
    python week7_agent.py --budget-demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as gtypes

from rag_app.config import load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("week7_agent")

# Pricing — Gemini 2.5 Flash (USD per 1M tokens)
PRICE_INPUT_PER_M = 0.30
PRICE_OUTPUT_PER_M = 2.50

# Budget constants
MAX_ITERATIONS: int = 8
MAX_TOKENS: int = 6000
MAX_COST_USD: float = 0.05
MAX_WALL_SECS: float = 30.0

# Serving size reference
BASE_SERVINGS: dict[str, int] = {
    "R001": 8, "R002": 6, "R003": 12, "R004": 8, "R005": 6, "R006": 4,
}

RECIPE_INGREDIENTS_RAW: dict[str, dict[str, float]] = {
    "R001": {"strong bread flour": 1000, "water": 720, "ripe sourdough starter": 200, "fine sea salt": 20},
    "R002": {"bread flour": 500, "water": 400, "olive oil": 25, "fine sea salt": 7},
    "R003": {"jalapeño peppers": 800, "water": 200, "fine sea salt": 30, "garlic cloves": 40},
    "R004": {"napa cabbage": 1200, "coarse sea salt": 60, "water": 900, "gochugaru": 45, "garlic": 30, "ginger": 15, "soy sauce": 20},
    "R005": {"bread flour": 750, "water": 540, "ripe sourdough starter": 150, "fine sea salt": 15},
    "R006": {"caputo 00 flour": 600, "water": 420, "sourdough starter": 120, "fine sea salt": 12},
}

ALLERGEN_DB: dict[str, dict[str, dict]] = {
    "R001": {
        "gluten": {
            "profile": "Strong bread flour contains wheat gluten. Essential for dough structure.",
            "substitutes": {
                "gluten_free": {"name": "buckwheat flour", "note": "GF; add 1 tsp xanthan gum per 200g.", "itself_allergen": False},
                "nut_free":    {"name": "rice flour blend", "note": "70% white rice + 30% tapioca. GF and nut-free.", "itself_allergen": False},
                "vegan":       {"name": "oat flour", "note": "Vegan but may contain gluten cross-contamination.", "itself_allergen": True, "cascade_allergen": "gluten"},
                "any":         {"name": "almond flour", "note": "1:1 swap; denser crumb. Contains tree nuts.", "itself_allergen": True, "cascade_allergen": "nuts"},
            },
        },
        "nuts": {
            "profile": "No nuts in original recipe.",
            "substitutes": {"any": {"name": "sunflower seed flour", "note": "Nut-free, similar fat.", "itself_allergen": False}},
        },
    },
    "R002": {
        "gluten": {
            "profile": "Bread flour provides gluten for chew in focaccia.",
            "substitutes": {
                "gluten_free": {"name": "chickpea flour", "note": "High protein; 1:1 by weight.", "itself_allergen": False},
                "nut_free":    {"name": "rice flour", "note": "Lighter crumb; add psyllium husk.", "itself_allergen": False},
                "any":         {"name": "almond flour", "note": "Dense crumb. Contains tree nuts.", "itself_allergen": True, "cascade_allergen": "nuts"},
                "vegan":       {"name": "oat flour", "note": "Vegan; verify certified GF if needed.", "itself_allergen": True, "cascade_allergen": "gluten"},
            },
        },
    },
    "R003": {
        "gluten": {"profile": "No gluten in original recipe.", "substitutes": {"any": {"name": "N/A — already gluten-free", "note": "", "itself_allergen": False}}},
    },
    "R004": {
        "soy": {
            "profile": "Soy sauce contains soy protein — major allergen.",
            "substitutes": {
                "any":         {"name": "coconut aminos", "note": "Soy-free; slightly sweeter. Check coconut allergy.", "itself_allergen": True, "cascade_allergen": "coconut"},
                "gluten_free": {"name": "fish sauce", "note": "GF (check label); rich umami. Use 75% quantity.", "itself_allergen": False},
                "vegan":       {"name": "chickpea miso thinned with water", "note": "Soy-free and vegan.", "itself_allergen": False},
                "nut_free":    {"name": "sunflower liquid aminos", "note": "Nut-free and soy-free. Increase by 20%.", "itself_allergen": False},
            },
        },
        "gluten": {"profile": "Kimchi is naturally gluten-free; soy sauce may have trace gluten.", "substitutes": {"any": {"name": "certified GF tamari", "note": "Drop-in replacement.", "itself_allergen": False}}},
    },
    "R005": {
        "gluten": {
            "profile": "Bread flour is structural for baguette crust; high hydration dough.",
            "substitutes": {
                "gluten_free": {"name": "oat flour (certified GF)", "note": "Denser; needs psyllium + xanthan.", "itself_allergen": True, "cascade_allergen": "gluten"},
                "nut_free":    {"name": "brown rice flour + tapioca blend", "note": "Nut-free; GF. 3:1 ratio.", "itself_allergen": False},
                "any":         {"name": "almond flour", "note": "Dense; no crust formation. Contains nuts.", "itself_allergen": True, "cascade_allergen": "nuts"},
                "vegan":       {"name": "sorghum flour", "note": "Vegan, mild flavour; blend with tapioca.", "itself_allergen": False},
            },
        },
    },
    "R006": {
        "gluten": {
            "profile": "Caputo 00 flour: fine particle, moderate protein for pizza extensibility.",
            "substitutes": {
                "gluten_free": {"name": "cassava flour", "note": "Best GF pizza sub; 1:1 swap.", "itself_allergen": False},
                "any":         {"name": "almond flour", "note": "Dense; no stretchability. Contains tree nuts.", "itself_allergen": True, "cascade_allergen": "nuts"},
                "nut_free":    {"name": "cassava flour", "note": "Nut-free and GF. Recommended.", "itself_allergen": False},
                "vegan":       {"name": "sorghum flour blend", "note": "Vegan; mix with potato starch 2:1.", "itself_allergen": False},
            },
        },
    },
}


class BudgetExceeded(Exception):
    def __init__(self, budget_name: str, value: Any, limit: Any):
        self.budget_name = budget_name
        self.value = value
        self.limit = limit
        super().__init__(
            f"[BUDGET] {budget_name} exceeded — value={value}, limit={limit}"
        )


@dataclass
class RunMetrics:
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    passed: bool = False
    answer: str = ""
    tool_calls_log: list[dict] = field(default_factory=list)
    budget_hit: str | None = None
    latency_secs: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_usage(self, usage) -> None:
        if usage is None:
            return
        it = getattr(usage, "prompt_token_count", 0) or 0
        ot = getattr(usage, "candidates_token_count", 0) or 0
        self.input_tokens += it
        self.output_tokens += ot


def _tool_search_recipes(recipe_name: str, query: str) -> str:
    name_map = {
        "sourdough": "R001", "country sourdough": "R001", "r001": "R001",
        "focaccia": "R002", "r002": "R002",
        "hot sauce": "R003", "jalapeno hot sauce": "R003", "jalapeño hot sauce": "R003", "r003": "R003",
        "kimchi": "R004", "baechu kimchi": "R004", "r004": "R004",
        "baguette": "R005", "baguettes": "R005", "r005": "R005",
        "pizza": "R006", "pizza dough": "R006", "r006": "R006",
    }
    recipe_id = name_map.get(recipe_name.lower().strip())
    if not recipe_id:
        return json.dumps({"error": f"Recipe '{recipe_name}' not found."})

    id_to_file = {
        "R001": "R001_country_sourdough.md", "R002": "R002_focaccia.md",
        "R003": "R003_hot_sauce.md", "R004": "R004_kimchi.md",
        "R005": "R005_baguettes.md", "R006": "R006_pizza_dough.md",
    }
    cards_dir = Path(__file__).parent / "data" / "cards"
    card_text = ""
    card_path = cards_dir / id_to_file.get(recipe_id, "")
    if card_path.exists():
        card_text = card_path.read_text(encoding="utf-8")

    allergens = list(ALLERGEN_DB.get(recipe_id, {}).keys())
    return json.dumps({
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "base_servings": BASE_SERVINGS[recipe_id],
        "ingredients_g": RECIPE_INGREDIENTS_RAW.get(recipe_id, {}),
        "known_allergens": allergens,
        "card_excerpt": card_text[:600],
    })


def _tool_scale_recipe(recipe_id: str, target_servings: int) -> str:
    rid = recipe_id.upper().strip()
    base = BASE_SERVINGS.get(rid)
    if base is None:
        return json.dumps({"error": f"Unknown recipe_id '{recipe_id}'"})
    if target_servings <= 0:
        return json.dumps({"error": "target_servings must be positive"})
    factor = target_servings / base
    raw = RECIPE_INGREDIENTS_RAW.get(rid, {})
    scaled = {ing: round(g * factor, 1) for ing, g in raw.items()}
    return json.dumps({
        "recipe_id": rid,
        "base_servings": base,
        "target_servings": target_servings,
        "scale_factor": round(factor, 4),
        "scaled_ingredients_g": scaled,
    })


def _tool_get_allergen_profile(recipe_id: str, allergen: str, diet_constraint: str) -> str:
    rid = recipe_id.upper().strip()
    allergen_key = allergen.lower().strip()
    diet_key = diet_constraint.lower().strip()
    recipe_db = ALLERGEN_DB.get(rid)
    if recipe_db is None:
        return json.dumps({"error": f"No allergen data for '{recipe_id}'"})
    allergen_data = recipe_db.get(allergen_key)
    if allergen_data is None:
        return json.dumps({"recipe_id": rid, "allergen": allergen_key,
                           "profile": f"No '{allergen_key}' allergen in this recipe.",
                           "substitute_name": None, "substitute_itself_allergen": False, "cascade_allergen": None})
    substitutes = allergen_data.get("substitutes", {})
    sub = substitutes.get(diet_key) or substitutes.get("any") or {}
    return json.dumps({
        "recipe_id": rid,
        "allergen": allergen_key,
        "profile": allergen_data["profile"],
        "substitute_name": sub.get("name", "No substitute found"),
        "substitute_note": sub.get("note", ""),
        "substitute_itself_allergen": sub.get("itself_allergen", False),
        "cascade_allergen": sub.get("cascade_allergen"),
    })


TOOL_SEARCH_RECIPES = gtypes.FunctionDeclaration(
    name="search_recipes",
    description=(
        "Retrieve the full recipe card for a named recipe from the index, "
        "including ingredients, method steps, base serving count, and known allergens. "
        "Use this tool to fetch recipe context before scaling or allergen work."
    ),
    parameters=gtypes.Schema(
        type=gtypes.Type.OBJECT,
        properties={
            "recipe_name": gtypes.Schema(type=gtypes.Type.STRING,
                description="The recipe common name (e.g. 'sourdough', 'focaccia', 'kimchi')."),
            "query": gtypes.Schema(type=gtypes.Type.STRING,
                description="The user adaptation question for relevance context."),
        },
        required=["recipe_name", "query"],
    ),
)

TOOL_SCALE_RECIPE = gtypes.FunctionDeclaration(
    name="scale_recipe",
    description=(
        "Compute scaled ingredient quantities (in grams) for a recipe when the serving count changes. "
        "Input the recipe ID and desired target serving count; returns each ingredient new weight. "
        "Does not modify ingredient identity or suggest substitutions — arithmetic scaling only."
    ),
    parameters=gtypes.Schema(
        type=gtypes.Type.OBJECT,
        properties={
            "recipe_id": gtypes.Schema(type=gtypes.Type.STRING,
                description="Recipe identifier (R001–R006)."),
            "target_servings": gtypes.Schema(type=gtypes.Type.INTEGER,
                description="Target number of servings. Must be a positive integer."),
        },
        required=["recipe_id", "target_servings"],
    ),
)

# TOOL 3 — get_allergen_profile
# One job: allergen identity lookup + substitute recommendation.
# Enum parameters for allergen and diet_constraint.
# No overlap: search_recipes retrieves text; scale_recipe does arithmetic; this does allergen identity.
TOOL_GET_ALLERGEN_PROFILE = gtypes.FunctionDeclaration(
    name="get_allergen_profile",
    description=(
        "Return the allergen profile for a single named ingredient in a recipe "
        "and list one safe substitute that satisfies the given dietary constraint. "
        "This tool performs allergen identity lookup and substitution recommendation only — "
        "it does not retrieve recipe text and does not compute ingredient quantities."
    ),
    parameters=gtypes.Schema(
        type=gtypes.Type.OBJECT,
        properties={
            "recipe_id": gtypes.Schema(type=gtypes.Type.STRING,
                description="Recipe identifier (R001–R006)."),
            "allergen": gtypes.Schema(
                type=gtypes.Type.STRING,
                description="The allergen to look up in the recipe.",
                enum=["gluten", "soy", "nuts", "dairy", "egg", "shellfish", "sesame"],
            ),
            "diet_constraint": gtypes.Schema(
                type=gtypes.Type.STRING,
                description="Household dietary constraint the substitute must satisfy.",
                enum=["vegan", "vegetarian", "gluten_free", "nut_free", "any"],
            ),
        },
        required=["recipe_id", "allergen", "diet_constraint"],
    ),
)

ALL_TOOLS = gtypes.Tool(function_declarations=[
    TOOL_SEARCH_RECIPES,
    TOOL_SCALE_RECIPE,
    TOOL_GET_ALLERGEN_PROFILE,
])


def dispatch_tool(name: str, args: dict) -> str:
    if name == "search_recipes":
        return _tool_search_recipes(args["recipe_name"], args["query"])
    elif name == "scale_recipe":
        return _tool_scale_recipe(args["recipe_id"], int(args["target_servings"]))
    elif name == "get_allergen_profile":
        return _tool_get_allergen_profile(args["recipe_id"], args["allergen"], args["diet_constraint"])
    return json.dumps({"error": f"Unknown tool: {name}"})


SYSTEM_INSTRUCTION = (
    "You are a recipe adaptation assistant. When asked to adapt a recipe:\n"
    "1. Call search_recipes to retrieve the recipe card.\n"
    "2. Call scale_recipe if a different serving count is requested.\n"
    "3. Call get_allergen_profile if an allergen swap or dietary substitution is needed.\n"
    "4. If get_allergen_profile returns substitute_itself_allergen=true, "
    "call get_allergen_profile again with the cascade_allergen to find a truly safe substitute.\n"
    "5. Return a final adapted recipe with scaled quantities and the confirmed safe substitute.\n"
    "Always prefer tool results over your own knowledge. Be concise."
)


def run_agent(
    request: str,
    api_key: str,
    model: str,
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = MAX_TOKENS,
    max_cost_usd: float = MAX_COST_USD,
    max_wall_secs: float = MAX_WALL_SECS,
    verbose: bool = True,
) -> RunMetrics:
    client = genai.Client(api_key=api_key)
    metrics = RunMetrics()
    start_wall = time.monotonic()

    messages: list[gtypes.Content] = [
        gtypes.Content(role="user", parts=[gtypes.Part(text=request)])
    ]

    if verbose:
        log.info("=== AGENT START ===")
        log.info("Request: %s", request)
        log.info("Budgets: max_iter=%d  max_tok=%d  max_cost=$%.4f  max_wall=%.1fs",
                 max_iterations, max_tokens, max_cost_usd, max_wall_secs)

    try:
        while True:
            elapsed = time.monotonic() - start_wall

            # ── Check ALL FOUR budgets before every model call ─────────────
            if metrics.iterations >= max_iterations:
                raise BudgetExceeded("max_iterations", metrics.iterations, max_iterations)
            if metrics.total_tokens >= max_tokens:
                raise BudgetExceeded("max_tokens", metrics.total_tokens, max_tokens)
            if metrics.cost_usd >= max_cost_usd:
                raise BudgetExceeded("max_cost_usd", metrics.cost_usd, max_cost_usd)
            if elapsed >= max_wall_secs:
                raise BudgetExceeded("max_wall_secs", elapsed, max_wall_secs)

            if verbose:
                log.info("[lap %d] model call — tok=%d cost=$%.5f elapsed=%.1fs",
                         metrics.iterations + 1, metrics.total_tokens, metrics.cost_usd, elapsed)

            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=gtypes.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[ALL_TOOLS],
                    temperature=0,
                    max_output_tokens=800,
                ),
            )

            metrics.iterations += 1
            metrics.add_usage(response.usage_metadata)
            # Recompute cost after accumulating tokens
            metrics.cost_usd = (
                metrics.input_tokens * PRICE_INPUT_PER_M +
                metrics.output_tokens * PRICE_OUTPUT_PER_M
            ) / 1_000_000

            messages.append(response.candidates[0].content)

            tool_calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if part.function_call
            ]

            if not tool_calls:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        metrics.answer = part.text.strip()
                        break
                if verbose:
                    log.info("[lap %d] Final answer (first 200 chars): %s",
                             metrics.iterations, metrics.answer[:200])
                break

            tool_response_parts = []
            for fc in tool_calls:
                tool_args = dict(fc.args) if fc.args else {}
                if verbose:
                    log.info("[lap %d] tool=%s args=%s", metrics.iterations, fc.name, json.dumps(tool_args))
                result = dispatch_tool(fc.name, tool_args)
                metrics.tool_calls_log.append({
                    "lap": metrics.iterations, "tool": fc.name, "args": tool_args
                })
                if verbose:
                    log.info("[lap %d] result=%s", metrics.iterations, result[:180])
                tool_response_parts.append(
                    gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                )
            messages.append(gtypes.Content(role="user", parts=tool_response_parts))

    except BudgetExceeded as exc:
        metrics.budget_hit = exc.budget_name
        log.warning("=== BUDGET TERMINATION ===")
        log.warning(str(exc))
        log.warning("Iterations completed : %d / %d", metrics.iterations, max_iterations)
        log.warning("Total tokens         : %d / %d", metrics.total_tokens, max_tokens)
        log.warning("Total cost           : $%.5f / $%.4f", metrics.cost_usd, max_cost_usd)
        log.warning("Wall-clock elapsed   : %.2fs / %.1fs", time.monotonic() - start_wall, max_wall_secs)
        log.warning("Tool calls completed : %s", metrics.tool_calls_log)
        log.warning("=== TERMINATING CLEANLY — no more model calls will be made ===")
        metrics.answer = f"[Budget exceeded: {exc.budget_name}]"

    metrics.latency_secs = time.monotonic() - start_wall
    if verbose:
        log.info("=== AGENT DONE === iter=%d tok=%d cost=$%.5f latency=%.2fs budget=%s",
                 metrics.iterations, metrics.total_tokens, metrics.cost_usd,
                 metrics.latency_secs, metrics.budget_hit)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 Recipe Adaptation Agent")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--budget-demo", action="store_true")
    parser.add_argument("request", nargs="?", default=None)
    args = parser.parse_args()

    settings = load_settings()

    if args.budget_demo:
        print("\n" + "=" * 60)
        print("BUDGET TERMINATION DEMO  (max_iterations=2)")
        print("=" * 60)
        req = (
            "Adapt the sourdough recipe for a nut-free household. "
            "The substitute almond flour contains nuts — find a nut-free alternative. "
            "Scale to 4 servings."
        )
        m = run_agent(req, settings.api_key, settings.llm_model,
                      max_iterations=2, verbose=True)
        print(f"\n>>> Budget fired: {m.budget_hit}")
        print(f"    Iterations={m.iterations}  Tokens={m.total_tokens}  Cost=${m.cost_usd:.5f}")
        return

    req = args.request or (
        "Adapt the kimchi recipe for a soy-free vegan household and scale to 4 servings."
        if args.demo else
        "Adapt the focaccia recipe to be gluten-free for 8 servings."
    )
    print("\n" + "=" * 60)
    print("RECIPE ADAPTATION AGENT")
    print("=" * 60)
    m = run_agent(req, settings.api_key, settings.llm_model, verbose=True)
    print("\n" + "─" * 60)
    print("FINAL ANSWER:")
    print("─" * 60)
    print(m.answer)
    print("─" * 60)
    print(f"iter={m.iterations} tok={m.total_tokens} cost=${m.cost_usd:.5f} latency={m.latency_secs:.2f}s")
    if m.budget_hit:
        print(f"Budget fired: {m.budget_hit}")


if __name__ == "__main__":
    main()
