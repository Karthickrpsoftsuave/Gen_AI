"""Week 9 - MCP Server 2: Ingredient Database
==============================================
Exposes allergen flags and per-100g nutrition for individual ingredients.
This is the SECOND server bolted on via config only - zero agent code changes.

Tools exposed
-------------
lookup_ingredient   - nutrition + allergen flags for an ingredient by name
list_ingredients    - list all ingredient names in the database
search_ingredients  - fuzzy-search ingredients by partial name

Architecture note
-----------------
The AI (LLM) runs on the HOST side (week9_mcp_agent.py).
This server never calls an LLM - it only answers JSON-RPC tool calls.
"""

from __future__ import annotations
import json
from fastmcp import FastMCP

# Ingredient database: name -> {allergens, nutrition per 100g}
INGREDIENT_DB = {
    "strong bread flour": {
        "allergens": ["gluten", "wheat"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 341, "protein_g": 13.2, "fat_g": 1.7, "carbs_g": 70.2, "fibre_g": 2.7, "sodium_mg": 2},
        "notes": "High-protein wheat flour; primary gluten source in bread.",
    },
    "bread flour": {
        "allergens": ["gluten", "wheat"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 341, "protein_g": 12.0, "fat_g": 1.7, "carbs_g": 70.5, "fibre_g": 2.5, "sodium_mg": 2},
        "notes": "Standard high-protein flour; used in focaccia and baguettes.",
    },
    "caputo 00 flour": {
        "allergens": ["gluten", "wheat"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 339, "protein_g": 11.5, "fat_g": 1.3, "carbs_g": 71.0, "fibre_g": 2.2, "sodium_mg": 1},
        "notes": "Finely milled Italian wheat flour; low ash for pizza extensibility.",
    },
    "ripe sourdough starter": {
        "allergens": ["gluten", "wheat"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 148, "protein_g": 4.5, "fat_g": 0.7, "carbs_g": 30.1, "fibre_g": 1.2, "sodium_mg": 310},
        "notes": "Live culture of flour and water; fermented. Contains residual gluten.",
    },
    "sourdough starter": {
        "allergens": ["gluten", "wheat"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 148, "protein_g": 4.5, "fat_g": 0.7, "carbs_g": 30.1, "fibre_g": 1.2, "sodium_mg": 310},
        "notes": "Alias for ripe sourdough starter.",
    },
    "water": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0, "fibre_g": 0, "sodium_mg": 0},
        "notes": "No allergens. Filtered tap or bottled.",
    },
    "fine sea salt": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0, "fibre_g": 0, "sodium_mg": 38758},
        "notes": "No allergens. High sodium; use in recipe amounts only.",
    },
    "coarse sea salt": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0, "fibre_g": 0, "sodium_mg": 38758},
        "notes": "No allergens.",
    },
    "olive oil": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 884, "protein_g": 0, "fat_g": 100.0, "carbs_g": 0, "fibre_g": 0, "sodium_mg": 2},
        "notes": "No common allergens. Cold-pressed extra virgin preferred for focaccia.",
    },
    "jalapeño peppers": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 29, "protein_g": 0.9, "fat_g": 0.4, "carbs_g": 6.5, "fibre_g": 2.5, "sodium_mg": 3},
        "notes": "No allergens. Contains capsaicin - handle with gloves.",
    },
    "garlic cloves": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 149, "protein_g": 6.4, "fat_g": 0.5, "carbs_g": 33.1, "fibre_g": 2.1, "sodium_mg": 17},
        "notes": "No common allergens.",
    },
    "garlic": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 149, "protein_g": 6.4, "fat_g": 0.5, "carbs_g": 33.1, "fibre_g": 2.1, "sodium_mg": 17},
        "notes": "No common allergens.",
    },
    "napa cabbage": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 13, "protein_g": 1.2, "fat_g": 0.2, "carbs_g": 2.2, "fibre_g": 1.0, "sodium_mg": 9},
        "notes": "No allergens. Rinse well before use.",
    },
    "gochugaru": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 282, "protein_g": 12.0, "fat_g": 9.0, "carbs_g": 50.0, "fibre_g": 27.0, "sodium_mg": 30},
        "notes": "Korean red pepper flakes. No common allergens; verify processing facility for cross-contact.",
    },
    "ginger": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 80, "protein_g": 1.8, "fat_g": 0.8, "carbs_g": 18.0, "fibre_g": 2.0, "sodium_mg": 13},
        "notes": "No common allergens.",
    },
    "soy sauce": {
        "allergens": ["soy", "gluten", "wheat"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 60, "protein_g": 5.6, "fat_g": 0.1, "carbs_g": 9.5, "fibre_g": 0, "sodium_mg": 5720},
        "notes": "Contains soy (major allergen) and wheat gluten. Use tamari as GF substitute.",
    },
    "buckwheat flour": {
        "allergens": [],
        "may_contain": ["gluten"],
        "nutrition_per_100g": {"energy_kcal": 335, "protein_g": 13.3, "fat_g": 3.4, "carbs_g": 71.5, "fibre_g": 10.0, "sodium_mg": 1},
        "notes": "Naturally gluten-free but often processed alongside wheat. Verify GF certification.",
    },
    "almond flour": {
        "allergens": ["tree nuts", "nuts"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 571, "protein_g": 21.4, "fat_g": 50.0, "carbs_g": 17.9, "fibre_g": 10.7, "sodium_mg": 1},
        "notes": "Contains tree nuts (almonds) - major allergen. Dense, high fat.",
    },
    "rice flour": {
        "allergens": [],
        "may_contain": ["gluten"],
        "nutrition_per_100g": {"energy_kcal": 366, "protein_g": 6.0, "fat_g": 0.8, "carbs_g": 80.0, "fibre_g": 2.4, "sodium_mg": 0},
        "notes": "Naturally gluten-free. Check facility cross-contact.",
    },
    "chickpea flour": {
        "allergens": ["legumes"],
        "may_contain": ["gluten"],
        "nutrition_per_100g": {"energy_kcal": 387, "protein_g": 22.4, "fat_g": 6.7, "carbs_g": 57.8, "fibre_g": 10.7, "sodium_mg": 64},
        "notes": "High protein GF flour. Legume allergy risk; check cross-contact for GF.",
    },
    "oat flour": {
        "allergens": ["gluten"],
        "may_contain": ["wheat"],
        "nutrition_per_100g": {"energy_kcal": 404, "protein_g": 17.0, "fat_g": 8.7, "carbs_g": 62.0, "fibre_g": 8.5, "sodium_mg": 2},
        "notes": "Contains avenin (oat gluten). Use certified GF oat flour to avoid wheat cross-contact.",
    },
    "cassava flour": {
        "allergens": [],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 333, "protein_g": 0.3, "fat_g": 0.5, "carbs_g": 82.4, "fibre_g": 1.8, "sodium_mg": 14},
        "notes": "No common allergens. Best 1:1 GF pizza substitute.",
    },
    "coconut aminos": {
        "allergens": ["coconut"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 42, "protein_g": 0.5, "fat_g": 0, "carbs_g": 10.3, "fibre_g": 0, "sodium_mg": 1230},
        "notes": "Soy-free soy sauce substitute. Contains coconut - check tree-nut allergy policy.",
    },
    "creme fraiche": {
        "allergens": ["dairy", "milk"],
        "may_contain": [],
        "nutrition_per_100g": {"energy_kcal": 292, "protein_g": 2.2, "fat_g": 30.0, "carbs_g": 2.8, "fibre_g": 0, "sodium_mg": 29},
        "notes": "French cultured cream. Major allergen: dairy/milk.",
    },
}

# Build a lower-case lookup for fuzzy matching
_DB_LOWER = {k.lower(): k for k in INGREDIENT_DB}

mcp = FastMCP(
    name="ingredient-database-server",
    instructions=(
        "Ingredient database for the recipe assistant. "
        "Look up allergen flags and per-100g nutrition for any ingredient. "
        "Tools: lookup_ingredient, list_ingredients, search_ingredients."
    ),
)


@mcp.tool()
def lookup_ingredient(ingredient_name: str) -> str:
    """Look up allergen flags and per-100g nutrition for a named ingredient.

    Call this tool when the user asks about allergens or nutrition for a
    specific ingredient, or when you need to verify whether a substitute
    ingredient is safe for a given dietary constraint.

    Recoverable errors
    ------------------
    If the ingredient is not found, the tool returns a helpful suggestion
    with close matches rather than a bare error. Do NOT hallucinate flags
    for unknown ingredients - surface the 'not_found' response to the user
    and suggest the closest known name.

    Parameters
    ----------
    ingredient_name : str
        The ingredient to look up, e.g. 'soy sauce', 'almond flour'.
        Case-insensitive. Spell it as it appears in the recipe.
    """
    key = ingredient_name.lower().strip()
    canonical = _DB_LOWER.get(key)

    if canonical is None:
        # Suggest close matches (simple substring search)
        close = [k for k in INGREDIENT_DB if key in k or k in key]
        if not close:
            # wider search: any word overlap
            words = set(key.split())
            close = [k for k in INGREDIENT_DB if words & set(k.split())]
        suggestion = close[0] if close else None
        return json.dumps({
            "found": False,
            "queried": ingredient_name,
            "message": (
                f"No ingredient matched '{ingredient_name}': "
                + (f"try '{suggestion}'" if suggestion else "no close match found — check spelling")
            ),
            "close_matches": close[:5],
        })

    data = INGREDIENT_DB[canonical]
    return json.dumps({
        "found": True,
        "ingredient": canonical,
        "allergens": data["allergens"],
        "may_contain": data["may_contain"],
        "nutrition_per_100g": data["nutrition_per_100g"],
        "notes": data["notes"],
    })


@mcp.tool()
def list_ingredients() -> str:
    """Return the full list of ingredient names in the database.

    Use this tool to discover available ingredients before calling
    lookup_ingredient, or to show the user which ingredients are indexed.
    """
    return json.dumps({"ingredient_count": len(INGREDIENT_DB), "ingredients": sorted(INGREDIENT_DB.keys())})


@mcp.tool()
def search_ingredients(partial_name: str) -> str:
    """Search ingredients by partial or approximate name.

    Use this before lookup_ingredient when you are unsure of the exact name.
    Returns all ingredients whose name contains the search term.

    Parameters
    ----------
    partial_name : str
        Partial ingredient name to search for, e.g. 'flour', 'sauce'.
    """
    term = partial_name.lower().strip()
    matches = [k for k in INGREDIENT_DB if term in k.lower()]
    results = []
    for name in matches:
        d = INGREDIENT_DB[name]
        results.append({"ingredient": name, "allergens": d["allergens"], "notes": d["notes"]})
    if not results:
        return json.dumps({"found": False, "message": f"No ingredients match '{partial_name}'.", "results": []})
    return json.dumps({"found": True, "count": len(results), "results": results})


if __name__ == "__main__":
    mcp.run()
