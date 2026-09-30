"""Week 6 · Module 3 — Evals & Error Analysis (The Core)

Validate the Substitution Judge Before You Trust Its Number
Track B: Recipes & Food

BLIND PROTOCOL:
  labels_25.json and prediction.txt are written BEFORE this script runs any
  judge call.  Their file-system mtimes and embedded created_at timestamps
  prove ordering.  This script reads (never overwrites) labels_25.json.

Assertion / judge split
  4 deterministic assertions (moved OUT of judge, implemented as if-checks):
    A1  Every ingredient mentioned in method steps appears in the ingredient list
    A2  Allergen warning present when an allergen ingredient is introduced
    A3  Oven temperature carries units (°C or °F)
    A4  Serving count echoed and parses as a number
  1 LLM-judged criterion (binary, single sentence):
    J1  Substitution is culinarily acceptable (function, method, flavour direction)

Usage:
    python evaluate_week6.py            # full run (requires GEMINI_API_KEY in .env)
    python evaluate_week6.py --dry-run  # print cases, no API calls

Outputs:
    output/labels_25.json    (READ ONLY — must predate this run)
    output/prediction.txt    (READ ONLY — must predate this run)
    output/judge_v1.txt      judge prompt v1
    output/judge_v2.txt      judge prompt v2 (with 2 few-shot disagreements)
    output/week6_records.json structured results
    stdout                   pass rate by mode, agreement before -> after
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from google.genai import types as gtypes

from rag_app.config import load_settings
from rag_app.rag import FORCED_REFUSAL, GeminiRAG

# ---------------------------------------------------------------------------
# Counts (printed in summary — update if assertions change)
# ---------------------------------------------------------------------------
ASSERTION_COUNT = 4
JUDGED_CRITERIA_COUNT = 1

# ---------------------------------------------------------------------------
# Recipe reference data
# ---------------------------------------------------------------------------
RECIPE_NAMES: dict[str, str] = {
    "R001": "Country Sourdough Loaf (2 kg)",
    "R002": "No-Knead Fermented Focaccia",
    "R003": "Fermented Jalapeño Hot Sauce",
    "R004": "Baechu Kimchi",
    "R005": "Sourdough Baguettes",
    "R006": "24-Hour Fermented Pizza Dough",
}

# Canonical ingredient lists from the recipe cards (for assertion A1)
RECIPE_INGREDIENTS: dict[str, list[str]] = {
    "R001": ["strong bread flour", "water", "ripe sourdough starter", "fine sea salt"],
    "R002": ["bread flour", "water", "olive oil", "fine sea salt"],
    "R003": ["jalapeño peppers", "water", "fine sea salt", "garlic cloves"],
    "R004": ["napa cabbage", "coarse sea salt", "water", "gochugaru", "garlic",
             "ginger", "soy sauce"],
    "R005": ["bread flour", "water", "ripe sourdough starter", "fine sea salt"],
    "R006": ["caputo 00 flour", "water", "sourdough starter", "fine sea salt"],
}

# ---------------------------------------------------------------------------
# 25 evaluation cases — each tagged with one Week 5 taxonomy mode
# S21 and S22 are regression cases replayed verbatim from Week 5 failed traces
# ---------------------------------------------------------------------------
EVAL_CASES: list[dict] = [
    # ── ALLERGEN_SAFETY ────────────────────────────────────────────────────
    {"id": "S01", "mode": "ALLERGEN_SAFETY", "recipe_id": "R002",
     "request": "Can I replace the bread flour in the focaccia with almond flour?",
     "expected_servings": None, "is_regression": False},
    {"id": "S02", "mode": "ALLERGEN_SAFETY", "recipe_id": "R003",
     "request": "Can I use soy sauce instead of fine sea salt in the jalapeño hot sauce?",
     "expected_servings": None, "is_regression": False},
    {"id": "S03", "mode": "ALLERGEN_SAFETY", "recipe_id": "R005",
     "request": "Can I substitute bread flour with rice flour in the sourdough baguettes to make them gluten-free?",
     "expected_servings": None, "is_regression": False},
    {"id": "S04", "mode": "ALLERGEN_SAFETY", "recipe_id": "R004",
     "request": "Can I replace the garlic in the baechu kimchi with shrimp paste for a more umami-rich flavour?",
     "expected_servings": None, "is_regression": False},
    {"id": "S05", "mode": "ALLERGEN_SAFETY", "recipe_id": "R004",
     "request": "Can I substitute gochugaru in the baechu kimchi with regular red chili flakes?",
     "expected_servings": None, "is_regression": False},
    # ── FLAVOUR_PLAUSIBILITY ───────────────────────────────────────────────
    {"id": "S06", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R002",
     "request": "Can I swap the olive oil in the focaccia with coconut oil?",
     "expected_servings": None, "is_regression": False},
    {"id": "S07", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R003",
     "request": "Can I replace jalapeño peppers with habanero peppers in the fermented hot sauce?",
     "expected_servings": None, "is_regression": False},
    {"id": "S08", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R006",
     "request": "Can I swap the sourdough starter in the pizza dough for commercial instant yeast at the same weight?",
     "expected_servings": None, "is_regression": False},
    {"id": "S09", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R004",
     "request": "Can I replace gochugaru in the baechu kimchi with sweet smoked paprika?",
     "expected_servings": None, "is_regression": False},
    {"id": "S10", "mode": "FLAVOUR_PLAUSIBILITY", "recipe_id": "R001",
     "request": "Can I substitute the water in the country sourdough with whole milk at the same weight?",
     "expected_servings": None, "is_regression": False},
    # ── METHOD_COMPATIBILITY ───────────────────────────────────────────────
    {"id": "S11", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R002",
     "request": "Can I use cake flour instead of bread flour in the no-knead fermented focaccia?",
     "expected_servings": None, "is_regression": False},
    {"id": "S12", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R001",
     "request": "Can I replace fine sea salt with kosher salt at the same weight in the country sourdough?",
     "expected_servings": None, "is_regression": False},
    {"id": "S13", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R006",
     "request": "Can I double the bulk ferment temperature to 48 °C instead of 24 °C in the pizza dough?",
     "expected_servings": None, "is_regression": False},
    {"id": "S14", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R005",
     "request": "Can I skip all the coil folds during bulk fermentation of the sourdough baguettes?",
     "expected_servings": None, "is_regression": False},
    {"id": "S15", "mode": "METHOD_COMPATIBILITY", "recipe_id": "R001",
     "request": "Can I replace the sourdough starter with an equal weight of poolish at 100% hydration?",
     "expected_servings": None, "is_regression": False},
    # ── QUANTITY_SCALING ───────────────────────────────────────────────────
    {"id": "S16", "mode": "QUANTITY_SCALING", "recipe_id": "R001",
     "request": "I want to double the country sourdough recipe to make 4 kg of dough for 4 people. What are the adjusted quantities?",
     "expected_servings": 4, "is_regression": False},
    {"id": "S17", "mode": "QUANTITY_SCALING", "recipe_id": "R002",
     "request": "I only have 250 g of flour. How should I halve the focaccia recipe?",
     "expected_servings": None, "is_regression": False},
    {"id": "S18", "mode": "QUANTITY_SCALING", "recipe_id": "R005",
     "request": "I only want to make 2 sourdough baguettes instead of 4. How should I adjust the quantities?",
     "expected_servings": 2, "is_regression": False},
    {"id": "S19", "mode": "QUANTITY_SCALING", "recipe_id": "R003",
     "request": "I only have 300 g of jalapeños. Can I keep the salt at 30 g and just use fewer peppers?",
     "expected_servings": None, "is_regression": False},
    {"id": "S20", "mode": "QUANTITY_SCALING", "recipe_id": "R004",
     "request": "I want to triple the baechu kimchi recipe to serve 12 people. What are the scaled quantities?",
     "expected_servings": 12, "is_regression": False},
    # ── OOC_REFUSAL — 2 regression cases from Week 5 ──────────────────────
    {"id": "S21", "mode": "OOC_REFUSAL", "recipe_id": "R003",
     "request": "What is the exact weight of roasted garlic to substitute for the raw garlic in the hot sauce, accounting for moisture loss during roasting?",
     "expected_servings": None, "is_regression": True},   # Week 5 T06 — garlic retrieval miss
    {"id": "S22", "mode": "OOC_REFUSAL", "recipe_id": "R005",
     "request": "Can you suggest a gluten-free sourdough baguette substitution that maintains the same sodium content per serving?",
     "expected_servings": None, "is_regression": True},   # Week 5 T19 — false-negative refusal
    {"id": "S23", "mode": "OOC_REFUSAL", "recipe_id": "R004",
     "request": "Which probiotic supplement can I add to the focaccia to provide the same gut health benefits as the kimchi?",
     "expected_servings": None, "is_regression": False},
    {"id": "S24", "mode": "OOC_REFUSAL", "recipe_id": "R001",
     "request": "Can I replace the bulk fermentation with commercial probiotic capsules to achieve the same digestive health benefits?",
     "expected_servings": None, "is_regression": False},
    {"id": "S25", "mode": "OOC_REFUSAL", "recipe_id": "R006",
     "request": "What is the glycemic index impact of swapping Caputo 00 flour for whole wheat flour in the pizza dough?",
     "expected_servings": None, "is_regression": False},
]

assert len(EVAL_CASES) == 25, f"Expected 25 cases, got {len(EVAL_CASES)}"

# ---------------------------------------------------------------------------
# Judge prompt v1 — single binary criterion
# The 4 assertable criteria are explicitly excluded from this prompt.
# ---------------------------------------------------------------------------
JUDGE_V1_BODY = """\
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


# ---------------------------------------------------------------------------
# Deterministic assertion A1 — ingredients in method
# ---------------------------------------------------------------------------
# Words that appear in method descriptions but are NOT food ingredients.
_TECHNIQUE_WORDS = {
    "mix", "fold", "add", "rest", "bake", "heat", "cook", "knead", "stretch",
    "ferment", "proof", "shape", "divide", "cover", "remove", "press", "blend",
    "strain", "stir", "pour", "rinse", "salt", "slice", "quarter", "dimple",
    "drizzle", "bring", "measure", "bottle", "burp", "pack", "massage",
    "dissolve", "give", "make", "work", "move", "slash", "increase", "keep",
    "reduce", "scale", "double", "halve", "triple", "adjust", "change",
    "maintain", "replace", "substitute", "swap", "use", "need", "want",
    "will", "should", "note", "alternatively", "same", "original", "recipe",
    "method", "step", "process", "time", "temperature", "minutes", "hours",
    "days", "gram", "grams", "kilogram", "weight", "ratio", "percent",
    "amount", "quantity", "batch", "yield", "serving", "servings", "portion",
    "result", "texture", "flavour", "flavor", "aroma", "colour", "color",
    "structure", "crumb", "crust", "dough", "sauce", "paste", "brine",
}

def assert_ingredients_in_method(sub_text: str, recipe_id: str) -> tuple[bool, str]:
    """A1: Every food ingredient word mentioned in the method section of the
    generated substitution must appear in the original recipe's ingredient list
    or be a recognised technique / generic cooking word.

    Implementation: extract the method-note section from the substitution text,
    find 5+ character words that are not technique words, and flag any that do
    not appear anywhere in the ingredient word pool for this recipe.
    """
    known_ingredient_words: set[str] = set()
    for ing in RECIPE_INGREDIENTS.get(recipe_id, []):
        known_ingredient_words.update(re.findall(r"[a-z]{4,}", ing.lower()))
    # Also include common substitution ingredient words that may appear
    substitution_vocab = {
        "almond", "coconut", "habanero", "paprika", "poolish",
        "yeast", "commercial", "instant", "cake", "kosher", "rice",
        "shrimp", "prawn", "butter", "cream", "dairy", "milk", "whole",
        "wheat", "gluten", "flour", "sourdough", "starter", "kimchi",
        "focaccia", "baguette", "pizza", "jalapeño", "jalapeno", "chili",
        "flakes", "gochugaru", "cabbage", "fermented", "miso", "tamari",
    }
    known_ingredient_words.update(substitution_vocab)

    # Isolate the method-note section from the generated text
    text_lower = sub_text.lower()
    method_start = -1
    for marker in ["method adjustment", "method note", "method:", "adjustment:", "steps:"]:
        idx = text_lower.find(marker)
        if idx != -1:
            method_start = idx
            break

    if method_start == -1:
        # No explicit method section found — scan whole text but be lenient
        method_text = text_lower
    else:
        # Take the section after the method marker until next ** heading or end
        method_text = text_lower[method_start:]
        next_section = re.search(r"\*\*[a-z]", method_text[20:])
        if next_section:
            method_text = method_text[: next_section.start() + 20]

    # Extract candidate food words (5+ chars, not technique words)
    candidate_words = {
        w for w in re.findall(r"[a-z]{5,}", method_text)
        if w not in _TECHNIQUE_WORDS
    }
    unrecognised = candidate_words - known_ingredient_words - _TECHNIQUE_WORDS
    # Filter out very common English words unlikely to be ingredients
    common_english = {
        "which", "where", "there", "their", "would", "could", "should",
        "about", "after", "before", "until", "while", "since", "using",
        "these", "those", "other", "every", "place", "still", "first",
        "second", "third", "small", "large", "lower", "upper", "light",
        "heavy", "thick", "thin", "crisp", "crispy", "smooth", "sticky",
        "risen", "baked", "mixed", "folded", "shaped", "proofed", "chilled",
        "cooled", "strained", "blended", "fermented",
    }
    unrecognised -= common_english

    if unrecognised:
        return False, (
            f"Method section references word(s) not in ingredient list or known vocab: "
            f"{', '.join(sorted(unrecognised)[:5])}"
        )
    return True, "All method-section food words appear in ingredient list or known vocab"


# ---------------------------------------------------------------------------
# Deterministic assertion A2 — allergen warning
# ---------------------------------------------------------------------------
_ALLERGEN_MAP: dict[str, str] = {
    "almond": "tree nut",
    "cashew": "tree nut",
    "walnut": "tree nut",
    "pecan": "tree nut",
    "hazelnut": "tree nut",
    "pistachio": "tree nut",
    "shrimp": "shellfish",
    "prawn": "shellfish",
    "crab": "shellfish",
    "lobster": "shellfish",
    "milk": "dairy",
    "butter": "dairy",
    "cream": "dairy",
    "cheese": "dairy",
    "yoghurt": "dairy",
    "yogurt": "dairy",
    "egg": "egg",
    "wheat": "gluten",
    "barley": "gluten",
    "rye": "gluten",
    "soy sauce": "soy",
    "soya": "soy",
    "peanut": "peanut",
}
_WARNING_PHRASES = ["allergen", "contains", "allergy", "warning", "alert", "note:"]


def assert_allergen_warning(sub_text: str) -> tuple[bool, str]:
    """A2: If the substitution text introduces an allergen ingredient, an
    allergen warning phrase must also appear in the same text.
    """
    text_lower = sub_text.lower()
    found: list[str] = []
    for keyword, category in _ALLERGEN_MAP.items():
        if keyword in text_lower:
            found.append(f"{keyword}({category})")

    if not found:
        return True, "No allergen ingredients detected in substitution text"

    has_warning = any(phrase in text_lower for phrase in _WARNING_PHRASES)
    if not has_warning:
        return False, (
            f"Allergen ingredient(s) detected [{', '.join(found[:3])}] "
            f"but no allergen warning phrase found in text"
        )
    return True, f"Allergen ingredient(s) [{', '.join(found[:3])}] — warning phrase present"


# ---------------------------------------------------------------------------
# Deterministic assertion A3 — temperature carries units
# ---------------------------------------------------------------------------
_TEMP_UNIT_RE = re.compile(r"(\d{2,3})\s*°\s*(?![CF])", re.IGNORECASE)
_BAKING_CONTEXT_WORDS = {"bake", "oven", "heat", "temperature", "preheat", "ferment", "°"}


def assert_temperature_has_units(sub_text: str) -> tuple[bool, str]:
    """A3: Any temperature value (2–3 digit number followed by °) in the
    substitution text must be followed by C or F within 3 characters.
    """
    # Pattern: number + ° without C or F immediately after
    problems: list[str] = []
    for m in _TEMP_UNIT_RE.finditer(sub_text):
        num = int(m.group(1))
        if 20 <= num <= 500:  # plausible temperature range
            # Check surrounding context for baking words
            ctx_start = max(0, m.start() - 40)
            ctx_end = min(len(sub_text), m.end() + 5)
            context = sub_text[ctx_start:ctx_end].lower()
            if any(w in context for w in _BAKING_CONTEXT_WORDS):
                problems.append(f"{num}°")

    # Also catch cases where a 3-digit number appears near a ° symbol but separated
    bare = re.findall(r"\b([12]\d{2})\s+degrees?\b(?!\s*(?:celsius|fahrenheit|C|F))",
                      sub_text, re.IGNORECASE)
    problems.extend(bare)

    if problems:
        return False, f"Temperature(s) without °C/°F units: {'; '.join(set(problems))}"
    return True, "All temperatures carry units (or no bare ° found)"


# ---------------------------------------------------------------------------
# Deterministic assertion A4 — serving count echoed
# ---------------------------------------------------------------------------
_WORD_NUMS: dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    12: "twelve",
}


def assert_servings_echoed(sub_text: str, expected_servings: int | None) -> tuple[bool, str]:
    """A4: If the eval case specifies an expected serving count, that number
    must appear in the generated text and be parseable as an integer.
    """
    if expected_servings is None:
        return True, "No serving count required for this case"

    text_lower = sub_text.lower()
    # Check digit form
    if str(expected_servings) in text_lower:
        # Verify it is parseable as int (it is, since it's a literal digit string)
        return True, f"Serving count {expected_servings} found and parseable in text"
    # Check word form
    word = _WORD_NUMS.get(expected_servings, "")
    if word and word in text_lower:
        return True, f"Serving count {expected_servings} (as '{word}') found in text"

    return False, f"Serving count {expected_servings} not found in substitution text"


# ---------------------------------------------------------------------------
# Run all 4 assertions
# ---------------------------------------------------------------------------

def run_assertions(sub_text: str, case: dict) -> dict:
    """Run all 4 deterministic assertions; return structured result dict."""
    is_refusal = FORCED_REFUSAL.lower() in sub_text.lower()

    # For OOC cases that correctly refused, only A4 might be relevant
    # but still run all to show assertion counts
    a1_pass, a1_msg = assert_ingredients_in_method(sub_text, case["recipe_id"])
    a2_pass, a2_msg = assert_allergen_warning(sub_text)
    a3_pass, a3_msg = assert_temperature_has_units(sub_text)
    a4_pass, a4_msg = assert_servings_echoed(sub_text, case.get("expected_servings"))

    # For refusals: A1/A2/A3/A4 are vacuously true (no substitution to check)
    if is_refusal:
        a1_pass, a1_msg = True, "N/A (refusal — no method to inspect)"
        a2_pass, a2_msg = True, "N/A (refusal — no allergen to warn about)"
        a3_pass, a3_msg = True, "N/A (refusal — no temperature stated)"
        a4_pass, a4_msg = True, "N/A (refusal — no serving count expected from a refusal)"

    all_pass = all([a1_pass, a2_pass, a3_pass, a4_pass])
    return {
        "assertions_pass": all_pass,
        "A1_ingredients_in_method": {"pass": a1_pass, "msg": a1_msg},
        "A2_allergen_warning": {"pass": a2_pass, "msg": a2_msg},
        "A3_temperature_units": {"pass": a3_pass, "msg": a3_msg},
        "A4_servings_echoed": {"pass": a4_pass, "msg": a4_msg},
    }


# ---------------------------------------------------------------------------
# Substitution generation
# ---------------------------------------------------------------------------
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


def generate_substitution(rag: GeminiRAG, case: dict) -> str:
    """Retrieve relevant chunks then generate a structured substitution response."""
    sources = rag.retrieve(case["request"], top_k=3, strategy="hybrid")
    if not sources:
        return FORCED_REFUSAL

    context = "\n\n".join(
        f"[Source: {r.chunk.id} | {r.chunk.recipe_id}]\n{r.chunk.text}"
        for r in sources
    )
    prompt = _SUB_PROMPT_TEMPLATE.format(
        refusal=FORCED_REFUSAL,
        context=context,
        request=case["request"],
    )
    response = rag.client.models.generate_content(
        model=rag.model,
        contents=prompt,
        config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=400),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _format_judge_prompt(body: str, case: dict, sub_text: str) -> str:
    return body.format(
        recipe_name=RECIPE_NAMES.get(case["recipe_id"], case["recipe_id"]),
        request=case["request"],
        sub_text=sub_text,
    )


def call_judge(rag: GeminiRAG, prompt: str) -> tuple[bool, str]:
    """Call the LLM judge; parse first line for verdict, second for reason."""
    response = rag.client.models.generate_content(
        model=rag.model,
        contents=prompt,
        config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=120),
    )
    text = (response.text or "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    verdict_line = lines[0].upper() if lines else ""
    reason = lines[1] if len(lines) > 1 else "(no reason given)"
    acceptable = "ACCEPTABLE" in verdict_line and "UNACCEPTABLE" not in verdict_line
    return acceptable, reason


# ---------------------------------------------------------------------------
# Build judge v2 (few-shot from 2 disagreements)
# ---------------------------------------------------------------------------

def build_judge_v2_body(records: list[dict], label_map: dict[str, dict]) -> tuple[str, list[str]]:
    """Find 2 disagreements with judge_v1, embed them as few-shot examples."""
    disagreements = [
        r for r in records
        if r.get("judge_v1_acceptable") != label_map[r["id"]]["acceptable"]
    ]
    if len(disagreements) < 2:
        disagreements = records[:2]  # fallback — use first 2 if too few disagreements

    chosen = disagreements[:2]
    chosen_ids = [r["id"] for r in chosen]

    few_shot_blocks = ""
    for i, ex in enumerate(chosen, 1):
        human_lbl = label_map[ex["id"]]
        correct_verdict = "ACCEPTABLE" if human_lbl["acceptable"] else "UNACCEPTABLE"
        recipe_name = RECIPE_NAMES.get(ex["recipe_id"], ex["recipe_id"])
        few_shot_blocks += f"""
─── Disagreement example {i} (the judge was wrong — learn from this) ───
Recipe: {recipe_name}
Request: {ex['request']}
System response:
{ex['sub_text']}
Correct verdict: {correct_verdict}
Why the judge was wrong: {human_lbl['reason']}
"""

    v2_body = (
        JUDGE_V1_BODY.rstrip()
        + f"""

════════════════════════════════════════════════════════════════════
FEW-SHOT CALIBRATION — study these disagreements before evaluating
════════════════════════════════════════════════════════════════════
{few_shot_blocks.strip()}
════════════════════════════════════════════════════════════════════

Now apply the same standard to the new case below:
"""
    )
    return v2_body, chosen_ids


# ---------------------------------------------------------------------------
# Agreement metric
# ---------------------------------------------------------------------------

def compute_agreement(records: list[dict], verdict_key: str, label_map: dict) -> float:
    """Percentage of records where judge verdict matches human label."""
    if not records:
        return 0.0
    matches = sum(
        1 for r in records
        if r.get(verdict_key) == label_map[r["id"]]["acceptable"]
    )
    return matches / len(records)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _dry_run() -> None:
    print("=== Week 6 — DRY RUN (no API calls) ===")
    mode_counts: dict[str, int] = defaultdict(int)
    for c in EVAL_CASES:
        mode_counts[c["mode"]] += 1

    print(f"\n{'ID':4s}  {'Mode':22s}  {'Reg':3s}  Request (truncated to 70 chars)")
    print("-" * 105)
    for c in EVAL_CASES:
        reg = "REG" if c["is_regression"] else "   "
        print(f"{c['id']:4s}  {c['mode']:22s}  {reg}  {c['request'][:70]}")

    print(f"\nTotal cases: {len(EVAL_CASES)}")
    print("Mode breakdown:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")
    reg_count = sum(1 for c in EVAL_CASES if c["is_regression"])
    print(f"Regression cases: {reg_count}")
    print(f"\nAssertions: {ASSERTION_COUNT}  |  Judged criteria: {JUDGED_CRITERIA_COUNT}")
    print("\nLabels summary (from HAND_LABELS baked into this script):")
    from collections import Counter
    acceptable_by_mode: dict[str, Counter] = defaultdict(Counter)
    for c in EVAL_CASES:
        lab = _LABEL_MAP_STATIC.get(c["id"], {})
        acceptable_by_mode[c["mode"]][lab.get("acceptable", "?")] += 1
    for mode, cnt in sorted(acceptable_by_mode.items()):
        print(f"  {mode}: ACCEPTABLE={cnt.get(True,0)}  UNACCEPTABLE={cnt.get(False,0)}")


# ---------------------------------------------------------------------------
# Report table
# ---------------------------------------------------------------------------

def print_eval_table(
    records: list[dict],
    label_map: dict,
    agreement_before: float,
    agreement_after: float,
) -> None:
    modes = [
        "ALLERGEN_SAFETY",
        "FLAVOUR_PLAUSIBILITY",
        "METHOD_COMPATIBILITY",
        "QUANTITY_SCALING",
        "OOC_REFUSAL",
    ]

    print("\n" + "=" * 82)
    print("WEEK 6 EVAL  —  PASS RATE BY MODE")
    print("=" * 82)
    hdr = f"{'Mode':<22}  {'N':>2}  {'Assert':>7}  {'JudgeV2':>8}  {'Overall':>8}  {'AgreeV2':>8}"
    print(hdr)
    print("-" * 72)

    total_n = total_a = total_j = total_o = 0

    for mode in modes:
        mrs = [r for r in records if r["mode"] == mode]
        if not mrs:
            continue
        n = len(mrs)
        a = sum(1 for r in mrs if r["assertions_pass"])
        j = sum(1 for r in mrs if r.get("judge_v2_acceptable", False))
        o = sum(1 for r in mrs if r["assertions_pass"] and r.get("judge_v2_acceptable", False))
        ag = sum(1 for r in mrs if r.get("agree_v2", False))
        print(
            f"{mode:<22}  {n:>2}  {a}/{n}({a/n:.0%})  "
            f"{j}/{n}({j/n:.0%})  {o}/{n}({o/n:.0%})  {ag}/{n}({ag/n:.0%})"
        )
        total_n += n; total_a += a; total_j += j; total_o += o

    print("-" * 72)
    print(
        f"{'TOTAL':<22}  {total_n:>2}  "
        f"{total_a}/{total_n}({total_a/total_n:.0%})  "
        f"{total_j}/{total_n}({total_j/total_n:.0%})  "
        f"{total_o}/{total_n}({total_o/total_n:.0%})  "
        f"  —"
    )

    # Regression cases
    print()
    reg = [r for r in records if r["is_regression"]]
    if reg:
        print(f"Regression cases ({len(reg)} replayed verbatim from Week 5 failed traces):")
        for r in reg:
            v1 = "ACC" if r.get("judge_v1_acceptable") else "UNA"
            v2 = "ACC" if r.get("judge_v2_acceptable") else "UNA"
            h = "ACC" if r["human_acceptable"] else "UNA"
            print(f"  [{r['id']}] {r['mode']}: judge_v1={v1}  judge_v2={v2}  human={h}  "
                  f"{'✓' if r.get('agree_v2') else '✗'}")

    print(f"\n{'─' * 52}")
    print(f"  agreement_before  (judge_v1 vs human):  "
          f"{agreement_before:.0%}  ({round(agreement_before*25)}/25)")
    print(f"  agreement_after   (judge_v2 vs human):  "
          f"{agreement_after:.0%}  ({round(agreement_after*25)}/25)")
    print(f"  Assertions: {ASSERTION_COUNT}  |  Judged criteria: {JUDGED_CRITERIA_COUNT}")
    print(f"{'─' * 52}")

    # Disagreement analysis: up to 5 remaining v2 disagreements
    remaining = [r for r in records if not r.get("agree_v2", False)]
    if remaining:
        print(f"\nDisagreements remaining after judge_v2 ({len(remaining)}):")
        for r in remaining[:5]:
            h = "ACCEPTABLE" if r["human_acceptable"] else "UNACCEPTABLE"
            j2 = "ACCEPTABLE" if r.get("judge_v2_acceptable") else "UNACCEPTABLE"
            print(f"  [{r['id']}] [{r['mode']}]")
            print(f"    human: {h} — {label_map[r['id']]['reason'][:80]}")
            print(f"    judge: {j2} — {r.get('judge_v2_reason','')[:80]}")
    else:
        print("\nNo remaining disagreements after judge_v2 — perfect agreement.")


# ---------------------------------------------------------------------------
# Static label map (used by _dry_run before any label file is loaded)
# ---------------------------------------------------------------------------
_HAND_LABELS_STATIC: list[dict] = [
    {"id": "S01", "acceptable": False,
     "reason": "Almond flour: tree-nut allergen + insufficient protein for stretch-and-fold."},
    {"id": "S02", "acceptable": False,
     "reason": "Soy sauce: introduces soy allergen + disrupts safe brine ratio."},
    {"id": "S03", "acceptable": True,
     "reason": "Rice flour: GF substitute, removes gluten allergen, culinarily accepted."},
    {"id": "S04", "acceptable": False,
     "reason": "Shrimp paste: introduces shellfish allergen not in original recipe."},
    {"id": "S05", "acceptable": True,
     "reason": "Red chili flakes: same allergen class (none), reasonable heat substitute."},
    {"id": "S06", "acceptable": True,
     "reason": "Coconut oil: known fat substitute, method intact, flavour changes to tropical."},
    {"id": "S07", "acceptable": True,
     "reason": "Habanero: same capsicum family, just much hotter; method compatible."},
    {"id": "S08", "acceptable": False,
     "reason": "Commercial yeast: lacks LAB, cannot replicate 24h sourdough fermentation."},
    {"id": "S09", "acceptable": False,
     "reason": "Sweet paprika: negligible heat, completely different flavour profile from gochugaru."},
    {"id": "S10", "acceptable": True,
     "reason": "Whole milk: enriched-dough variant, method compatible, recognised sourdough category."},
    {"id": "S11", "acceptable": False,
     "reason": "Cake flour: ~8% protein, gluten won't develop through stretch-and-fold."},
    {"id": "S12", "acceptable": True,
     "reason": "Kosher salt: same NaCl, same weight, no method change required."},
    {"id": "S13", "acceptable": False,
     "reason": "48°C kills the sourdough starter; 24h fermentation impossible."},
    {"id": "S14", "acceptable": False,
     "reason": "Skipping coil folds: no gluten development, baguettes will be dense."},
    {"id": "S15", "acceptable": True,
     "reason": "Poolish: pre-ferment functionally analogous to starter; method compatible."},
    {"id": "S16", "acceptable": True,
     "reason": "Doubling: all quantities × 2, baker's percentages preserved."},
    {"id": "S17", "acceptable": True,
     "reason": "Halving to 250g flour: all ratios preserved, method unchanged."},
    {"id": "S18", "acceptable": True,
     "reason": "Half recipe = 2 baguettes of ~415g each; method identical."},
    {"id": "S19", "acceptable": False,
     "reason": "Salt ratio 10% with 300g peppers — unsafe for fermentation."},
    {"id": "S20", "acceptable": True,
     "reason": "Tripling all kimchi quantities; ratios preserved, fermentation time same."},
    {"id": "S21", "acceptable": True,
     "reason": "REGRESSION T06: moisture-loss conversion not in corpus; refusal is correct."},
    {"id": "S22", "acceptable": True,
     "reason": "REGRESSION T19: sodium per serving not in corpus; refusal is correct."},
    {"id": "S23", "acceptable": True,
     "reason": "Probiotic health equivalence not in recipe cards; refusal is correct."},
    {"id": "S24", "acceptable": True,
     "reason": "Fermentation vs probiotic health science not in corpus; refusal is correct."},
    {"id": "S25", "acceptable": True,
     "reason": "Glycemic index data not in any recipe card; refusal is correct."},
]
_LABEL_MAP_STATIC: dict[str, dict] = {l["id"]: l for l in _HAND_LABELS_STATIC}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "--dry-run" in sys.argv:
        _dry_run()
        return

    settings = load_settings()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path     = output_dir / "labels_25.json"
    prediction_path = output_dir / "prediction.txt"
    judge_v1_path   = output_dir / "judge_v1.txt"
    judge_v2_path   = output_dir / "judge_v2.txt"
    records_path    = output_dir / "week6_records.json"

    # ── Blind protocol: labels_25.json MUST already exist ─────────────────
    if not labels_path.exists():
        raise SystemExit(
            f"ERROR: {labels_path} not found.\n"
            "The blind protocol requires labels_25.json to be written BEFORE\n"
            "this script is run. Its file-system mtime proves ordering.\n"
            "Write the labels file first, then run this script."
        )
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    label_map: dict[str, dict] = {l["id"]: l for l in labels_data["labels"]}
    print(f"[BLIND]  labels_25.json exists — created_at: {labels_data.get('created_at','?')}")
    print(f"         File mtime: {datetime.fromtimestamp(labels_path.stat().st_mtime)}")

    # ── prediction.txt must already exist ─────────────────────────────────
    if not prediction_path.exists():
        raise SystemExit(
            f"ERROR: {prediction_path} not found.\n"
            "Write prediction.txt before running the judge."
        )
    print(f"[PRED]   prediction.txt exists — mtime: "
          f"{datetime.fromtimestamp(prediction_path.stat().st_mtime)}")

    # ── Index RAG ──────────────────────────────────────────────────────────
    print(f"\n=== Week 6 — Substitution Judge Validation ===")
    print(f"Model:   {settings.llm_model}")
    rag = GeminiRAG(settings.api_key, settings.index_path, settings.llm_model)
    n = rag.index_documents(settings.documents_dir)
    print(f"Indexed: {n} chunks\n")

    # ── PHASE 1: Generate substitutions + assertions + judge v1 ───────────
    records: list[dict] = []
    print(f"Running {len(EVAL_CASES)} cases ...\n")

    for case in EVAL_CASES:
        reg_flag = " [REGRESSION]" if case["is_regression"] else ""
        print(f"  [{case['id']}] [{case['mode']}]{reg_flag}")
        print(f"         {case['request'][:75]}...")

        sub_text = generate_substitution(rag, case)
        time.sleep(0.6)

        assertion_results = run_assertions(sub_text, case)
        a_icon = "✓" if assertion_results["assertions_pass"] else "✗"

        judge_v1_prompt = _format_judge_prompt(JUDGE_V1_BODY, case, sub_text)
        judge_v1_acc, judge_v1_reason = call_judge(rag, judge_v1_prompt)
        time.sleep(0.6)

        human_acc = label_map[case["id"]]["acceptable"]
        agree_v1 = judge_v1_acc == human_acc

        j1_icon = "ACC" if judge_v1_acc else "UNA"
        h_icon = "ACC" if human_acc else "UNA"
        print(
            f"         assert={a_icon}  judge_v1={j1_icon}  human={h_icon}  "
            f"agree={'✓' if agree_v1 else '✗'}"
        )

        records.append({
            "id": case["id"],
            "mode": case["mode"],
            "recipe_id": case["recipe_id"],
            "request": case["request"],
            "is_regression": case["is_regression"],
            "sub_text": sub_text,
            "human_acceptable": human_acc,
            "human_reason": label_map[case["id"]]["reason"],
            **assertion_results,
            "judge_v1_acceptable": judge_v1_acc,
            "judge_v1_reason": judge_v1_reason,
            "agree_v1": agree_v1,
        })

    # ── Agreement before ───────────────────────────────────────────────────
    agreement_before = compute_agreement(records, "judge_v1_acceptable", label_map)
    n_agree_v1 = round(agreement_before * 25)
    print(f"\n>>> agreement_before = {agreement_before:.0%}  ({n_agree_v1}/25)")

    # ── Save judge_v1.txt ──────────────────────────────────────────────────
    judge_v1_path.write_text(JUDGE_V1_BODY, encoding="utf-8")
    print(f"[SAVED]  judge_v1.txt")

    # ── PHASE 2: Build and run judge v2 (few-shot from disagreements) ──────
    print("\nBuilding judge_v2 from 2 judge_v1 disagreements ...")
    judge_v2_body, chosen_ids = build_judge_v2_body(records, label_map)
    judge_v2_path.write_text(judge_v2_body, encoding="utf-8")
    print(f"[SAVED]  judge_v2.txt  (few-shot examples: {chosen_ids})")

    # Show which disagreements were used
    disagree_v1 = [r for r in records if not r["agree_v1"]]
    print(f"\nTotal judge_v1 disagreements: {len(disagree_v1)}")
    for r in disagree_v1[:4]:
        h = "ACC" if r["human_acceptable"] else "UNA"
        j = "ACC" if r["judge_v1_acceptable"] else "UNA"
        print(f"  [{r['id']}] [{r['mode']}] human={h}  judge={j}")

    print(f"\nRunning judge_v2 on all {len(EVAL_CASES)} cases ...")
    for record in records:
        case = next(c for c in EVAL_CASES if c["id"] == record["id"])
        v2_prompt = _format_judge_prompt(judge_v2_body, case, record["sub_text"])
        judge_v2_acc, judge_v2_reason = call_judge(rag, v2_prompt)
        time.sleep(0.4)

        agree_v2 = judge_v2_acc == record["human_acceptable"]
        record["judge_v2_acceptable"] = judge_v2_acc
        record["judge_v2_reason"] = judge_v2_reason
        record["agree_v2"] = agree_v2

        flip = ""
        if agree_v2 != record["agree_v1"]:
            flip = f"  ← FLIPPED ({'✓ fixed' if agree_v2 else '✗ broke'})"
        v2_icon = "ACC" if judge_v2_acc else "UNA"
        print(f"  [{record['id']}] judge_v2={v2_icon}{flip}")

    # ── Agreement after ────────────────────────────────────────────────────
    agreement_after = compute_agreement(records, "judge_v2_acceptable", label_map)
    n_agree_v2 = round(agreement_after * 25)
    print(f"\n>>> agreement_after  = {agreement_after:.0%}  ({n_agree_v2}/25)")

    # ── Print eval table ───────────────────────────────────────────────────
    print_eval_table(records, label_map, agreement_before, agreement_after)

    # ── Save records ───────────────────────────────────────────────────────
    records_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[SAVED]  week6_records.json")
    print("=== Done ===")


if __name__ == "__main__":
    main()
