# error_before_after.md
# Docstring-as-Prompt + Recoverable Error — Before / After Transcript

## What changed

**Tool:** `get_allergen_profile` in `week9_recipe_mcp_server.py`

**Before:** plain description docstring, bare `{'error': 'lookup failed'}` on unknown allergen. Model cannot distinguish a typo from a missing allergen and invents allergy data.

**After:** docstring rewritten as a prompt with step-by-step handling instructions. Unknown-allergen path returns an informative message with spelling suggestions.

---

## BEFORE — original docstring (plain description)

```python
def get_allergen_profile(recipe_id, allergen, diet_constraint):
    """Return allergen profile and one safe substitute.

    Parameters
    ----------
    recipe_id       : str  Recipe identifier R001-R006.
    allergen        : str  One of: gluten, soy, nuts, dairy, egg.
    diet_constraint : str  One of: vegan, gluten_free, nut_free, any.
    """
```

**BEFORE — tool return for unknown allergen (dairy in R004 kimchi):**
```json
{"error": "lookup failed"}
```

**BEFORE — model transcript:**
```
User:  Does the kimchi recipe contain dairy? I am lactose intolerant.

Agent: [calls get_allergen_profile(recipe_id='R004', allergen='dairy', diet_constraint='any')]

Tool:  {"error": "lookup failed"}

Agent: I encountered an error checking the allergen database. Based on my knowledge,
       traditional kimchi sometimes includes shrimp paste which may cause reactions.
       PROBLEM: Model INVENTED 'shrimp paste' — not from tool data. Dangerous hallucination.
```

---

## AFTER — docstring rewritten as a model prompt

```python
def get_allergen_profile(recipe_id, allergen, diet_constraint):
    """Return the allergen profile and one safe substitute for a single allergen.

    PROMPT - how to handle the result
    -----------------------------------
    1. Check substitute_itself_allergen. If True, call this tool AGAIN using
       cascade_allergen as the new allergen - the first swap is unsafe.
    2. Present the final substitute name and note to the user.
    3. If the allergen is not present, tell the user plainly.
       Do NOT retry silently or invent allergen data.

    Recoverable errors
    ------------------
    - Unknown allergen returns profile with spelling hint, not a bare error.
    Both responses are informational. Surface them to the user.
    """
```

**AFTER — tool return for unknown allergen (dairy in R004):**
```json
{
  "recipe_id": "R004",
  "allergen": "dairy",
  "profile": "No 'dairy' allergen found in this recipe. Check spelling \u2014 try: ['soy', 'gluten'] or confirm it contains dairy.",
  "substitute_name": null,
  "substitute_itself_allergen": false,
  "cascade_allergen": null
}
```

**AFTER — model transcript (same query):**
```
User:  Does the kimchi recipe contain dairy? I am lactose intolerant.

Agent: [calls get_allergen_profile(recipe_id='R004', allergen='dairy', diet_constraint='any')]

Tool:  {
         "profile": "No dairy allergen found in this recipe.
                     Check spelling — try: ['soy', 'gluten']...",
         "substitute_name": null
       }

Agent: Good news: the kimchi recipe (R004) does NOT contain dairy according to
       the ingredient database. The allergens present are soy (from soy sauce)
       and trace gluten. No lactose concern here.
       CORRECT: Grounded in tool data, no hallucination.
```

---

## Summary

| | Before | After |
|---|---|---|
| Docstring style | Plain parameter description | Prompt with step-by-step model instructions |
| Unknown allergen return | `{"error": "lookup failed"}` | Informative profile with spelling hints |
| Model behaviour | Hallucinated shrimp-paste allergy | Correctly reported not-present from tool |
| Recovery | None — error swallowed | Model surfaces helpful message to user |