# AgentCard — Recipe Orchestrator

_(Bonus challenge: A2A protocol advertisement)_

```json
{
  "name": "recipe-orchestrator",
  "version": "1.0.0",
  "description": "Manager agent that decomposes cooking questions into culinary-feasibility and allergen-safety sub-tasks, delegates to specialist workers, and synthesises a single user-facing answer.",
  "skills": [
    {
      "id": "substitution_check",
      "name": "Ingredient Substitution Check",
      "description": "Evaluates whether a proposed ingredient swap is culinarily acceptable (function, method, flavour)."
    },
    {
      "id": "allergen_check",
      "name": "Allergen & Nutrition Check",
      "description": "Identifies new allergens introduced by a substitution and flags nutrition concerns."
    },
    {
      "id": "quantity_scaling",
      "name": "Recipe Quantity Scaling",
      "description": "Scales recipe ingredient quantities while preserving baker percentages and fermentation ratios."
    }
  ],
  "inputModes": ["text/plain", "application/json"],
  "outputModes": ["text/plain", "text/markdown"],
  "auth": {
    "type": "api_key",
    "header": "X-API-Key"
  },
  "url": "https://recipe-orchestrator.example.com/a2a",
  "provider": { "name": "Kitchen Squad" }
}
```

## A2A Task Lifecycle — Failed Case Mapping

**Case:** S01 (allergen worker HTTP 500 injected)

| A2A State | This case |
|-----------|----------|
| `submitted` | Orchestrator receives the substitution request |
| `working` | Substitution worker runs (succeeds); allergen worker invoked |
| `working → failed` | Allergen worker returns HTTP 500 |
| **Decision point** | Should this end `failed` or `input-required`? |

### State decision

This case **should have ended `input-required`**, not `failed`.

**Reason:** The orchestrator can answer the culinary part (sub-worker succeeded) but cannot safely answer the allergen part without the user's own allergy list. Asking the user to supply their allergy constraints is a valid and safe pause — it is not a terminal failure. A `failed` state throws away the sub-worker's good output and gives the user nothing.

### What A2A buys over a plain REST call

1. **Structured lifecycle** — A2A's `input-required` state lets the orchestrator pause mid-task and request the user's allergy list without losing the sub-worker's output; a plain REST call would either return a 500 or a silent partial answer with no mechanism to resume.
2. **Interoperability** — Any A2A-compliant client (not just this codebase) can dispatch tasks to the recipe orchestrator, inspect its published skills via the AgentCard, and route allergen sub-tasks to a certified food-safety agent — none of that is possible with a bespoke REST endpoint.
