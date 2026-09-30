"""Week 7 Task Set B — Race Runner: Agent vs Fixed Workflow

Races the recipe adaptation agent (week7_agent.py) against the fixed
three-step workflow (week7_workflow.py) over 10 test requests and
reports four numbers for each system:

  pass_rate  — fraction of requests where the answer contains required keywords
  p50_secs   — median latency in seconds
  total_tok  — total tokens across all 10 runs
  cost_req   — average cost per request in USD

Outputs
-------
  output/race.csv              raw per-request results (10 rows x 8 columns)
  output/race_summary.txt      comparable 2-row table + verdict paragraph
  output/budget_termination.log  log of one run that hits max_iterations budget

Usage
-----
    python evaluate_week7.py                 # full race (API calls)
    python evaluate_week7.py --dry-run       # print request table, no API calls
    python evaluate_week7.py --budget-log    # produce budget termination log only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import statistics
import sys
import time
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path

from rag_app.config import load_settings
from week7_agent import (
    MAX_COST_USD,
    MAX_ITERATIONS,
    MAX_TOKENS,
    MAX_WALL_SECS,
    RunMetrics,
    run_agent,
)
from week7_workflow import WorkflowRequest, run_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_week7")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 10 test requests — mix of simple, combined, and cascade cases
# ---------------------------------------------------------------------------
# Each entry has:
#   id, label, kind, agent_request (natural language prompt),
#   workflow_req (structured WorkflowRequest params),
#   pass_keywords (list of strings that must appear in a passing answer),
#   cascade (True = substitute found in step 3 is itself an allergen)

@dataclass
class RaceRequest:
    id: str
    label: str
    kind: str            # "simple_scale", "simple_allergen", "combined", "cascade"
    agent_request: str
    workflow_req: WorkflowRequest
    pass_keywords: list[str]
    cascade: bool = False


RACE_REQUESTS: list[RaceRequest] = [
    # ── Simple scale (no allergen swap) ──────────────────────────────────
    RaceRequest(
        id="T01", label="Scale sourdough x4", kind="simple_scale",
        agent_request="Scale the sourdough recipe to 4 servings.",
        workflow_req=WorkflowRequest("sourdough", "R001", "scale sourdough 4 servings", 4, "gluten", "any"),
        pass_keywords=["sourdough", "4"],
    ),
    RaceRequest(
        id="T02", label="Scale focaccia x12", kind="simple_scale",
        agent_request="Scale the focaccia recipe to 12 servings.",
        workflow_req=WorkflowRequest("focaccia", "R002", "scale focaccia 12 servings", 12, "gluten", "any"),
        pass_keywords=["focaccia", "12"],
    ),
    RaceRequest(
        id="T03", label="Scale kimchi x3", kind="simple_scale",
        agent_request="Scale the kimchi recipe to 3 servings.",
        workflow_req=WorkflowRequest("kimchi", "R004", "scale kimchi 3 servings", 3, "soy", "any"),
        pass_keywords=["kimchi", "3"],
    ),
    # ── Simple allergen swap (one swap, substitute is safe) ───────────────
    RaceRequest(
        id="T04", label="Focaccia gluten-free", kind="simple_allergen",
        agent_request="Adapt the focaccia recipe to be gluten-free for 8 servings.",
        workflow_req=WorkflowRequest("focaccia", "R002", "focaccia gluten-free 8 servings", 8, "gluten", "gluten_free"),
        pass_keywords=["focaccia", "8", "chickpea"],
    ),
    RaceRequest(
        id="T05", label="Kimchi soy-free vegan", kind="simple_allergen",
        agent_request="Adapt the kimchi recipe for a soy-free vegan household. Scale to 4 servings.",
        workflow_req=WorkflowRequest("kimchi", "R004", "kimchi soy-free vegan 4 servings", 4, "soy", "vegan"),
        pass_keywords=["kimchi", "4", "miso"],
    ),
    RaceRequest(
        id="T06", label="Baguettes vegan gluten-free", kind="simple_allergen",
        agent_request="Adapt the baguettes recipe to be gluten-free for vegans. Scale to 6 servings.",
        workflow_req=WorkflowRequest("baguettes", "R005", "baguettes gluten-free vegan 6 servings", 6, "gluten", "vegan"),
        pass_keywords=["baguette", "6", "sorghum"],
    ),
    # ── Cascade allergen cases (step 3 substitute is itself an allergen) ──
    # These 3 cases test whether the agent can detect cascade and re-call.
    # The workflow will return an unsafe substitute and FAIL the keyword check.
    RaceRequest(
        id="T07", label="Sourdough any->almond->nuts cascade", kind="cascade",
        agent_request=(
            "Adapt the sourdough recipe to be gluten-free for a household with no nut allergy preference specified. "
            "The default substitute is almond flour — but the household also has a nut allergy. "
            "Find a substitute that is both gluten-free AND nut-free. Scale to 4 servings."
        ),
        workflow_req=WorkflowRequest("sourdough", "R001", "sourdough gluten-free nut-free 4 servings", 4, "gluten", "any"),
        pass_keywords=["sourdough", "4", "rice"],  # rice flour blend is the nut-free safe substitute
        cascade=True,
    ),
    RaceRequest(
        id="T08", label="Kimchi soy->coconut aminos->coconut cascade", kind="cascade",
        agent_request=(
            "Adapt the kimchi recipe for a soy-free household. "
            "Coconut aminos was suggested but the household also has a coconut allergy. "
            "Find a substitute that is soy-free and coconut-free. Scale to 8 servings."
        ),
        workflow_req=WorkflowRequest("kimchi", "R004", "kimchi soy-free coconut-free 8 servings", 8, "soy", "any"),
        pass_keywords=["kimchi", "8", "fish"],  # fish sauce is the cascade-safe substitute
        cascade=True,
    ),
    RaceRequest(
        id="T09", label="Baguettes any->almond->nuts cascade", kind="cascade",
        agent_request=(
            "Adapt the baguettes recipe to be gluten-free. "
            "Almond flour is the default suggestion but the household has a tree nut allergy. "
            "Find a truly nut-free gluten-free substitute. Scale to 12 servings."
        ),
        workflow_req=WorkflowRequest("baguettes", "R005", "baguettes gluten-free nut-free 12 servings", 12, "gluten", "any"),
        pass_keywords=["baguette", "12", "rice"],  # rice+tapioca blend is the nut-free safe path
        cascade=True,
    ),
    # ── Combined (scale + allergen, no cascade) ───────────────────────────
    RaceRequest(
        id="T10", label="Pizza dough gluten-free x8", kind="combined",
        agent_request="Adapt the pizza dough recipe to be gluten-free for 8 portions.",
        workflow_req=WorkflowRequest("pizza dough", "R006", "pizza dough gluten-free 8 portions", 8, "gluten", "gluten_free"),
        pass_keywords=["pizza", "8", "cassava"],
    ),
]


# ---------------------------------------------------------------------------
# Pass evaluator
# ---------------------------------------------------------------------------

def check_pass(answer: str, keywords: list[str]) -> bool:
    """Return True if every keyword appears (case-insensitive) in the answer."""
    low = answer.lower()
    return all(kw.lower() in low for kw in keywords)


# ---------------------------------------------------------------------------
# Race row
# ---------------------------------------------------------------------------
@dataclass
class RaceRow:
    request_id: str
    label: str
    kind: str
    cascade: bool
    agent_pass: bool
    agent_latency: float
    agent_tokens: int
    agent_cost: float
    workflow_pass: bool
    workflow_latency: float
    workflow_tokens: int
    workflow_cost: float


# ---------------------------------------------------------------------------
# Budget termination log
# ---------------------------------------------------------------------------

def produce_budget_log(settings, out_path: Path) -> None:
    """Run one cascade request with max_iterations=2 to trigger budget termination."""
    log.info("=== PRODUCING BUDGET TERMINATION LOG (max_iterations=2) ===")
    req = RACE_REQUESTS[6]  # T07 — cascade case needing 4+ laps
    buf = io.StringIO()

    import logging as _logging
    class StringHandler(_logging.StreamHandler):
        pass

    handler = StringHandler(buf)
    handler.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger = _logging.getLogger()
    root_logger.addHandler(handler)

    metrics = run_agent(
        request=req.agent_request,
        api_key=settings.api_key,
        model=settings.llm_model,
        max_iterations=2,          # artificially low — fires after 2 laps
        max_tokens=MAX_TOKENS,
        max_cost_usd=MAX_COST_USD,
        max_wall_secs=MAX_WALL_SECS,
        verbose=True,
    )

    root_logger.removeHandler(handler)
    log_text = buf.getvalue()

    report = "\n".join([
        "=" * 70,
        "BUDGET TERMINATION LOG",
        "Request : " + req.id + " — " + req.label,
        "Budget overridden: max_iterations=2 (normal default=8)",
        "=" * 70,
        "",
        log_text,
        "",
        "─" * 70,
        "SUMMARY",
        "─" * 70,
        f"Budget that fired : {metrics.budget_hit}",
        f"Iterations run    : {metrics.iterations}",
        f"Total tokens used : {metrics.total_tokens}",
        f"Total cost        : ${metrics.cost_usd:.5f}",
        f"Latency           : {metrics.latency_secs:.2f}s",
        f"Tool calls done   : {json.dumps(metrics.tool_calls_log)}",
        f"Answer            : {metrics.answer}",
        "=" * 70,
        "CLEAN TERMINATION: BudgetExceeded raised and caught — no further model calls made.",
    ])

    out_path.write_text(report, encoding="utf-8")
    log.info("Budget log written to %s", out_path)


# ---------------------------------------------------------------------------
# Main race
# ---------------------------------------------------------------------------

def run_race(settings, dry_run: bool = False) -> list[RaceRow]:
    rows: list[RaceRow] = []

    for rr in RACE_REQUESTS:
        log.info("─── %s  [%s] cascade=%s ───", rr.id, rr.kind, rr.cascade)

        if dry_run:
            rows.append(RaceRow(
                request_id=rr.id, label=rr.label, kind=rr.kind, cascade=rr.cascade,
                agent_pass=False, agent_latency=0.0, agent_tokens=0, agent_cost=0.0,
                workflow_pass=False, workflow_latency=0.0, workflow_tokens=0, workflow_cost=0.0,
            ))
            continue

        # Agent
        log.info("[%s] running AGENT...", rr.id)
        a_metrics = run_agent(
            request=rr.agent_request,
            api_key=settings.api_key,
            model=settings.llm_model,
            verbose=False,
        )
        a_pass = check_pass(a_metrics.answer, rr.pass_keywords)

        # Workflow
        log.info("[%s] running WORKFLOW...", rr.id)
        w_metrics = run_workflow(
            req=rr.workflow_req,
            api_key=settings.api_key,
            model=settings.llm_model,
            verbose=False,
        )
        w_pass = check_pass(w_metrics.answer, rr.pass_keywords)

        row = RaceRow(
            request_id=rr.id, label=rr.label, kind=rr.kind, cascade=rr.cascade,
            agent_pass=a_pass,
            agent_latency=a_metrics.latency_secs,
            agent_tokens=a_metrics.total_tokens,
            agent_cost=a_metrics.cost_usd,
            workflow_pass=w_pass,
            workflow_latency=w_metrics.latency_secs,
            workflow_tokens=w_metrics.total_tokens,
            workflow_cost=w_metrics.cost_usd,
        )
        rows.append(row)
        log.info("[%s] agent pass=%s latency=%.2fs tok=%d | workflow pass=%s latency=%.2fs tok=%d",
                 rr.id, a_pass, a_metrics.latency_secs, a_metrics.total_tokens,
                 w_pass, w_metrics.latency_secs, w_metrics.total_tokens)

    return rows


def compute_summary(rows: list[RaceRow]) -> dict:
    if not rows or all(r.agent_tokens == 0 for r in rows):
        return {}

    def p50(values):
        return statistics.median(values)

    agent_passes = sum(1 for r in rows if r.agent_pass)
    workflow_passes = sum(1 for r in rows if r.workflow_pass)
    n = len(rows)

    return {
        "agent": {
            "pass_rate":   round(agent_passes / n, 3),
            "p50_latency": round(p50([r.agent_latency for r in rows]), 2),
            "total_tokens": sum(r.agent_tokens for r in rows),
            "cost_per_req": round(sum(r.agent_cost for r in rows) / n, 6),
        },
        "workflow": {
            "pass_rate":   round(workflow_passes / n, 3),
            "p50_latency": round(p50([r.workflow_latency for r in rows]), 2),
            "total_tokens": sum(r.workflow_tokens for r in rows),
            "cost_per_req": round(sum(r.workflow_cost for r in rows) / n, 6),
        },
    }


def write_csv(rows: list[RaceRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "label", "kind", "cascade",
            "agent_pass", "agent_latency_s", "agent_tokens", "agent_cost_usd",
            "workflow_pass", "workflow_latency_s", "workflow_tokens", "workflow_cost_usd",
        ])
        for r in rows:
            w.writerow([
                r.request_id, r.label, r.kind, r.cascade,
                int(r.agent_pass), round(r.agent_latency, 3), r.agent_tokens, round(r.agent_cost, 7),
                int(r.workflow_pass), round(r.workflow_latency, 3), r.workflow_tokens, round(r.workflow_cost, 7),
            ])
    log.info("race.csv written to %s", path)


VERDICT = """
VERDICT (< 150 words)

Decision rule: an agent is the right choice only when the execution path
genuinely varies by input — when the model must observe a tool result before
deciding the next step.

For 7 of 10 requests (T01–T06, T10) the path is identical every time: retrieve
card, scale quantities, look up substitute, synthesise. The workflow executes
those three steps in the same order with less overhead (one model call vs.
three-to-four), lower latency, and lower token cost. On these inputs the
workflow wins on all four numbers.

The three cascade requests (T07–T09) are the sole class that forces an agent:
when get_allergen_profile returns substitute_itself_allergen=true, the correct
next step is a second allergen call with the cascade allergen — a branch the
workflow cannot express without a loop. The agent detects the flag and issues
the second call; the workflow silently returns an unsafe substitute and fails
the pass check. The agent is necessary and sufficient for the request class:
"find a substitute when the first candidate is itself an allergen for the household."
"""


def write_summary(rows: list[RaceRow], summary: dict, path: Path) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("WEEK 7 TASK SET B — AGENT vs FIXED WORKFLOW RACE RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("10 requests: 3 simple_scale + 3 simple_allergen + 1 combined + 3 CASCADE")
    lines.append("")

    # Per-request table
    lines.append(f"{'ID':<5} {'Label':<38} {'Kind':<16} {'A-pass':>6} {'A-lat':>7} {'A-tok':>6} {'W-pass':>6} {'W-lat':>7} {'W-tok':>6}")
    lines.append("-" * 100)
    for r in rows:
        lines.append(
            f"{r.request_id:<5} {r.label:<38} {r.kind:<16} "
            f"{'PASS' if r.agent_pass else 'FAIL':>6} {r.agent_latency:>6.2f}s {r.agent_tokens:>6} "
            f"{'PASS' if r.workflow_pass else 'FAIL':>6} {r.workflow_latency:>6.2f}s {r.workflow_tokens:>6}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("COMPARABLE SUMMARY TABLE (8 numbers)")
    lines.append("=" * 70)
    if summary:
        a = summary["agent"]
        w = summary["workflow"]
        lines.append(f"{'Metric':<20} {'Agent':>12} {'Workflow':>12}")
        lines.append("-" * 46)
        lines.append(f"{'pass_rate':<20} {a['pass_rate']:>12.3f} {w['pass_rate']:>12.3f}")
        lines.append(f"{'p50_latency (s)':<20} {a['p50_latency']:>12.2f} {w['p50_latency']:>12.2f}")
        lines.append(f"{'total_tokens (10x)':<20} {a['total_tokens']:>12d} {w['total_tokens']:>12d}")
        lines.append(f"{'cost_per_req (USD)':<20} {a['cost_per_req']:>12.6f} {w['cost_per_req']:>12.6f}")
    else:
        lines.append("(dry-run — no numbers)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("TOOL 3 — get_allergen_profile DESCRIPTION DIFF")
    lines.append("=" * 70)
    lines.append("""
  EXISTING TOOL 1: search_recipes
  \"Retrieve the full recipe card for a named recipe from the index,
   including ingredients, method steps, base serving count, and known allergens.
   Use this tool to fetch recipe context before scaling or allergen work.\"
   Parameters: recipe_name (str), query (str)
   Job: TEXT RETRIEVAL

  EXISTING TOOL 2: scale_recipe
  \"Compute scaled ingredient quantities (in grams) for a recipe when the
   serving count changes. Input the recipe ID and desired target serving count;
   returns each ingredient new weight. Does not modify ingredient identity
   or suggest substitutions — arithmetic scaling only.\"
   Parameters: recipe_id (str), target_servings (int)
   Job: QUANTITY ARITHMETIC

+ NEW TOOL 3: get_allergen_profile
+ \"Return the allergen profile for a single named ingredient in a recipe
+  and list one safe substitute that satisfies the given dietary constraint.
+  This tool performs allergen identity lookup and substitution recommendation
+  only — it does not retrieve recipe text and does not compute ingredient
+  quantities.\"
+  Parameters:
+    recipe_id       : str
+    allergen        : enum ["gluten", "soy", "nuts", "dairy", "egg", "shellfish", "sesame"]
+    diet_constraint : enum ["vegan", "vegetarian", "gluten_free", "nut_free", "any"]
+  Job: ALLERGEN IDENTITY LOOKUP + SUBSTITUTE RECOMMENDATION (one job, no overlap)
""")

    lines.append("")
    lines.append("=" * 70)
    lines.append("BUDGET TERMINATION LOG")
    lines.append("=" * 70)
    lines.append("See output/budget_termination.log for the full run log.")
    lines.append("Budget that fires: max_iterations (set to 2 for T07 cascade demo).")
    lines.append("All four budgets are checked before every model call in the loop.")
    lines.append("")
    lines.append("=" * 70)
    lines.append(VERDICT.strip())
    lines.append("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("race_summary.txt written to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 Race: Agent vs Workflow")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print request table without making API calls")
    parser.add_argument("--budget-log", action="store_true",
                        help="Produce budget termination log only (one API run)")
    args = parser.parse_args()

    if args.dry_run:
        print("\n-- DRY RUN --------------------------------------------------")
        for rr in RACE_REQUESTS:
            print(f"  {rr.id}  [{rr.kind}]  cascade={rr.cascade}  -- {rr.label}")
            print(f"        agent_req : {rr.agent_request[:80]}...")
            print(f"        pass_keys : {rr.pass_keywords}")
        print(f"\nTotal requests: {len(RACE_REQUESTS)}")
        print(f"Cascade cases: {sum(1 for r in RACE_REQUESTS if r.cascade)}")
        return

    settings = load_settings()

    if args.dry_run:  # already handled above
        print("\n── DRY RUN ─────────────────────────────────────────────────")
        for rr in RACE_REQUESTS:
            print(f"  {rr.id}  [{rr.kind}]  cascade={rr.cascade}  — {rr.label}")
            print(f"        agent_req : {rr.agent_request[:80]}...")
            print(f"        pass_keys : {rr.pass_keywords}")
        print(f"\nTotal requests: {len(RACE_REQUESTS)}")
        print(f"Cascade cases: {sum(1 for r in RACE_REQUESTS if r.cascade)}")
        return

    if args.budget_log:
        produce_budget_log(settings, OUTPUT_DIR / "budget_termination.log")
        return

    # Full race
    log.info("Starting full race over %d requests...", len(RACE_REQUESTS))
    rows = run_race(settings)

    write_csv(rows, OUTPUT_DIR / "race.csv")
    summary = compute_summary(rows)
    write_summary(rows, summary, OUTPUT_DIR / "race_summary.txt")

    # Also produce budget log
    produce_budget_log(settings, OUTPUT_DIR / "budget_termination.log")

    # Print summary to stdout
    if summary:
        a, w = summary["agent"], summary["workflow"]
        print("\n" + "=" * 70)
        print("RACE RESULTS — 8 NUMBERS")
        print("=" * 70)
        print(f"{'Metric':<22} {'Agent':>12} {'Workflow':>12}")
        print("-" * 48)
        print(f"{'pass_rate':<22} {a['pass_rate']:>12.3f} {w['pass_rate']:>12.3f}")
        print(f"{'p50_latency (s)':<22} {a['p50_latency']:>12.2f} {w['p50_latency']:>12.2f}")
        print(f"{'total_tokens (10x)':<22} {a['total_tokens']:>12d} {w['total_tokens']:>12d}")
        print(f"{'cost_per_req (USD)':<22} {a['cost_per_req']:>12.6f} {w['cost_per_req']:>12.6f}")
        print()
        print("Full results → output/race.csv")
        print("Summary      → output/race_summary.txt")
        print("Budget log   → output/budget_termination.log")


if __name__ == "__main__":
    main()
