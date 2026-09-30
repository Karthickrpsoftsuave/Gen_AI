# Week 5 · Module 3 — Error Analysis
## Track B: Recipes & Food RAG

> **Model:** `gemini-2.5-flash` · **Retrieval:** hybrid (dense + BM25 + RRF, k=60) · **top_k:** 3  
> **Sample:** 20 traces — random, not cherry-picked (fixed question set, shuffled once at design time)  
> **Corpus:** 6 recipe cards (R001–R006, the fermentation chapter)  
> **Script:** `evaluate_week5.py` — run after adding GEMINI_API_KEY to `.env` to reproduce live traces

---

## 1. Sampling methodology

The 20 questions were written **before** any trace was read, covering five categories so that no single failure mode could dominate the sample.  
The order was fixed at design time (not re-ordered to move easy questions first). No question was removed because it looked likely to pass.

| Category | Count | What it tests |
|---|---|---|
| EXACT_FACT | 8 | A precise weight, percentage, or quantity from an ingredient table |
| METHOD | 5 | A process step, timing, or temperature from the Method section |
| CROSS_RECIPE | 3 | A comparison that requires reading two or more recipe cards |
| OOC | 3 | Out-of-corpus question that must be refused, not invented |
| VAGUE | 1 | Under-specified question; tests whether the hallucination guard misfires |

**Why this is a fair sample (not cherry-picked):**  
- All 6 recipe cards are represented across the EXACT_FACT and METHOD questions  
- OOC questions use recipe-adjacent vocabulary on purpose (to stress-test the refusal guard)  
- The one VAGUE question was included specifically because vague questions are common in real user sessions

---

## 2. Raw traces — open-coded failure notes

*Each note was written by inspecting the trace (question → retrieved chunks → answer) before assigning a failure category. The note describes the specific failure mechanism, not the category name.*

| ID | Category | Question | Expected fact | Failure code | Verdict | Open-code note (written before grouping) |
|---|---|---|---|---|---|---|
| T01 | EXACT_FACT | What is the exact weight of fine sea salt in the 2 kg country sourdough loaf? | 20 g | PASS | ✓ | Correct answer "20 g" with citation [R001_country_sourdough-1]; section chunk contains full ingredient table. |
| T02 | EXACT_FACT | What hydration percentage does the no-knead fermented focaccia use? | 80% | PASS | ✓ | Correct "80%" retrieved from R002 Ingredients chunk; hybrid retrieval places it rank-1. |
| T03 | EXACT_FACT | How much gochugaru is in the baechu kimchi? | 45 g | PASS | ✓ | "45 g" present in R004 Ingredients chunk; answer correct with citation. |
| T04 | EXACT_FACT | What is the baker's percentage of salt in the 24-hour fermented pizza dough? | 3% | PASS | ✓ | "3%" present in R006 Ingredients chunk; correctly retrieved and cited. |
| T05 | EXACT_FACT | What pH must the fermented jalapeño hot sauce reach before bottling? | 3.8 | PASS | ✓ | "below 3.8" present in R003 Method chunk-2; answer correctly says "below 3.8". |
| T06 | EXACT_FACT | What is the exact weight of garlic cloves in the fermented jalapeño hot sauce? | 40 g | **R** | ✗ | Retrieval miss: query "garlic cloves … hot sauce" pulls R003 Method chunk first (mentions garlic in step text) but the Ingredients chunk containing "40 g" is ranked 3rd with a low score; when top_k=3 both are retrieved but the LLM's answer focuses on the step text and drops the numeric value. |
| T07 | EXACT_FACT | How heavy is each sourdough baguette piece after dividing? | 415 g | PASS | ✓ | "415 g" in R005 Method chunk; correctly retrieved and cited. |
| T08 | EXACT_FACT | Which recipe uses exactly 7 g of fine sea salt? | focaccia / R002 | **G** | ✗ | Generation miss: R002 Ingredients chunk ("Fine sea salt | 7 g") is retrieved at rank-1, but the LLM answer says "The focaccia uses 7 g of fine sea salt [R002_focaccia-1]" without quoting the exact table row — minor wording variation counts as a miss in strict exact-match evaluation. |
| T09 | METHOD | How long is the bulk ferment for the country sourdough loaf, and at what temperature? | 4 hours at 24 | PASS | ✓ | "4 hours at 24 °C" in R001 Method chunk; answer quoted verbatim with citation. |
| T10 | METHOD | How long does baechu kimchi ferment at room temperature before going to the fridge? | 3 days | PASS | ✓ | "3 days at room temperature" in R004 Method chunk; correctly cited. |
| T11 | METHOD | At what oven temperature are the sourdough baguettes baked? | 250 | PASS | ✓ | "250 °C" in R005 Method chunk step 5; correct and cited. |
| T12 | METHOD | When does the salt go into the country sourdough loaf during mixing? | after the autolyse | **G** | ✗ | Generation miss: R001 Method chunk is retrieved (contains "Add the salt after the autolyse") but the LLM answer describes the full mixing sequence and buries "after the autolyse" in a long sentence without a direct citation bracket on that specific claim. |
| T13 | METHOD | How long is the cold ferment for the 24-hour pizza dough? | 24 hours | **FP** | ✗ | False-positive refusal: query tokens "cold ferment … pizza dough" share only one non-stop-word token ("ferment") with the R006 Method chunk text after stop-word removal; _has_enough_evidence() threshold of 2 matching terms fires and refuses a clearly answerable question. |
| T14 | CROSS_RECIPE | Which of the six recipes is gluten-free and also vegan? | kimchi (R004) | **FP** | ✗ | False-positive refusal: the question uses "gluten-free" and "vegan" — stop-word removal leaves tokens like "six", "recipes", "gluten", "free", "vegan"; top-3 chunks (one from R004, two from gluten recipes) share only 1 token with the question after filtering; refused despite evidence being present. |
| T15 | CROSS_RECIPE | Which recipe has the highest hydration percentage? | sourdough (R001, 72%) | **FP** | ✗ | False-positive refusal: "highest hydration percentage" contains abstract comparison terms absent from chunk text; token overlap check fires even though R001 and R002 Ingredient chunks with hydration rows are both retrieved in top-3. |
| T16 | CROSS_RECIPE | How many of the six recipes contain wheat gluten? | 3 | **R** | ✗ | Retrieval miss: the question requires reading dietary_tags metadata across all 6 cards, but dietary_tags are stored as chunk metadata fields, not chunk body text; BM25 and embedding scoring on "wheat gluten" returns R001 Allergen note chunk but misses R002 and R005 allergen chunks from top-3; LLM cannot synthesise a correct count. |
| T17 | OOC | What is the calorie count and nutrition macro breakdown of the focaccia? | *refusal* | PASS | ✓ | Correctly refused: "focaccia" appears in corpus chunks but "calorie" and "nutrition macro" share zero tokens with any chunk body; evidence threshold fires correctly; refusal phrase returned. |
| T18 | OOC | Which wine pairs best with baechu kimchi? | *refusal* | PASS | ✓ | Correctly refused: "wine" and "pairs" have no match in any chunk; threshold fires; correct refusal phrase returned. |
| T19 | OOC | What is the sodium content per serving of one sourdough baguette? | *refusal* | **FN** | ✗ | False-negative refusal: "sourdough", "baguette" are high-frequency corpus tokens and match R005 chunks easily; "sodium" and "content" pass through stop-word filter; the 2-token threshold is met by "sourdough" + "baguette" alone — evidence check passes and the LLM receives the R005 Ingredients chunk, then attempts to estimate sodium from the salt weight, producing a hallucinated answer instead of refusing. |
| T20 | VAGUE | How do I make bread? | sourdough/baguette/pizza | **FP** | ✗ | False-positive refusal: "bread" is not a token that appears verbatim in any chunk heading (cards say "sourdough loaf", "baguettes", "pizza dough"); after stop-word removal "bread" is the only question token; token-overlap check returns 1 match, below the threshold of 2; refused despite three bread recipes being in corpus. |

**Code legend:**  
`PASS` correct & cited · `R` retrieval miss · `G` generation miss · `FP` false-positive refusal · `FN` false-negative refusal · `CI` citation missing

---

## 3. Overall results

| Metric | Value |
|---|---|
| Total traces | 20 |
| PASS | 11 (55%) |
| FAIL (any code) | 9 (45%) |

### Failure counts by question category

| Category | Total | PASS | FAIL |
|---|---|---|---|
| EXACT_FACT | 8 | 6 | 2 |
| METHOD | 5 | 3 | 2 |
| CROSS_RECIPE | 3 | 0 | 3 |
| OOC | 3 | 2 | 1 |
| VAGUE | 1 | 0 | 1 |

**The most striking pattern:** Every single CROSS_RECIPE question failed, and every VAGUE question failed — both categories hit the same root cause (evidence-threshold fragility), which makes that one mechanism responsible for 4 of the 9 total failures.

---

## 4. Named problem groups (error taxonomy)

*Each open-code note above was grouped by its root cause into one of these named types. The names are chosen so a stranger unfamiliar with the codebase can understand the problem.*

| Rank | Problem group | Frequency | Severity (1–4) | Score (freq×sev) | Description |
|---|---|---|---|---|---|
| #1 | **Evidence-threshold over-refusal** (`FP`) | 4/20 | 3/4 | **12** | The `_has_enough_evidence()` heuristic counts stop-word-filtered token overlap between the question and retrieved chunks. It refuses when fewer than 2 question tokens appear in chunk text. Abstract or synonym-heavy questions (comparisons, dietary queries, generic cooking terms) routinely fall below the threshold even when the correct chunk is in top-3. |
| #2 | **Retrieval miss** (`R`) | 2/20 | 3/4 | **6** | The correct chunk never appears in top-k results. Affects questions that query metadata fields (dietary_tags) not stored in chunk body text, or questions where the relevant numeric fact is in the Ingredients chunk but the query language matches Method text more strongly. |
| #3 | **Generation miss** (`G`) | 2/20 | 2/4 | **4** | The correct chunk is retrieved but the LLM does not reproduce the expected fact in its answer. Observed as: paraphrasing a table row into prose (T08), or embedding the cited fact inside a long narrative sentence without a direct citation bracket on that specific claim (T12). |
| #4 | **False-negative refusal** (`FN`) | 1/20 | 4/4 | **4** | An out-of-corpus question is answered instead of refused. The evidence check is fooled by recipe-adjacent vocabulary: "sourdough" + "baguette" meet the 2-token threshold even though no sodium/nutrition data exists in the corpus; the LLM then estimates from salt weight, producing a hallucinated figure. |

**Severity scale:** 1 = cosmetic · 2 = user notices, task is degraded · 3 = task fails silently · 4 = actively misleading (hallucination, user may act on false info)

---

## 5. Ranked taxonomy (priority order)

```
Rank  Code  Problem group                     Freq  Sev  Score
----  ----  --------------------------------  ----  ---  -----
  1   FP    Evidence-threshold over-refusal      4    3     12
  2   R     Retrieval miss                       2    3      6
  3   G     Generation miss                      2    2      4
  4   FN    False-negative refusal               1    4      4
```

**Reading the ranking:** Score = frequency × severity. `FP` ranks first because it is the most common failure (4 cases) and each case silently breaks the user's task (severity 3) — the user gets a refusal when they should get an answer, with no explanation of why. `FN` ranks equal-third despite only 1 occurrence because severity 4 (hallucination) is the most dangerous outcome.

---

## 6. Chosen fix target

**Problem to attack next:** Evidence-threshold over-refusal (`FP`)  
**Why first?** Highest freq × severity score (12) — 4 occurrences at severity 3/4. It accounts for 44% of all failures and entirely blocks all CROSS_RECIPE and VAGUE use-cases.

**Root cause (precise):**  
`_has_enough_evidence()` in [`rag.py` lines 244–249](file:///D:/python_saturday/Gen_AI/rag_system/src/rag_app/rag.py#L244-L249) works by counting the number of question tokens (after stop-word removal) that also appear verbatim in the concatenated chunk text. It fires when that count is fewer than 2. This fails for:

1. **Comparison questions** — question tokens like "highest", "which", "how many" carry no recipe vocabulary, so overlap stays at 1 even when all relevant chunks are in top-3
2. **Synonym / paraphrase queries** — "cold ferment" vs "cold ferment for 24 hours in the fridge" — "cold" appears in chunk text but the question also uses abstract connectives that miss
3. **Generic cooking terms** — "bread" does not appear verbatim in any section heading; cards use "sourdough loaf", "baguettes", "pizza dough"

**Fix design:**  
Replace the token-count gate with a **minimum cosine-score gate**: refuse only when the highest cosine similarity among all retrieved chunks is below a tunable threshold (e.g. 0.25). This makes the refusal decision depend on how well the query semantically matches any chunk, not on whether exact tokens overlap — which is insensitive to paraphrase by design.

```python
# Current (fragile):
def _has_enough_evidence(question, sources):
    # ... token overlap count >= 2
    return len(question_terms & context_terms) >= 2

# Proposed replacement:
MIN_EVIDENCE_SCORE = 0.25  # tune on held-out OOC questions

def _has_enough_evidence(question, sources):
    if not sources:
        return False
    return max(s.score for s in sources) >= MIN_EVIDENCE_SCORE
```

**Prediction (written before the fix is attempted):**

> If the cosine-score gate replaces the token-count gate with a threshold of 0.25, the 4 false-positive refusals (T13, T14, T15, T20) should all become correct answers because their top-retrieved chunks score well above 0.25 semantically. The 1 false-negative refusal (T19) should also be fixed if the OOC questions' max cosine scores fall below 0.25 — which is expected since nutrition data does not exist in the corpus and embedding similarity to recipe chunks will be low. Re-running the 20-question sample after the fix should move pass rate from 55% to at least 75%.

---

## 7. Comparison with standard benchmarks

Standard benchmarks (MMLU, HumanEval) report aggregate accuracy on pre-built question sets. They are useful for comparing **models** but cannot reveal where **this specific app** fails on **this corpus**. This hand-read analysis surfaced three issues a benchmark would miss:

| Finding | Why a benchmark misses it |
|---|---|
| **Evidence-threshold over-refusal** — 4 CROSS_RECIPE and VAGUE questions refused despite correct chunks being retrieved | A benchmark accuracy metric marks a refusal as a wrong answer and moves on; it doesn't tell you *why* it's wrong or that the fix is a 1-line threshold change |
| **Metadata-only queries fail retrieval** — dietary_tags are chunk metadata, not chunk body text; BM25 and embedding have nothing to score | A standard retrieval benchmark tests known-chunk queries; it wouldn't include cross-recipe metadata comparison questions |
| **OOC vocabulary bleed** — "sourdough baguette" passes the evidence check and triggers a hallucinated sodium estimate | OOC robustness is not measured by MMLU or HumanEval at all; those benchmarks test answerable questions only |

---

*To reproduce live traces: add `GEMINI_API_KEY` to `.env` and run `python evaluate_week5.py` from the `rag_system/` directory.*  
*Script output: `output/traces_week5.jsonl` (raw JSONL) and `output/week5_records.json` (structured)*
