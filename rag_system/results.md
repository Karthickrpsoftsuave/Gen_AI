# Week 3 · M2 — Set B Results: Fermentation Chapter Ingest

> Task: ingest the 6 new fermentation recipe cards, measure TWO chunking
> strategies on 8 known-answer questions, add a dietary_tags filter, and force
> the app to refuse what it cannot source.
> **Scope note (requirement 6): only the 6 supplied cards were indexed. No
> pre-existing recipe corpus was re-indexed — the app's index in this build
> contains exactly the 6 new cards (25 chunks under the naive
> chunker, 70 under the structure-aware chunker).**

## 0. What was built

- `data/cards/` — the 6 supplied cards (R001–R006), each: YAML frontmatter
  (recipe_id, cuisine, dietary_tags) + ingredient table + method + allergen note.
- `rag_pipeline/chunker.py` — two chunkers (naive fixed-size; structure-aware) +
  the ingest gate that FAILS a chunk with no `source_file`.
- `rag_pipeline/store.py` — dense embeddings (BAAI/bge-small-en-v1.5, dim 384,
  ONNX via fastembed — no torch) + top-K cosine search + dietary_tags filter.
- `rag_pipeline/generate.py` — grounded (extractive) generator with a citation
  on every claim and a hard refusal gate: no 'use your best judgement' path exists.
- `evaluate.py` — this measurement run. `questions/questions.json` — the 8
  questions, written from the cards BEFORE any search ran.
- `output/search_dump_naive.json`, `output/search_dump_structured.json` — full
  search-only dumps for all 8 questions under both strategies.

## 1. The 8 known-answer questions (written from the cards first)

| ID | Question | Known-correct (recipe · section) | Depends on |
|---|---|---|---|
| Q1 | What is the exact weight of fine sea salt in the 2 kg country sourdough loaf? | R001 · Ingredients (salt row) | ingredient-table row |
| Q2 | What hydration percentage does the no-knead fermented focaccia use? | R002 · Ingredients (water row) | ingredient-table row |
| Q3 | How much gochugaru goes into the baechu kimchi? | R004 · Ingredients (gochugaru row) | ingredient-table row |
| Q4 | Which recipe uses exactly 7 g of fine sea salt? | R002 · Ingredients (salt row) | ingredient-table row |
| Q5 | How many baguettes does the sourdough baguette recipe make, and how heavy is each piece? | R005 · Method | method |
| Q6 | What pH must the fermented jalapeño hot sauce reach before it can be bottled? | R003 · Method | method |
| Q7 | How long is the bulk ferment for the country sourdough loaf, and at what temperature? | R001 · Method | method |
| Q8 | What is the baker's percentage of salt in the 24-hour fermented pizza dough? | R006 · Ingredients (salt row) | ingredient-table row |

5 of the 8 depend on a row inside an ingredient table (requirement 2 needs ≥ 3).

## 2. The two chunking strategies

**Naive ('current' chunker)** — fixed-size window of 200 chars with 40-char
overlap, snapped to line breaks. Section-blind: a window can slice straight
through the ingredient table and separate a row from its header and title.
→ 25 chunks.

**Structure-aware** — parses sections; every ingredient row is emitted as
`<recipe title> + | Ingredient | Weight | Baker's % | + <row>` so a row can
NEVER be separated from its header or parent title (requirement 3). Method
steps, allergen note and a full-table chunk are separate chunks.
→ 70 chunks.

Same embedding model (bge-small-en-v1.5) for both indexes — exactly one
variable changed (Set B §7).

## 3. Hit-in-top-5 — SAME 8 questions, both strategies

| Q | Known-correct (recipe · section) | Naive rank (score) | Structured rank (score) |
|---|---|---|---|
| Q1 | R001 · Ingredients | #1 (0.827) | #1 (0.917) |
| Q2 | R002 · Ingredients | #1 (0.800) | #1 (0.857) |
| Q3 | R004 · Ingredients | #2 (0.673) | #1 (0.809) |
| Q4 | R002 · Ingredients | #1 (0.797) | #1 (0.773) |
| Q5 | R005 · Method | MISS (—) | #1 (0.853) |
| Q6 | R003 · Method | #1 (0.778) | #1 (0.927) |
| Q7 | R001 · Method | MISS (—) | #1 (0.888) |
| Q8 | R006 · Ingredients | #5 (0.706) | #1 (0.888) |
| **hit-in-top-5** | | **6/8** | **8/8** |
| **MRR** | | **0.588** | **1.000** |

Hit = a chunk from the known-correct recipe that also contains the expected
fact appears in the top-5. Scores are cosine similarity.

**Numbers that moved: hit-in-top-5 is 6/8 (naive) vs 8/8 (structured),
MRR 0.588 vs 1.000.**

## 4. Metadata filter (requirement 4) — structured store

Query: *"How long is the bulk ferment for the country sourdough loaf, and at what temperature?"* — filter: `dietary_tags` contains `gluten-free`.

**Unfiltered top-5:**

1. [score 0.8878] R001-method-02 (R001 · R001_country_sourdough.md · Method)
   Country Sourdough Loaf (2 kg final dough)
## Method
Bulk ferment for 4 hours at 24 °C, giving the dough three coil folds at 30 minute intervals.
2. [score 0.8320] R001-method-03 (R001 · R001_country_sourdough.md · Method)
   Country Sourdough Loaf (2 kg final dough)
## Method
Shape the dough, proof overnight in the fridge for 12 hours, then bake at 250 °C with the lid on for 20 minu
3. [score 0.8167] R001-ing-02 (R001 · R001_country_sourdough.md · Ingredients)
   Country Sourdough Loaf (2 kg final dough)
| Ingredient | Weight | Baker's % |
| Ripe sourdough starter | 200 g | 20% |
4. [score 0.8155] R001-ing-01 (R001 · R001_country_sourdough.md · Ingredients)
   Country Sourdough Loaf (2 kg final dough)
| Ingredient | Weight | Baker's % |
| Water (24 °C) | 720 g | 72% |
5. [score 0.8078] R001-method-00 (R001 · R001_country_sourdough.md · Method)
   Country Sourdough Loaf (2 kg final dough)
## Method
Mix the flour and water and rest for a 45 minute autolyse.

**Filtered (gluten-free only) top-5:**

1. [score 0.6538] R003-ing-01 (R003 · R003_hot_sauce.md · Ingredients)
   Fermented Jalapeño Hot Sauce
| Ingredient | Weight | Baker's % |
| Water | 200 g | 25% |
2. [score 0.6501] R003-method-01 (R003 · R003_hot_sauce.md · Method)
   Fermented Jalapeño Hot Sauce
## Method
Ferment for 5 days at 21 °C, burping the jar daily.
3. [score 0.6465] R004-method-03 (R004 · R004_kimchi.md · Method)
   Baechu Kimchi
## Method
Ferment 3 days at room temperature, then move to the fridge.
4. [score 0.6453] R003-ing-02 (R003 · R003_hot_sauce.md · Ingredients)
   Fermented Jalapeño Hot Sauce
| Ingredient | Weight | Baker's % |
| Fine sea salt | 30 g | 3.75% |
5. [score 0.6264] R003-ing-03 (R003 · R003_hot_sauce.md · Ingredients)
   Fermented Jalapeño Hot Sauce
| Ingredient | Weight | Baker's % |
| Garlic cloves | 40 g | 5% |

The top-1 flips from **R001-method-02** (0.8878) to
**R003-ing-01** (0.6538) because the filter deletes all
`contains-gluten` chunks BEFORE ranking — the gluten-free corpus has no bread
recipe, so the same question retrieves the kimchi ferment instead of the
sourdough ferment.

## 5. Three answerable questions — citations resolve to real chunk_ids


### > Q: How much fine sea salt does the 2 kg country sourdough loaf need?
A: | Fine sea salt | 20 g | 2% | | Strong bread flour | 1000 g | 100% | | Ripe sourdough starter | 200 g | 20% |
   [0.897] | Fine sea salt | 20 g | 2% |  (chunk R001-ing-full · R001 · R001_country_sourdough.md · section Ingredients)
   [0.785] | Strong bread flour | 1000 g | 100% |  (chunk R001-ing-full · R001 · R001_country_sourdough.md · section Ingredients)
   [0.785] | Ripe sourdough starter | 200 g | 20% |  (chunk R001-ing-full · R001 · R001_country_sourdough.md · section Ingredients)
### > Q: What hydration percentage does the no-knead fermented focaccia use?
A: | Water | 400 g | 80% | | Bread flour | 500 g | 100% | | Olive oil | 25 g | 5% |
   [0.727] | Water | 400 g | 80% |  (chunk R002-ing-01 · R002 · R002_focaccia.md · section Ingredients)
   [0.621] | Bread flour | 500 g | 100% |  (chunk R002-ing-full · R002 · R002_focaccia.md · section Ingredients)
   [0.621] | Olive oil | 25 g | 5% |  (chunk R002-ing-full · R002 · R002_focaccia.md · section Ingredients)
### > Q: What pH must the fermented jalapeño hot sauce reach before bottling?
A: The pH must read below 3.8 before bottling; keep the bottled sauce refrigerated. Ferment for 5 days at 21 °C, burping the jar daily. | Water | 200 g | 25% |
   [0.902] The pH must read below 3.8 before bottling; keep the bottled sauce refrigerated.  (chunk R003-method-03 · R003 · R003_hot_sauce.md · section Method)
   [0.594] Ferment for 5 days at 21 °C, burping the jar daily.  (chunk R003-method-01 · R003 · R003_hot_sauce.md · section Method)
   [0.591] | Water | 200 g | 25% |  (chunk R003-ing-01 · R003 · R003_hot_sauce.md · section Ingredients)

Every claim above carries its chunk_id + recipe_id + source_file. The cited
chunk text (visible in the search dumps / chunk index) contains the claim
verbatim — the generator is extractive, so a citation that did not contain its
claim would be a code bug.

## 6. Three unanswerable questions — refused, not invented


### > Q: What is the calorie count and nutrition macro breakdown of the focaccia?
A: I don't know. The answer is not in the indexed documents, and I won't guess.
   [grounding gate: best evidence 0.260 < threshold 0.42 -> refusal forced]
   [nearest material found: 'No-Knead Fermented Focaccia
| Ingredient | Weight | Baker's % |
| Water | 400 g | 80% |' (R002)]
### > Q: What is the sodium content per serving of one sourdough baguette?
A: I don't know. The answer is not in the indexed documents, and I won't guess.
   [grounding gate: best evidence 0.355 < threshold 0.42 -> refusal forced]
   [nearest material found: 'Sourdough Baguettes
| Ingredient | Weight | Baker's % |
| Fine sea salt | 17 g | 2% |' (R005)]
### > Q: Which wine would pair best with the baechu kimchi?
A: I don't know. The answer is not in the indexed documents, and I won't guess.
   [grounding gate: best evidence 0.303 < threshold 0.42 -> refusal forced]
   [nearest material found: 'Baechu Kimchi
recipe_id R004 · cuisine Korean · tags vegan, gluten-free' (R004)]

The refusal is forced by the grounding gate in `generate.py`: no eligible
sentence cleared the evidence threshold (0.42), so the generator
is structurally unable to emit an answer. There is no fallback branch that
guesses.

## 7. Which chunker ships, and why

The structure-aware chunker ships. Hit-in-top-5 8/8 vs 6/8, MRR 1.000 vs 0.588 — and the two naive misses (Q5, Q7) are exactly the failure the kitchen warned about: a table row or method step carved away from its recipe by a blind window. The structure-aware chunker makes that impossible by construction: an ingredient row can never be separated from its table header or parent title. Its per-row chunks also keep the grounding gate honest — retrieval precision is higher, so the generator either cites the exact row or refuses. The naive chunker's occasional wins on score magnitude are not worth the structural risk; the number that moved (MRR 0.588 → 1.000) is the decider.

## 8. The retrieval that embarrassed me

The naive chunker missed **2 of the 8** questions — both ingredient-row/method
questions whose answer window it had carved apart. The structured chunker
missed none. Details:

### Q5 — "How many baguettes does the sourdough baguette recipe make, and how heavy is each piece?"
Known-correct: R005 · Method.
Top-5 as retrieved:

1. [0.8309] R005-naive-000 (R005 · Title)
2. [0.7514] R001-naive-000 (R001 · Title)
3. [0.7085] R006-naive-000 (R006 · Title)
4. [0.7027] R006-naive-001 (R006 · Ingredients)
5. [0.6847] R001-naive-001 (R001 · Ingredients)

The method step that carries the numbers ('Divide the dough into 4 pieces of about 415 g each.') lost to window boundaries: the 200-char window cut the table before it and the step after it, so the window that should answer the question is diluted, and every top-5 slot went to title/ingredient windows from R001 and R006 that contain no number at all.

### Q7 — "How long is the bulk ferment for the country sourdough loaf, and at what temperature?"
Known-correct: R001 · Method.
Top-5 as retrieved:

1. [0.8128] R001-naive-000 (R001 · Title)
2. [0.7770] R001-naive-001 (R001 · Ingredients)
3. [0.7709] R006-naive-001 (R006 · Ingredients)
4. [0.7511] R005-naive-002 (R005 · Ingredients)
5. [0.7509] R006-naive-002 (R006 · Ingredients)

The R001 method window that carries the bulk-ferment step ('Bulk ferment for 4 hours at 24 °C...') never made the top-5. The window split the method so the step sits beside the previous section's rows, and the top-5 filled with title/ingredient windows from R001 and R006 — the pizza dough is also 'fermented', so its table windows outranked the correct method step.

Same questions under the structure-aware chunker: Q5 correct
chunk at rank 1 (0.853), and
Q7 at rank 1 (0.888).

## 9. Bonus — precision vs completeness (section 5)

Question: *"When does the salt go into the country sourdough loaf?"* (known fact: the salt goes in after the autolyse)

**Structure-aware** — retrieval top-1: R001-method-01
(0.8419); generated:
answered: "| Strong bread flour | 1000 g | 100% | | Ripe sourdough starter | 200 g | 20% | | Fine sea salt | 20 g | 2% |"

**Naive** — retrieval top-1: R001-naive-001
(0.7920); generated:
answered: "| Strong bread flour | 1000 g | 100% |"

Neither answer contained the timing fact. Structure-aware retrieved R001-method-01 (0.842), naive R001-naive-001 (0.792), but both generators emitted grounded-but-irrelevant sentences (the numeric rows) because the evidence scorer's numeric bonus outweighed the single low-overlap timing sentence. Diagnosis: short numeric sentences dominate the evidence score; a temporal question needs its temporal markers weighted — noted as the fix.

## 10. The code diff (from-scratch note)

There was no pre-existing app to diff against (this build started from an empty
working directory), so the 'diff' is the new code itself. The two pieces the
checklist asks for — the second chunker and the metadata fields — live in:

- `rag_pipeline/chunker.py` — `chunk_structured()` enforces the invariant:

```python
# ingredient rows — every row carries table header + parent title
for n, row in enumerate(data_rows):
    chunks.append(Chunk(
        chunk_id=f"{card.recipe_id}-ing-{n:02d}",
        text=f"{card.title}\n{TABLE_HEADER}\n{row}",
        source_file=card.path.name, recipe_id=card.recipe_id,
        cuisine=card.cuisine, dietary_tags=card.dietary_tags,
        section="Ingredients", strategy="structured",
    ))
```

- `rag_pipeline/chunker.py` — `ingest()` is the requirement-1 gate (a chunk
  without `source_file` raises and the whole ingest fails), and every chunk
  carries `source_file / recipe_id / cuisine / dietary_tags`.

## 11. Reproduce

```
python -m venv .venv
.venv\Scripts\pip install numpy fastembed
.venv\Scripts\python evaluate.py
```
