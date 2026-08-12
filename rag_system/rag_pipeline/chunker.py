"""Chunking for the Week 3 / M2 'ask my documents' pipeline (Set B — recipes).

Two strategies are implemented so their retrieval can be measured on the SAME
8 known-answer questions:

  naive      - the 'current' chunker: fixed-size character window with overlap.
               It is section-blind: a window can slice straight through an
               ingredient table and separate a row from its table header and
               from the parent recipe title.

  structured - the new, structure-aware chunker. It parses each card into
               sections and enforces the invariant from Set B requirement 3:
               an ingredient row is NEVER separated from its table header
               (| Ingredient | Weight | Baker's % |) or from its parent
               recipe title. Every row chunk therefore carries both.

Every chunk carries the full metadata contract from requirement 1:
source_file, recipe_id, cuisine, dietary_tags. Ingest validates that
source_file is present on every chunk and FAILS the ingest otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Card model
# --------------------------------------------------------------------------

# Document order matters: the naive chunker flattens sections in this order,
# so it must be a tuple, not a set (set iteration order is randomized per
# process by Python's string hash randomization — that made window boundaries
# flip between runs).
SECTION_ORDER = ("Ingredients", "Method", "Allergen note")
TABLE_HEADER = "| Ingredient | Weight | Baker's % |"


@dataclass
class Card:
    path: Path
    recipe_id: str
    cuisine: str
    dietary_tags: str          # comma-separated, e.g. "vegetarian, contains-gluten"
    title: str
    sections: dict[str, str]   # section name -> body text (table / steps / note)
    raw: str


def parse_card(path: Path) -> Card:
    raw = path.read_text(encoding="utf-8")
    front = {}
    body_start = 0
    m = re.match(r"^---\n(.*?)\n---\n", raw, flags=re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                front[k.strip()] = v.strip()
        body_start = m.end()

    body = raw[body_start:]
    title_m = re.search(r"^#\s+(.+)$", body, flags=re.M)
    title = title_m.group(1).strip() if title_m else path.stem

    sections: dict[str, str] = {}
    cur = None
    for line in body.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            sections.setdefault(cur, "")
        elif cur is not None:
            sections[cur] += line + "\n"

    return Card(
        path=path,
        recipe_id=front.get("recipe_id", ""),
        cuisine=front.get("cuisine", ""),
        dietary_tags=front.get("dietary_tags", ""),
        title=title,
        sections={k: v.strip() for k, v in sections.items()},
        raw=raw,
    )


def load_cards(cards_dir: Path) -> list[Card]:
    return [parse_card(p) for p in sorted(cards_dir.glob("*.md"))]


# --------------------------------------------------------------------------
# Chunk model
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    recipe_id: str
    cuisine: str
    dietary_tags: str
    section: str
    strategy: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "recipe_id": self.recipe_id,
            "cuisine": self.cuisine,
            "dietary_tags": self.dietary_tags,
            "section": self.section,
            "strategy": self.strategy,
        }


def ingest(chunks: list[Chunk]) -> list[Chunk]:
    """Requirement 1 gate: a chunk with no source_file is a failed ingest."""
    missing = [c.chunk_id for c in chunks if not c.source_file]
    if missing:
        raise ValueError(f"FAILED INGEST: {len(missing)} chunk(s) without source_file: {missing}")
    return chunks


# --------------------------------------------------------------------------
# Strategy 1 — the existing (naive) chunker: fixed-size window + overlap
# --------------------------------------------------------------------------

def _split_windows(text: str, window: int, overlap: int) -> list[str]:
    if len(text) <= window:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + window, len(text))
        # snap the end back to a line boundary when possible
        if end < len(text):
            nl = text.rfind("\n", start, end)
            sp = text.rfind(" ", start, end)
            cut = max(nl, sp)
            if cut > start + window // 2:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [t for t in out if t]


def chunk_naive(cards: list[Card], window: int = 200, overlap: int = 40) -> list[Chunk]:
    """Current production chunker: blind fixed-size windows over each card.

    It does not know what a table is. The only thing that saves a table row
    from losing its recipe identity is the metadata stamped below — which is
    exactly the failure mode Set B asks us to measure.
    """
    chunks: list[Chunk] = []
    for card in cards:
        # title + sections flattened in reading order, same as the old pipeline
        body = f"# {card.title}\n"
        for name in SECTION_ORDER:
            if name in card.sections:
                body += f"\n## {name}\n{card.sections[name]}\n"
        for i, text in enumerate(_split_windows(body, window, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{card.recipe_id}-naive-{i:03d}",
                text=text,
                source_file=card.path.name,
                recipe_id=card.recipe_id,
                cuisine=card.cuisine,
                dietary_tags=card.dietary_tags,
                section=_section_of(body, text) or "mixed",
                strategy="naive",
            ))
    return ingest(chunks)


def _section_of(body: str, text: str) -> str | None:
    """Best-effort section label for a naive window (informational only)."""
    pos = body.find(text)
    if pos < 0:
        return None
    prefix = body[:pos]
    for name in SECTION_ORDER:
        if f"## {name}" in prefix:
            return name
    return "Title"


# --------------------------------------------------------------------------
# Strategy 2 — structure-aware chunker
# --------------------------------------------------------------------------

def _table_rows(section_text: str) -> list[str]:
    out = []
    for ln in section_text.splitlines():
        if not ln.strip().startswith("|"):
            continue
        # skip the markdown delimiter row (|---|---|---|)
        if re.fullmatch(r"\|?[\s:|-]+\|?", ln.strip()):
            continue
        out.append(ln)
    return out


def chunk_structured(cards: list[Card]) -> list[Chunk]:
    """Structure-aware chunker.

    Invariant (Set B requirement 3): an ingredient row is never separated
    from its table header or its parent recipe title. Each row chunk is:

        <recipe title>
        | Ingredient | Weight | Baker's % |
        <the row itself>

    Method steps and the allergen note are kept whole. A full-table chunk is
    also emitted so 'whole recipe' questions still match.
    """
    chunks: list[Chunk] = []
    for card in cards:
        meta = f"recipe_id {card.recipe_id} · cuisine {card.cuisine} · tags {card.dietary_tags}"

        # title chunk
        chunks.append(Chunk(
            chunk_id=f"{card.recipe_id}-title",
            text=f"{card.title}\n{meta}",
            source_file=card.path.name, recipe_id=card.recipe_id,
            cuisine=card.cuisine, dietary_tags=card.dietary_tags,
            section="Title", strategy="structured",
        ))

        # ingredient rows — every row carries table header + parent title
        rows = _table_rows(card.sections.get("Ingredients", ""))
        header_rows = [r for r in rows if "Ingredient" in r]
        data_rows = [r for r in rows if "Ingredient" not in r]
        for n, row in enumerate(data_rows):
            chunks.append(Chunk(
                chunk_id=f"{card.recipe_id}-ing-{n:02d}",
                text=f"{card.title}\n{TABLE_HEADER}\n{row}",
                source_file=card.path.name, recipe_id=card.recipe_id,
                cuisine=card.cuisine, dietary_tags=card.dietary_tags,
                section="Ingredients", strategy="structured",
            ))
        if rows:
            full = f"{card.title}\n{TABLE_HEADER}\n" + "\n".join(data_rows)
            chunks.append(Chunk(
                chunk_id=f"{card.recipe_id}-ing-full",
                text=full,
                source_file=card.path.name, recipe_id=card.recipe_id,
                cuisine=card.cuisine, dietary_tags=card.dietary_tags,
                section="Ingredients", strategy="structured",
            ))

        # method steps — one chunk per step, header kept
        for n, step in enumerate(_numbered_steps(card.sections.get("Method", ""))):
            chunks.append(Chunk(
                chunk_id=f"{card.recipe_id}-method-{n:02d}",
                text=f"{card.title}\n## Method\n{step}",
                source_file=card.path.name, recipe_id=card.recipe_id,
                cuisine=card.cuisine, dietary_tags=card.dietary_tags,
                section="Method", strategy="structured",
            ))

        # allergen note
        if card.sections.get("Allergen note"):
            chunks.append(Chunk(
                chunk_id=f"{card.recipe_id}-allergen",
                text=f"{card.title}\n## Allergen note\n{card.sections['Allergen note']}",
                source_file=card.path.name, recipe_id=card.recipe_id,
                cuisine=card.cuisine, dietary_tags=card.dietary_tags,
                section="Allergen note", strategy="structured",
            ))

    return ingest(chunks)


def _numbered_steps(text: str) -> list[str]:
    if not text:
        return []
    steps = re.split(r"(?m)^\d+\.\s+", text)
    return [s.strip() for s in steps if s.strip()]
