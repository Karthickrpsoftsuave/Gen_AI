"""Recipe-card parsing and the two Week 3 chunking strategies."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

TABLE_HEADER = "| Ingredient | Weight | Baker's % |"


@dataclass
class Card:
    path: Path
    recipe_id: str
    cuisine: str
    dietary_tags: str
    title: str
    sections: dict[str, str]
    raw: str


@dataclass
class Chunk:
    """A retrievable text unit with the Week 3 metadata contract."""

    id: str
    text: str
    source_file: str
    recipe_id: str
    cuisine: str
    dietary_tags: str
    section: str
    strategy: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_cards(documents_dir: Path) -> list[Card]:
    """Load exactly the supplied markdown recipe cards."""
    cards = [parse_card(path) for path in sorted(documents_dir.glob("*.md"))]
    if not cards:
        raise ValueError(f"No markdown recipe cards found in {documents_dir}")
    return cards


def parse_card(path: Path) -> Card:
    raw = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(raw)
    title = next((line[2:].strip() for line in body.splitlines() if line.startswith("# ")), path.stem)
    sections: dict[str, str] = {}
    current_section: str | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = ""
        elif current_section is not None:
            sections[current_section] += f"{line}\n"
    return Card(
        path=path,
        recipe_id=front_matter.get("recipe_id", path.stem),
        cuisine=front_matter.get("cuisine", ""),
        dietary_tags=front_matter.get("dietary_tags", ""),
        title=title,
        sections={name: value.strip() for name, value in sections.items()},
        raw=raw,
    )


def build_chunks(cards: list[Card], strategy: str) -> list[Chunk]:
    """Build and validate a chunk set for the requested strategy.

    Strategies
    ----------
    section    One chunk per markdown section; ID = {file_stem}-{section_index}.
               This is the default production strategy: simple, deterministic,
               and aligned with the Week 4 golden-set chunk IDs.
    naive      Fixed-size sliding windows (200 tokens, 40-token overlap).
               Does not understand table structure.
    structured Per ingredient-row chunks plus per-step method chunks.
               Keeps every row attached to its table header and recipe title.
    """
    if strategy == "section":
        chunks = chunk_section(cards)
    elif strategy == "naive":
        chunks = chunk_naive(cards)
    elif strategy == "structured":
        chunks = chunk_structured(cards)
    else:
        raise ValueError("strategy must be 'section', 'naive', or 'structured'")
    return validate_ingest(chunks)


def chunk_section(cards: list[Card]) -> list[Chunk]:
    """Default strategy: one chunk per markdown section.

    IDs follow the pattern ``{file_stem}-{section_index}`` (1-based), e.g.
    ``R001_country_sourdough-1`` for the Ingredients section of R001.
    This matches the Week 4 golden-set expected_chunk_id values.
    """
    chunks: list[Chunk] = []
    for card in cards:
        for index, (name, text) in enumerate(card.sections.items(), start=1):
            chunk_id = f"{card.path.stem}-{index}"
            full_text = f"# {card.title}\n\n## {name}\n{text}"
            chunks.append(_chunk(card, chunk_id, full_text, name, "section"))
    return chunks


def chunk_naive(cards: list[Card], window: int = 200, overlap: int = 40) -> list[Chunk]:
    """Baseline: fixed-size sliding windows that do not understand tables."""
    chunks: list[Chunk] = []
    for card in cards:
        body = f"# {card.title}\n\n"
        for name, section in card.sections.items():
            body += f"## {name}\n{section}\n\n"
        for index, text in enumerate(_windows(body, window, overlap)):
            chunks.append(_chunk(card, f"{card.recipe_id}-naive-{index:03d}", text, "mixed", "naive"))
    return chunks


def chunk_structured(cards: list[Card]) -> list[Chunk]:
    """Keep every ingredient row with its table header and parent recipe title."""
    chunks: list[Chunk] = []
    for card in cards:
        ingredient_rows = _ingredient_rows(card.sections.get("Ingredients", ""))
        for index, row in enumerate(ingredient_rows):
            text = f"{card.title}\n## Ingredients\n{TABLE_HEADER}\n{row}"
            chunks.append(
                _chunk(card, f"{card.recipe_id}-structured-ing-{index:02d}", text, "Ingredients", "structured")
            )

        for index, step in enumerate(_numbered_steps(card.sections.get("Method", ""))):
            text = f"{card.title}\n## Method\n{step}"
            chunks.append(
                _chunk(card, f"{card.recipe_id}-structured-method-{index:02d}", text, "Method", "structured")
            )

        allergen_note = card.sections.get("Allergen note", "")
        if allergen_note:
            text = f"{card.title}\n## Allergen note\n{allergen_note}"
            chunks.append(
                _chunk(card, f"{card.recipe_id}-structured-allergen", text, "Allergen note", "structured")
            )
    return chunks


def validate_ingest(chunks: list[Chunk]) -> list[Chunk]:
    """Reject an ingest whenever any chunk has no source_file metadata."""
    missing = [chunk.id for chunk in chunks if not chunk.source_file]
    if missing:
        raise ValueError(f"Failed ingest: chunks without source_file: {missing}")
    return chunks


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n", raw, flags=re.DOTALL)
    if not match:
        return {}, raw
    metadata = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata, raw[match.end():]


def _chunk(card: Card, chunk_id: str, text: str, section: str, strategy: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source_file=card.path.name,
        recipe_id=card.recipe_id,
        cuisine=card.cuisine,
        dietary_tags=card.dietary_tags,
        section=section,
        strategy=strategy,
    )


def _ingredient_rows(section: str) -> list[str]:
    rows = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "Ingredient" in stripped or re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
            continue
        rows.append(stripped)
    return rows


def _numbered_steps(section: str) -> list[str]:
    return [step.strip() for step in re.split(r"(?m)^\d+\.\s+", section) if step.strip()]


def _windows(text: str, window: int, overlap: int) -> list[str]:
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        if end < len(text):
            line_break = text.rfind("\n", start, end)
            if line_break > start + window // 2:
                end = line_break
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [window for window in windows if window]
