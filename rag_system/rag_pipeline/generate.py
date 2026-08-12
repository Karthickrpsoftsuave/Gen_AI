"""Grounded generation with citations, and FORCED refusal (Set B requirement 5).

No API key is available on this machine, so generation is deterministic
extractive grounding: the answer sentence must be pulled verbatim from a
retrieved chunk, and every claim carries a citation that resolves to that
chunk's chunk_id + recipe.

The refusal is forced, not suggested. There is no 'if the context is
insufficient, use your best judgement' escape hatch anywhere in the pipeline
(Set B, section 7: that sentence is how an invented gram weight ends up in a
reader's dough). If no sentence clears the evidence gate, the generator is
structurally incapable of emitting an answer — it can only refuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_pipeline.chunker import Chunk
from rag_pipeline.store import VectorStore

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "at", "for", "and", "or", "does", "do", "what", "which", "how", "much",
    "many", "can", "it", "its", "this", "that", "with", "from", "by", "be",
    "use", "uses", "must", "before", "goes", "reach", "need", "out", "per",
    "each", "then", "over", "under", "while", "until", "into", "about",
    "no", "not", "your",
}

# Synonym expansion used ONLY by the evidence scorer, so a question about
# "hydration" can ground on a chunk that says "water". Kept tiny and curated;
# retrieval itself is still pure dense cosine.
_SYNONYMS = {
    "hydration": ["water"],
    "water": ["hydration"],
    "loaf": ["bread"],
    "percent": ["percentage", "%"],
}

_NUMERIC_CUES = ("how much", "how many", "weight", "percentage", "percent",
                 "hydration", "ph", "baker's", "temperature", "long", "heat")


@dataclass
class Claim:
    text: str
    chunk_id: str
    recipe_id: str
    source_file: str
    section: str
    evidence: float

    def to_dict(self) -> dict:
        return {
            "claim": self.text,
            "citation_chunk_id": self.chunk_id,
            "citation_recipe": self.recipe_id,
            "citation_source_file": self.source_file,
            "citation_section": self.section,
            "evidence": round(self.evidence, 4),
        }


@dataclass
class GenerationResult:
    question: str
    refused: bool
    answer: str | None
    claims: list[Claim]
    evidence: float
    gate: float
    retrieved: list[tuple[Chunk, float]]

    def transcript(self) -> str:
        lines = [f"> Q: {self.question}"]
        if self.refused:
            lines.append(
                "A: I don't know. The answer is not in the indexed documents, "
                "and I won't guess."
            )
            lines.append(
                f"   [grounding gate: best evidence {self.evidence:.3f} < "
                f"threshold {self.gate:.2f} -> refusal forced]"
            )
            if self.retrieved:
                top = self.retrieved[0][0]
                lines.append(
                    f"   [nearest material found: '{top.text[:120]}' "
                    f"({top.recipe_id})]"
                )
        else:
            lines.append(f"A: {self.answer}")
            for c in self.claims:
                lines.append(
                    f"   [{c.evidence:.3f}] {c.text}  "
                    f"(chunk {c.chunk_id} · {c.recipe_id} · {c.source_file} · "
                    f"section {c.section})"
                )
        return "\n".join(lines)


class GroundedGenerator:
    """Extractive, citation-forced generator.

    Pipeline:
      1. retrieve top-K chunks (dense cosine)
      2. split chunks into sentences (table rows count as sentences)
      3. score every sentence by lexical overlap with the question + retrieval
         weight + numeric bonus; best sentence = the claim
      4. if the best evidence score is below the gate -> REFUSE (forced)
    """

    def __init__(self, store: VectorStore, top_k: int = 5, gate: float = 0.42):
        self.store = store
        self.top_k = top_k
        self.gate = gate

    # ------------------------------------------------------------------ public
    def answer(self, question: str, tag_filter: str | None = None) -> GenerationResult:
        hits = self.store.search(question, top_k=self.top_k, tag_filter=tag_filter)
        candidates = []  # (sentence, chunk, chunk_score, evidence)
        for chunk, score in hits:
            first_line = chunk.text.splitlines()[0].strip()
            for sent in self._sentences(chunk.text):
                if not self._is_claim_candidate(sent, first_line):
                    continue
                ev = self._evidence(question, sent, chunk.text, score)
                candidates.append((sent, chunk, score, ev))

        if not candidates:
            return GenerationResult(question, True, None, [], 0.0, self.gate, hits)

        best = max(candidates, key=lambda t: t[3])
        _, best_chunk, _, best_ev = best

        if best_ev < self.gate:
            return GenerationResult(question, True, None, [], best_ev, self.gate, hits)

        # every eligible sentence that clears the gate becomes a cited claim (max 3)
        claims = []
        for sent, chunk, score, ev in sorted(candidates, key=lambda t: -t[3]):
            if ev < self.gate or len(claims) >= 3:
                continue
            if any(sent == c.text for c in claims):
                continue
            claims.append(Claim(
                text=sent, chunk_id=chunk.chunk_id, recipe_id=chunk.recipe_id,
                source_file=chunk.source_file, section=chunk.section, evidence=ev,
            ))

        answer = " ".join(c.text for c in claims)
        return GenerationResult(question, False, answer, claims, best_ev, self.gate, hits)

    # ----------------------------------------------------------------- internal
    @staticmethod
    def _is_claim_candidate(sent: str, first_line: str) -> bool:
        """A sentence can be a claim only if it is not the chunk's own title
        line, is not a bare step number, and carries >=3 significant tokens or
        at least one digit. A 2-word title like 'Baechu Kimchi' is context,
        not an answer claim."""
        if sent == first_line:
            return False
        if re.fullmatch(r"\d+\.?", sent.strip()):
            return False
        sig = {t for t in re.findall(r"[a-z0-9'%]+", sent.lower())} - _STOP
        if len(sig) < 3 and not re.search(r"\d", sent):
            return False
        return True
    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split prose on sentence boundaries; keep table rows intact."""
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("|"):
                out.append(line)
            else:
                out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
        return out

    def _evidence(self, question: str, sentence: str, chunk_text: str, chunk_score: float) -> float:
        q_tokens = {t.lower() for t in re.findall(r"[a-z0-9'%]+", question.lower())} - _STOP
        s_tokens = {t.lower() for t in re.findall(r"[a-z0-9'%]+", sentence.lower())} - _STOP
        c_tokens = {t.lower() for t in re.findall(r"[a-z0-9'%]+", chunk_text.lower())} - _STOP
        if not q_tokens:
            return chunk_score

        q_expanded = set(q_tokens)
        for t in q_tokens:
            q_expanded.update(_SYNONYMS.get(t, []))

        # sentence-level and chunk-level lexical fit
        sent_overlap = len(q_expanded & s_tokens) / len(q_tokens)
        chunk_overlap = len(q_expanded & c_tokens) / len(q_tokens)

        numeric_bonus = 0.0
        if any(cue in question.lower() for cue in _NUMERIC_CUES) and re.search(r"\d", sentence):
            numeric_bonus = 0.18

        # evidence = retrieval weight + sentence lexical fit + chunk lexical fit + numeric bonus
        return (
            chunk_score * 0.30
            + sent_overlap * 0.45
            + chunk_overlap * 0.25
            + numeric_bonus
        )
