"""Offline unit tests for document processing and retrieval helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from rag_app.rag import BM25, _rrf_scores, read_recipe_chunks


class RAGUnitTests(unittest.TestCase):
    def test_recipe_cards_create_expected_section_chunks(self) -> None:
        root = Path(__file__).resolve().parents[1]
        chunks = read_recipe_chunks(root / "data" / "cards")
        self.assertEqual(18, len(chunks))
        self.assertEqual("R001_country_sourdough-1", chunks[0].id)

    def test_bm25_scores_each_document(self) -> None:
        bm25 = BM25(["xanthan gum", "pizza dough", "xanthan gum brioche"])
        scores = bm25.scores("xanthan gum")
        self.assertEqual(3, len(scores))
        self.assertGreater(scores[0], scores[1])

    def test_rrf_returns_one_score_per_document(self) -> None:
        fused = _rrf_scores([0.9, 0.2, 0.5], [0.2, 0.8, 0.3])
        self.assertEqual(3, len(fused))
        self.assertTrue(all(score > 0 for score in fused))


if __name__ == "__main__":
    unittest.main()
