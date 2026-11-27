"""Test hybrid search ranking, overlapping windows, and cache validation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import hybrid_search as hs


def chunk(doc, number, text):
    return {"doc_name": doc, "chunk_id": f"{doc}:p{number}:c1", "pdf_page": number, "page_index": number - 1,
            "company": "Demo", "fiscal_year": 2020, "text": text}


class HybridSearchTests(unittest.TestCase):
    def test_window_coverage_includes_middle_and_tail(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "row": 1}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        text = " ".join(["row"] * (3 * hs.WINDOW + 1))
        spans = list(hs.token_windows(text, tokenizer))
        self.assertGreater(len(spans), 3)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        self.assertTrue(all(start < previous_end for (_, previous_end), (start, _) in zip(spans, spans[1:])))

    def test_cross_encoder_reorders_candidates_and_preserves_entire_evidence(self):
        from unittest.mock import Mock

        chunks = [chunk("SAMPLE", 1, "revenue revenue"), chunk("SAMPLE", 2, "company sales 175")]
        spans = np.array([[0, 0, 15], [1, 0, 7], [1, 8, 17]])
        vectors = np.array([[1., 0.], [0., 1.], [0., 1.]])
        search = hs.HybridSearch(chunks, spans, vectors, Path("."))
        dense, cross = Mock(), Mock()
        dense.encode.return_value = np.array([[1., 0.]])
        cross.encode.return_value = np.array([1., 2., 8.])
        with patch.object(hs, "encoder", side_effect=[dense, cross]):
            first, second = search.search("revenue", 2)
        self.assertEqual((first["chunk_id"], first["text"]), (chunks[1]["chunk_id"], chunks[1]["text"]))
        self.assertEqual((first["rank"], first["rrf_rank"], first["rerank_score"]), (1, 2, 8.))
        self.assertEqual(second["bm25_rank"], 1)
        self.assertIsNone(first["bm25_rank"])
        cross.encode.assert_called_once_with([("revenue", "revenue revenue"), ("revenue", "company"), ("revenue", "sales 175")])

    def test_stale_or_wrong_filing_cannot_use_index(self):
        chunks = [chunk("SAMPLE", 1, "sales 25")]
        with tempfile.TemporaryDirectory() as tmp, patch.object(hs, "asset_hashes", return_value={"fake_model": "v1"}):
            index_dir = Path(tmp) / "data/index"
            path = hs.cache_path("SAMPLE", index_dir)
            path.parent.mkdir(parents=True)
            np.savez(path, fingerprint=hs.fingerprint(chunks, {"fake_model": "v1"}),
                     spans=np.asarray([[0, 0, 8]]), vectors=np.asarray([[1., 0.]]))
            self.assertEqual(len(hs.HybridSearch.for_document("SAMPLE", index_dir, chunks).chunks), 1)
            with self.assertRaisesRegex(ValueError, "Stale"):
                hs.HybridSearch.for_document("SAMPLE", index_dir, [dict(chunks[0], text="sales 42")])
            with self.assertRaisesRegex(ValueError, "assigned filing"):
                hs.HybridSearch.for_document("OTHER", index_dir, chunks)
            with self.assertRaises(ValueError):
                hs.cache_path("../OTHER", index_dir)


if __name__ == "__main__":
    unittest.main()
