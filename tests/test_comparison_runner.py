"""Small synthetic fixtures for split integrity, checkpointing and retrieval scope."""
import json
import tempfile
import unittest
from pathlib import Path

from filing_tools import FilingTools
from run_comparison import DEMO_ID, current_trace, planned_ids, run_lock, store_or_match_manifest


class ComparisonRunnerTests(unittest.TestCase):
    def test_selects_disjoint_split_and_rejects_overlap(self):
        split = {"dev": [DEMO_ID], "held_out": ["synthetic_test"]}
        questions = {qid: {} for qid in (DEMO_ID, "synthetic_test")}
        self.assertEqual(planned_ids("held-out", split, questions), ["synthetic_test"])
        self.assertEqual(planned_ids("dev", split, questions), [DEMO_ID])
        split["held_out"].append(DEMO_ID)
        with self.assertRaises(ValueError):
            planned_ids("held-out", split, questions)

    def test_resume_preserves_failures_and_rejects_conflicting_trace_or_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = {"run_sha256": "run_a", "model": "synthetic", "scope": "held-out"}
            with run_lock(folder):
                store_or_match_manifest(folder, manifest, False)
                with self.assertRaises(RuntimeError):
                    with run_lock(folder):
                        self.fail("Second writer acquired the same run lock")
            with run_lock(folder):
                store_or_match_manifest(folder, manifest, True)
                with self.assertRaises(ValueError):
                    store_or_match_manifest(folder, dict(manifest, scope="dev"), True)
            path = folder / "trace.json"
            self.assertFalse(current_trace(path, "q", "Ours", manifest))
            trace = {"question_id": "q", "system": "Ours", "status": "api_error",
                     "run_sha256": "run_a", "model": "synthetic"}
            path.write_text(json.dumps(trace), encoding="utf-8")
            self.assertTrue(current_trace(path, "q", "Ours", manifest))
            with self.assertRaises(ValueError):
                current_trace(path, "q", "B2", manifest)
            with self.assertRaises(ValueError):
                current_trace(path, "q", "Ours", dict(manifest, run_sha256="run_b"))

    def test_filing_metadata_filters_do_not_expose_a_different_year(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "SAMPLE.jsonl"
            chunk = {"doc_name": "SAMPLE", "company": "Sample Co", "fiscal_year": 2023}
            path.write_text(json.dumps(chunk), encoding="utf-8")
            self.assertEqual(FilingTools.for_document("SAMPLE", name, company="Sample Co", fiscal_year=2023).chunks, [chunk])
            with self.assertRaises(ValueError):
                FilingTools.for_document("SAMPLE", name, fiscal_year=2024)
            with self.assertRaises(ValueError):
                FilingTools.for_document("SAMPLE", name, company="Other Co")
