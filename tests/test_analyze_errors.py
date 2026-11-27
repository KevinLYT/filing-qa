"""Diagnostic labels must be grounded in archived evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from analyze_errors import analyze, diagnose, reference_coverage


def synthetic_score(system, outcome="incorrect"):
    return {"system": system, "question_id": "q1", "outcome": outcome,
            "gold_pdf_pages": [["SAMPLE", 12]], "gold_page_hits": [],
            "numeric_comparison": None, "automatic_citation": {"status": "fail", "checks": []}}


class ErrorAnalysisTests(unittest.TestCase):
    def test_alternative_page_contains_reference_literals_despite_gold_page_miss(self):
        excerpt = {"doc_name": "SAMPLE", "pdf_page": 3, "chunk_id": "SAMPLE:p3:c1", "text": "Outflows (17)"}
        block = "[SAMPLE:p3:c1 | PDF page 3]\nOutflows (17)"
        trace = {"system": "B1", "retrieved_chunks": [excerpt], "messages": [{"role":"user","content":block}]}
        row = diagnose(synthetic_score("B1", "correct"), trace, [["(17)", "17"]])
        self.assertEqual(row["reference_page_signal"], "annotated_page_not_returned")
        self.assertEqual(row["reference_literals_in_visible_text"], "complete_path_present")
        self.assertIn("final_citation_failed", row["automatic_signals"])
        self.assertEqual(row["manual_primary_cause"], "")
        trace["messages"] = []
        invisible = diagnose(synthetic_score("B1"), trace, [["(17)", "17"]])
        self.assertEqual(invisible["reference_literals_in_visible_text"], "no_complete_annotated_path")
        self.assertEqual(reference_coverage({"annotation_status": "partial"}, set()), "unassessed")

    def test_bad_prediction_does_not_imply_bad_arithmetic(self):
        calc = {"name": "calculate", "arguments":{"operation":"subtract","operands":["17","2"]},
                "result":{"result":"15"}}
        trace = {"system":"B2","status":"answered","value":42,"steps":[{"tool_call":calc}]}
        row = diagnose(synthetic_score("B2"), trace)
        self.assertEqual((row["arithmetic_recomputed"],row["arithmetic_mismatches"]), (1,0))
        self.assertEqual(row["prediction_matches_any_raw_tool_result"], "no")
        calc["result"]["result"] = "999"
        self.assertIn("tool_arithmetic_mismatch", diagnose(synthetic_score("B2"),trace)["automatic_signals"])

    def test_missing_predictions_are_included_even_when_no_annotation_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/"manifest.json").write_text(json.dumps({"mode":"live","scope":"dev","systems":["B1"],"question_ids":["q1"]}))
            row = synthetic_score("B1","missing_prediction")
            (base/"scores.json").write_text(json.dumps({"rows":[row]}))
            rows = analyze(base, reference_dir=base)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["reference_page_signal"], "unknown")
            self.assertEqual(rows[0]["execution_issue"], "missing_prediction")


if __name__ == "__main__":
    unittest.main()
