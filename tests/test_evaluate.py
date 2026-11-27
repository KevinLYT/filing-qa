"""Exercise the offline scoring rules using synthetic predictions."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate import citation_audit, compare_value, score_run


def matches(value, unit, target, target_unit):
    return compare_value({"value": value, "unit": unit}, {"value": target, "unit": target_unit})[0]


class NumericTests(unittest.TestCase):
    def test_currency_scales_and_negative_gold(self):
        self.assertTrue(matches(-1.01, "USD_billions", -1000, "USD_millions"))
        self.assertFalse(matches(-1.010001, "USD_billions", -1000, "USD_millions"))
        self.assertTrue(matches(1000, "USD_thousands", 1000000, "USD"))

    def test_rate_boundary_fraction_and_percentage_points(self):
        self.assertTrue(matches(20.08, "percent", 20, "percent"))
        self.assertTrue(matches(0.201, "fraction", 20, "percent"))
        self.assertFalse(matches(20.100001, "percent", 20, "percent"))
        self.assertFalse(matches(20, "percent", 20, "percentage_points"))
        self.assertTrue(matches(0.1, "percentage_points", 0.2, "percentage_points"))

    def test_zero(self):
        self.assertTrue(matches(0, "USD_billions", 0, "USD"))
        self.assertFalse(matches(0.01, "USD", 0, "USD"))
        self.assertTrue(matches(-0.1, "percent", 0, "percent"))


class CitationTests(unittest.TestCase):
    def setUp(self):
        chunk = {"doc_name": "mockdoc", "pdf_page": 4, "chunk_id": "mock:p4:c1", "text": "Cash (1,025)\nReceivables 4,200"}
        self.cite = {"doc_name": "mockdoc", "pdf_page": 4, "chunk_id": "mock:p4:c1", "label": "mock", "raw_number": "(1,025)"}
        self.trace = {"system": "B1", "status": "answered", "retrieved_chunks": [chunk], "citations": [self.cite],
                      "messages": [{"role": "user", "content": "Question\n[mock:p4:c1 | PDF page 4]\n" + chunk["text"]}]}

    def test_visible_means_sent(self):
        self.assertEqual(citation_audit(self.trace, ["(1,025)"])["status"], "pass")
        self.trace["messages"][0]["content"] = "Question only"
        self.assertFalse(citation_audit(self.trace, [])["checks"][0]["visible"])

    def test_wrong_page_token_and_sign_fail(self):
        self.cite["pdf_page"] = 3
        self.assertEqual(citation_audit(self.trace, [])["status"], "fail")
        self.cite["pdf_page"] = 4
        for token in ("25", "200", "1,025"):
            self.cite["raw_number"] = token
            self.assertFalse(citation_audit(self.trace, [])["checks"][0]["number_found"])

    def test_each_operand_needs_support(self):
        self.assertEqual(citation_audit(self.trace, ["(1,025)", "4,200"])["missing_reference_operands"], ["4,200"])
        self.trace["citations"].append(dict(self.cite, raw_number="4200"))
        self.assertEqual(citation_audit(self.trace, [["(1,025)", "-1025"], "4,200"])["status"], "pass")


class RunTests(unittest.TestCase):
    def test_fixed_denominator_and_preserved_human_review(self):
        with TemporaryDirectory() as work:
            labels = Path(work) / "labels"
            folder = Path(work) / "run"
            labels.mkdir()
            (folder / "B1").mkdir(parents=True)
            ids = ["mock_a", "mock_b", "mock_c", "mock_d", "mock_e"]
            manifest = {"mode": "live", "systems": ["B1"], "question_ids": ids}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            gold = [{"question_id": q, "value": "1", "unit": "USD_millions", "evidence": []} for q in ids]
            (labels / "gold.jsonl").write_text("\n".join(map(json.dumps, gold)))
            (labels / "dev_operands.json").write_text(json.dumps({q: [] for q in ids}))
            for q, status in zip(ids, ["answered", "refusal", "api_error", "parse_error"]):
                record = {"question_id": q, "system": "B1", "status": status, "citations": [],
                          "value": 1, "unit": "USD_millions", "messages": [], "retrieved_chunks": [],
                          "estimated_api_usd": "unknown", "model_calls": 1, "elapsed_seconds": 1}
                (folder / "B1" / (q + ".json")).write_text(json.dumps(record))
            result = score_run(folder, labels)
            summary = result["summary"]["B1"]
            self.assertEqual((summary["accuracy"], summary["n"]), (0.2, 5))
            self.assertEqual([summary[k] for k in ("correct", "refusal", "api_error", "parse_error", "missing_prediction")], [1]*5)
            self.assertEqual(summary["estimated_api_usd"], "unknown")
            self.assertEqual(result["rows"][0]["automatic_citation"]["status"], "fail")
            review = {"B1/mock_a": {"reviewer": "Mock reviewer", "year": True, "line_item": True, "unit": False, "notes": "mock"}}
            (folder / "human_reviews.json").write_text(json.dumps(review))
            updated = score_run(folder, labels)
            self.assertEqual(updated["rows"][0]["human_review"]["status"], "fail")
            self.assertEqual(json.loads((folder / "human_reviews.json").read_text()), review)
            manifest["mode"] = "mock"
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                score_run(folder, labels)


if __name__ == "__main__":
    unittest.main()