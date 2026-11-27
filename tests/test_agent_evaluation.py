"""Offline B2 scoring and replay use synthetic records."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate import citation_audit, score_run
from replay import replay


class AgentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.hit = {"chunk_id": "TEST_10K:p2:c1", "doc_name": "TEST_10K", "pdf_page": 2,
                    "text": "Current assets 17\nCurrent liabilities 3", "rank": 1, "score": 3.5}
        question = {"role": "user", "content": "What was net working capital?"}
        excerpt = "[TEST_10K:p2:c1 | PDF page 2]\n" + self.hit["text"]
        seen = [question, {"role": "user", "content": excerpt}]
        citation = {"chunk_id": self.hit["chunk_id"], "doc_name": "TEST_10K", "pdf_page": 2,
                    "raw_number": "17", "label": "Current assets"}
        self.trace = {"system": "B2", "question_id": "mock", "status": "answered", "model": "mock",
                      "value": 14, "unit": "USD", "explanation": "17 minus 3", "elapsed_seconds": 1,
                      "retrieved_chunks": [self.hit], "messages": seen, "model_calls": 3, "estimated_api_usd": "0",
                      "citations": [citation, dict(citation, raw_number="3", label="Current liabilities")],
                      "steps": [{"phase": "decide", "messages": [question],
                                 "tool_call": {"name": "search_filings", "arguments": {"query": "assets"},
                                               "result": {"chunks": [self.hit]}}},
                                {"phase": "decide", "messages": seen,
                                 "tool_call": {"name": "calculate", "arguments": {"operation": "subtract", "operands": ["17", "3"]},
                                               "result": {"operation": "subtract", "operands": ["17", "3"], "result": "14"}}},
                                {"phase": "answer", "messages": seen}], "end_reason": "finish"}

    def test_only_model_visible_search_results_can_support_operands(self):
        self.assertEqual(citation_audit(self.trace, ["17", "3"])["status"], "pass")
        # The server returned the chunk, but the model issued no request after the search.
        self.trace["steps"] = self.trace["steps"][:1]
        self.assertEqual(citation_audit(self.trace, ["17", "3"])["missing_reference_operands"], ["17", "3"])

    def test_agent_score_and_replay_use_the_same_saved_run(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            labels = Path(tmp) / "labels"
            (run / "B2").mkdir(parents=True)
            labels.mkdir()
            manifest = {"mode": "live", "question_ids": ["mock", "missing"], "systems": ["B2"]}
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "B2/mock.json").write_text(json.dumps(self.trace))
            gold = {"question_id": "mock", "value": "14", "unit": "USD",
                    "evidence": [{"doc_name": "TEST_10K", "evidence_page_num": 1}]}
            (labels / "gold.jsonl").write_text(json.dumps(gold) + "\n" + json.dumps(dict(gold, question_id="missing")))
            (labels / "dev_operands.json").write_text(json.dumps({"mock": ["17", "3"], "missing": []}))
            result = score_run(run, labels)
            summary = result["summary"]["B2"]
            self.assertEqual((summary["n"], summary["accuracy"], summary["missing_prediction"]), (2, 0.5, 1))
            row = result["rows"][0]
            self.assertEqual(row["automatic_citation"]["status"], "pass")
            self.assertEqual(row["human_review"]["status"], "pending")
            self.assertEqual(row["gold_page_hits"], [("TEST_10K", 2)])
            display = replay(run, "mock")
            self.assertIn("What was net working capital?", display)
            self.assertLess(display.index("1. search_filings"), display.index("2. calculate"))
            self.assertIn("final answer request", display)
            self.assertIn("prediction file missing", replay(run, "missing"))
