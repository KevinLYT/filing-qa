"""Check paired multi-agent traces with a small synthetic filing."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate import score_run
from replay import graph_actions, replay


class MultiEvaluationTests(unittest.TestCase):
    def test_same_numeric_rules_and_independent_visibility_check_for_both_variants(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            qid = "mock"
            systems = ["Ours", "NoVerifier"]
            chunk = {"chunk_id": "DOC:p2:c1", "doc_name": "DOC", "pdf_page": 2, "text": "Current assets 7",
                     "rank": 1, "score": 1.0}
            header = "[DOC:p2:c1 | PDF page 2]\n"
            citation = {"chunk_id": "DOC:p2:c1", "doc_name": "DOC", "pdf_page": 2, "label": "Current assets", "raw_number": "7"}
            gold = {"question_id": qid, "unit": "USD", "value": "7", "evidence": [{"doc_name": "DOC", "evidence_page_num": 1}]}
            (folder / "gold.jsonl").write_text(json.dumps(gold), encoding="utf-8")
            (folder / "dev_operands.json").write_text(json.dumps({qid: ["7"]}), encoding="utf-8")
            (folder / "manifest.json").write_text(json.dumps({"mode": "live", "question_ids": [qid], "systems": systems}), encoding="utf-8")
            for system in systems:
                content = header + chunk["text"] if system == "NoVerifier" else "The search completed."
                step = {"phase": "Calculator", "messages": [{"role": "user", "content": content}], "raw_response": "{}"}
                trace = {"system": system, "question_id": qid, "question": "How much?", "status": "answered",
                         "model": "mock", "value": 7, "unit": "USD", "citations": [citation], "steps": [step],
                         "retrieved_chunks": [chunk], "messages": [], "model_calls": 1, "elapsed_seconds": 1,
                         "estimated_api_usd": "unknown", "end_reason": "test", "graph_events": []}
                (folder / system).mkdir()
                (folder / system / f"{qid}.json").write_text(json.dumps(trace), encoding="utf-8")
            first, second = score_run(folder, folder)["rows"]
            self.assertEqual([first["outcome"], second["outcome"]], ["correct", "correct"])
            self.assertEqual([first["automatic_citation"]["status"], second["automatic_citation"]["status"]], ["fail", "pass"])
            for row in [first, second]:
                self.assertEqual(row["gold_page_hits"], [("DOC", 2)])
                self.assertEqual(row["human_review"]["status"], "pending")
            self.assertIn("How much?", replay(folder, qid))
            self.assertIn("NoVerifier | mock | answered", replay(folder, qid))

    def test_replay_shows_checks_and_retry_in_order(self):
        events = [{"event": "node", "attempt": 1, "role": "Verifier"},
                  {"event": "tool", "name": "check_citation", "arguments": {"figure": "7"}, "result": {"number_found": False}},
                  {"event": "model"}, {"event": "route", "decision": "retry", "reason": "wrong year"},
                  {"event": "node", "attempt": 2, "role": "Retriever"}]
        trace = {"steps": [{"raw_response": '{"approved":false}'}], "graph_events": events, "end_reason": "test"}
        output = "\n".join(graph_actions(trace))
        self.assertIn('"approved": false', output)
        self.assertLess(output.index("number_found"), output.index("Route: retry | wrong year"))
        self.assertLess(output.index("Route: retry"), output.index("Attempt 2 | Retriever"))
