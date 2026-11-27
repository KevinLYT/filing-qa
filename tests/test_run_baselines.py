"""No model calls or paid requests: verify experiment input boundaries and accounting."""
import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import run_baselines as runner


class PairedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.question = {"question_id": "mock_id", "question": "What were sales?", "doc_name": "mock_filing",
                         "gold_should_never_be_sent": "secret annotation"}
        self.config = runner.configuration(hosted=False, needs_key=False)
        self.settings = runner.generation_settings(self.config[1])

    def test_only_b1_receives_excerpts(self):
        hit = {"chunk_id": "mock_filing:p12:c1", "doc_name": "mock_filing", "pdf_page": 12,
               "score": 12.3, "text": "FY2022 sales: 24,365."}
        with patch("run_baselines.read_rows", return_value=[hit]) as read:
            b0 = runner.build_input(self.question, "B0")
            read.assert_not_called()
            with patch("run_baselines.retrieve", return_value=[hit]):
                b1 = runner.build_input(self.question, "B1")
        self.assertEqual(b0["messages"][0], b1["messages"][0])
        self.assertEqual(b0["messages"][1]["content"], self.question["question"])
        self.assertIn("[mock_filing:p12:c1 | PDF page 12]\nFY2022 sales: 24,365.", b1["messages"][1]["content"])
        self.assertNotIn("secret annotation", json.dumps(b1["messages"]))
        self.assertEqual(b1["retrieved_chunks"][0]["rank"], 1)

    def test_wire_schema_constrains_refusal_and_unwraps_answer(self):
        schema = self.settings["response_format"]["json_schema"]["schema"]
        answered, refusal = schema["properties"]["answer"]["anyOf"]
        for branch in (answered, refusal):
            self.assertEqual(list(branch["properties"])[:2], ["explanation", "status"])
            self.assertEqual(branch["properties"]["explanation"]["minLength"], 1)
        self.assertIn('"answered"', runner.SYSTEM_PROMPT)
        self.assertIn('"refusal"', runner.SYSTEM_PROMPT)
        for unit in runner.UNITS:
            self.assertIn(unit, runner.SYSTEM_PROMPT)
        self.assertEqual(answered["properties"]["unit"]["enum"], runner.UNITS)
        self.assertEqual(refusal["properties"]["unit"]["type"], "null")
        self.assertEqual(refusal["properties"]["value"]["type"], "null")
        self.assertEqual(refusal["properties"]["citations"]["maxItems"], 0)
        result = {"status": "refusal", "explanation": "Mock: insufficient evidence.",
                  "value": None, "unit": None, "citations": []}
        self.assertEqual(runner.parse_answer(json.dumps({"answer": result})), result)
        result["unit"] = "USD"
        with self.assertRaises(ValueError):
            runner.parse_answer(json.dumps({"answer": result}))

    @patch("run_baselines.build_opener")
    def test_api_error_vs_malformed_completion(self, opener):
        opener.return_value.open.side_effect = HTTPError("mock", 429, "mock", None, None)
        trace = runner.run_one(self.question, "B0", self.config, self.settings, export=False)
        self.assertEqual((trace["status"], trace["model_calls"], trace["http_status"]), ("api_error", 1, 429))
        opener.return_value.open.side_effect = None
        body = {"choices": [{"message": {"content": "mock malformed reply"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
        opener.return_value.open.return_value = BytesIO(json.dumps(body).encode())
        trace = runner.run_one(self.question, "B0", self.config, self.settings, export=False)
        self.assertEqual(trace["status"], "parse_error")
        self.assertEqual(trace["raw_response"], "mock malformed reply")
        self.assertEqual(trace["usage"], body["usage"])

    def test_cached_input_and_unknown_cost(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 20}}
        rates = {"input": "0.5", "output": "2", "cached_input": "0.1", "per_tokens": 1_000_000}
        self.assertEqual(runner.estimate_cost(usage, rates), "0.000062")
        self.assertEqual(runner.estimate_cost(None, self.config[3]), "unknown")
        del rates["cached_input"]
        self.assertEqual(runner.estimate_cost(usage, rates), "unknown")


if __name__ == "__main__":
    unittest.main()
