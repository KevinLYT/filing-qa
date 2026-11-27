"""Tool contracts on tiny fictitious filings."""
import json
import tempfile
import unittest
from pathlib import Path

from filing_tools import FilingTools, calculate


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {"chunk_id": "DEMO:p2:c1", "doc_name": "DEMO", "pdf_page": 2, "page_index": 1,
                      "company": "Demo Co", "fiscal_year": 2020, "text": "Net loss (1,500); revenue growth 20%."}
        self.tools = FilingTools([self.chunk])

    def test_citation_uses_visible_chunks_and_printed_sign(self):
        chunk_id = self.chunk["chunk_id"]
        self.assertFalse(self.tools.check_citation(chunk_id, "(1,500)")["visible"])
        hit = self.tools.search_filings("net loss")["chunks"][0]
        self.assertEqual(hit["text"], self.chunk["text"])
        self.assertEqual(hit["rank"], 1)
        self.assertTrue(self.tools.check_citation(chunk_id, "-1500")["number_found"])
        self.assertFalse(self.tools.check_citation(chunk_id, "1500")["number_found"])
        self.assertFalse(self.tools.check_citation(chunk_id, "20")["number_found"])
        self.assertTrue(self.tools.check_citation(chunk_id, "20%")["number_found"])
        self.assertFalse(self.tools.check_citation("DEMO:p3:c1", "20%")["visible"])
        self.assertFalse(FilingTools([self.chunk]).check_citation(chunk_id, "20%")["visible"])

    def test_scope_binds_the_requested_document(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "DEMO.jsonl"
            index.write_text(json.dumps(self.chunk), encoding="utf-8")
            self.assertEqual(FilingTools.for_document("DEMO", directory).chunks, [self.chunk])
            with self.assertRaises(ValueError):
                FilingTools.for_document("../DEMO", directory)
            index.write_text(json.dumps(dict(self.chunk, doc_name="OTHER")), encoding="utf-8")
            with self.assertRaises(ValueError):
                FilingTools.for_document("DEMO", directory)

    def test_decimal_operations_and_invalid_operands(self):
        self.assertEqual(calculate("subtract", ["19,815", "13,997"])["result"], "5818")
        self.assertEqual(calculate("percent_change", ["120", "100"])["result"], "20")
        self.assertEqual(calculate("add", ["0.10", "(0.30)"])["result"], "-0.20")
        for operation, args in [("divide", ["1", "0"]), ("add", ["1", "NaN"]), ("add", ["1e9999", "2"]),
                                ("subtract", ["0"]), ("power", ["2", "3"]), ("multiply", ["9" * 97, "1"])]:
            with self.subTest(operation=operation, args=args), self.assertRaises(ValueError):
                calculate(operation, args)
