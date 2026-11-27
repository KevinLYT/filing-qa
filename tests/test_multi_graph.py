"""Exercise the graph routes using scripted model responses."""
import json
import unittest

from filing_tools import FilingTools, calculate
from multi_graph import run_multi_dialogue
from run_baselines import configuration


DOC = "TEST_2020_10K"
CHUNK = {"chunk_id": DOC + ":p3:c1", "doc_name": DOC, "pdf_page": 3, "page_index": 2,
         "company": "Test", "fiscal_year": 2020, "text": "2020 (USD millions)\nSales 120\nCost of sales 30"}
QUESTION = {"doc_name": DOC, "question": "What were 2020 sales in USD millions?"}
SEARCH = {"query": "2020 sales"}
ACCEPT = {"reason": "The row and year match.", "approved": True}
REJECT = {"reason": "Review the year column.", "approved": False}


def plan(operation, raw_values):
    operands = [{"label": label, "raw_number": raw, "chunk_id": CHUNK["chunk_id"], "pdf_page": 3}
                for label, raw in raw_values]
    return {"operation": operation, "operands": operands, "source_unit": "USD_millions", "unit": "USD_millions"}


async def exercise(replies, verify=True):
    tools = FilingTools([CHUNK])
    scripted = iter(replies)
    trace = {"steps": [], "model_calls": 0, "retrieved_chunks": []}

    def complete(config, settings, messages):
        content = json.dumps(next(scripted))
        return {"model": config[1], "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6}}

    async def call_tool(name, arguments):
        functions = {"search_filings": tools.search_filings, "calculate": calculate, "check_citation": tools.check_citation}
        return functions[name](**arguments)

    await run_multi_dialogue(trace, QUESTION, configuration(False, True), call_tool, complete, verify)
    return trace


class MultiGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_rejection_sends_feedback_to_retrieval_once(self):
        lookup = plan("lookup", [("Sales, 2020", "120")])
        trace = await exercise([SEARCH, lookup, REJECT, SEARCH, lookup, ACCEPT])
        self.assertEqual(trace["value"], 120)
        self.assertEqual(trace["end_reason"], "verified")
        self.assertEqual([s["attempt"] for s in trace["steps"]], [1, 1, 1, 2, 2, 2])
        self.assertIn(REJECT["reason"], trace["steps"][3]["messages"][1]["content"])
        decisions = [e["decision"] for e in trace["graph_events"] if e["event"] == "route"]
        self.assertEqual(decisions, ["retry", "accept"])

    async def test_two_rejections_end_in_refusal(self):
        lookup = plan("lookup", [("Sales, 2020", "120")])
        trace = await exercise([SEARCH, lookup, REJECT, SEARCH, lookup, REJECT])
        self.assertEqual(trace["status"], "refusal")
        self.assertIsNone(trace["value"])
        self.assertEqual(trace["citations"], [])
        self.assertEqual(trace["model_calls"], 6)

    async def test_full_tool_budget_cannot_approve_an_unchecked_retry(self):
        difference = plan("subtract", [("Sales, 2020", "120"), ("Cost of sales, 2020", "30")])
        trace = await exercise([SEARCH, difference, REJECT, SEARCH, difference])
        self.assertEqual(trace["status"], "refusal")
        self.assertEqual(trace["end_reason"], "tool_limit")
        self.assertEqual(trace["tool_call_count"], 6)
        self.assertIsNone(trace["value"])

    async def test_ablation_keeps_candidate_and_skips_all_verification(self):
        difference = plan("subtract", [("Sales, 2020", "120"), ("Cost of sales, 2020", "30")])
        trace = await exercise([SEARCH, difference], verify=False)
        self.assertEqual(trace["value"], 90)
        self.assertEqual(trace["end_reason"], "verifier_removed")
        self.assertEqual([s["phase"] for s in trace["steps"]], ["Retriever", "Calculator"])
        self.assertEqual([e["name"] for e in trace["graph_events"] if e["event"] == "tool"], ["search_filings", "calculate"])
