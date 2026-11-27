"""Synthetic actions exercise the loop without a model, filing, MCP process or API bill."""
import json
import unittest

from run_agent import account_for_calls, parse_action, run_dialogue
from run_baselines import configuration


QUESTION = {"question": "What was the difference between sales and costs?", "doc_name": "TEST_10K"}
CHUNK = {"doc_name": "TEST_10K", "pdf_page": 3, "chunk_id": "TEST_10K:p3:c1",
         "text": "Sales 120\nCosts 30", "rank": 1, "score": 3.0}
ANSWER = {"answer": {"status": "answered", "explanation": "Sales less costs.", "citations": [],
                     "value": 90, "unit": "USD"}}


def initial_trace():
    return {"steps": [], "retrieved_chunks": [], "model_calls": 0, "estimated_api_usd": "unknown", "usage": None}


def scripted_model(replies):
    responses = iter(replies)

    def complete(config, settings, messages):
        call = {"content": json.dumps(next(responses))}
        return {"choices": [{"message": call, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    return complete


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_tools_and_the_exact_visible_evidence_are_archived(self):
        trace = initial_trace()
        responses = [
            {"action": {"name": "search_filings", "query": "sales costs"}},
            {"action": {"name": "calculate", "operation": "subtract", "operands": ["120", "30"]}},
            {"action": {"name": "check_citation", "chunk_id": CHUNK["chunk_id"], "figure": "120"}},
            {"action": {"name": "finish"}}, ANSWER,
        ]
        tool_replies = {"search_filings": {"chunks": [CHUNK]}, "calculate": {"result": "90"},
                        "check_citation": {"visible": True, "number_found": True}}
        called = []

        async def call_tool(name, arguments):
            called.append((name, arguments))
            return tool_replies[name]

        await run_dialogue(trace, QUESTION, configuration(False, True), call_tool, scripted_model(responses))
        account_for_calls(trace)
        self.assertEqual([step[0] for step in called], list(tool_replies))
        self.assertEqual(called[1][1], {"operation": "subtract", "operands": ["120", "30"]})
        self.assertEqual(trace["model_calls"], 5)
        self.assertEqual(trace["usage"]["total_tokens"], 60)
        self.assertEqual(trace["estimated_api_usd"], "0")
        self.assertEqual(trace["end_reason"], "finish")
        self.assertEqual(trace["retrieved_chunks"], [CHUNK])
        prior_request = trace["steps"][0]["messages"]
        self.assertFalse(any(CHUNK["text"] in m["content"] for m in prior_request))
        visible = f"[{CHUNK['chunk_id']} | PDF page 3]\n{CHUNK['text']}"
        self.assertTrue(any(visible in m["content"] for m in trace["messages"] if m["role"] == "user"))

    async def test_budget_exhaustion_can_still_produce_a_model_answer(self):
        trace = initial_trace()
        action = {"action": {"name": "search_filings", "query": "sales"}}

        async def call_tool(name, arguments):
            return {"chunks": [CHUNK]}

        await run_dialogue(trace, QUESTION, configuration(False, True), call_tool,
                           scripted_model([action, action, ANSWER]), limit=2)
        self.assertEqual(trace["end_reason"], "tool_limit")
        self.assertEqual(trace["value"], 90)
        self.assertEqual(trace["model_calls"], 3)
        self.assertEqual(trace["retrieved_chunks"], [CHUNK])
        self.assertEqual(trace["steps"][-1]["phase"], "answer")

    def test_unreported_call_keeps_aggregate_unknown(self):
        trace = initial_trace()
        trace["steps"] = [{"usage": {"prompt_tokens": 5, "completion_tokens": 1}, "estimated_api_usd": "0.1"}, {}]
        account_for_calls(trace)
        self.assertIsNone(trace["usage"])
        self.assertEqual(trace["estimated_api_usd"], "unknown")

    def test_unknown_model_action_is_a_parse_failure(self):
        with self.assertRaises(ValueError):
            parse_action('{"action":{"name":"estimate"}}')
