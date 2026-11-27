"""Exercise the MCP transport on a public filing without calling a model."""
import importlib.util
import json
import sys
import unittest
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/questions.jsonl"
DOC = "LOCKHEEDMARTIN_2021_10K"


@unittest.skipUnless(importlib.util.find_spec("mcp"), "Install requirements.txt for the MCP transport test")
@unittest.skipUnless((ROOT / "data/index" / f"{DOC}.jsonl").exists() and QUESTIONS.exists(),
                     "Prepare the FinanceBench filing first")
class MCPTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_question_to_verified_subtraction_over_stdio(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        records = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
        question = next(row["question"] for row in records if row["question_id"] == "financebench_id_03031")
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "filing_mcp.py"), "--doc", DOC])
        # The test chooses the arguments, so its result is never a model prediction.
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                listing = await session.list_tools()
                self.assertEqual({t.name for t in listing.tools}, {"search_filings", "calculate", "check_citation"})
                search = await session.call_tool("search_filings", {"query": question})
                self.assertFalse(search.isError)
                hit = search.structuredContent["chunks"][0]
                self.assertEqual(hit["pdf_page"], 68)
                operands = ["19,815", "13,997"]
                for operand in operands:
                    checked = await session.call_tool("check_citation", {"chunk_id": hit["chunk_id"], "figure": operand})
                    self.assertFalse(checked.isError)
                    self.assertTrue(checked.structuredContent["number_found"])
                answer = await session.call_tool("calculate", {"operation": "subtract", "operands": operands})
                self.assertFalse(answer.isError)
                self.assertEqual(answer.structuredContent["result"], "5818")
