"""Test each agent role on synthetic table entries."""
import unittest

from filing_tools import calculate
from multi_roles import calculate_candidate, requested_unit


class ScriptedRoleContext:
    def __init__(self, operation, literals, source, target):
        refs = [{"raw_number": v, "label": "Synthetic entry", "chunk_id": "DOC:p1:c1", "pdf_page": 1}
                for v in literals]
        self.plan = {"operation": operation, "operands": refs, "source_unit": source, "unit": target}
        self.question = {"question": "Read the sample table.", "doc_name": "DOC"}
        self.operations = []

    def excerpts(self):
        return "[DOC:p1:c1 | PDF page 1]\nSynthetic table."

    async def ask(self, role, prompt, content, schema):
        return self.plan

    async def tool(self, name, args):
        if name != "calculate":
            raise AssertionError(name)
        self.operations.append(args["operation"])
        return calculate(**args)


class RoleUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_equivalent_dollar_scale_is_converted_and_distinct_dimension_is_rejected(self):
        ctx = ScriptedRoleContext("lookup", ["25,500,000"], "USD", "USD")
        ctx.question["question"] = "Report in USD millions."
        answer, reason = await calculate_candidate(ctx)
        self.assertEqual((answer["value"], answer["unit"], reason), (25.5, "USD_millions", ""))
        self.assertEqual(ctx.operations, ["divide"])
        ctx = ScriptedRoleContext("lookup", ["25"], "percent", "percent")
        ctx.question["question"] = "Report in percentage points."
        answer, reason = await calculate_candidate(ctx)
        self.assertIsNone(answer)
        self.assertIn("percentage_points", reason)

    def test_requested_percentage_points_are_distinct_from_percent(self):
        self.assertEqual(requested_unit("Give the difference in percentage points."), "percentage_points")
        self.assertEqual(requested_unit("Answer in units of percents."), "percent")
        self.assertEqual(requested_unit("Please express the amount in USD billions."), "USD_billions")

    async def test_multiply_by_percent_or_fraction_through_shared_tool(self):
        for raw, operations in [("12.5%", ["multiply", "divide"]), ("0.125", ["multiply"])]:
            with self.subTest(raw=raw):
                ctx = ScriptedRoleContext("multiply", ["200", raw], "USD_millions", "USD_millions")
                answer, reason = await calculate_candidate(ctx)
                self.assertEqual(reason, "")
                self.assertEqual(answer["value"], 25)
                self.assertEqual(ctx.operations, operations)
                self.assertEqual(answer["citations"][1]["raw_number"], raw)

    async def test_rounded_change_and_lookup_scale_use_mcp(self):
        ctx = ScriptedRoleContext("percent_change", ["4", "3"], "USD_millions", "percent")
        ctx.question["question"] = "Express growth in percent and round to one decimal place."
        answer, _ = await calculate_candidate(ctx)
        self.assertEqual(answer["value"], 33.3)
        self.assertEqual(ctx.operations, ["percent_change"])
        ctx = ScriptedRoleContext("lookup", ["3,250"], "USD_millions", "USD_billions")
        answer, _ = await calculate_candidate(ctx)
        self.assertEqual(answer["value"], 3.25)
        self.assertEqual(ctx.operations, ["divide"])
