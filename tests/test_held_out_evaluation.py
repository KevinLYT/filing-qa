"""Check alternative operand paths and held-out reference integrity."""
import json
from pathlib import Path
import unittest

from evaluate import citation_audit, compare_value
from filing_numbers import NUMBER, raw_key

ROOT = Path(__file__).resolve().parents[1]


def evidence_trace(claims):
    chunk_id = "EXAMPLE:p1:c1"
    text = "2019 total assets 100, including cash 40 and property 60. Cash outflow (25)."
    hit = {"chunk_id": chunk_id, "doc_name": "EXAMPLE", "pdf_page": 1, "text": text}
    header = f"[{chunk_id} | PDF page 1]\n"
    citations = [dict(chunk_id=chunk_id, doc_name="EXAMPLE", pdf_page=1, raw_number=n, label="synthetic") for n in claims]
    return {"system": "B1", "status": "answered", "retrieved_chunks": [hit], "citations": citations,
            "messages": [{"role": "user", "content": header + text}]}


class HeldOutEvaluationTests(unittest.TestCase):
    def test_complete_alternative_operand_paths(self):
        reference = {"annotation_status": "verified", "paths": [[{"figures": ["100"]}], [{"figures": ["40", "60"]}]]}
        for inputs, match in [(["100"], 1), (["40", "60"], 2)]:
            result = citation_audit(evidence_trace(inputs), reference)
            self.assertEqual((result["status"], result["matched_reference_paths"]), ("pass", [match]))
        self.assertEqual(citation_audit(evidence_trace(["40"]), reference)["status"], "fail")
        ref = {"annotation_status": "verified", "paths": [[{"figures": ["(25)"], "allow_magnitude": True}]]}
        self.assertEqual(citation_audit(evidence_trace(["25"]), ref)["status"], "fail")
        self.assertEqual(citation_audit(evidence_trace(["(25)"]), ref)["status"], "pass")

    def test_partial_or_absent_reference_never_counts_as_a_pass(self):
        trace = evidence_trace(["100"])
        self.assertEqual(citation_audit(trace, None)["status"], "unassessable")
        self.assertEqual(citation_audit(trace, {"annotation_status": "verified", "paths": [[]]})["status"], "unassessable")
        partial = {"annotation_status": "partial", "paths": [[{"figures": ["100"]}]]}
        self.assertEqual(citation_audit(trace, partial)["status"], "partial")
        self.assertEqual(citation_audit(evidence_trace(["900"]), partial)["literal_status"], "fail")

    @unittest.skipUnless((ROOT / "data/splits.json").is_file(), "Prepare the FinanceBench dataset first")
    def test_source_figures_and_recomputed_values_for_every_held_out_id(self):
        data = ROOT / "data"
        references = json.loads((data / "evaluation/held_out_operands.json").read_text(encoding="utf-8"))["questions"]
        held_out = json.loads((data / "splits.json").read_text(encoding="utf-8"))["held_out"]
        gold = {r["question_id"]: r for r in map(json.loads, (data / "evaluation/gold.jsonl").read_text(encoding="utf-8").splitlines())}
        self.assertEqual(set(references), set(held_out))
        differences = []
        for qid in held_out:
            ref = references[qid]
            pages = {(e["doc_name"], e["evidence_page_num"] + 1): e["evidence_text"] for e in gold[qid]["evidence"]}
            self.assertEqual(ref["annotation_status"], "verified")
            self.assertTrue(ref["paths"])
            for path in ref["paths"]:
                self.assertTrue(path)
                for row in path:
                    evidence = pages[ref["doc_name"], row["pdf_page"]]
                    printed = {raw_key(m.group()) for m in NUMBER.finditer(evidence)}
                    self.assertEqual(len(row["figures"]), len(row["years"]))
                    for number in row["figures"]:
                        self.assertIn(raw_key(number), printed, (qid, row["line_item"], number))
            source_value = {"value": ref["recomputed_value"], "unit": ref["result_unit"]}
            if not compare_value(source_value, gold[qid])[0]:
                differences.append(qid)
        self.assertEqual(differences, ["financebench_id_05718"])
