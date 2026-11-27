"""Classify observable outcomes and record supporting evidence from a scored held-out run."""
import argparse
import csv
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from analyze_errors import tool_calls
from evaluate import normalize, read_json


def by_key(rows, expected, source):
    indexed = {(r["system"], r["question_id"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError(f"Planned system/question keys disagree with {source}")
    return indexed


def get_ai_evidence(run, run_hash, scores):
    path = run / "ai_review_suggestions.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    if payload["run_sha256"] != run_hash:
        raise ValueError("The draft semantic annotations belong to a different run")
    expected = {key for key, row in scores.items()
                if row["system"] != "B0" and row["outcome"] in ("correct", "incorrect")}
    items = []
    for row in payload["records"]:
        system, qid = row["key"].split("/")
        if not all(row[k] is None or type(row[k]) is bool for k in ("year", "line_item", "unit")):
            raise ValueError(f"Unexpected draft field in {row['key']}")
        items.append(dict(row, system=system, question_id=qid))
    return by_key(items, expected, path)


def observed_disposition(score, trace, ai):
    """Assign one observed endpoint to each prediction. Several causes may be consistent with it."""
    outcome, system = score["outcome"], score["system"]
    if outcome == "correct":
        return "numeric_correct"
    if outcome == "parse_error":
        final_step = (trace.get("steps") or [{}])[-1]
        finish = trace.get("finish_reason") or final_step.get("finish_reason")
        return "parse_failure_at_length_limit" if finish == "length" else "parse_or_schema_failure"
    if outcome == "refusal":
        if system == "NoVerifier" and trace.get("end_reason") == "no_candidate":
            return "refused_no_candidate"
        if trace.get("end_reason") in ("tool_limit", "model_limit"):
            return "refused_at_budget"
        if system == "Ours" and trace.get("end_reason") == "verification_failed":
            return "refused_after_retries"
        return "model_declared_refusal"
    if outcome != "incorrect":
        return outcome
    if system == "B0":
        return "incorrect_without_filing_context"
    ref = score.get("reference_recalculation")
    if ref and ref["within_official_tolerance"] is False:
        if normalize(trace["value"], trace["unit"]) == normalize(ref["value"], ref["unit"]):
            return "matches_filing_recalculation_only"
    if ai and any(ai[field] is False for field in ("year", "line_item", "unit")):
        return "incorrect_with_semantic_counterexample"
    return "incorrect_source_unresolved"


def documented_calculation_case(trace, ref):
    """Verify three selected calculations against the saved traces."""
    key = (trace.get("system"), trace.get("question_id"))
    if key == ("B2", "financebench_id_06741"):
        claim = trace.get("explanation", "")
        cited_rates = ("2.8", "4.0", "4.4")
        if "average" in claim and all(rate + "%" in claim for rate in cited_rates):
            mean = sum(map(Decimal, cited_rates)) / 3
            if trace["unit"] == "percent" and trace["value"] == 3.4 and abs(mean - Decimal("3.4")) > Decimal("0.1"):
                return "response_arithmetic", "The stated mean of 2.8%, 4.0% and 4.4% is 3.73%. The model submitted 3.4%."
    if key == ("B2", "financebench_id_09724") and "value" in ref:
        calculations = [c for c in tool_calls(trace) if c["name"] == "calculate"]
        if calculations:
            args = calculations[-1]["arguments"]
            supplied = args.get("operands", [])
            if args.get("operation") == "add" and len(supplied) == 2 and supplied[-1] == "100" and trace.get("unit") == "percent":
                try:
                    value = Decimal(supplied[0])
                except ArithmeticError:
                    return "", ""
                if abs(value - Decimal(ref["value"])) <= Decimal("0.1") and abs(Decimal(str(trace["value"])) - value - 100) < Decimal("0.00001"):
                    return "formula_selection", "The correct margin was already 39.728%. The final tool instruction added 100."
    if key == ("NoVerifier", "financebench_id_07966"):
        for call in tool_calls(trace):
            if call["name"] == "calculate" and call["arguments"] == {"operation": "percent_change", "operands": ["-1.0", "-1.0"]}:
                return "formula_selection", "A percentage change of two equal figures cannot calculate the three-year mean of capex/revenue."
    return "", ""


def evidence_flags(diag, ai):
    flags = []
    if diag["reference_literals_in_visible_text"] == "no_complete_annotated_path":
        flags.append("retrieval_coverage_gap")
    if diag["citation_status"] == "fail":
        flags.append("final_citation_failed")
    if int(diag["runtime_checks_failed"]):
        flags.append("runtime_citation_failed")
    if int(diag["calculator_call_failures"]):
        flags.append("calculator_rejected_input")
    if int(diag["arithmetic_mismatches"]):
        flags.append("tool_arithmetic_mismatch")
    if "plan_unit_validation_failed" in diag["automatic_signals"].split(" | "):
        flags.append("plan_unit_validation_failed")
    if ai:
        flags.extend("draft_" + field + "_counterexample" for field in ("year", "line_item", "unit")
                     if ai[field] is False)
    return flags


def classify(run):
    manifest = read_json(run / "manifest.json")
    if manifest["mode"] != "live" or manifest["scope"] != "held-out":
        raise ValueError("Choose a scored, saved held-out run")
    expected = [(s, q) for s in manifest["systems"] for q in manifest["question_ids"]]
    scores = by_key(read_json(run / "scores.json")["rows"], expected, "scores.json")
    with (run / "error_analysis.csv").open(encoding="utf-8-sig", newline="") as handle:
        diagnostics = by_key(list(csv.DictReader(handle)), expected, "error_analysis.csv")
    draft = get_ai_evidence(run, manifest["run_sha256"], scores)
    for system, qid in expected:
        score = scores[system, qid]
        diag = diagnostics[system, qid]
        if score["outcome"] != diag["outcome"] or score["automatic_citation"]["status"] != diag["citation_status"]:
            raise ValueError(f"Diagnostics need to be regenerated for {system}/{qid}")
        trace = {}
        path = run / system / (qid + ".json")
        if path.is_file():
            try:
                trace = read_json(path)
            except (UnicodeError, json.JSONDecodeError):
                if score["outcome"] != "parse_error":
                    raise
            if trace and (trace["system"], trace["question_id"], trace["run_sha256"]) != (system, qid, manifest["run_sha256"]):
                raise ValueError(f"Mismatched trace for {system}/{qid}")
        elif score["outcome"] != "missing_prediction":
            raise ValueError(f"Trace missing for {system}/{qid}")
        ai = draft.get((system, qid))
        disposition = observed_disposition(score, trace, ai)
        calc_tag, calc_note = documented_calculation_case(trace, score.get("reference_recalculation") or {})
        flags = evidence_flags(diag, ai)
        if calc_tag:
            flags.append(calc_tag)
        # Record routing, independent arithmetic checks, and later assessments separately so the source of each label is clear.
        yield {"system": system, "question_id": qid, "outcome": score["outcome"], "observed_disposition": disposition,
               "evidence_flags": " | ".join(flags), "trace_path": f"{system}/{qid}.json",
               "visible_evidence": diag["reference_literals_in_visible_text"], "automatic_citation": diag["citation_status"],
               "runtime_check_failures": diag["runtime_checks_failed"], "arithmetic_calls_recomputed": diag["arithmetic_recomputed"],
               "arithmetic_mismatches": diag["arithmetic_mismatches"], "end_reason": trace.get("end_reason") or "",
               "last_return_or_route": diag["last_route_reason"] or trace.get("explanation") or "",
               "error_type": trace.get("error_type") or "",
               "draft_semantic_evidence": ai["reason"] if ai else "", "calculation_case_evidence": calc_note,
               "official_human_status": score["human_review"]["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new CSV path")
    args = parser.parse_args()
    rows = list(classify(args.run))
    if not rows:
        raise ValueError("Empty manifest")
    with args.out.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(Counter(row["observed_disposition"] for row in rows), sort_keys=True))


if __name__ == "__main__":
    main()
