"""Export one diagnostic row for each planned prediction using saved traces and scores."""
import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

from evaluate import AGENT_SYSTEMS, NUMBER, raw_key, read_json

ROOT = Path(__file__).resolve().parent
RUN_ISSUES = {"api_error", "parse_error", "input_error", "missing_prediction"}


def tool_calls(trace):
    if "graph_events" in trace:
        return [event for event in trace["graph_events"] if event["event"] == "tool"]
    return [step["tool_call"] for step in trace.get("steps", []) if step.get("tool_call")]


def visible_literals(trace):
    if trace.get("system") in AGENT_SYSTEMS:
        messages = [m for step in trace.get("steps", []) for m in step.get("messages", [])]
    else:
        messages = trace.get("messages", [])
    sent = [m["content"] for m in messages if m["role"] in ("user", "tool")]
    visible = []
    literals = set()
    for c in trace.get("retrieved_chunks", []):
        block = f"[{c['chunk_id']} | PDF page {c['pdf_page']}]\n{c['text']}"
        if any(block in content for content in sent):
            visible.append(c["chunk_id"])
            literals.update(raw_key(m.group()) for m in NUMBER.finditer(c["text"]))
    literals.discard(None)
    return visible, literals


def reference_coverage(reference, available):
    # Lexical coverage does not establish that equal numbers refer to the same account or year.
    if isinstance(reference, list) and reference:
        groups = [forms if isinstance(forms, list) else [forms] for forms in reference]
        paths = [[{raw_key(form) for form in group} for group in groups]]
    elif isinstance(reference, dict) and reference.get("annotation_status") == "verified" and reference.get("paths"):
        paths = []
        for path in reference["paths"]:
            groups = []
            for row in path:
                for text in row["figures"]:
                    key = raw_key(text)
                    if key is None:
                        raise ValueError(f"Invalid reference operand: {text}")
                    forms = {key}
                    if row.get("allow_magnitude"):
                        forms.add((abs(key[0]), key[1]))
                    groups.append(forms)
            if groups:
                paths.append(groups)
    else:
        return "unassessed"
    if not paths:
        return "unassessed"
    if any(all(not group.isdisjoint(available) for group in path) for path in paths):
        return "complete_path_present"
    return "no_complete_annotated_path"


def as_decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def replay_arithmetic(call):
    """Recompute the tool result from its recorded operands. Account selection is assessed separately."""
    result = as_decimal((call.get("result") or {}).get("result"))
    args = call.get("arguments", {})
    inputs = [raw_key(x) for x in args.get("operands", [])]
    if result is None or len(inputs) != 2 or any(x is None for x in inputs):
        return None
    a, b = (x[0] for x in inputs)
    operation = args.get("operation")
    with localcontext() as ctx:
        ctx.prec = 48  # The shared MCP calculator rounds to 48 significant decimal digits.
        if operation == "add":
            expected = a + b
        elif operation == "subtract":
            expected = a - b
        elif operation == "multiply":
            expected = a * b
        elif operation == "divide" and b:
            expected = a / b
        elif operation == "percent_change" and b:
            expected = (a - b) * 100 / b
        else:
            return None
    return result == expected


def page_list(pages):
    return " | ".join(f"{doc}:p{n}" for doc, n in pages)


def diagnose(score, trace, reference=None):
    system, qid, outcome = score["system"], score["question_id"], score["outcome"]
    tags = []
    issue = outcome if outcome in RUN_ISSUES else ""
    if issue:
        tags.append("execution_issue")
    has_trace = trace is not None
    trace = trace or {}
    visible, literals = visible_literals(trace)
    gold_pages = score.get("gold_pdf_pages") or []
    hits = score.get("gold_page_hits")
    if system == "B0":
        page_signal = coverage = "not_applicable"
    elif not has_trace:
        page_signal = coverage = "unknown"
    else:
        if not gold_pages:
            page_signal = "unassessed"
        else:
            page_signal = "annotated_page_returned" if hits else "annotated_page_not_returned"
        coverage = reference_coverage(reference, literals)
        if page_signal == "annotated_page_not_returned":
            tags.append("reference_page_not_returned")
        if coverage == "no_complete_annotated_path":
            tags.append("reference_literals_incomplete")
    metric = score.get("numeric_comparison") or {}
    if metric.get("reason") == "incompatible_units":
        tags.append("incompatible_answer_units")
    if metric.get("predicted_unit") == metric.get("gold_unit"):
        pred, gold = as_decimal(metric.get("predicted")), as_decimal(metric.get("gold"))
        if pred is not None and gold is not None and pred and gold and pred.is_signed() != gold.is_signed():
            tags.append("opposite_sign")

    calls = tool_calls(trace)
    calculations = [c for c in calls if c["name"] == "calculate"]
    tool_errors = [c for c in calculations if "result" not in (c.get("result") or {})]
    tool_values = [as_decimal(c["result"]["result"]) for c in calculations if "result" in (c.get("result") or {})]
    if tool_errors:
        tags.append("calculator_call_failed")
    arithmetic = [replay_arithmetic(c) for c in calculations]
    if False in arithmetic:
        tags.append("tool_arithmetic_mismatch")
    chosen_literals = []
    for step in trace.get("steps", []):
        if step.get("phase") == "Calculator" and step.get("raw_response"):
            try:
                plan = json.loads(step["raw_response"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(plan, dict) and isinstance(plan.get("operands"), list):
                chosen_literals.extend(c.get("raw_number") for c in plan["operands"] if isinstance(c, dict))
    absent_plan_literals = [str(x) for x in chosen_literals if raw_key(x) not in literals]
    if absent_plan_literals:
        tags.append("planned_operand_absent_from_visible_text")
    matches = "not_applicable"
    if trace.get("status") == "answered" and tool_values:
        # A raw difference can reflect rounding or units. It is not proof of an arithmetic mistake.
        matches = "yes" if as_decimal(trace.get("value")) in tool_values else "no"
        if matches == "no":
            tags.append("final_value_differs_from_all_raw_calculator_results")

    citation = score["automatic_citation"]
    checks = citation.get("checks", [])
    invisible = sum(not c["visible"] for c in checks)
    absent = sum(c["visible"] and not c["number_found"] for c in checks)
    if citation["status"] == "fail":
        tags.append("final_citation_failed")
    if invisible:
        tags.append("invisible_citation")
    if absent:
        tags.append("number_absent_from_cited_excerpt")
    failed_checks = sum(c["name"] == "check_citation" and
                        (c.get("result") or {}).get("number_found") is False for c in calls)
    if failed_checks:
        tags.append("runtime_citation_check_failed")
    routes = [e for e in trace.get("graph_events", []) if e["event"] == "route"]
    reasons = [e.get("reason", "") for e in routes]
    if any(reason.startswith(("Cannot express a ", "The question requests ", "A cited percent sign conflicts")) for reason in reasons):
        tags.append("plan_unit_validation_failed")
    if any(e.get("decision") == "retry" for e in routes):
        tags.append("retrieval_retry")
    if trace.get("end_reason") in ("tool_limit", "model_limit"):
        tags.append("budget_reached")
    if outcome == "refusal":
        tags.append("refusal")
    human = score.get("human_review") or {}
    row = {"system": system, "question_id": qid, "outcome": outcome, "prediction": trace.get("value"),
           "prediction_unit": trace.get("unit"), "gold_normalized": metric.get("gold"),
           "gold_unit": metric.get("gold_unit"), "absolute_error": metric.get("absolute_error"),
           "tolerance": metric.get("allowed_absolute_error"), "reference_pages": page_list(gold_pages),
           "reference_page_signal": page_signal, "visible_chunk_ids": " | ".join(visible),
           "reference_literals_in_visible_text": coverage, "calculator_calls": len(calculations),
           "calculator_call_failures": len(tool_errors), "raw_calculator_results": " | ".join(str(x) for x in tool_values),
           "arithmetic_recomputed": sum(x is not None for x in arithmetic), "arithmetic_mismatches": arithmetic.count(False),
           "absent_planned_operand_literals": " | ".join(absent_plan_literals),
           "prediction_matches_any_raw_tool_result": matches, "citation_status": citation["status"],
           "invisible_final_citations": invisible, "visible_citations_missing_number": absent,
           "runtime_checks_failed": failed_checks, "end_reason": trace.get("end_reason"),
           "last_route_reason": reasons[-1] if reasons else "", "execution_issue": issue,
           "failure_stage": trace.get("failure_stage"), "error_type": trace.get("error_type"),
           "automatic_signals": " | ".join(tags), "saved_citation_review": human.get("status"),
           "manual_primary_cause": "", "manual_notes": "", "manual_reviewer": "",
           "trace_path": f"{system}/{qid}.json"}
    return row


def analyze(folder, reference_dir=ROOT / "data/evaluation"):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("mode") != "live":
        raise ValueError("Choose a scored live run")
    scores = read_json(folder / "scores.json")["rows"]
    expected = [(s, q) for s in manifest["systems"] for q in manifest["question_ids"]]
    lookup = {(r["system"], r["question_id"]): r for r in scores}
    if len(expected) != len(scores) or set(expected) != set(lookup):
        raise ValueError("Scores do not match the planned questions and systems")
    reference_file = "held_out_operands.json" if manifest["scope"] == "held-out" else "dev_operands.json"
    path = reference_dir / reference_file
    references = read_json(path) if path.exists() else {}
    if manifest["scope"] == "held-out":
        references = references.get("questions", {})
    rows = []
    for s, qid in expected:
        trace_path = folder / s / f"{qid}.json"
        trace = None
        if trace_path.is_file():
            try:
                trace = read_json(trace_path)
            except (json.JSONDecodeError, UnicodeError):
                if lookup[s, qid]["outcome"] != "parse_error":
                    raise
        elif lookup[s, qid]["outcome"] != "missing_prediction":
            raise ValueError(f"Trace missing after grading: {trace_path}")
        if trace is not None and (trace["system"], trace["question_id"]) != (s, qid):
            raise ValueError(f"Inconsistent trace: {trace_path}")
        rows.append(diagnose(lookup[s, qid], trace, references.get(qid)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New CSV relative to the current directory")
    args = parser.parse_args()
    rows = analyze(args.run)
    with args.out.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(tag for row in rows for tag in row["automatic_signals"].split(" | ") if tag)
    print(f"{len(rows)} planned answers. Overlapping signals: {json.dumps(counts, sort_keys=True)}")
    print("Unreviewed cause fields are blank. Missing an official page alone does not establish a retrieval failure.")


if __name__ == "__main__":
    main()
