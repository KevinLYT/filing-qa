"""Score a saved run without loading a model."""
import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from filing_numbers import NUMBER, raw_key

ROOT = Path(__file__).resolve().parent
AMOUNT_SCALES = {"USD": 1, "USD_thousands": 1000, "USD_millions": 10**6, "USD_billions": 10**9}
OUTCOMES = ("correct", "incorrect", "refusal", "api_error", "parse_error", "input_error", "missing_prediction")
AGENT_SYSTEMS = {"B2", "Ours", "NoVerifier"}
RETRIEVAL_SYSTEMS = AGENT_SYSTEMS | {"B1"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        stream.write(text)
    Path(stream.name).replace(path)


def normalize(value, unit):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Nonfinite number")
    if unit in AMOUNT_SCALES:
        return number * AMOUNT_SCALES[unit], "USD"
    if unit == "fraction":
        return number * 100, "percent"
    if unit in ("percent", "percentage_points"):
        return number, unit
    raise ValueError(f"Unsupported unit {unit}")


def compare_value(prediction, gold):
    actual, unit = normalize(prediction["value"], prediction["unit"])
    target, reference_unit = normalize(gold["value"], gold["unit"])
    info = {"predicted": str(actual), "predicted_unit": unit, "gold": str(target), "gold_unit": reference_unit}
    if unit != reference_unit:
        return False, dict(info, reason="incompatible_units")
    # Allow 1% of the absolute reference amount, or 0.1 percentage points for rates. A zero reference amount requires zero.
    tolerance = abs(target) * Decimal("0.01") if unit == "USD" else Decimal("0.1")
    error = abs(actual - target)
    return error <= tolerance, dict(info, absolute_error=str(error), allowed_absolute_error=str(tolerance))


def held_out_citation_grade(checks, supported, reference):
    # Accept a complete set of cited operands from any supported calculation path, including a subtotal or its components.
    annotation = (reference or {}).get("annotation_status", "unavailable")
    literal_pass = bool(checks) and all(check["number_found"] for check in checks)
    missing_by_path = []
    for path in (reference or {}).get("paths", []):
        missing = []
        figures = 0
        for row in path:
            for figure in row["figures"]:
                figures += 1
                key = raw_key(figure)
                if key is None:
                    raise ValueError(f"Invalid reference figure {figure!r}")
                forms = {key}
                if row.get("allow_magnitude"):
                    forms.add((abs(key[0]), key[1]))
                if forms.isdisjoint(supported):
                    missing.append(figure)
        if figures:
            missing_by_path.append(missing)
    matches = [i + 1 for i, missing in enumerate(missing_by_path) if not missing]
    if not literal_pass:
        status = "fail"
    elif annotation == "partial":
        status = "partial"
    elif annotation != "verified" or not missing_by_path:
        status = "unassessable"
    else:
        status = "pass" if matches else "fail"
    return {"status": status, "checks": checks, "reference_annotation_status": annotation,
            "literal_status": "pass" if literal_pass else "fail", "matched_reference_paths": matches,
            "missing_reference_operands": min(missing_by_path, key=len) if missing_by_path else None}


def citation_audit(trace, operands):
    checks = []
    if trace.get("status") != "answered" or (trace["system"] == "B0" and not trace.get("citations")):
        return {"status": "not_applicable", "checks": checks}
    # A tool result counts as evidence only after its excerpt was sent to the model.
    if trace["system"] in AGENT_SYSTEMS:
        messages = (m for step in trace.get("steps", []) for m in step.get("messages", []))
    else:
        messages = trace["messages"]
    sent = [m["content"] for m in messages if m["role"] in ("user", "tool")]
    visible_chunks = {}
    for chunk in trace["retrieved_chunks"]:
        key = (chunk["doc_name"], chunk["pdf_page"], chunk["chunk_id"])
        header = f"[{chunk['chunk_id']} | PDF page {chunk['pdf_page']}]\n"
        if any(header + chunk["text"] in message for message in sent):
            visible_chunks[key] = chunk["text"]
    supported = set()
    for citation in trace["citations"]:
        key = (citation["doc_name"], citation["pdf_page"], citation["chunk_id"])
        text = visible_chunks.get(key)
        claimed = raw_key(citation["raw_number"])
        available = {raw_key(m.group()) for m in NUMBER.finditer(text or "")}
        found = claimed is not None and claimed in available
        if found:
            supported.add(claimed)
        checks.append(dict(citation, visible=text is not None, number_found=found))
    if operands is None or isinstance(operands, dict) or not operands:
        return held_out_citation_grade(checks, supported, operands or None)
    missing = []
    for operand in operands:
        forms = operand if isinstance(operand, list) else [operand]
        if not any(raw_key(form) in supported for form in forms):
            missing.append(operand)
    passed = bool(checks) and all(c["number_found"] for c in checks) and not missing
    return {"status": "pass" if passed else "fail", "checks": checks, "missing_reference_operands": missing}


def grade(trace, gold, operands):
    status = trace["status"]
    comparison = None
    if status == "answered":
        correct, comparison = compare_value(trace, gold)
        outcome = "correct" if correct else "incorrect"
    elif status in OUTCOMES[2:]:
        outcome = status
    else:
        raise ValueError(f"Unrecognized saved status {status}")
    pages = {(e["doc_name"], int(e["evidence_page_num"]) + 1) for e in gold["evidence"]}
    retrieved = {(c["doc_name"], c["pdf_page"]) for c in trace.get("retrieved_chunks", [])}
    return {"question_id": trace["question_id"], "system": trace["system"], "outcome": outcome,
            "numeric_comparison": comparison, "automatic_citation": citation_audit(trace, operands),
            "gold_pdf_pages": sorted(pages), "gold_page_hits": sorted(pages & retrieved) if trace["system"] in RETRIEVAL_SYSTEMS else None}


def summarize(rows, traces):
    n = len(rows)
    counts = Counter(row["outcome"] for row in rows)
    elapsed = [t["elapsed_seconds"] for t in traces if t.get("elapsed_seconds") is not None]
    usage = [t["usage"] for t in traces if isinstance(t.get("usage"), dict)]
    costs = [t.get("estimated_api_usd") for t in traces]
    known_cost = all(c not in (None, "unknown") for c in costs)
    total = sum(Decimal(str(c)) for c in costs) if known_cost else None
    return {"n": n, **{k: counts[k] for k in OUTCOMES}, "accuracy": counts["correct"] / n,
            "automatic_citation": dict(Counter(r["automatic_citation"]["status"] for r in rows)),
            "human_review": dict(Counter(r["human_review"]["status"] for r in rows)),
            "model_calls_recorded": sum(t.get("model_calls", 0) for t in traces), "latencies_recorded": len(elapsed),
            "mean_elapsed_seconds": round(sum(elapsed) / len(elapsed), 4) if elapsed else None,
            "usage_records": len(usage), "prompt_tokens_recorded": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens_recorded": sum(u.get("completion_tokens", 0) for u in usage),
            "estimated_api_usd": str(total) if known_cost else "unknown",
            "mean_estimated_api_usd": str(total / n) if known_cost else "unknown"}


def score_run(folder, evaluation_dir=ROOT / "data/evaluation"):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("mode") != "live":
        raise ValueError("Only live predictions are eligible for model scores")
    ids = manifest["question_ids"]
    systems = manifest["systems"]
    if not ids or len(set(ids)) != len(ids) or not systems or len(set(systems)) != len(systems):
        raise ValueError("The evaluation manifest needs nonempty, unique questions and systems")
    labels = (evaluation_dir / "gold.jsonl").read_text(encoding="utf-8-sig").splitlines()
    gold = {r["question_id"]: r for r in map(json.loads, filter(str.strip, labels))}
    operands = read_json(evaluation_dir / "dev_operands.json")
    if any(qid not in operands for qid in ids):
        held_out_file = evaluation_dir / "held_out_operands.json"
        if held_out_file.exists():
            extra = read_json(held_out_file)["questions"]
            if operands.keys() & extra.keys():
                raise ValueError("Overlapping development and held-out operand annotations")
            operands.update(extra)
    review_file = folder / "human_reviews.json"
    reviews = read_json(review_file) if review_file.exists() else {}
    previous_keys = set(reviews)
    rows, summary = [], {}
    for system in systems:
        traces, group = [], []
        # Every planned question contributes to accuracy, including absent records.
        for qid in ids:
            path = folder / system / (qid + ".json")
            trace = {"question_id": qid, "system": system, "status": "missing_prediction"}
            if path.exists():
                try:
                    trace = read_json(path)
                except (json.JSONDecodeError, UnicodeError):
                    trace["status"] = "parse_error"
                if (trace["question_id"], trace["system"]) != (qid, system):
                    raise ValueError(f"Prediction ID or system mismatch in {path}")
            reference = operands.get(qid)
            row = grade(trace, gold[qid], reference)
            if isinstance(reference, dict) and "recomputed_value" in reference:
                amount = {"value": reference["recomputed_value"], "unit": reference["result_unit"]}
                consistent, _ = compare_value(amount, gold[qid])
                row["reference_recalculation"] = dict(amount, within_official_tolerance=consistent)
            review = {"year": None, "line_item": None, "unit": None, "reviewer": None, "notes": ""}
            review_status = "not_applicable"
            if system in RETRIEVAL_SYSTEMS and trace["status"] == "answered":
                review = reviews.setdefault(f"{system}/{qid}", review)
                fields = [review.get(k) for k in ("year", "line_item", "unit")]
                review_status = "pending"
                if review.get("reviewer") and all(type(v) is bool for v in fields):
                    review_status = "pass" if all(fields) else "fail"
            row["human_review"] = dict(review, status=review_status)
            group.append(row)
            traces.append(trace)
        rows.extend(group)
        summary[system] = summarize(group, traces)
    if not review_file.exists() or set(reviews) != previous_keys:
        write_json(review_file, reviews)
    result = {"summary": summary, "rows": rows}
    write_json(folder / "scores.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Directory containing the saved manifest and traces")
    args = parser.parse_args()
    print(json.dumps(score_run(args.run)["summary"], indent=2))
