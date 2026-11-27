"""Replay the saved question, evidence, answers and grades without a model or network."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = "financebench_id_03031"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tool_lines(tool):
    args = json.dumps(tool["arguments"], ensure_ascii=False)
    result = tool.get("result")
    if tool["name"] == "search_filings" and isinstance(result, dict) and "chunks" in result:
        observed = ", ".join(chunk["chunk_id"] for chunk in result["chunks"])
    else:
        observed = json.dumps(result, ensure_ascii=False)
    return f"{tool['name']} {args}", "     returned: " + observed


def agent_actions(trace):
    # Replay decisions from the archived requests and results, without calling MCP again.
    lines = ["Agent steps (in order):"]
    for index, step in enumerate(trace.get("steps", []), 1):
        tool = step.get("tool_call")
        if not tool:
            phase = "final answer request" if step["phase"] == "answer" else "decision without a tool call"
            lines.append(f"  {index}. {phase}")
            continue
        call, result = tool_lines(tool)
        lines.extend([f"  {index}. {call}", result])
    if trace.get("end_reason"):
        lines.append(f"  Stopping reason: {trace['end_reason']}")
    return lines


def graph_actions(trace):
    # Events also record checks and budget stops that precede a role's model request.
    lines = ["Graph steps (in order):"]
    decisions = iter(trace.get("steps", []))
    for event in trace.get("graph_events", []):
        kind = event["event"]
        if kind == "node":
            lines.append(f"  Attempt {event['attempt']} | {event['role']}")
        elif kind == "model":
            step = next(decisions, {})
            raw = step.get("raw_response")
            if raw:
                try:
                    decision = json.dumps(json.loads(raw), ensure_ascii=False)
                    lines.append("     model: " + decision)
                except (TypeError, ValueError):
                    lines.append("     model: incomplete or malformed response; see --full")
        elif kind == "tool":
            call, result = tool_lines(event)
            lines.extend(["     " + call, result])
        elif kind == "route":
            lines.append(f"     Route: {event['decision']} | {event.get('reason', '')}")
    if trace.get("end_reason"):
        lines.append(f"  Stopping reason: {trace['end_reason']}")
    if trace.get("end_reason") == "verification_failed" and not any(
        event["event"] == "node" and event["role"] == "Verifier" for event in trace.get("graph_events", [])
    ):
        lines.append("  No candidate reached the Verifier; the Calculator plan failed validation.")
    return lines


def replay(folder, question_id, full=False):
    manifest = read(folder / "manifest.json")
    if manifest["mode"] != "live" or question_id not in manifest["question_ids"]:
        raise ValueError("Choose a question in a saved live run.")
    grades = read(folder / "scores.json")
    by_system = {r["system"]: r for r in grades["rows"] if r["question_id"] == question_id}
    lines = [f"SAVED RUN | {question_id}"]
    for index, system in enumerate(manifest["systems"]):
        row = by_system[system]
        path = folder / system / f"{question_id}.json"
        if not path.exists():
            lines.extend(["", f"{system} | {row['outcome']} | prediction file missing"])
            continue
        try:
            trace = read(path)
        except (json.JSONDecodeError, UnicodeError):
            lines.extend(["", f"{system} | {row['outcome']} | unreadable prediction file"])
            continue
        if index == 0:
            lines.extend(["", "QUESTION"])
            question = trace.get("question") or next((m["content"] for m in trace.get("messages", []) if m["role"] == "user"), None)
            lines.append(question or "(no question sent)")
        lines.extend(["", f"{system} | {trace['model']} | {trace['status']}"])
        if system == "B2":
            lines.extend(agent_actions(trace))
        elif system in ("Ours", "NoVerifier"):
            lines.extend(graph_actions(trace))
        # Use only the archived excerpt text.
        hits = trace.get("retrieved_chunks", [])
        for hit in hits:
            if "rerank_score" in hit:
                scores = (f"cross-encoder={hit['rerank_score']:.4f}  RRF rank={hit['rrf_rank']}  "
                          f"BM25 rank={hit['bm25_rank']}  dense rank={hit['dense_rank']}")
            else:
                scores = f"BM25={hit['score']:.4f}"
            lines.append(f"  #{hit['rank']} {hit['chunk_id']}  " + scores)
        if question_id == DEMO and hits:
            for hit in hits:
                text = hit["text"]
                if "Consolidated Balance Sheets" in text and "Total current liabilities" in text:
                    lines.append(f"  Extract from {hit['chunk_id']}:")
                    labels = ("in millions", "2021  2020", "total current")
                    if system in ("B2", "Ours", "NoVerifier"):
                        labels += ("total liabilities ",)
                    lines.extend("    " + line for line in text.splitlines()
                                 if any(t in line.lower() for t in labels))
                    break
        lines.append(f"Prediction: {trace.get('value')} {trace.get('unit')}")
        lines.append(f"Explanation: {trace.get('explanation')}")
        lines.append(f"Numeric grade: {row['outcome']}  {json.dumps(row['numeric_comparison'])}")
        lines.append(f"Citations: auto={row['automatic_citation']['status']} | human={row['human_review']['status']}")
        for citation in trace.get("citations", []):
            lines.append(f"  {citation['label']} | {citation['raw_number']} | {citation['chunk_id']}")
        lines.append(f"Calls: {trace['model_calls']} | seconds: {trace['elapsed_seconds']} | API USD: {trace['estimated_api_usd']}")
        lines.append(f"Usage: {json.dumps(trace.get('usage'))}")
        if full and system in ("B2", "Ours", "NoVerifier"):
            lines.extend(["Exact model requests, completions and tool results:",
                          json.dumps(trace.get("steps", []), ensure_ascii=False, indent=2)])
        elif full:
            lines.extend(["Exact messages:", json.dumps(trace.get("messages"), ensure_ascii=False, indent=2),
                          "Raw completion:", str(trace.get("raw_response"))])
    if question_id == DEMO:
        reference = read(ROOT / "data" / "evaluation" / f"{DEMO}.json")
        lines.extend(["", "AFTER SCORING | reference visible only to the evaluator",
                      f"Official answer: {reference['official_answer']} ({reference['unit']})",
                      f"19,815 - 13,997 = 5,818 USD millions | PDF page {reference['pdf_page']}"])
    elif manifest.get("scope") == "held-out":
        # Show the reference only in offline replay, after the archived predictions and grades.
        labels = ROOT / "data/evaluation"
        gold_rows = (json.loads(line) for line in (labels / "gold.jsonl").read_text(encoding="utf-8-sig").splitlines() if line)
        gold = next(r for r in gold_rows if r["question_id"] == question_id)
        lines.extend(["", "AFTER SCORING | reference visible only to the evaluator",
                      f"Official answer: {gold['official_answer']} ({gold['unit']})"])
        if (labels / "held_out_operands.json").is_file():
            ref = read(labels / "held_out_operands.json")["questions"].get(question_id, {})
            if ref.get("paths"):
                lines.append(f"Reference calculation: {ref['method']} | filing result {ref['recomputed_value']} {ref['result_unit']}")
                for entry in ref["paths"][0]:
                    observations = ", ".join(f"{year} = {value}" for year, value in zip(entry["years"], entry["figures"]))
                    lines.append(f"  PDF p.{entry['pdf_page']} | {entry['line_item']} | {observations}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results" / "phase1-final")
    parser.add_argument("--id", help="Question ID; defaults to the first ID in the saved manifest")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    question_id = args.id or read(args.run / "manifest.json")["question_ids"][0]
    print(replay(args.run, question_id, args.full))
