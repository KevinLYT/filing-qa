"""Summarize a scored paired run without contacting the model."""
import argparse
from pathlib import Path

from evaluate import read_json


def render_report(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    scores = read_json(folder / "scores.json")
    systems = manifest["systems"]
    ids = manifest["question_ids"]
    rows = {(r["system"], r["question_id"]): r for r in scores["rows"]}
    if set(rows) != {(s, q) for s in systems for q in ids}:
        raise ValueError("Scores must cover every planned system and question")
    lines = ["# Paired filing QA results", "", f"Model: {manifest['model']} · {len(ids)} questions · {manifest['scope']}", "",
             "| System | Correct | Incorrect | Refusal | Run issues | Citation passes / answered | Mean seconds | Model calls | Input / output tokens | API USD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for system in systems:
        s = scores["summary"][system]
        answered = s["correct"] + s["incorrect"]
        issues = s["n"] - answered - s["refusal"]
        citations = f"{s['automatic_citation'].get('pass', 0)}/{answered}" if answered else "N/A (0 answered)"
        seconds = s["mean_elapsed_seconds"]
        latency = f"{seconds:.2f}" if seconds is not None else "unknown"
        lines.append(f"| {system} | {s['correct']}/{s['n']} | {s['incorrect']} | {s['refusal']} | {issues} "
                     f"| {citations} | {latency} | {s['model_calls_recorded']} "
                     f"| {s['prompt_tokens_recorded']:,} / {s['completion_tokens_recorded']:,} | {s['estimated_api_usd']} |")
    lines.extend(["", "Citation passes are automatic. The mean latency covers recorded attempts. Token counts are reported usage. "
                  "The API column excludes electricity and hardware.", "",
                  "| Question ID | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)])
    for qid in ids:
        lines.append("| " + qid + " | " + " | ".join(rows[s, qid]["outcome"] for s in systems) + " |")
    # Count paired changes, including refusals and execution failures in the denominator.
    for reference in ("B2", "NoVerifier"):
        if "Ours" in systems and reference in systems:
            wins = sum(rows["Ours", q]["outcome"] == "correct" and rows[reference, q]["outcome"] != "correct" for q in ids)
            losses = sum(rows["Ours", q]["outcome"] != "correct" and rows[reference, q]["outcome"] == "correct" for q in ids)
            lines.extend(["", f"Against {reference}, Ours gained {wins} correct and lost {losses} correct "
                          f"on the same {len(ids)} questions."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(render_report(args.run), end="")
