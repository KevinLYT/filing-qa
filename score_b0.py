"""Grade a single saved B0 response to the working-capital question in USD millions."""
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTION_ID = "financebench_id_03031"


def grade(record, gold):
    # Read the stated conclusion; the explanation also contains two operands.
    conclusion = record["answer"].strip().split("\n")[-1].strip()
    result = {
        "question_id": gold["question_id"],
        "system": record["system"],
        "conclusion": conclusion,
        "unit": gold["unit"],
    }
    # The saved B0 runs report their final answer in an explicit conclusion.
    pattern = r"(?:Therefore,\s*)?Lockheed Martin's FY2021 net working capital is \$(-?[\d,]+(?:\.\d+)?) million\.?"
    match = re.fullmatch(pattern, conclusion, flags=re.IGNORECASE)
    if match is None:
        return dict(result, outcome="needs_review", reason="No supported final amount statement.")

    predicted = Decimal(match.group(1).replace(",", ""))
    target = Decimal(gold["official_answer"].removeprefix("$").replace(",", ""))
    # Compare in the same unit, including the tolerance boundary.
    error = abs(predicted - target)
    tolerance = abs(target) * Decimal("0.01")
    outcome = "correct" if error <= tolerance else "incorrect"
    return dict(result, predicted=str(predicted), gold=str(target), absolute_error=str(error),
                allowed_error=str(tolerance), outcome=outcome)


def main():
    gold_path = ROOT / "data" / "evaluation" / f"{QUESTION_ID}.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    output_dir = ROOT / "results" / "scores" / "b0"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "results" / "b0").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["system"] != "B0" or record["question_id"] != gold["question_id"]:
            continue
        score = grade(record, gold)
        score["prediction_file"] = path.relative_to(ROOT).as_posix()
        output = output_dir / path.name
        output.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
        print(f"{output.relative_to(ROOT)}: {score['outcome']}")


if __name__ == "__main__":
    main()
