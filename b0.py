"""B0: ask the local model one FinanceBench question using only its built-in knowledge."""
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.request import ProxyHandler, Request, build_opener

API = "http://127.0.0.1:18091/v1"
QUESTION = (
    "What is Lockheed Martin's FY2021 net working capital? "
    "Define net working capital as total current assets less total current liabilities. "
    "Answer in USD millions. Respond to the question by assuming the perspective "
    "of an investment analyst who can only use the details shown within the balance sheet."
)


def save_record(payload, result, elapsed_seconds, system="B0", retrieved_chunks=None):
    # Save the request and the original model response together for replay.
    record = {
        "question_id": "financebench_id_03031",
        "system": system,
        "request": payload,
        "answer": result["choices"][0]["message"]["content"],
        "usage": result.get("usage"),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    if retrieved_chunks is not None:
        record["retrieved_chunks"] = retrieved_chunks
    record["finish_reason"] = result["choices"][0].get("finish_reason")
    folder = Path(__file__).resolve().parent / "results" / system.lower()
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"{timestamp}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    return path


def ask_model(message):
    # Bypass HTTP proxies when connecting to the local model server.
    client = build_opener(ProxyHandler({}))
    with client.open(f"{API}/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]

    # Use the same generation settings for B0 and B1.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
        "max_tokens": 300,
    }
    request = Request(
        f"{API}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with client.open(request, timeout=120) as response:
        result = json.load(response)

    return payload, result


def main():
    start = perf_counter()
    payload, result = ask_model(QUESTION)
    path = save_record(payload, result, perf_counter() - start)
    print("Saved:", path)
    print("Model:", payload["model"])
    print("Question:", QUESTION)
    print("Answer:", result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
