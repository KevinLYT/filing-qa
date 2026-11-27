"""Paired B0/B1 experiment. Gold annotations are read only by evaluate.py."""
import argparse
import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from retrieve_b1 import retrieve

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEMO_ID = "financebench_id_03031"
LOCAL_MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_API = "http://127.0.0.1:18091/v1"
UNITS = ["USD", "USD_thousands", "USD_millions", "USD_billions", "percent", "fraction", "percentage_points"]
PROTOCOL_VERSION = "explanation-first-v2"
SYSTEM_PROMPT = """Answer the financial question using the supplied filing excerpts when available.
Use a JSON object with a single answer property. Inside it write explanation, status, citations, value, and unit.
Begin explanation with a short factual statement of the relevant line items, years and arithmetic or missing data.
Use status "answered" when the requested number is present or you can compute it from the available operands.
When a formula and its inputs are known, compute the result even if that result is not printed in the filing.
With no excerpts, give an answer if you know the figures with confidence. Set citations to an empty list in that case.
For an answered response, value is the final JSON number. Use one of the following unit names:
USD, USD_thousands, USD_millions, USD_billions, percent, fraction, percentage_points.
A value of 20 percent and 0.20 fraction represent the same rate. Use percentage_points for a difference of rates.
For a lookup cite the source value. For a calculation cite each source operand. Each citation contains doc_name,
pdf_page (integer), chunk_id, label (line item and year), and raw_number (a string with the printed number and sign).
Copy the identifiers from the excerpt header. Its chunk_id begins with the doc_name followed by a colon.
Use status "refusal" when essential information is missing. State what is missing in explanation, set citations to [],
and set value and unit to null."""


def object_schema(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def generation_settings(model):
    text = {"type": "string"}
    citation = object_schema({"doc_name": text, "pdf_page": {"type": "integer"}, "chunk_id": text,
                              "label": text, "raw_number": text})
    # Both baselines use the same schema. A refusal requires a null value, a null unit, and an empty citation list.
    answered = object_schema({
        "explanation": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["answered"]},
        "citations": {"type": "array", "items": citation},
        "value": {"type": "number"}, "unit": {"type": "string", "enum": UNITS},
    })
    refusal = object_schema({
        "explanation": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["refusal"]},
        "citations": {"type": "array", "items": citation, "maxItems": 0},
        "value": {"type": "null"}, "unit": {"type": "null"},
    })
    schema = object_schema({"answer": {"anyOf": [answered, refusal]}})
    return {"model": model, "temperature": 0, "max_tokens": 800,
            "response_format": {"type": "json_schema", "json_schema": {"name": "filing_answer", "strict": True, "schema": schema}}}


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Expose only finished traces to the offline scorer.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_input(question, system):
    # The question row supplies the filing; the retriever must discover the page.
    hits = []
    if system == "B1":
        chunks = read_rows(DATA / "index" / (question["doc_name"] + ".jsonl"))
        if not chunks:
            raise ValueError("Empty filing index")
        hits = [dict(hit, rank=i) for i, hit in enumerate(retrieve(question["question"], chunks, limit=4), 1)]
    blocks = [f"[{c['chunk_id']} | PDF page {c['pdf_page']}]\n{c['text']}" for c in hits]
    context = "\n\n".join(blocks)
    content = question["question"]
    if context:
        content += "\n\nFiling excerpts, in retrieval order:\n" + context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    return {"messages": messages, "retrieved_chunks": hits, "context": context}


def hosted_env():
    # Environment variables override settings from the local .env file.
    settings = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                settings[name.strip()] = value.strip().strip("\"'")
    return {**settings, **os.environ}


def configuration(hosted, needs_key):
    if not hosted:
        rates = {"source": "Self-hosted llama.cpp on 127.0.0.1; no provider token charge.",
                 "input": "0", "output": "0", "cached_input": "0", "per_tokens": 1_000_000, "currency": "USD"}
        return LOCAL_API, LOCAL_MODEL, None, rates
    env = hosted_env()
    model = env.get("QA_MODEL")
    key = env.get("OPENAI_API_KEY")
    if not model or (needs_key and not key):
        raise ValueError("Set QA_MODEL and OPENAI_API_KEY in .env or PowerShell for hosted prediction.")
    base = env.get("QA_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an HTTPS API root without embedded credentials, query or fragment.")
    rates = None
    if env.get("QA_PRICE_MODEL") == model and env.get("QA_PRICE_SOURCE"):
        rates = {"source": env["QA_PRICE_SOURCE"], "per_tokens": 1_000_000, "currency": "USD"}
        for role in ("input", "output", "cached_input"):
            raw = env.get(f"QA_{role.upper()}_PRICE")
            if raw:
                number = Decimal(raw)
                if not number.is_finite() or number < 0:
                    raise ValueError("Prices must be finite non-negative numbers.")
                rates[role] = str(number)
    return base, model, key, rates


def estimate_cost(usage, rates):
    if not isinstance(usage, dict) or not rates or "input" not in rates or "output" not in rates:
        return "unknown"
    prompt = usage.get("prompt_tokens")
    output = usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if not all(type(n) is int and n >= 0 for n in (prompt, output, cached)) or cached > prompt:
        return "unknown"
    if cached and "cached_input" not in rates:
        return "unknown"
    total = Decimal(rates["input"]) * (prompt - cached) + Decimal(rates["output"]) * output
    if cached:
        total += Decimal(rates["cached_input"]) * cached
    return str(total / rates["per_tokens"])


def parse_answer(raw):
    envelope = json.loads(raw)
    if not isinstance(envelope, dict) or set(envelope) != {"answer"}:
        raise ValueError("Expected an answer object")
    answer = envelope["answer"]
    if not isinstance(answer, dict) or answer.get("status") not in ("answered", "refusal"):
        raise ValueError("Invalid answer status")
    keys = {"status", "value", "unit", "explanation", "citations"}
    if set(answer) != keys or not isinstance(answer["explanation"], str) or not answer["explanation"].strip():
        raise ValueError("Missing or unexpected answer fields")
    if not isinstance(answer["citations"], list):
        raise ValueError("Invalid citations list")
    if answer["status"] == "refusal":
        if answer["value"] is not None or answer["unit"] is not None or answer["citations"]:
            raise ValueError("Invalid refusal fields")
    elif type(answer["value"]) not in (int, float) or not math.isfinite(answer["value"]) or answer["unit"] not in UNITS:
        raise ValueError("Invalid value or unit")
    for citation in answer["citations"]:
        strings = ("doc_name", "chunk_id", "raw_number", "label")
        if not isinstance(citation, dict) or set(citation) != {*strings, "pdf_page"}:
            raise ValueError("Invalid citation fields")
        if not all(isinstance(citation[k], str) and citation[k].strip() for k in strings):
            raise ValueError("Invalid citation text")
        if type(citation["pdf_page"]) is not int or citation["pdf_page"] < 1:
            raise ValueError("Invalid citation page")
    return answer


def run_one(question, system, config, settings, export):
    start = perf_counter()
    base, model, key, rates = config
    trace = {"question_id": question["question_id"], "system": system, "model": model, "status": "input_error",
             "value": None, "unit": None, "explanation": None, "citations": [], "messages": [], "context": "",
             "retrieved_chunks": [], "raw_response": None, "model_calls": 0, "usage": None, "estimated_api_usd": "unknown"}
    stage = "input_error"
    try:
        trace.update(build_input(question, system))
        if export:
            trace["status"] = "prompt_export"
        else:
            stage = "api_error"
            payload = json.dumps({**settings, "messages": trace["messages"]}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = "Bearer " + key
            request = Request(base + "/chat/completions", payload, headers)
            client = build_opener() if key else build_opener(ProxyHandler({}))
            trace["model_calls"] = 1
            with client.open(request, timeout=180) as response:
                body = response.read()
            stage = "parse_error"
            response = json.loads(body)
            trace["usage"] = response.get("usage")
            trace["estimated_api_usd"] = estimate_cost(trace["usage"], rates)
            trace["served_model"] = response.get("model")
            choice = response["choices"][0]
            trace["finish_reason"] = choice.get("finish_reason")
            trace["raw_response"] = choice["message"]["content"]
            # Some hosted providers represent a safety refusal separately from model text.
            refusal = choice["message"].get("refusal")
            if refusal and not trace["raw_response"]:
                trace.update(status="refusal", explanation=str(refusal), provider_refusal=True)
            else:
                trace.update(parse_answer(trace["raw_response"]))
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        trace["status"] = stage
        trace["error_type"] = type(error).__name__
        if isinstance(error, HTTPError):
            trace["http_status"] = error.code
        elif stage == "parse_error" and type(error) is ValueError:
            trace["parse_detail"] = str(error)
    trace["elapsed_seconds"] = round(perf_counter() - start, 4)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["predict", "export"])
    parser.add_argument("--scope", choices=["demo", "dev"], default="demo")
    parser.add_argument("--out", type=Path, help="New directory. Default: timestamped folder under results.")
    parser.add_argument("--hosted", action="store_true", help="Use the model in .env; default is local Qwen.")
    args = parser.parse_args()
    export = args.action == "export"
    config = configuration(args.hosted, not export)
    base, model, _, rates = config
    settings = generation_settings(model)
    questions = {q["question_id"]: q for q in read_rows(DATA / "questions.jsonl")}
    ids = [DEMO_ID] if args.scope == "demo" else json.loads((DATA / "splits.json").read_text(encoding="utf-8"))["dev"]
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(questions):
        parser.error("Invalid question IDs in the frozen split")
    now = datetime.now(timezone.utc)
    folder = args.out or ROOT / "results" / f"{args.action}-{args.scope}-{now:%Y%m%dT%H%M%S%fZ}"
    folder.mkdir(exist_ok=False, parents=True)
    manifest = {"protocol_version": PROTOCOL_VERSION, "mode": "prompt_export" if export else "live", "scope": args.scope, "started_utc": now.isoformat(),
                "question_ids": ids, "systems": ["B0", "B1"], "model": model, "endpoint": base,
                "pricing": rates, "request_settings": settings}
    write_json(folder / "manifest.json", manifest)
    for qid in ids:
        for system in manifest["systems"]:
            trace = run_one(questions[qid], system, config, settings, export)
            write_json(folder / system / (qid + ".json"), trace)
            print(f"{qid} {system} {trace['status']}", flush=True)
    print(folder.resolve())


if __name__ == "__main__":
    main()
