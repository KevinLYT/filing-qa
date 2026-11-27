"""Run matched filing QA systems on a frozen split."""
import argparse
import asyncio
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path

from langsmith import tracing_context
from multi_graph import run_multi_dialogue
from run_agent import TOOL_LIMIT, run_dialogue, run_question
from run_baselines import DATA, DEMO_ID, PROTOCOL_VERSION, ROOT, configuration, generation_settings
from run_baselines import read_rows, run_one, write_json

AGENTS = {"B2": run_dialogue, "Ours": partial(run_multi_dialogue, verify=True),
          "NoVerifier": partial(run_multi_dialogue, verify=False)}
SYSTEMS = ("B0", "B1", *AGENTS)
FINAL_STATUSES = ("answered", "refusal", "api_error", "input_error", "parse_error")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(data):
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def planned_ids(scope, split, questions):
    dev, held_out = split["dev"], split["held_out"]
    if len(set(dev)) != len(dev) or len(set(held_out)) != len(held_out) or set(dev) & set(held_out):
        raise ValueError("Development and held-out IDs must be unique and disjoint")
    ids = {"demo": [DEMO_ID], "dev": dev, "held-out": held_out}[scope]
    if not ids or set(ids) - set(questions) or (scope == "demo" and DEMO_ID not in dev):
        raise ValueError("Frozen question IDs are missing")
    return ids


@contextmanager
def run_lock(folder):
    # OS locks release on process exit even when a run is interrupted.
    with (folder / ".run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another writer is active in this run directory") from error
        yield


def make_manifest(args, config, ids, questions):
    docs = sorted({questions[qid]["doc_name"] for qid in ids})
    sources = ["run_baselines.py", "run_agent.py", "run_comparison.py", "multi_graph.py", "multi_roles.py",
               "retrieve_b1.py", "filing_mcp.py", "filing_tools.py", "filing_numbers.py", "start_qwen.ps1"]
    packages = ["langgraph", "langchain-core", "langsmith", "mcp", "pypdf"]
    retrieval_protocol = {"method": args.retrieval, "agent_chunks_per_search": 2}
    cache_signature = {}
    if args.retrieval == "hybrid":
        from hybrid_search import hybrid_manifest
        hybrid = hybrid_manifest(docs, DATA / "index")
        sources.extend(["hybrid_search.py", "setup_retrieval.py"])
        packages.extend(["numpy", "onnxruntime", "tokenizers"])
        retrieval_protocol["hybrid"] = hybrid["protocol"]
        cache_signature = hybrid["documents"]
    # This digest excludes the question set so a held-out run can prove a protocol match to development.
    protocol = {"protocol_version": "paired-filing-qa-v4", "baseline_protocol": PROTOCOL_VERSION,
                "b2_protocol": "single-agent-mcp-v2", "model": config[1], "endpoint": config[0], "pricing": config[3],
                "retrieval": retrieval_protocol, "baseline_retrieval": {"B0": "none", "B1": "BM25 top 4"},
                "budget": {"max_tool_calls": TOOL_LIMIT, "max_model_calls": TOOL_LIMIT + 1, "baseline_model_calls": 1,
                           "temperature": 0, "max_output_tokens_per_call": 800, "max_verification_retries": 1},
                "python": platform.python_version(), "packages": {p: version(p) for p in packages},
                "source_sha256": {f: sha256(ROOT / f) for f in sources}}
    if not args.hosted:
        protocol["local_gguf_sha256"] = sha256(ROOT / "models" / config[1])
        protocol["local_llama_server_sha256"] = sha256(ROOT / "tools/llama/llama-server.exe")
    protocol_digest = fingerprint(protocol)
    if args.frozen_from:
        prior = json.loads(args.frozen_from.read_text(encoding="utf-8"))
        if prior["protocol_sha256"] != protocol_digest:
            raise ValueError("Model, source, retrieval settings or dependencies differ from the frozen manifest")
    order = {}
    for i, qid in enumerate(ids):
        offset = i % len(args.systems)
        order[qid] = args.systems[offset:] + args.systems[:offset]
    run = {**protocol, "mode": "live", "protocol_sha256": protocol_digest, "scope": args.scope, "systems": args.systems,
           "order": order, "question_ids": ids, "input_sha256": fingerprint([questions[qid] for qid in ids]),
           "filing_index_sha256": {d: sha256(DATA / "index" / f"{d}.jsonl") for d in docs},
           "hybrid_index_signatures": cache_signature}
    return {**run, "run_sha256": fingerprint(run)}


def current_trace(path, qid, system, manifest):
    if not path.exists():
        return False
    trace = json.loads(path.read_text(encoding="utf-8"))
    if (trace.get("run_sha256") != manifest["run_sha256"] or trace.get("question_id") != qid
            or trace.get("system") != system or trace.get("model") != manifest["model"]
            or trace.get("status") not in FINAL_STATUSES):
        raise ValueError(f"Unexpected prediction in {path}. Resume requires an intact finished trace.")
    return True


def store_or_match_manifest(folder, manifest, resume):
    path = folder / "manifest.json"
    if resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.pop("started_utc")
        if previous != manifest:
            raise ValueError("Run configuration changed. Resume requires the original source, data, model and settings.")
    else:
        write_json(path, {**manifest, "started_utc": datetime.now(timezone.utc).isoformat()})


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("demo", "dev", "held-out"), default="demo")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(AGENTS))
    parser.add_argument("--retrieval", choices=("bm25", "hybrid"), default="bm25", help="Shared agent retrieval backend")
    parser.add_argument("--out", type=Path, help="New output directory, or original directory with --resume")
    parser.add_argument("--resume", action="store_true", help="Keep all completed traces, including recorded failures")
    parser.add_argument("--frozen-from", type=Path, help="Require the same inference protocol as this earlier manifest")
    parser.add_argument("--hosted", action="store_true", help="Use the configured remote model instead of local Qwen")
    args = parser.parse_args()
    if len(set(args.systems)) != len(args.systems):
        parser.error("Each system must appear once")
    if args.resume and not args.out:
        parser.error("Choose the existing run directory with --out when resuming")
    questions = {q["question_id"]: q for q in read_rows(DATA / "questions.jsonl")}
    split = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    ids = planned_ids(args.scope, split, questions)
    config = configuration(args.hosted, True)
    manifest = make_manifest(args, config, ids, questions)
    now = datetime.now(timezone.utc)
    folder = args.out or ROOT / "results" / f"comparison-{args.scope}-{now:%Y%m%dT%H%M%S%fZ}"
    if not args.resume:
        folder.mkdir(parents=True, exist_ok=False)
    settings = generation_settings(config[1])
    with run_lock(folder), tracing_context(enabled=False):
        store_or_match_manifest(folder, manifest, args.resume)
        for qid, systems in manifest["order"].items():
            for system in systems:
                path = folder / system / f"{qid}.json"
                if current_trace(path, qid, system, manifest):
                    continue
                if system in AGENTS:
                    trace = await run_question(questions[qid], config, system=system,
                                               dialogue=AGENTS[system], retrieval=args.retrieval)
                else:
                    trace = run_one(questions[qid], system, config, settings, export=False)
                trace["run_sha256"] = manifest["run_sha256"]
                write_json(path, trace)
                print(f"{qid} {system} {trace['status']} | {trace['model_calls']} model calls", flush=True)
    print(folder.resolve())


if __name__ == "__main__":
    asyncio.run(main())
