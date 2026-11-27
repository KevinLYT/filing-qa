"""Build the pinned FinanceBench sample, evaluation labels and complete filing indexes."""
import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
RAW_URL = f"https://raw.githubusercontent.com/patronus-ai/financebench/{REVISION}/"
DEV_TASKS = {
    "financebench_id_03031": "balance-sheet subtraction",
    "financebench_id_03029": "cash-flow lookup",
    "financebench_id_04672": "balance-sheet lookup with scale conversion",
    "financebench_id_07507": "year-over-year operating income growth",
    "financebench_id_04854": "free cash flow calculation",
    "financebench_id_08286": "income statement lookup",
    "financebench_id_03531": "current assets lookup",
}
QUESTION_FILE = "financebench_open_source.jsonl"
DOCUMENT_FILE = "financebench_document_information.jsonl"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path, content):
    path.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(content)
    Path(temp.name).replace(path)


def save_json(path, data):
    save(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


def save_rows(path, data):
    save(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data).encode())


def fetch_file(entry):
    name = Path(entry["path"]).name
    location = DATA / ("raw" if entry["path"].startswith("data/") else "filings") / name
    local = location.is_file()
    if local:
        content = location.read_bytes()
    else:
        with urlopen(RAW_URL + quote(entry["path"]), timeout=120) as response:
            content = response.read()
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if len(content) != entry["size"] or digest != entry["sha"]:
        raise ValueError(f"Pinned release mismatch for {location}. Check existing file before replacing it.")
    if not local:
        save(location, content)


def fetch():
    # The lock records a Git blob hash for each file at the pinned commit.
    lock = json.loads((DATA / "raw/source_tree.json").read_text(encoding="utf-8-sig"))
    if lock["revision"] != REVISION:
        raise ValueError("Source lock revision mismatch.")
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch_file, lock["files"]))
    print(f"Verified {len(lock['files'])} files at {REVISION}.", flush=True)


def score_target(row, document):
    if document["doc_type"].lower() != "10k":
        return None, "not_10k"
    answer = row["answer"].strip()
    question = row["question"].lower()
    if re.fullmatch(r"-?\d+(?:\.\d+)?%", answer):
        # Margin differences use percentage points. Relative growth rates use percent.
        unit = "percentage_points" if "change in unadjusted operating income % margin" in question else "percent"
        return {"value": answer[:-1], "unit": unit}, None
    scale = re.search(r"\busd (millions|billions)\b", question)
    if scale and re.fullmatch(r"\$-?\d+(?:\.\d+)?", answer):
        return {"value": answer[1:], "unit": "USD_" + scale[1]}, None
    return None, "outside_single_explicit_usd_or_percent_rule"


def prepare():
    raw = rows(DATA / "raw" / QUESTION_FILE)
    documents = {doc["doc_name"]: doc for doc in rows(DATA / "raw" / DOCUMENT_FILE)}
    public, gold, excluded = [], [], {}
    for row in raw:
        qid = row["financebench_id"]
        document = documents[row["doc_name"]]
        target, reason = score_target(row, document)
        if reason:
            excluded[qid] = reason
            continue
        # Route each question to its full assigned filing and store reference answers and evidence separately.
        public.append({"question_id": qid, "question": row["question"], "doc_name": document["doc_name"],
                       "company": document["company"], "fiscal_year": document["doc_period"]})
        gold.append({"question_id": qid, **target, "official_answer": row["answer"], "evidence": row["evidence"]})
    ids = {q["question_id"] for q in public}
    if len(raw) != 150 or len(public) != len(ids) or not set(DEV_TASKS) <= ids:
        raise ValueError("Unexpected release, duplicate IDs, or unknown dev IDs.")
    split = {"source_count": len(raw), "eligible_count": len(public), "dev": list(DEV_TASKS),
             "dev_tasks": DEV_TASKS, "held_out": sorted(ids.difference(DEV_TASKS)), "excluded": excluded}
    save_json(DATA / "splits.json", split)
    save_rows(DATA / "questions.jsonl", public)
    save_rows(DATA / "evaluation/gold.jsonl", gold)
    print(f"{len(raw)} source rows -> {len(public)} eligible = {len(DEV_TASKS)} dev + {len(split['held_out'])} held out.", flush=True)


def build_one(document):
    from retrieve_b1 import read_filing

    name = document["doc_name"]
    path = DATA / "filings" / (name + ".pdf")
    page_count, chunks = read_filing(path, name, document["company"], document["fiscal_year"])
    save_rows(DATA / "index" / (name + ".jsonl"), chunks)
    with_text = {c["pdf_page"] for c in chunks}
    return {"doc_name": name, "pdf_pages": page_count, "chunks": len(chunks),
            "empty_pdf_pages": sorted(set(range(1, page_count + 1)) - with_text)}


def build_indexes():
    public = rows(DATA / "questions.jsonl")
    docs = {q["doc_name"]: q for q in public}
    dev_docs = {q["doc_name"] for q in public if q["question_id"] in DEV_TASKS}
    ordered = sorted(docs, key=lambda name: (name not in dev_docs, name))
    with ProcessPoolExecutor(max_workers=4) as pool:
        coverage = []
        for item in pool.map(build_one, [docs[name] for name in ordered]):
            coverage.append(item)
            print(f"{item['doc_name']}: {item['pdf_pages']} pages, {item['chunks']} chunks", flush=True)
    save_json(DATA / "index/coverage.json", coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "prepare", "index", "all"], nargs="?", default="all")
    stage = parser.parse_args().stage
    if stage in ("fetch", "all"):
        fetch()
    if stage in ("prepare", "all"):
        prepare()
    if stage in ("index", "all"):
        build_indexes()

