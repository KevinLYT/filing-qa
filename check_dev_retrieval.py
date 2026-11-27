"""Compare two retrieval methods on the seven development questions."""
import argparse
import json
from pathlib import Path

from hybrid_search import HybridSearch
from retrieve_b1 import retrieve
from run_baselines import DATA, read_rows, write_json


def rank_dev():
    ids = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))["dev"]
    questions = {q["question_id"]: q for q in read_rows(DATA / "questions.jsonl")}
    rows = []
    for qid in ids:
        question = questions[qid]
        doc = question["doc_name"]
        chunks = read_rows(DATA / "index" / f"{doc}.jsonl")
        model = HybridSearch.for_document(doc, DATA / "index", chunks)
        for method in ("bm25", "hybrid"):
            hits = (model.search(question["question"], limit=2) if method == "hybrid"
                    else retrieve(question["question"], chunks, limit=2))
            rows.append({"question_id": qid, "method": method, "doc_name": doc,
                         "hits": [{k: v for k, v in hit.items() if k != "text"} for hit in hits]})
    return rows


def evaluate(rows):
    # Read reference pages after ranking. Other pages can repeat the same financial statement item.
    gold = {r["question_id"]: r for r in read_rows(DATA / "evaluation/gold.jsonl")}
    summary = {}
    for row in rows:
        refs = {(e["doc_name"], int(e["evidence_page_num"]) + 1) for e in gold[row["question_id"]]["evidence"]}
        hits = [(c["doc_name"], c["pdf_page"]) for c in row["hits"]]
        ranks = [i for i, item in enumerate(hits, 1) if item in refs]
        row["reference_pages"] = sorted(refs)
        row["first_matching_rank"] = min(ranks) if ranks else None
    for method in ("bm25", "hybrid"):
        group = [r["first_matching_rank"] for r in rows if r["method"] == method]
        summary[method] = {"questions": len(group), "any_gold_page_at_2": sum(r is not None for r in group),
                           "mrr_at_2": sum(1 / r for r in group if r is not None) / len(group)}
    return {"scope": "dev", "query": "original question verbatim", "top_k": 2, "summary": summary, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    result = evaluate(rank_dev())
    write_json(args.out, result)
    print(json.dumps(result["summary"], indent=2))
