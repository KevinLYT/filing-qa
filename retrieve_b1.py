"""Rank excerpts from the complete filing for the single-question B1 baseline."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from b0 import QUESTION

ROOT = Path(__file__).resolve().parent
# The FinanceBench question specifies this document; the search must discover the page.
DOCUMENT = "LOCKHEEDMARTIN_2021_10K"
PDF = ROOT / "data" / "filings" / f"{DOCUMENT}.pdf"
SOURCE = "https://github.com/patronus-ai/financebench/blob/cc39aeb4afdf33909ee1412188bf89035950c2eb/pdfs/"
STOP_WORDS = set("a an and are as at be by for from how in into is it of on or that the this to was what which with".split())


def split_page(text, limit=3200):
    # Split between rows and overlap two lines when a long page spans multiple chunks.
    block = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]{2,}", "  ", raw_line.strip())
        if not line:
            continue
        if block and len("\n".join(block + [line])) > limit:
            yield "\n".join(block)
            block = block[-2:] if len(block) > 2 else []
        block.append(line)
    if block:
        yield "\n".join(block)


def read_filing(pdf_path, doc_name=DOCUMENT, company="Lockheed Martin", fiscal_year=2021):
    reader = PdfReader(pdf_path)
    chunks = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
        for part, excerpt in enumerate(split_page(text), start=1):
            chunks.append({
                "chunk_id": f"{doc_name}:p{index + 1}:c{part}", "doc_name": doc_name,
                "company": company, "fiscal_year": fiscal_year,
                "page_index": index, "pdf_page": index + 1, "text": excerpt,
            })
    return len(reader.pages), chunks


def terms(text):
    words = re.findall(r"[a-z]+|\d+", text.lower())
    single = [w for w in words if w not in STOP_WORDS]
    pairs = [a + "_" + b for a, b in zip(words, words[1:]) if a not in STOP_WORDS and b not in STOP_WORDS]
    return single + pairs


def retrieve(question, chunks, limit=4):
    # BM25 favors distinctive matching terms and accounts for excerpt length.
    if not chunks:
        return []
    frequency = [Counter(terms(chunk["text"])) for chunk in chunks]
    lengths = [counts.total() for counts in frequency]
    average_length = sum(lengths) / len(chunks) or 1
    weights = {}
    for term in sorted(set(terms(question))):
        matches = sum(term in counts for counts in frequency)
        weights[term] = math.log1p((len(chunks) - matches + 0.5) / (matches + 0.5))

    ranked = []
    for chunk, counts, length in zip(chunks, frequency, lengths):
        length_penalty = 1.5 * (0.25 + 0.75 * length / average_length)
        score = 0.0
        for term, weight in weights.items():
            count = counts[term]
            score += weight * 2.5 * count / (count + length_penalty)
        if score > 0:
            ranked.append(dict(chunk, score=score))
    ranked.sort(key=lambda item: (-item["score"], item["pdf_page"], item["chunk_id"]))
    return ranked[:limit]


def main():
    pages, chunks = read_filing(PDF)
    hits = retrieve(QUESTION, chunks)
    index_file = ROOT / "data" / "index" / f"{DOCUMENT}.jsonl"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with index_file.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    report = ROOT / "results" / "b1" / "retrieval.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# B1 retrieval", "", QUESTION, "", f"Searched {pages} PDF pages and {len(chunks)} chunks.",
             f"[Source filing]({SOURCE}{PDF.name})", ""]
    for rank, hit in enumerate(hits, start=1):
        heading = f"Rank {rank} | {hit['chunk_id']} | BM25 {hit['score']:.4f}"
        print(heading)
        lines.extend(["## " + heading, "", "~~~text", hit["text"], "~~~", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Searched {pages} PDF pages, {len(chunks)} chunks. Saved: {report}")


if __name__ == "__main__":
    main()
