"""B1: send the original question and the four highest-ranked filing excerpts to Qwen."""
import json
from time import perf_counter

from b0 import QUESTION, ask_model, save_record
from retrieve_b1 import DOCUMENT, ROOT, retrieve


def build_message(question, hits):
    # Place the chunk ID and page beside the excerpt sent to the model.
    blocks = []
    for hit in hits:
        header = f"[{hit['chunk_id']} | PDF page {hit['pdf_page']}]"
        blocks.append(header + "\n" + hit["text"])
    return question + "\n\nFiling excerpts:\n" + "\n\n".join(blocks)


def main():
    start = perf_counter()
    index = ROOT / "data" / "index" / f"{DOCUMENT}.jsonl"
    with index.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]

    # Search the complete index with the original question.
    hits = retrieve(QUESTION, chunks)
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
    message = build_message(QUESTION, hits)
    payload, result = ask_model(message)
    path = save_record(payload, result, perf_counter() - start, system="B1", retrieved_chunks=hits)
    print("Pages sent:", [hit["pdf_page"] for hit in hits])
    print("Answer:", result["choices"][0]["message"]["content"])
    print("Saved:", path)


if __name__ == "__main__":
    main()
