"""Offline BM25, dense passage retrieval and cross-encoder reranking."""
import argparse
import hashlib
import json
import os
import re
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from retrieve_b1 import retrieve

ROOT = Path(__file__).resolve().parent
WINDOW = 192
OVERLAP = 32
RRF_K = 60
CANDIDATES = 8
BATCH = 12
THREADS = 4
CONFIG = {"method": "bm25-minilm-rrf-crossencoder-v1", "window_tokens": WINDOW, "overlap_tokens": OVERLAP,
          "dense_max_tokens": 256, "cross_encoder_max_tokens": 512, "rrf_k": RRF_K, "rerank_candidates": CANDIDATES,
          "window_aggregation": "max", "backend": "CPUExecutionProvider", "threads": THREADS, "batch_size": BATCH}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_path(doc, index_dir):
    if not re.fullmatch(r"[A-Za-z0-9_]+", doc):
        raise ValueError("Invalid document identifier")
    return Path(index_dir).parent / "hybrid_index" / f"{doc}.npz"


def read_chunks(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def fingerprint(chunks, dense_assets):
    # Bind vector offsets to the exact ordered text and tokenization used at build time.
    digest = hashlib.sha256(json.dumps([CONFIG, dense_assets, version("onnxruntime"), version("tokenizers")],
                                       sort_keys=True).encode())
    for chunk in chunks:
        row = [chunk["doc_name"], chunk["chunk_id"], chunk["pdf_page"], chunk["text"]]
        digest.update(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")
    return digest.hexdigest()


def model_dir(index_dir):
    return Path(index_dir).resolve().parents[1] / "models" / "retrieval"


@lru_cache(maxsize=4)
def asset_hashes(folder):
    return {file: file_hash(folder / file) for file in ("model.onnx", "tokenizer.json")}


def normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("The dense model produced an invalid vector")
    return vectors / norms


def token_windows(text, tokenizer):
    # Score overlapping windows so late rows in long tables can surface their complete source chunk.
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    for first in range(0, len(offsets), WINDOW - OVERLAP):
        last = min(first + WINDOW, len(offsets)) - 1
        yield offsets[first][0], offsets[last][1]
        if last == len(offsets) - 1:
            break


class OnnxEncoder:
    def __init__(self, folder, role):
        self.role = role
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=CONFIG["dense_max_tokens"] if role == "dense" else CONFIG["cross_encoder_max_tokens"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"])
        expected = {"last_hidden_state", "token_embeddings"} if role == "dense" else {"logits"}
        self.output_name = next(output.name for output in self.session.get_outputs() if output.name in expected)
        self.inputs = [item.name for item in self.session.get_inputs()]

    def encode(self, sequences):
        batches = []
        for start in range(0, len(sequences), BATCH):
            tokens = self.tokenizer.encode_batch(sequences[start:start + BATCH])
            attributes = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
            inputs = {name: np.asarray([getattr(t, attributes[name]) for t in tokens], dtype=np.int64) for name in self.inputs}
            output = self.session.run([self.output_name], inputs)[0]
            if self.role == "dense":
                mask = inputs["attention_mask"][:, :, None].astype(np.float32)
                pooled = (output * mask).sum(axis=1) / mask.sum(axis=1)
                batches.append(normalize_rows(pooled))
            else:
                if output.shape != (len(tokens), 1) or not np.isfinite(output).all():
                    raise ValueError("Expected one finite relevance logit per query/window pair")
                batches.append(output[:, 0])
        return np.concatenate(batches, axis=0)


@lru_cache(maxsize=4)
def encoder(folder, role):
    return OnnxEncoder(folder / role, role)


def validate_scope(doc, chunks):
    if not chunks or any(c["doc_name"] != doc for c in chunks):
        raise ValueError("Expected nonempty chunks from one assigned filing")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Each filing chunk needs a unique identifier")


def build_index(doc, index_dir, chunks):
    path = cache_path(doc, index_dir)
    validate_scope(doc, chunks)
    folder = model_dir(index_dir)
    signature = fingerprint(chunks, asset_hashes(folder / "dense"))
    if path.is_file():
        with np.load(path, allow_pickle=False) as previous:
            if previous["fingerprint"].item() == signature:
                return path
    tokenizer = Tokenizer.from_file(str(folder / "dense" / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    spans = []
    passages = []
    for owner, chunk in enumerate(chunks):
        text = chunk["text"]
        for start, end in token_windows(text, tokenizer):
            spans.append((owner, start, end))
            passages.append(text[start:end])
    if len({s[0] for s in spans}) != len(chunks):
        raise ValueError("Each indexed chunk must contain tokenizable text")
    vectors = encoder(folder, "dense").encode(passages)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, fingerprint=signature, spans=np.asarray(spans, dtype=np.int32), vectors=vectors)
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return path


class HybridSearch:
    def __init__(self, chunks, spans, vectors, folder):
        self.chunks = chunks
        self.spans = spans
        self.vectors = vectors
        self.folder = folder

    @classmethod
    def for_document(cls, doc, index_dir, chunks):
        path = cache_path(doc, index_dir)
        validate_scope(doc, chunks)
        folder = model_dir(index_dir)
        signature = fingerprint(chunks, asset_hashes(folder / "dense"))
        if not path.is_file():
            raise FileNotFoundError(f"Build the dense filing index first: {path}")
        with np.load(path, allow_pickle=False) as cache:
            if cache["fingerprint"].item() != signature:
                raise ValueError(f"Stale dense index for {doc}; run hybrid_search.py build")
            spans, vectors = cache["spans"], cache["vectors"]
        return cls(chunks, spans, vectors, folder)

    def search(self, query, limit=2):
        if not query.strip() or not 1 <= limit <= CANDIDATES:
            raise ValueError("Supply a nonempty query and a supported result limit")
        count = len(self.chunks)
        lexical = retrieve(query, self.chunks, limit=count)
        bm25 = {hit["chunk_id"]: (i, hit["score"]) for i, hit in enumerate(lexical, 1)}
        vector = encoder(self.folder, "dense").encode([query])[0]
        similarities = self.vectors @ vector
        dense_scores = np.full(count, -np.inf, dtype=np.float32)
        np.maximum.at(dense_scores, self.spans[:, 0], similarities)
        dense_order = sorted(range(count), key=lambda i: (-float(dense_scores[i]), self.chunks[i]["chunk_id"]))
        # Fuse ranks to avoid directly mixing BM25 values and cosine similarities.
        candidates = []
        for dense_rank, i in enumerate(dense_order, 1):
            chunk = self.chunks[i]
            bm_rank, bm_score = bm25.get(chunk["chunk_id"], (None, 0.0))
            rrf = 1 / (RRF_K + dense_rank)
            if bm_rank is not None:
                rrf += 1 / (RRF_K + bm_rank)
            hit = dict(chunk, bm25_rank=bm_rank, bm25_score=bm_score, dense_rank=dense_rank,
                       dense_score=float(dense_scores[i]), rrf_score=rrf)
            candidates.append((i, hit))
        candidates.sort(key=lambda pair: (-pair[1]["rrf_score"], pair[1]["chunk_id"]))
        shortlist = dict(candidates[:CANDIDATES])
        pairs, owners = [], []
        for i, left, right in self.spans:
            if i in shortlist:
                pairs.append((query, self.chunks[i]["text"][left:right]))
                owners.append(i)
        logits = encoder(self.folder, "reranker").encode(pairs)
        reranked = np.full(count, -np.inf, dtype=np.float32)
        np.maximum.at(reranked, owners, logits)
        for rrf_rank, (i, hit) in enumerate(shortlist.items(), 1):
            hit.update(rrf_rank=rrf_rank, rerank_score=float(reranked[i]), score=float(reranked[i]))
        ranked = sorted(shortlist.values(), key=lambda hit: (-hit["rerank_score"], hit["rrf_rank"]))
        return [dict(hit, rank=rank, retrieval_method="hybrid") for rank, hit in enumerate(ranked[:limit], 1)]


def hybrid_manifest(doc_names, index_dir):
    folder = model_dir(index_dir)
    models = {name: asset_hashes(folder / name) for name in ("dense", "reranker")}
    protocol = dict(CONFIG, assets=models, packages={k: version(k) for k in ("numpy", "onnxruntime", "tokenizers")})
    docs = {}
    for doc in sorted(doc_names):
        path = cache_path(doc, index_dir)
        with np.load(path, allow_pickle=False) as cache:
            docs[doc] = {"embedding_fingerprint": cache["fingerprint"].item(), "npz_sha256": file_hash(path)}
    return {"protocol": protocol, "documents": docs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--scope", choices=["dev", "eligible"], default="dev")
    args = parser.parse_args()
    index_dir = ROOT / "data/index"
    questions = read_chunks(ROOT / "data/questions.jsonl")
    split = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))
    ids = set(split["dev"])
    if args.scope == "eligible":
        ids.update(split["held_out"])
    docs = sorted({q["doc_name"] for q in questions if q["question_id"] in ids})
    for i, doc in enumerate(docs, 1):
        chunks = read_chunks(index_dir / f"{doc}.jsonl")
        print(f"[{i}/{len(docs)}] {doc}: {len(chunks)} chunks", flush=True)
        build_index(doc, index_dir, chunks)


if __name__ == "__main__":
    main()
