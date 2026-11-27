"""Shared tools for one filing QA session."""
import json
import re
from decimal import localcontext
from pathlib import Path

from filing_numbers import NUMBER, raw_key
from retrieve_b1 import retrieve


class FilingTools:
    def __init__(self, chunks, hybrid=None):
        self.chunks = chunks
        self.visible = {}
        self.hybrid = hybrid

    @classmethod
    def for_document(cls, doc_name, index_dir, *, retrieval="bm25", company=None, fiscal_year=None):
        # Select the whole filing from the question metadata before retrieval.
        if not re.fullmatch(r"[A-Za-z0-9_]+", doc_name):
            raise ValueError("Invalid filing identifier")
        path = Path(index_dir) / (doc_name + ".jsonl")
        with path.open(encoding="utf-8") as stream:
            chunks = [json.loads(line) for line in stream if line.strip()]
        if not chunks or any(chunk["doc_name"] != doc_name for chunk in chunks):
            raise ValueError("Index must contain chunks from exactly the requested filing")
        if company is not None and any(chunk["company"] != company for chunk in chunks):
            raise ValueError("Company metadata does not match the question's filing")
        if fiscal_year is not None and any(chunk["fiscal_year"] != fiscal_year for chunk in chunks):
            raise ValueError("Fiscal-year metadata does not match the question's filing")
        if retrieval == "bm25":
            return cls(chunks)
        if retrieval != "hybrid":
            raise ValueError(f"Unknown retrieval method: {retrieval}")
        from hybrid_search import HybridSearch
        return cls(chunks, HybridSearch.for_document(doc_name, index_dir, chunks))

    def search_filings(self, query: str) -> dict[str, list]:
        """Search the assigned filing. Returns up to two complete chunks with IDs, PDF pages and text."""
        if not query.strip() or len(query) > 1600:
            raise ValueError("Supply a search query of 1 to 1600 characters")
        if self.hybrid is None:
            ranked = retrieve(query, self.chunks, limit=2)
        else:
            ranked = self.hybrid.search(query, limit=2)
        chunks = [dict(hit, rank=rank) for rank, hit in enumerate(ranked, start=1)]
        self.visible.update({chunk["chunk_id"]: chunk for chunk in chunks})
        return {"chunks": chunks}

    def check_citation(self, chunk_id: str, figure: str) -> dict[str, object]:
        """Check a printed number in a chunk returned by search. Accounting negatives and percent signs matter."""
        # Only chunks returned for this question can support a citation. Account, year, and unit require separate review.
        chunk = self.visible.get(chunk_id)
        claimed = raw_key(figure)
        found = False
        if chunk is not None and claimed is not None:
            found = any(raw_key(match.group()) == claimed for match in NUMBER.finditer(chunk["text"]))
        return {"chunk_id": chunk_id, "figure": figure, "visible": chunk is not None, "number_found": found}


def calculate(operation: str, operands: list[str]) -> dict[str, object]:
    """Use two numeric strings. Operations: add, subtract, multiply, divide, percent_change [new,old]. 20% has magnitude 20."""
    if len(operands) != 2:
        raise ValueError("Supply exactly two printed numbers or decimal results")
    values = []
    for operand in operands:
        key = raw_key(operand) if isinstance(operand, str) and len(operand) <= 96 else None
        if key is None or not key[0].is_finite():
            raise ValueError("Use finite numeric strings with at most 96 characters")
        values.append(key[0])
    left, right = values
    if operation in ("divide", "percent_change") and right == 0:
        raise ValueError("The denominator must be nonzero")
    # Decimal avoids binary floating-point artifacts when comparing financial quantities.
    with localcontext() as precision:
        precision.prec = 48
        if operation == "add":
            result = left + right
        elif operation == "subtract":
            result = left - right
        elif operation == "multiply":
            result = left * right
        elif operation == "divide":
            result = left / right
        elif operation == "percent_change":
            result = (left - right) * 100 / right
        else:
            raise ValueError("Choose add, subtract, multiply, divide, or percent_change")
    return {"operation": operation, "operands": operands, "result": format(result, "f")}
