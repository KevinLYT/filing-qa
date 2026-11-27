"""Expose tools for one assigned filing through an MCP stdio server."""
import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from filing_tools import FilingTools, calculate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, help="doc_name from the question metadata")
    parser.add_argument("--retrieval", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument("--company")
    parser.add_argument("--fiscal-year", type=int)
    args = parser.parse_args()
    tools = FilingTools.for_document(args.doc, Path(__file__).resolve().parent / "data/index",
                                     retrieval=args.retrieval, company=args.company, fiscal_year=args.fiscal_year)
    # Both agent systems can discover and call these same functions through MCP.
    server = FastMCP("Financial filing tools")
    server.add_tool(tools.search_filings)
    server.add_tool(calculate)
    server.add_tool(tools.check_citation)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
