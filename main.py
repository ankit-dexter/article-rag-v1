"""
CLI entry point — index articles and run interactive search.

This is the command-line interface for the Article Topic RAG system.
It provides two modes of operation:
  1. Interactive mode (default): presents a "search>" prompt where you
     can type queries and see results in real time.
  2. Single-query mode (--query): runs one search and exits, useful
     for scripting or quick lookups.

Before searching, it always ensures the ChromaDB index is built and up to date.

Usage:
    python main.py                          # index + interactive search
    python main.py --reindex                # force re-embed all articles
    python main.py --query "oil prices"     # single query, non-interactive
"""

# --- Standard library imports ---
# `argparse` provides the command-line argument parsing framework.
# It automatically generates help text and validates user input.
import argparse

# `os` is used for file path construction and reading environment variables.
import os

# --- Local imports ---
# `ArticleIndexer` reads raw article files, generates embeddings, and
# stores them in ChromaDB. Used in the indexing step.
from indexer import ArticleIndexer

# `ArticleSearcher` performs semantic search against the ChromaDB index.
# Used in both interactive and single-query search modes.
from searcher import ArticleSearcher

# ──────────────────────────────────────────────────────────────────────
# Configuration — paths for data and the ChromaDB persistence directory
# ──────────────────────────────────────────────────────────────────────

# Resolve the absolute path of the directory this script lives in.
# __file__ is the path to this script; abspath makes it absolute;
# dirname extracts just the directory part.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the raw article JSONL data files.
# Checks for a CLASSIFIER_DATA_PATH environment variable first (useful for
# Docker or production). Falls back to the sibling project's data directory.
DATA_DIR = os.environ.get(
    "CLASSIFIER_DATA_PATH",
    os.path.join(PROJECT_ROOT, "..", "article-topic-classifier", "data"),
)

# Path where ChromaDB persists its vector index on disk.
# This is shared between the CLI (main.py) and the API (api.py) so both
# use the same index.
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_store")


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def build_index(force: bool = False) -> int:
    """
    Build or load the article search index.

    Creates an ArticleIndexer, points it at the data and ChromaDB directories,
    and runs the indexing pipeline. On first run this embeds all articles (slow).
    On subsequent runs it detects the existing index and skips (fast).

    Args:
        force: If True, re-embed all articles even if the index already exists.
               Use this when the underlying data files have changed.

    Returns:
        The total number of articles in the index.
    """
    # Create the indexer with our configured paths
    indexer = ArticleIndexer(data_dir=DATA_DIR, persist_dir=PERSIST_DIR)

    # Run the indexing pipeline and get back the article count
    count = indexer.index(force=force)
    return count


def run_search(query: str, top_k: int = 5, split=None, topic=None):
    """
    Execute a single semantic search query.

    Creates an ArticleSearcher, connects to the ChromaDB index, and
    performs a similarity search for articles matching the query.

    Args:
        query:  Natural language search text (e.g., "oil prices rising")
        top_k:  How many results to return (default: 5)
        split:  Optional filter by data split ("train", "val", "test")
        topic:  Optional filter by news category ("World", "Sports", etc.)

    Returns:
        List of matching article dicts, each with title, topic, split,
        similarity score, and a text snippet.
    """
    # Create a new searcher instance connected to our ChromaDB store
    searcher = ArticleSearcher(persist_dir=PERSIST_DIR)

    # Perform the semantic search and return the results
    results = searcher.search(query=query, top_k=top_k, split=split, topic=topic)
    return results


def print_results(results: list[dict]):
    """
    Display search results in a human-readable format in the terminal.

    Each result is printed with a separator line, its similarity score,
    topic, data split, title, and a text snippet.

    Args:
        results: List of article dicts as returned by run_search()
    """
    # Handle the case where the search returned no matches
    if not results:
        print("No results found.")
        return

    # Iterate over results with 1-based numbering (enumerate starts at 1)
    for i, r in enumerate(results, 1):
        # Print a separator with the result number and similarity score
        print(f"\n--- Result {i} (similarity: {r['similarity']}) ---")

        # Print the metadata fields: news topic and data split
        print(f"Topic: {r['topic']}  |  Split: {r['split']}")

        # Print the article title
        print(f"Title: {r['title']}")

        # Print a preview of the article text (first 300 chars from searcher)
        print(f"Snippet: {r['snippet']}")


# ──────────────────────────────────────────────────────────────────────
# Interactive search mode
# ──────────────────────────────────────────────────────────────────────

def interactive_search():
    """
    Run an interactive search loop in the terminal.

    Presents a "search>" prompt where the user can type queries and see
    results immediately. Supports inline filters using key=value syntax:
        search> oil prices topic=Business split=test

    The loop continues until the user types "quit", "exit", or "q".
    """
    # Print instructions for the user
    print("\nInteractive search (type 'quit' to exit)")
    print("Optional filters: topic=World|Sports|Business|Sci/Tech  split=train|val|test\n")

    # Main input loop — keeps asking for queries until the user quits
    while True:
        # Read a line of input from the user and remove leading/trailing whitespace
        raw = input("search> ").strip()

        # Check if the user wants to exit the search loop
        if raw.lower() in ("quit", "exit", "q"):
            break

        # Skip empty input (user just pressed Enter)
        if not raw:
            continue

        # ── Parse optional filters from the query string ──
        # The user can mix search terms with filter expressions like:
        #   "oil prices topic=Business split=test"
        # We need to separate the actual search query from the filters.
        parts = raw.split()       # Split input into whitespace-separated tokens
        query_parts = []          # Will hold the actual search words
        split = None              # Will hold the split filter value, if any
        topic = None              # Will hold the topic filter value, if any

        for p in parts:
            if p.startswith("split="):
                # Extract the value after "split=" (e.g., "train" from "split=train")
                # split("=", 1) splits on first "=" only, in case value contains "="
                split = p.split("=", 1)[1]
            elif p.startswith("topic="):
                # Extract the value after "topic=" (e.g., "Sports" from "topic=Sports")
                topic = p.split("=", 1)[1]
            else:
                # This token is a regular search word, not a filter
                query_parts.append(p)

        # Rejoin the search words into a single query string
        query = " ".join(query_parts)

        # If all tokens were filters and no actual query was provided, prompt again
        if not query:
            print("Please provide a search query.")
            continue

        # Run the search with the parsed query and any filters
        results = run_search(query, split=split, topic=topic)

        # Display the results in the terminal
        print_results(results)


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    """
    Parse command-line arguments, build the index, and run search.

    This function orchestrates the two main steps:
      Step 1: Build/load the ChromaDB vector index (always runs)
      Step 2: Either run a single query (--query) or start interactive mode
    """
    # ── Set up the argument parser ──
    # argparse handles --help, validates types, and provides defaults.
    parser = argparse.ArgumentParser(description="Article Topic RAG Search")

    # --reindex: boolean flag (no value needed). When present, forces the indexer
    # to re-embed all articles even if the index already exists.
    parser.add_argument("--reindex", action="store_true", help="Force re-index all articles")

    # --query: takes a string value. If provided, runs a single search instead
    # of entering interactive mode. Useful for scripting.
    parser.add_argument("--query", type=str, help="Single search query (non-interactive)")

    # --top-k: how many results to return. Defaults to 5 if not specified.
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    # --split: filter results to a specific data split
    parser.add_argument("--split", type=str, help="Filter by split: train, val, test")

    # --topic: filter results to a specific news category
    parser.add_argument("--topic", type=str, help="Filter by topic: World, Sports, Business, Sci/Tech")

    # Parse the command-line arguments into a namespace object.
    # e.g., args.reindex=True, args.query="oil prices", args.top_k=5
    args = parser.parse_args()

    # ── Step 1: Build or load the index ──
    # This always runs first, regardless of search mode.
    # If the index already exists on disk, this is fast (just loads metadata).
    # If it's the first run or --reindex is passed, this embeds all articles.
    print("Building/loading index...")
    count = build_index(force=args.reindex)
    print(f"Index ready — {count} articles indexed.")

    # ── Step 2: Search ──
    if args.query:
        # Single-query mode: run one search with the provided query and filters,
        # print results, and exit. Useful for scripts and one-off lookups.
        results = run_search(args.query, top_k=args.top_k, split=args.split, topic=args.topic)
        print_results(results)
    else:
        # Interactive mode (default): start the search loop where the user
        # can type multiple queries at the "search>" prompt.
        interactive_search()


# This is the standard Python idiom for "run main() only when this file
# is executed directly (python main.py), not when it's imported as a module."
# It allows other files to import functions like build_index() and run_search()
# without triggering the CLI.
if __name__ == "__main__":
    main()
