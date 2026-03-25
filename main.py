"""
CLI entry point — index articles and run interactive search.

Usage:
    python main.py                          # index + interactive search
    python main.py --reindex                # force re-embed all articles
    python main.py --query "oil prices"     # single query, non-interactive
"""

import argparse
import os
from indexer import ArticleIndexer
from searcher import ArticleSearcher

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "CLASSIFIER_DATA_PATH",
    os.path.join(PROJECT_ROOT, "..", "article-topic-classifier", "data"),
)
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_store")


def build_index(force: bool = False) -> int:
    indexer = ArticleIndexer(data_dir=DATA_DIR, persist_dir=PERSIST_DIR)
    count = indexer.index(force=force)
    return count


def run_search(query: str, top_k: int = 5, split=None, topic=None):
    searcher = ArticleSearcher(persist_dir=PERSIST_DIR)
    results = searcher.search(query=query, top_k=top_k, split=split, topic=topic)
    return results


def print_results(results: list[dict]):
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (similarity: {r['similarity']}) ---")
        print(f"Topic: {r['topic']}  |  Split: {r['split']}")
        print(f"Title: {r['title']}")
        print(f"Snippet: {r['snippet']}")


def interactive_search():
    print("\nInteractive search (type 'quit' to exit)")
    print("Optional filters: topic=World|Sports|Business|Sci/Tech  split=train|val|test\n")

    while True:
        raw = input("search> ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        # Parse optional filters from the query
        parts = raw.split()
        query_parts = []
        split = None
        topic = None
        for p in parts:
            if p.startswith("split="):
                split = p.split("=", 1)[1]
            elif p.startswith("topic="):
                topic = p.split("=", 1)[1]
            else:
                query_parts.append(p)

        query = " ".join(query_parts)
        if not query:
            print("Please provide a search query.")
            continue

        results = run_search(query, split=split, topic=topic)
        print_results(results)


def main():
    parser = argparse.ArgumentParser(description="Article Topic RAG Search")
    parser.add_argument("--reindex", action="store_true", help="Force re-index all articles")
    parser.add_argument("--query", type=str, help="Single search query (non-interactive)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--split", type=str, help="Filter by split: train, val, test")
    parser.add_argument("--topic", type=str, help="Filter by topic: World, Sports, Business, Sci/Tech")
    args = parser.parse_args()

    # Step 1: Index
    print("Building/loading index...")
    count = build_index(force=args.reindex)
    print(f"Index ready — {count} articles indexed.")

    # Step 2: Search
    if args.query:
        results = run_search(args.query, top_k=args.top_k, split=args.split, topic=args.topic)
        print_results(results)
    else:
        interactive_search()


if __name__ == "__main__":
    main()
