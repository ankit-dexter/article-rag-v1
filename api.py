"""
FastAPI search endpoint for the article-topic RAG system.

This module creates a REST API that allows users to perform semantic
(meaning-based) searches over a dataset of news articles. It uses
ChromaDB as a vector store and sentence-transformer embeddings to
find articles similar to a given query.

Usage:
    uvicorn api:app --port 8001

Once running, visit http://localhost:8001/docs for the interactive
Swagger UI where you can test the endpoints in your browser.
"""

# --- Standard library imports ---
# `os` is used to work with file paths and environment variables
import os

# `httpx` is used to make HTTP requests to the LLM proxy server (llm_server.py)
# for the answer generation step of the RAG pipeline.
import httpx

# `asynccontextmanager` lets us define startup/shutdown logic for the FastAPI app.
# It runs code before the app starts serving requests (startup) and after it
# stops (shutdown), using a single function with a `yield` in the middle.
from contextlib import asynccontextmanager

# --- Third-party imports ---
# `FastAPI` is the web framework that creates our API server.
# `Query` is a helper that lets us define query parameter validation and metadata
# (like default values, min/max constraints, and descriptions for the docs page).
from fastapi import FastAPI, Query

# --- Local imports ---
# `ArticleSearcher` handles semantic search against the ChromaDB vector store.
# `ArticleIndexer` reads raw article CSV files, generates embeddings, and stores
# them in ChromaDB so they can be searched later.
from searcher import ArticleSearcher
from indexer import ArticleIndexer

# ──────────────────────────────────────────────────────────────────────
# Configuration — paths for data and the ChromaDB persistence directory
# ──────────────────────────────────────────────────────────────────────

# Resolve the absolute path of the directory this file lives in.
# This is used as the base for constructing other paths below.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the raw article CSV data files.
# First checks for a CLASSIFIER_DATA_PATH environment variable (useful in
# production or Docker). If not set, falls back to the sibling project's
# data folder: ../article-topic-classifier/data/
DATA_DIR = os.environ.get(
    "CLASSIFIER_DATA_PATH",
    os.path.join(PROJECT_ROOT, "..", "article-topic-classifier", "data"),
)

# Path where ChromaDB stores its vector index on disk.
# This allows the index to persist between restarts so we don't have to
# re-embed all articles every time the server starts.
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_store")

# URL of the LLM proxy server (llm_server.py) that wraps `claude -p`.
# The /ask endpoint calls this to generate answers from retrieved articles.
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:8002")

# ──────────────────────────────────────────────────────────────────────
# Global searcher instance
# ──────────────────────────────────────────────────────────────────────

# We declare this at module level so the search endpoint can use it.
# It starts as None and gets initialized during the lifespan startup
# (see below). Using a global here avoids re-creating the searcher
# object on every request, which would be slow.
searcher: ArticleSearcher | None = None


# ──────────────────────────────────────────────────────────────────────
# Lifespan — startup and shutdown logic for the application
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts up (everything before `yield`)
    and once when the server shuts down (everything after `yield`).

    Startup steps:
      1. Create an ArticleIndexer that points to our data and ChromaDB store.
      2. Call index() which reads the CSV files, generates embeddings for any
         new/missing articles, and stores them in ChromaDB. If the index already
         exists and is up to date, this is a fast no-op.
      3. Create an ArticleSearcher that connects to the same ChromaDB store,
         ready to handle search queries.

    Shutdown:
      Nothing to clean up in this case, so the code after `yield` is empty.
    """
    # `global` keyword lets us assign to the module-level `searcher` variable
    # from inside this function. Without it, Python would create a local variable.
    global searcher

    # Step 1: Build or load the vector index.
    # The indexer reads article CSVs, generates embeddings using a
    # sentence-transformer model, and upserts them into ChromaDB.
    indexer = ArticleIndexer(data_dir=DATA_DIR, persist_dir=PERSIST_DIR)

    # index() returns the total number of articles now in the index.
    # On first run this embeds everything (slow); on subsequent runs it
    # only adds new articles (fast).
    count = indexer.index()
    print(f"Index ready — {count} articles")

    # Step 2: Create the searcher that will be used by the /search endpoint.
    # It connects to the same ChromaDB collection the indexer just populated.
    searcher = ArticleSearcher(persist_dir=PERSIST_DIR)

    # `yield` hands control to FastAPI — the app is now ready to serve requests.
    # When the server is shutting down, execution resumes after this yield.
    yield

    # (Shutdown logic would go here if we needed to close connections, etc.)


# ──────────────────────────────────────────────────────────────────────
# Create the FastAPI application
# ──────────────────────────────────────────────────────────────────────

# `title` appears in the auto-generated Swagger docs at /docs.
# `lifespan` tells FastAPI to run our startup/shutdown function above.
app = FastAPI(title="Article Topic RAG Search", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/search")
def search(
    # `q` is a required query parameter (the `...` means no default — the caller
    # must provide it). Example: /search?q=oil+prices
    q: str = Query(..., description="Search query"),

    # `top_k` controls how many results to return. Defaults to 5.
    # `ge=1` means the minimum allowed value is 1.
    # `le=50` means the maximum allowed value is 50.
    # These constraints are enforced by FastAPI automatically and shown in the docs.
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),

    # Optional filter to only return articles from a specific data split.
    # Allowed values: "train", "val", or "test". None means no filter.
    split: str | None = Query(None, description="Filter by split: train, val, test"),

    # Optional filter to only return articles from a specific news topic.
    # Allowed values: "World", "Sports", "Business", "Sci/Tech". None means no filter.
    topic: str | None = Query(None, description="Filter by topic: World, Sports, Business, Sci/Tech"),
):
    """
    Semantic search over the article dataset.

    This endpoint takes a natural language query, converts it into an embedding
    vector, and finds the most similar articles in the ChromaDB vector store.
    Unlike keyword search, this finds articles by *meaning* — so a query like
    "crude oil prices rising" will match articles about petroleum costs even if
    they don't contain those exact words.

    Returns a JSON object with:
      - query:   the original search text
      - count:   how many results were found
      - results: a list of matching articles, each with title, snippet,
                 topic, split, and a similarity score (0 to 1, higher = better)
    """
    # Delegate the actual search to the ArticleSearcher instance.
    # It embeds the query, queries ChromaDB for the nearest neighbors,
    # applies any filters (split/topic), and returns formatted results.
    results = searcher.search(query=q, top_k=top_k, split=split, topic=topic)

    # Return a structured JSON response. FastAPI automatically converts
    # this Python dict into a JSON HTTP response with Content-Type: application/json.
    return {"query": q, "count": len(results), "results": results}


@app.get("/ask")
def ask(
    # The user's natural language question
    q: str = Query(..., description="Question to answer using RAG"),

    # How many articles to retrieve as context for the LLM
    top_k: int = Query(5, ge=1, le=20, description="Number of articles to retrieve as context"),

    # Optional filters (same as /search)
    split: str | None = Query(None, description="Filter by split: train, val, test"),
    topic: str | None = Query(None, description="Filter by topic: World, Sports, Business, Sci/Tech"),
):
    """
    Full RAG pipeline: Retrieve relevant articles, then generate an answer.

    This is the complete Retrieval-Augmented Generation endpoint:
      1. RETRIEVE — semantic search finds the top_k most relevant articles
      2. AUGMENT  — retrieved articles are formatted into an LLM prompt
      3. GENERATE — the prompt is sent to Claude via the LLM proxy server,
                    which runs `claude -p` under the hood

    Returns the generated answer along with the source articles used.
    """
    # ── Step 1: RETRIEVE — find relevant articles via semantic search ──
    results = searcher.search(query=q, top_k=top_k, split=split, topic=topic)

    if not results:
        return {"query": q, "answer": "No relevant articles found.", "sources": []}

    # ── Step 2: AUGMENT — build the prompt with retrieved context ──
    # Format each article as a numbered block so the LLM can reference them
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Article {i}] (Topic: {r['topic']}, Similarity: {r['similarity']})\n"
            f"Title: {r['title']}\n"
            f"Content: {r['snippet']}"
        )
    context = "\n\n".join(context_parts)

    # The prompt instructs the LLM to answer based ONLY on the provided articles
    # and to cite which articles it used — this is the standard RAG prompt pattern
    prompt = (
        f"You are a helpful assistant that answers questions based on news articles.\n\n"
        f"Here are the relevant articles retrieved from the database:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Based ONLY on the articles above, answer the following question. "
        f"Cite the article numbers you used (e.g., [Article 1]). "
        f"If the articles don't contain enough information, say so.\n\n"
        f"Question: {q}"
    )

    # ── Step 3: GENERATE — send prompt to the LLM proxy server ──
    try:
        # POST the prompt to the llm_server running on port 8002
        resp = httpx.post(
            f"{LLM_SERVER_URL}/generate",
            json={"prompt": prompt},
            timeout=130.0,  # slightly longer than the CLI's 120s timeout
        )
        resp.raise_for_status()
        answer = resp.json()["response"]
    except httpx.ConnectError:
        return {
            "query": q,
            "answer": "Error: LLM server not reachable. Start it with: uvicorn llm_server:app --port 8002",
            "sources": results,
        }
    except Exception as e:
        return {
            "query": q,
            "answer": f"Error generating answer: {str(e)}",
            "sources": results,
        }

    return {"query": q, "answer": answer, "sources": results}


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns the server status and whether the search index has been loaded.
    Useful for monitoring tools, load balancers, or Docker health checks
    to verify the service is up and ready to handle requests.

    - "indexed": true  means the searcher is initialized and ready
    - "indexed": false means the server is still starting up
    """
    return {"status": "ok", "indexed": searcher is not None}
