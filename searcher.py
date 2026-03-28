"""
Searcher — semantic search over indexed articles using ChromaDB.

This module handles the "retrieval" step of the RAG pipeline.
Given a natural language query, it converts the query into an embedding
vector (using the same model that was used during indexing), then finds
the most similar articles in the ChromaDB vector store using cosine
similarity. It also supports optional filtering by topic and data split.
"""

# --- Third-party imports ---
# `chromadb` is the vector database. We connect to the same persistent
# store that the indexer wrote to, so we can search over the indexed articles.
import chromadb

# `SentenceTransformer` is the same embedding model used during indexing.
# IMPORTANT: The search model MUST match the indexing model — if you change
# one, you must change the other and re-index, because different models
# produce incompatible embedding spaces.
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Must match the model used in indexer.py — embeddings are only comparable
# if they come from the same model.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Must match the collection name used in indexer.py — this is the "table"
# in ChromaDB where all article embeddings are stored.
COLLECTION_NAME = "articles"


class ArticleSearcher:
    """
    Performs semantic (meaning-based) search over the article index.

    Unlike keyword search (which matches exact words), semantic search
    understands meaning — so "crude oil prices rising" will match articles
    about "petroleum costs increase" even without shared words.
    """

    def __init__(self, persist_dir: str):
        """
        Initialize the searcher by loading the embedding model and
        connecting to the ChromaDB collection.

        Args:
            persist_dir: Path to the ChromaDB storage directory on disk
                         (same path that was used during indexing).
        """
        # Load the same sentence-transformer model used during indexing.
        # This ensures that query embeddings live in the same vector space
        # as the document embeddings, so cosine similarity comparisons are valid.
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to the persistent ChromaDB instance on disk.
        # This reads the index files that the indexer created.
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get a reference to the existing "articles" collection.
        # Note: get_collection() (not get_or_create_collection()) — this will
        # raise an error if the collection doesn't exist yet, which is the
        # correct behavior since you must index before you can search.
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def search(
        self,
        query: str,
        top_k: int = 5,
        split: str | None = None,
        topic: str | None = None,
    ) -> list[dict]:
        """
        Semantic search for articles matching the query.

        How it works:
        1. The query text is converted into an embedding vector (384 floats)
        2. ChromaDB compares this vector against all stored article embeddings
           using cosine similarity (configured during indexing)
        3. The top_k most similar articles are returned
        4. Optional filters (split, topic) narrow results before ranking

        Args:
            query: Natural language search query (e.g., "oil prices rising").
            top_k: Number of results to return (default 5).
            split:  Optional filter — "train", "val", or "test".
                    Only returns articles from the specified data split.
            topic:  Optional filter — "World", "Sports", "Business", or "Sci/Tech".
                    Only returns articles from the specified news category.

        Returns:
            List of matching articles, each as a dict with keys:
              - title:      the article's headline
              - topic:      news category (World, Sports, Business, Sci/Tech)
              - split:      which data split it belongs to (train, val, test)
              - similarity:  score from 0 to 1 (higher = more similar to query)
              - snippet:    first 300 characters of the article text
        """
        # ── Step 1: Embed the query ──
        # Convert the user's search text into a 384-dimensional vector.
        # .encode() runs the text through the neural network.
        # .tolist() converts from numpy array to plain Python list (ChromaDB format).
        query_embedding = self.embedder.encode(query).tolist()

        # ── Step 2: Build optional metadata filters ──
        # ChromaDB's `where` parameter filters results based on metadata fields
        # BEFORE ranking by similarity. This is efficient because it narrows
        # the search space before doing vector comparisons.
        where_filter = {}

        if split and topic:
            # If both filters are specified, combine them with $and.
            # ChromaDB uses MongoDB-style query operators.
            # This means: only return articles where split=X AND topic=Y
            where_filter = {"$and": [{"split": split}, {"topic": topic}]}
        elif split:
            # Filter by data split only (e.g., only "test" articles)
            where_filter = {"split": split}
        elif topic:
            # Filter by news topic only (e.g., only "Sports" articles)
            where_filter = {"topic": topic}

        # ── Step 3: Query ChromaDB ──
        # This is where the actual vector similarity search happens.
        # ChromaDB uses HNSW (Hierarchical Navigable Small World) algorithm
        # for approximate nearest neighbor search — it's very fast even
        # with millions of vectors.
        results = self.collection.query(
            # The query embedding to compare against all stored embeddings.
            # Wrapped in a list because ChromaDB supports batch queries.
            query_embeddings=[query_embedding],

            # How many results to return
            n_results=top_k,

            # Apply metadata filters (or None if no filters specified)
            where=where_filter if where_filter else None,

            # What data to include in the results:
            # - metadatas: the structured fields (title, topic, split)
            # - documents: the full text that was stored during indexing
            # - distances: the cosine distance between query and each result
            #   (distance = 1 - similarity, so smaller = more similar)
            include=["metadatas", "documents", "distances"],
        )

        # ── Step 4: Format the results ──
        # ChromaDB returns results in a nested structure because it supports
        # batch queries. Since we sent one query, our results are in index [0].
        # results["ids"][0]       = list of document IDs
        # results["metadatas"][0] = list of metadata dicts
        # results["documents"][0] = list of document texts
        # results["distances"][0] = list of cosine distances
        articles = []
        for i in range(len(results["ids"][0])):
            # Get the metadata dict for this result (title, topic, split)
            meta = results["metadatas"][0][i]

            # Get the full document text for this result
            doc = results["documents"][0][i]

            articles.append({
                "title": meta["title"],
                "topic": meta["topic"],
                "split": meta["split"],

                # Convert cosine distance to similarity score.
                # ChromaDB returns distance (0 = identical, 1 = completely different).
                # We convert to similarity (1 = identical, 0 = completely different)
                # by subtracting from 1. Round to 4 decimal places for readability.
                "similarity": round(1 - results["distances"][0][i], 4),

                # Return just the first 300 characters as a preview snippet.
                # This keeps API responses small when articles have long bodies.
                "snippet": doc[:300] + "..." if len(doc) > 300 else doc,
            })

        return articles
