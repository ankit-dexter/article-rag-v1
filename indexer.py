"""
Indexer — reads JSONL dataset files, embeds articles, and stores them in ChromaDB.

This module is responsible for the "ingestion" step of the RAG pipeline.
It reads raw article data from JSONL files (one JSON object per line),
converts each article's text into a numerical vector (embedding) using a
sentence-transformer model, and stores those embeddings in a ChromaDB
vector database for fast similarity search later.

On first run, embeds all articles and persists the index to disk.
On subsequent runs, skips indexing if the collection already exists
(unless force=True is passed).
"""

# --- Standard library imports ---
# `json` is used to parse each line of the JSONL files into Python dicts.
import json

# `os` is used for file path manipulation and checking if files exist.
import os

# --- Third-party imports ---
# `chromadb` is the vector database library. It stores embeddings and
# allows fast nearest-neighbor lookups. We use its PersistentClient
# so the index is saved to disk and survives server restarts.
import chromadb

# `SentenceTransformer` is from the sentence-transformers library.
# It wraps a pre-trained transformer model that converts text into
# fixed-size embedding vectors (384 dimensions for MiniLM).
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# The name of the pre-trained sentence-transformer model to use.
# "all-MiniLM-L6-v2" is a lightweight, fast model that produces 384-dimensional
# embeddings. It's a good balance between speed and quality for semantic search.
# The model is downloaded automatically on first use and cached locally.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# The name of the ChromaDB collection where article embeddings are stored.
# A collection is like a "table" in ChromaDB — all articles go into this one.
COLLECTION_NAME = "articles"

# The dataset is split into three parts: training, validation, and test sets.
# Each split has its own JSONL file (train.jsonl, val.jsonl, test.jsonl).
SPLITS = ["train", "val", "test"]


class ArticleIndexer:
    """
    Handles the full indexing pipeline:
    1. Load articles from JSONL files on disk
    2. Generate embeddings for each article using a sentence-transformer
    3. Store embeddings + metadata in ChromaDB for later search
    """

    def __init__(self, data_dir: str, persist_dir: str):
        """
        Initialize the indexer with paths and load the ML model.

        Args:
            data_dir:    Path to the folder containing train.jsonl, val.jsonl, test.jsonl
            persist_dir: Path where ChromaDB will save its index files to disk
        """
        # Store the directory paths for later use in _load_articles() and index()
        self.data_dir = data_dir
        self.persist_dir = persist_dir

        # Load the sentence-transformer model into memory.
        # This downloads the model on first run (~80MB) and caches it.
        # The model is used to convert article text into embedding vectors.
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Create a ChromaDB client that persists data to disk at `persist_dir`.
        # PersistentClient means the index survives process restarts — we don't
        # have to re-embed everything each time the server starts.
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _load_articles(self) -> list[dict]:
        """
        Load all articles from JSONL files across all splits (train, val, test).

        Each JSONL file has one JSON object per line with fields like:
            {"title": "...", "body": "...", "topic": "Sports"}

        This method reads every line from every split file, adds the split name
        and a unique doc_id to each article, and returns them all as a flat list.

        Returns:
            A list of dicts, each representing one article with keys:
            title, body, topic, split, doc_id
        """
        articles = []

        # Iterate over each split: "train", "val", "test"
        for split in SPLITS:
            # Construct the full path to this split's JSONL file
            # e.g., /data/train.jsonl, /data/val.jsonl, /data/test.jsonl
            path = os.path.join(self.data_dir, f"{split}.jsonl")

            # Skip if this split file doesn't exist (e.g., if only train data is available)
            if not os.path.exists(path):
                continue

            # Open the JSONL file and read it line by line.
            # Each line is a separate JSON object (one article per line).
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    # Parse the JSON string into a Python dict
                    obj = json.loads(line)

                    # Tag the article with which split it came from
                    # so we can filter by split during search
                    obj["split"] = split

                    # Create a unique ID for this article by combining
                    # the split name and line number (e.g., "train_0", "train_1", "val_0")
                    # ChromaDB requires each document to have a unique ID.
                    obj["doc_id"] = f"{split}_{i}"

                    articles.append(obj)

        return articles

    def index(self, force: bool = False) -> int:
        """
        Embed and store all articles in ChromaDB. Returns the total count of indexed articles.

        This is the main method that orchestrates the indexing pipeline:
        1. Get or create the ChromaDB collection
        2. Check if indexing is needed (skip if already done, unless force=True)
        3. Load all articles from disk
        4. Prepare documents, IDs, and metadata for ChromaDB
        5. Generate embeddings in batches (to manage memory)
        6. Upsert into ChromaDB in batches (ChromaDB has per-call limits)

        Args:
            force: If True, re-embed and re-index all articles even if the
                   collection already has data. Useful if the data files changed.

        Returns:
            The total number of articles in the collection after indexing.
        """
        # Get the collection if it exists, or create it if it doesn't.
        # The metadata {"hnsw:space": "cosine"} tells ChromaDB to use cosine
        # similarity for comparing embeddings. Cosine similarity measures the
        # angle between two vectors — it's the standard choice for text embeddings
        # because it focuses on direction (meaning) rather than magnitude (length).
        collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # If the collection already has articles and we're not forcing a re-index,
        # skip the expensive embedding step and just return the existing count.
        # This makes subsequent server starts very fast.
        if collection.count() > 0 and not force:
            return collection.count()

        # Load all articles from the JSONL files on disk
        articles = self._load_articles()

        # If no articles were found (empty data directory), return 0
        if not articles:
            return 0

        # Prepare the three parallel lists that ChromaDB needs:

        # 1. IDs — unique identifier for each document (e.g., "train_0", "val_42")
        ids = [a["doc_id"] for a in articles]

        # 2. Documents — the text content to store. We concatenate title + body
        #    so the embedding captures the full meaning of the article.
        #    This combined text is what gets embedded into a vector.
        documents = [a["title"] + " " + a["body"] for a in articles]

        # 3. Metadatas — structured data stored alongside each embedding.
        #    This is NOT embedded, but can be used for filtering search results
        #    (e.g., "only show Sports articles" or "only show test split").
        metadatas = [
            {"title": a["title"], "topic": a["topic"], "split": a["split"]}
            for a in articles
        ]

        # ── Step: Generate embeddings in batches ──
        # Embedding all articles at once could use too much RAM, so we process
        # them in batches of 256. The sentence-transformer model converts each
        # text string into a 384-dimensional float vector.
        batch_size = 256
        embeddings = []
        for start in range(0, len(documents), batch_size):
            # Slice out a batch of documents
            batch = documents[start : start + batch_size]

            # self.embedder.encode() runs the text through the neural network
            # and returns a numpy array of shape (batch_size, 384).
            # .tolist() converts it to a plain Python list of lists,
            # which is the format ChromaDB expects.
            embeddings.extend(self.embedder.encode(batch).tolist())

        # ── Step: Upsert into ChromaDB in batches ──
        # ChromaDB has a limit on how many items can be upserted in a single call
        # (typically around 5000), so we batch the upserts as well.
        # "Upsert" means insert-or-update: if a doc_id already exists, it's updated;
        # if it's new, it's inserted. This makes re-indexing safe and idempotent.
        chroma_batch = 5000
        for start in range(0, len(ids), chroma_batch):
            end = start + chroma_batch
            collection.upsert(
                ids=ids[start:end],             # Unique IDs for this batch
                embeddings=embeddings[start:end], # Corresponding embedding vectors
                documents=documents[start:end],   # Original text (stored for retrieval)
                metadatas=metadatas[start:end],   # Structured metadata for filtering
            )

        # Return the total number of documents now in the collection
        return collection.count()
