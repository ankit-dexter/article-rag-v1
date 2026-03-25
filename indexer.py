"""
Indexer — reads JSONL dataset files, embeds articles, and stores them in ChromaDB.

On first run, embeds all articles and persists the index to disk.
On subsequent runs, skips indexing if the collection already exists.
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "articles"
SPLITS = ["train", "val", "test"]


class ArticleIndexer:
    def __init__(self, data_dir: str, persist_dir: str):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _load_articles(self) -> list[dict]:
        """Load all articles from JSONL files across splits."""
        articles = []
        for split in SPLITS:
            path = os.path.join(self.data_dir, f"{split}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    obj = json.loads(line)
                    obj["split"] = split
                    obj["doc_id"] = f"{split}_{i}"
                    articles.append(obj)
        return articles

    def index(self, force: bool = False) -> int:
        """Embed and store articles. Returns count of indexed articles.

        Skips if collection already exists unless force=True.
        """
        collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if collection.count() > 0 and not force:
            return collection.count()

        articles = self._load_articles()
        if not articles:
            return 0

        ids = [a["doc_id"] for a in articles]
        documents = [a["title"] + " " + a["body"] for a in articles]
        metadatas = [
            {"title": a["title"], "topic": a["topic"], "split": a["split"]}
            for a in articles
        ]

        # Embed in batches
        batch_size = 256
        embeddings = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            embeddings.extend(self.embedder.encode(batch).tolist())

        # Upsert in batches (ChromaDB limit per call)
        chroma_batch = 5000
        for start in range(0, len(ids), chroma_batch):
            end = start + chroma_batch
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return collection.count()
