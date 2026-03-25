"""
Searcher — semantic search over indexed articles using ChromaDB.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "articles"


class ArticleSearcher:
    def __init__(self, persist_dir: str):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def search(
        self,
        query: str,
        top_k: int = 5,
        split: str | None = None,
        topic: str | None = None,
    ) -> list[dict]:
        """Semantic search for articles matching the query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default 5).
            split: Optional filter — "train", "val", or "test".
            topic: Optional filter — "World", "Sports", "Business", or "Sci/Tech".

        Returns:
            List of matching articles with title, topic, split, distance score,
            and a snippet of the document text.
        """
        query_embedding = self.embedder.encode(query).tolist()

        where_filter = {}
        if split and topic:
            where_filter = {"$and": [{"split": split}, {"topic": topic}]}
        elif split:
            where_filter = {"split": split}
        elif topic:
            where_filter = {"topic": topic}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["metadatas", "documents", "distances"],
        )

        articles = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            articles.append({
                "title": meta["title"],
                "topic": meta["topic"],
                "split": meta["split"],
                "similarity": round(1 - results["distances"][0][i], 4),
                "snippet": doc[:300] + "..." if len(doc) > 300 else doc,
            })

        return articles
