# Article-Topic RAG Search

Semantic search over the [article-topic-classifier](../article-topic-classifier) dataset using a RAG (Retrieval-Augmented Generation) pipeline.

Embeds 7,600 news articles using `sentence-transformers` and stores them in ChromaDB for fast cosine similarity search.

## How it works

1. **Index** — Reads JSONL files from the classifier dataset, embeds each article (title + body) using `all-MiniLM-L6-v2`, and stores vectors in ChromaDB on disk.
2. **Search** — Encodes your query into a vector and finds the most similar articles via cosine similarity. Supports filtering by topic and dataset split.

Embeddings are persisted to `chroma_store/` — indexing only runs once.

## Setup

```bash
pip install -r requirements.txt
```

By default, looks for dataset at `../article-topic-classifier/data/`. Override with:

```bash
export CLASSIFIER_DATA_PATH=/path/to/data
```

## Usage

```bash
# Interactive search (indexes on first run)
python main.py

# Force re-index
python main.py --reindex

# Single query
python main.py --query "oil prices middle east"

# With filters
python main.py --query "neural network" --topic "Sci/Tech" --split train --top-k 3
```

### Interactive mode filters

```
search> drone strikes split=test topic=World
search> stock market crash topic=Business
search> quit
```

## Project Structure

```
article-topic-rag/
├── main.py            # CLI entry point (index + search)
├── indexer.py         # Embeds articles and stores in ChromaDB
├── searcher.py        # Semantic search over the index
├── requirements.txt
├── chroma_store/      # Persisted vector index (gitignored)
└── README.md
```
