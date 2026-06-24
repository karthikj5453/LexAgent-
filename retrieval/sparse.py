import os
import pickle
import re
import logging
from rank_bm25 import BM25Okapi
from config import settings

logger = logging.getLogger("lexagent.retrieval.sparse")

class BM25Retriever:
    def __init__(self, persist_path: str = "./data/bm25_index.pkl"):
        self.persist_path = persist_path
        self.chunks_db = []  # list of dicts: {"id": str, "text": str, "metadata": dict}
        self.load()

    def load(self):
        """Load database from disk if it exists."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "rb") as f:
                    self.chunks_db = pickle.load(f)
                logger.info(f"Loaded {len(self.chunks_db)} chunks into BM25 retriever.")
            except Exception as e:
                logger.error(f"Error loading BM25 index: {e}")
                self.chunks_db = []
        else:
            self.chunks_db = []

    def save(self):
        """Save database to disk."""
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "wb") as f:
                pickle.dump(self.chunks_db, f)
            logger.info(f"Saved BM25 index with {len(self.chunks_db)} chunks.")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")

    def tokenize(self, text: str) -> list[str]:
        """Simple alphanumeric tokenizer."""
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def index_chunks(self, chunks: list[dict], doc_id: str):
        """
        Add chunks to the local database and save.
        chunks: list of {id, text, metadata}
        """
        if not chunks:
            return

        # Deduplicate: remove any existing chunks with same doc_id
        self.chunks_db = [c for c in self.chunks_db if c.get("metadata", {}).get("doc_id") != doc_id]

        for chunk in chunks:
            self.chunks_db.append({
                "id": f"{doc_id}_{chunk['id']}",
                "text": chunk["text"],
                "metadata": {**chunk.get("metadata", {}), "doc_id": doc_id}
            })

        self.save()
        logger.info(f"Indexed {len(chunks)} chunks in BM25 for doc_id: {doc_id}")

    def search(self, query: str, k: int = 10, doc_id: str | None = None) -> list[dict]:
        """
        Search for query using BM25, optionally filtered by doc_id.
        """
        if not self.chunks_db:
            return []

        # Filter chunks by doc_id if specified
        filtered_chunks = self.chunks_db
        if doc_id:
            filtered_chunks = [c for c in self.chunks_db if c.get("metadata", {}).get("doc_id") == doc_id]

        if not filtered_chunks:
            return []

        # Tokenize corpus for BM25
        tokenized_corpus = [self.tokenize(c["text"]) for c in filtered_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        # Get scores
        tokenized_query = self.tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Zip chunks with scores
        scored_chunks = []
        for i, chunk in enumerate(filtered_chunks):
            scored_chunks.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(scores[i]),
            })

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = scored_chunks[:k]

        # Normalize score (BM25 scores are unbounded, so normalize relative to highest score)
        max_score = top_chunks[0]["score"] if top_chunks and top_chunks[0]["score"] > 0 else 1.0
        for i, c in enumerate(top_chunks):
            # Scale score between 0 and 1
            raw_score = c["score"]
            normalized_score = raw_score / max_score if max_score > 0 else 0.0
            c["score"] = round(normalized_score, 4)
            c["rank"] = i

        return top_chunks
