import chromadb
from chromadb.config import Settings as ChromaSettings
from retrieval.embedder import embed_texts, embed_query
from config import settings
import logging

logger = logging.getLogger("lexagent.retrieval.dense")

class DenseRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def index_chunks(self, chunks: list[dict], doc_id: str):
        """
        chunks: list of {id, text, metadata}
        """
        if not chunks:
            logger.warning("No chunks to index in dense retriever.")
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = await embed_texts(texts)
        
        ids = [f"{doc_id}_{c['id']}" for c in chunks]
        metadatas = [{**c.get("metadata", {}), "doc_id": doc_id} for c in chunks]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(chunks)} chunks in ChromaDB for doc_id: {doc_id}")

    async def search(self, query: str, k: int = 10, doc_id: str | None = None) -> list[dict]:
        total_count = self.collection.count()
        if total_count == 0:
            logger.warning("ChromaDB collection is empty. Returning 0 dense search results.")
            return []
            
        query_embedding = await embed_query(query)
        
        where = {"doc_id": doc_id} if doc_id else None
        n_results = min(k, total_count)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []
            
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            return []
            
        output = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Convert L2 distance or cosine distance to similarity
            # Cosine distance in Chroma: 1 - cosine_similarity. So cosine_similarity = 1 - distance
            score = 1.0 - distance
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(score, 4),
                "rank": i,
            })
        return output
