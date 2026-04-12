"""
rag/vector_store.py
ChromaDB vector store for semantic search.
Stores text chunk embeddings and supports top-k retrieval.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from utils.logger import logger
from config.settings import settings


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with score."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: Dict


class VectorStore:
    """
    ChromaDB-backed vector store.
    Handles: indexing, querying, deletion, persistence.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self._embedding_fn = None
        self._init()

    def _init(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.PersistentClient(
                path=settings.vector_db.persist_dir,
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.vector_db.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB initialized: {settings.vector_db.collection_name} "
                f"({self.collection.count()} docs)"
            )
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")

    def add_chunks(self, chunks: List[Any]) -> int:
        """
        Add TextChunk objects to the vector store.
        Returns number of chunks added.
        """
        if not self.collection:
            logger.error("ChromaDB not available.")
            return 0

        from rag.embedder import embedding_model
        from ingestion.preprocessor import TextChunk

        if not chunks:
            return 0

        # Deduplicate: skip already-indexed chunks
        existing_ids = set()
        try:
            existing = self.collection.get(include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            pass

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All chunks already indexed.")
            return 0

        # Compute embeddings
        texts = [c.text for c in new_chunks]
        embeddings = embedding_model.encode(texts, batch_size=32)

        if embeddings is None:
            logger.error("Failed to compute embeddings.")
            return 0

        # Build ChromaDB batch
        ids = [c.chunk_id for c in new_chunks]
        metadatas = []
        for c in new_chunks:
            meta = {k: str(v) for k, v in c.metadata.items()}
            meta["doc_id"] = c.doc_id
            meta["chunk_index"] = str(c.chunk_index)
            metadatas.append(meta)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Indexed {len(new_chunks)} chunks into ChromaDB.")
            return len(new_chunks)
        except Exception as e:
            logger.error(f"ChromaDB add failed: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_doc_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Semantic search: returns top-k most similar chunks.
        """
        if not self.collection:
            return []

        top_k = top_k or settings.retrieval.top_k

        from rag.embedder import embedding_model
        query_embedding = embedding_model.encode_single(query)

        if query_embedding is None:
            return []

        where_filter = None
        if filter_doc_id:
            where_filter = {"doc_id": {"$eq": filter_doc_id}}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.collection.count() or 1),
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )

            retrieved = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    # Convert cosine distance to similarity (0-1)
                    score = max(0.0, 1.0 - distance)
                    content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                    retrieved.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        doc_id=metadata.get("doc_id", "unknown"),
                        content=content,
                        score=score,
                        metadata=metadata,
                    ))

            # Sort by score descending
            retrieved.sort(key=lambda x: x.score, reverse=True)
            return retrieved

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks from a document."""
        if not self.collection:
            return False
        try:
            self.collection.delete(where={"doc_id": {"$eq": doc_id}})
            logger.info(f"Deleted chunks for doc: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def get_stats(self) -> Dict:
        """Return vector store statistics."""
        count = 0
        if self.collection:
            try:
                count = self.collection.count()
            except Exception:
                pass
        return {
            "total_chunks": count,
            "collection": settings.vector_db.collection_name,
            "persist_dir": settings.vector_db.persist_dir,
        }

    def clear(self):
        """Delete all vectors."""
        if self.client and self.collection:
            try:
                self.client.delete_collection(settings.vector_db.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=settings.vector_db.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.warning("Vector store cleared!")
            except Exception as e:
                logger.error(f"Clear failed: {e}")


# Singleton
vector_store = VectorStore()

__all__ = ["vector_store", "VectorStore", "RetrievedChunk"]
