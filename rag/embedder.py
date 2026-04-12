"""
rag/embedder.py
Sentence-transformer based text embedding.
Converts text to dense vectors for semantic similarity search.
Caches embeddings to avoid redundant computation.
"""

from typing import List, Optional
import numpy as np

from utils.logger import logger
from utils.cache import cache_manager
from config.settings import settings


class EmbeddingModel:
    """
    Wrapper around sentence-transformers.
    Provides encode() with caching and batch support.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.vector_db.embedding_model
        self.model = None
        self.dimension = None
        self._load_model()

    def _load_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Test encode to get dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.dimension = test_embedding.shape[1]
            logger.info(f"Embedding model loaded: {self.model_name} (dim={self.dimension})")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Encode a list of texts to embeddings.
        Returns numpy array of shape (n_texts, dimension).
        """
        if not self.model:
            logger.error("Embedding model not available.")
            return None

        if not texts:
            return np.array([])

        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if use_cache:
            for i, text in enumerate(texts):
                cache_key = cache_manager._make_key("embedding", text[:500])
                cached = cache_manager.get(cache_key)
                if cached is not None:
                    results[i] = np.array(cached)
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Compute embeddings for uncached texts
        if uncached_texts:
            try:
                embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=len(uncached_texts) > 100,
                )

                for idx, (orig_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                    results[orig_idx] = embeddings[idx]
                    if use_cache:
                        cache_key = cache_manager._make_key("embedding", text[:500])
                        cache_manager.set(cache_key, embeddings[idx].tolist(), ttl=86400)  # 24hr

            except Exception as e:
                logger.error(f"Embedding computation failed: {e}")
                return None

        return np.array(results)

    def encode_single(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Encode a single text. Returns 1D array."""
        result = self.encode([text], use_cache=use_cache)
        if result is not None and len(result) > 0:
            return result[0]
        return None

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


# Singleton
embedding_model = EmbeddingModel()

__all__ = ["embedding_model", "EmbeddingModel"]
