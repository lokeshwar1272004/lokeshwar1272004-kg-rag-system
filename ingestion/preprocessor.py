"""
ingestion/preprocessor.py
Text preprocessing and chunking pipeline.
Converts raw document content into indexed chunks for RAG.
"""

from typing import List, Dict
from dataclasses import dataclass, field

from ingestion.loader import Document
from utils.text_utils import chunk_text, clean_text, extract_keywords
from utils.logger import logger
from config.settings import settings


@dataclass
class TextChunk:
    """A single text chunk ready for embedding."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: Dict = field(default_factory=dict)


class TextPreprocessor:
    """
    Transforms raw Document objects into TextChunk lists.
    Handles:
    - Cleaning
    - Sentence-aware chunking
    - Metadata enrichment
    - Keyword tagging
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.retrieval.chunk_size
        self.chunk_overlap = chunk_overlap or settings.retrieval.chunk_overlap

    def process(self, document: Document) -> List[TextChunk]:
        """
        Full preprocessing pipeline for a single document.
        Returns a list of TextChunk objects.
        """
        logger.info(f"Preprocessing document: {document.doc_id}")

        # Step 1: Clean
        cleaned_text = clean_text(document.content)

        if not cleaned_text:
            logger.warning(f"Empty content after cleaning: {document.doc_id}")
            return []

        # Step 2: Chunk
        raw_chunks = chunk_text(
            cleaned_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if not raw_chunks:
            logger.warning(f"No chunks generated for: {document.doc_id}")
            return []

        # Step 3: Enrich with metadata
        chunks = []
        for idx, chunk_text_content in enumerate(raw_chunks):
            keywords = extract_keywords(chunk_text_content, top_n=5)
            chunk = TextChunk(
                chunk_id=f"{document.doc_id}_chunk_{idx}",
                doc_id=document.doc_id,
                text=chunk_text_content,
                chunk_index=idx,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                    "keywords": keywords,
                    "char_count": len(chunk_text_content),
                }
            )
            chunks.append(chunk)

        logger.info(f"Generated {len(chunks)} chunks from '{document.doc_id}'")
        return chunks

    def process_batch(self, documents: List[Document]) -> List[TextChunk]:
        """Process multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.process(doc)
            all_chunks.extend(chunks)
        logger.info(f"Total chunks from batch: {len(all_chunks)}")
        return all_chunks


# Singleton
text_preprocessor = TextPreprocessor()

__all__ = ["text_preprocessor", "TextPreprocessor", "TextChunk"]
