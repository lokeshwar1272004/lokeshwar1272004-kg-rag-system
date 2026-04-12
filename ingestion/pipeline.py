"""
ingestion/pipeline.py
Full document ingestion pipeline orchestrator.
Connects: load → preprocess → embed → index (vector + graph).
"""

from typing import List, Dict, Optional, Union
from pathlib import Path

from ingestion.loader import document_loader, Document
from ingestion.preprocessor import text_preprocessor, TextChunk
from utils.logger import logger


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow:
    
    File/Text Input
         │
         ▼
    [Document Loader] → Document
         │
         ▼
    [Text Preprocessor] → TextChunks
         │
         ├──► [Vector Store] ChromaDB indexing
         │
         └──► [Graph Builder] Entity/Relation extraction → KG
    """

    def __init__(self):
        self._vector_store = None
        self._graph_builder = None

    def _get_vector_store(self):
        if self._vector_store is None:
            from rag.vector_store import vector_store
            self._vector_store = vector_store
        return self._vector_store

    def _get_graph_builder(self):
        if self._graph_builder is None:
            from graph.graph_builder import graph_builder
            self._graph_builder = graph_builder
        return self._graph_builder

    def ingest_file(
        self,
        file_path: Union[str, Path],
        build_graph: bool = True,
        use_llm_relations: bool = True,
    ) -> Dict:
        """Ingest a single file."""
        doc = document_loader.load_file(file_path)
        if not doc:
            return {"success": False, "error": f"Failed to load: {file_path}"}
        return self._process_document(doc, build_graph, use_llm_relations)

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        build_graph: bool = True,
        use_llm_relations: bool = True,
    ) -> Dict:
        """Ingest from uploaded bytes (Streamlit)."""
        doc = document_loader.load_bytes(file_bytes, filename)
        if not doc:
            return {"success": False, "error": f"Failed to load: {filename}"}
        return self._process_document(doc, build_graph, use_llm_relations)

    def ingest_text(
        self,
        text: str,
        source_name: str = "manual_input",
        build_graph: bool = True,
        use_llm_relations: bool = True,
    ) -> Dict:
        """Ingest raw text directly."""
        if not text or not text.strip():
            return {"success": False, "error": "Empty text provided"}
        doc = document_loader.load_text_directly(text, source_name)
        return self._process_document(doc, build_graph, use_llm_relations)

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        build_graph: bool = True,
        use_llm_relations: bool = False,  # Off by default for batch
    ) -> Dict:
        """Ingest all supported files from a directory."""
        docs = document_loader.load_directory(dir_path)
        if not docs:
            return {"success": False, "error": "No documents found"}

        total_stats = {
            "documents": 0,
            "chunks": 0,
            "vectors_indexed": 0,
            "entities_added": 0,
            "relations_added": 0,
        }

        for doc in docs:
            result = self._process_document(doc, build_graph, use_llm_relations)
            if result.get("success"):
                total_stats["documents"] += 1
                total_stats["chunks"] += result.get("chunks", 0)
                total_stats["vectors_indexed"] += result.get("vectors_indexed", 0)
                total_stats["entities_added"] += result.get("entities_added", 0)
                total_stats["relations_added"] += result.get("relations_added", 0)

        return {"success": True, **total_stats}

    def _process_document(
        self,
        doc: Document,
        build_graph: bool,
        use_llm_relations: bool,
    ) -> Dict:
        """Core processing for a single Document."""
        stats = {
            "success": True,
            "doc_id": doc.doc_id,
            "chunks": 0,
            "vectors_indexed": 0,
            "entities_added": 0,
            "relations_added": 0,
        }

        try:
            # Step 1: Chunk
            chunks = text_preprocessor.process(doc)
            stats["chunks"] = len(chunks)

            if not chunks:
                logger.warning(f"No chunks generated for {doc.doc_id}")
                return stats

            # Step 2: Index in vector store
            vs = self._get_vector_store()
            indexed = vs.add_chunks(chunks)
            stats["vectors_indexed"] = indexed

            # Step 3: Build knowledge graph (optional)
            if build_graph:
                gb = self._get_graph_builder()
                graph_stats = gb.build_from_chunks(chunks, use_llm_relations=use_llm_relations)
                stats["entities_added"] = graph_stats.get("entities_added", 0)
                stats["relations_added"] = graph_stats.get("relations_added", 0)

            logger.info(
                f"Ingested '{doc.doc_id}': "
                f"{stats['chunks']} chunks, "
                f"{stats['vectors_indexed']} vectors, "
                f"{stats['entities_added']} entities, "
                f"{stats['relations_added']} relations"
            )
            return stats

        except Exception as e:
            logger.error(f"Ingestion failed for {doc.doc_id}: {e}")
            return {"success": False, "error": str(e), "doc_id": doc.doc_id}


# Singleton
ingestion_pipeline = IngestionPipeline()

__all__ = ["ingestion_pipeline", "IngestionPipeline"]
