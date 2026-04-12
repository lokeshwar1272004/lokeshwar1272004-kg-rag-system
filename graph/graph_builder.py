"""
graph/graph_builder.py
Orchestrates building the Knowledge Graph from extracted entities and relations.
Populates Neo4j (or NetworkX) with nodes and edges.
"""

from typing import List, Dict
from tqdm import tqdm

from extraction.ner import Entity
from extraction.relation_extractor import Relation
from graph.neo4j_client import graph_client
from ingestion.preprocessor import TextChunk
from utils.logger import logger


class GraphBuilder:
    """
    Builds and maintains the Knowledge Graph.
    Takes TextChunks → extracts entities/relations → populates graph DB.
    """

    def __init__(self):
        self.graph = graph_client
        self._ner = None
        self._rel_extractor = None

    def _get_ner(self):
        if self._ner is None:
            from extraction.ner import ner_extractor
            self._ner = ner_extractor
        return self._ner

    def _get_rel_extractor(self):
        if self._rel_extractor is None:
            from extraction.relation_extractor import relation_extractor
            self._rel_extractor = relation_extractor
        return self._rel_extractor

    def build_from_chunks(
        self,
        chunks: List[TextChunk],
        use_llm_relations: bool = True,
    ) -> Dict:
        """
        Main pipeline: chunks → entities → relations → graph.
        Returns statistics about what was added.
        """
        logger.info(f"Building Knowledge Graph from {len(chunks)} chunks...")
        stats = {"entities_added": 0, "relations_added": 0, "chunks_processed": 0}
        ner = self._get_ner()
        rel_extractor = self._get_rel_extractor()

        for chunk in tqdm(chunks, desc="Building KG"):
            try:
                # Step 1: Extract entities
                entities = ner.extract(chunk.text)
                if not entities:
                    continue

                # Step 2: Add entities as nodes
                for entity in entities:
                    success = self.graph.add_entity(
                        name=entity.text,
                        entity_type=entity.label,
                        properties={
                            "source_doc": chunk.doc_id,
                            "chunk_id": chunk.chunk_id,
                        }
                    )
                    if success:
                        stats["entities_added"] += 1

                # Step 3: Extract relations
                relations = rel_extractor.extract(
                    text=chunk.text,
                    entities=entities,
                    use_llm=use_llm_relations,
                )

                # Step 4: Add relations as edges
                for relation in relations:
                    success = self.graph.add_relation(
                        subject=relation.subject,
                        predicate=relation.predicate,
                        obj=relation.object,
                        subject_type=relation.subject_type,
                        object_type=relation.object_type,
                        properties={"confidence": relation.confidence},
                    )
                    if success:
                        stats["relations_added"] += 1

                stats["chunks_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                continue

        logger.info(
            f"KG built: {stats['entities_added']} entities, "
            f"{stats['relations_added']} relations from "
            f"{stats['chunks_processed']} chunks"
        )
        return stats

    def add_manual_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: str = "Entity",
        object_type: str = "Entity",
    ) -> bool:
        """Manually add a knowledge triple."""
        return self.graph.add_relation(
            subject=subject,
            predicate=predicate,
            obj=obj,
            subject_type=subject_type,
            object_type=object_type,
        )

    def get_stats(self) -> Dict:
        return self.graph.get_stats()

    def clear_graph(self):
        """Remove all data from the graph."""
        logger.warning("Clearing Knowledge Graph!")
        self.graph.clear()


# Singleton
graph_builder = GraphBuilder()

__all__ = ["graph_builder", "GraphBuilder"]
