"""
query/retriever.py
Hybrid retrieval engine.
Combines Knowledge Graph traversal + Vector semantic search
based on query type classification.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from query.classifier import QueryType
from rag.vector_store import RetrievedChunk
from utils.logger import logger
from utils.text_utils import format_context_for_llm
from config.settings import settings


@dataclass
class RetrievalResult:
    """Combined result from hybrid retrieval."""
    query: str
    query_type: QueryType
    vector_results: List[RetrievedChunk]
    graph_results: List[Dict]
    formatted_context: str
    metadata: Dict


class HybridRetriever:
    """
    Routes retrieval to graph, vector, or both based on query type.
    
    GRAPH  → Only Neo4j/NetworkX traversal
    VECTOR → Only ChromaDB semantic search
    HYBRID → Both, merged into unified context
    """

    def __init__(self):
        self._vector_store = None
        self._graph_client = None
        self._cypher_generator = None

    def _get_vector_store(self):
        if self._vector_store is None:
            from rag.vector_store import vector_store
            self._vector_store = vector_store
        return self._vector_store

    def _get_graph(self):
        if self._graph_client is None:
            from graph.neo4j_client import graph_client
            self._graph_client = graph_client
        return self._graph_client

    def _get_cypher_gen(self):
        if self._cypher_generator is None:
            from graph.cypher_generator import cypher_generator
            self._cypher_generator = cypher_generator
        return self._cypher_generator

    def retrieve(
        self,
        query: str,
        query_type: QueryType,
        top_k: int = None,
    ) -> RetrievalResult:
        """
        Main retrieval method.
        Dispatches to the appropriate retrieval strategy.
        """
        top_k = top_k or settings.retrieval.top_k
        vector_results = []
        graph_results = []

        if query_type == QueryType.GRAPH:
            graph_results = self._retrieve_graph(query)

        elif query_type == QueryType.VECTOR:
            vector_results = self._retrieve_vector(query, top_k)

        elif query_type == QueryType.HYBRID:
            # Run both in sequence (could be parallelized)
            graph_results = self._retrieve_graph(query)
            vector_results = self._retrieve_vector(query, top_k)

        # Format into unified context string for LLM
        graph_context_dicts = [
            {"content": f"{r.get('subject', '')} --[{r.get('predicate', r.get('relation', ''))}]--> {r.get('object', '')}"}
            for r in graph_results
        ]
        formatted = format_context_for_llm(
            vector_results=[{"content": r.content} for r in vector_results],
            graph_results=[f"{r.get('subject', '')} --[{r.get('predicate', r.get('relation', ''))}]--> {r.get('object', '')}" for r in graph_results],
            max_tokens=settings.retrieval.max_context_tokens,
        )

        return RetrievalResult(
            query=query,
            query_type=query_type,
            vector_results=vector_results,
            graph_results=graph_results,
            formatted_context=formatted,
            metadata={
                "vector_count": len(vector_results),
                "graph_count": len(graph_results),
                "top_k": top_k,
            }
        )

    def _retrieve_vector(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Semantic vector search."""
        try:
            results = self._get_vector_store().search(query, top_k=top_k)
            logger.debug(f"Vector search returned {len(results)} chunks")
            return results
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    def _retrieve_graph(self, query: str) -> List[Dict]:
        """Knowledge Graph traversal via Cypher or direct API."""
        graph = self._get_graph()
        cypher_gen = self._get_cypher_gen()

        # Check if Neo4j (has run_cypher) or NetworkX
        if hasattr(graph, "run_cypher") and graph.driver:
            return self._retrieve_neo4j(query, graph, cypher_gen)
        else:
            return self._retrieve_networkx(query, graph)

    def _retrieve_neo4j(self, query: str, graph, cypher_gen) -> List[Dict]:
        """Neo4j retrieval: generate Cypher and execute."""
        try:
            schema = graph.get_schema() if hasattr(graph, "get_schema") else ""
            cypher = cypher_gen.generate(query, schema=schema)

            if not cypher:
                logger.warning("No Cypher generated; falling back to entity search.")
                return []

            logger.debug(f"Executing Cypher: {cypher[:100]}...")
            results = graph.run_cypher(cypher)
            logger.debug(f"Graph returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Neo4j retrieval failed: {e}")
            return []

    def _retrieve_networkx(self, query: str, graph) -> List[Dict]:
        """NetworkX retrieval: extract entities from query and traverse."""
        try:
            # Extract entity mentions from query
            import re
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            stop = {"Who", "What", "Where", "When", "How", "Why", "Is", "Are", "Does"}
            entities = [w for w in capitalized if w not in stop]

            results = []
            for entity in entities[:3]:
                rels = graph.query_relations(entity)
                results.extend(rels)

            # Deduplicate
            seen = set()
            unique = []
            for r in results:
                key = (r.get("subject"), r.get("predicate"), r.get("object"))
                if key not in seen:
                    seen.add(key)
                    unique.append(r)

            logger.debug(f"NetworkX returned {len(unique)} relations")
            return unique[:20]

        except Exception as e:
            logger.error(f"NetworkX retrieval failed: {e}")
            return []


# Singleton
hybrid_retriever = HybridRetriever()

__all__ = ["hybrid_retriever", "HybridRetriever", "RetrievalResult"]
