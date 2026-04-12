"""
query/pipeline.py
End-to-end Query Pipeline.
Orchestrates: classify → retrieve → generate → return answer.

This is the main entry point for answering user questions.
"""

from typing import Optional
from dataclasses import dataclass

from query.classifier import query_classifier, QueryType
from query.retriever import hybrid_retriever, RetrievalResult
from query.answer_generator import answer_generator, Answer
from utils.logger import logger
from utils.cache import cache_manager


@dataclass
class QueryResponse:
    """Complete response from the query pipeline."""
    question: str
    answer: str
    query_type: str
    confidence: str
    sources_used: int
    graph_facts_used: int
    llm_provider: str
    retrieval: Optional[RetrievalResult] = None


class QueryPipeline:
    """
    Full query pipeline:
    
    User Question
         │
         ▼
    [Query Classifier] → GRAPH / VECTOR / HYBRID
         │
         ▼
    [Hybrid Retriever]
     ├── Graph: Cypher → Neo4j/NetworkX
     └── Vector: Embedding → ChromaDB
         │
         ▼
    [Answer Generator] → LLM (Groq → Ollama fallback)
         │
         ▼
    Final Answer + Metadata
    """

    def __init__(self):
        self.classifier = query_classifier
        self.retriever = hybrid_retriever
        self.generator = answer_generator

    def run(
        self,
        question: str,
        force_query_type: Optional[QueryType] = None,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> QueryResponse:
        """
        Run the complete QA pipeline for a user question.
        
        Args:
            question: User's natural language question
            force_query_type: Override automatic classification
            top_k: Number of documents to retrieve
            use_cache: Whether to use response caching
            
        Returns:
            QueryResponse with answer and metadata
        """
        if not question or not question.strip():
            return QueryResponse(
                question=question,
                answer="Please enter a valid question.",
                query_type="UNKNOWN",
                confidence="low",
                sources_used=0,
                graph_facts_used=0,
                llm_provider="none",
            )

        # Check cache for identical question
        if use_cache:
            cache_key = cache_manager._make_key("pipeline_response", question.strip())
            cached_response = cache_manager.get(cache_key)
            if cached_response:
                logger.info(f"Pipeline cache HIT for: {question[:50]}...")
                return cached_response

        logger.info(f"Processing query: {question[:80]}...")

        # Step 1: Classify query type
        if force_query_type:
            query_type = force_query_type
            confidence_score = 1.0
        else:
            query_type, confidence_score = self.classifier.classify(question)

        # Step 2: Retrieve relevant context
        retrieval = self.retriever.retrieve(
            query=question,
            query_type=query_type,
            top_k=top_k,
        )

        # Step 3: Generate answer
        answer = self.generator.generate(retrieval)

        # Step 4: Package response
        response = QueryResponse(
            question=question,
            answer=answer.answer,
            query_type=query_type.value,
            confidence=answer.confidence,
            sources_used=answer.sources_used,
            graph_facts_used=answer.graph_facts_used,
            llm_provider=answer.llm_provider,
            retrieval=retrieval,
        )

        # Cache the response
        if use_cache:
            cache_manager.set(cache_key, response, ttl=300)

        logger.info(
            f"Query answered [{query_type.value}] "
            f"sources={answer.sources_used} graph={answer.graph_facts_used} "
            f"conf={answer.confidence} llm={answer.llm_provider}"
        )

        return response


# Singleton
query_pipeline = QueryPipeline()

__all__ = ["query_pipeline", "QueryPipeline", "QueryResponse"]
