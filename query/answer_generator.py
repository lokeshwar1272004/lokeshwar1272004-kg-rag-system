"""
query/answer_generator.py
Generates final answers using LLM + retrieved context.
Handles GRAPH, VECTOR, and HYBRID answer strategies.
"""

from typing import Optional
from dataclasses import dataclass

from query.classifier import QueryType
from query.retriever import RetrievalResult
from utils.logger import logger
from utils.text_utils import truncate_to_token_limit
from llm.prompts import (
    SYSTEM_ANSWER_GENERATION,
    build_answer_prompt,
    build_graph_summary_prompt,
    build_hybrid_prompt,
)
from config.settings import settings


@dataclass
class Answer:
    """Final answer with metadata."""
    question: str
    answer: str
    query_type: QueryType
    sources_used: int
    graph_facts_used: int
    llm_provider: str
    confidence: str  # "high", "medium", "low"


class AnswerGenerator:
    """
    Generates answers by combining retrieved context with LLM reasoning.
    
    Strategy per query type:
    - GRAPH: format graph triples → LLM summarize
    - VECTOR: format chunks → LLM answer
    - HYBRID: merge both → LLM synthesize
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from llm.llm_router import llm_router
            self._llm = llm_router
        return self._llm

    def generate(self, retrieval: RetrievalResult) -> Answer:
        """
        Generate an answer from a RetrievalResult.
        """
        llm = self._get_llm()
        question = retrieval.query
        query_type = retrieval.query_type

        # Check if we have any context
        has_vector = len(retrieval.vector_results) > 0
        has_graph = len(retrieval.graph_results) > 0

        if not has_vector and not has_graph:
            return Answer(
                question=question,
                answer=(
                    "I don't have enough information in the knowledge base to answer this question. "
                    "Please ingest relevant documents first."
                ),
                query_type=query_type,
                sources_used=0,
                graph_facts_used=0,
                llm_provider=llm.get_active_provider(),
                confidence="low",
            )

        # Build prompt based on query type
        prompt = self._build_prompt(retrieval)

        # Generate response
        response = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_ANSWER_GENERATION,
            max_tokens=800,
            temperature=0.1,
        )

        if not response:
            response = "I was unable to generate an answer. Please check your LLM configuration."

        # Assess confidence
        confidence = self._assess_confidence(retrieval, response)

        return Answer(
            question=question,
            answer=response,
            query_type=query_type,
            sources_used=len(retrieval.vector_results),
            graph_facts_used=len(retrieval.graph_results),
            llm_provider=llm.get_active_provider(),
            confidence=confidence,
        )

    def _build_prompt(self, retrieval: RetrievalResult) -> str:
        """Build the right prompt type based on query type and available context."""
        question = retrieval.query
        has_vector = len(retrieval.vector_results) > 0
        has_graph = len(retrieval.graph_results) > 0

        if has_graph and not has_vector:
            # Pure graph answer
            return build_graph_summary_prompt(retrieval.graph_results, question)

        elif has_vector and not has_graph:
            # Pure vector answer
            vector_context = "\n\n".join(
                f"[Source {i+1}] {r.content}"
                for i, r in enumerate(retrieval.vector_results[:settings.retrieval.top_k])
            )
            vector_context = truncate_to_token_limit(vector_context, 2500)
            return build_answer_prompt(question, vector_context)

        else:
            # Hybrid: both graph and vector
            graph_lines = [
                f"• {r.get('subject', '')} --[{r.get('predicate', r.get('relation', ''))}]--> {r.get('object', '')}"
                for r in retrieval.graph_results[:10]
            ]
            graph_context = "\n".join(graph_lines)

            vector_context = "\n\n".join(
                f"[Doc {i+1}] {r.content}"
                for i, r in enumerate(retrieval.vector_results[:3])
            )
            vector_context = truncate_to_token_limit(vector_context, 1500)

            return build_hybrid_prompt(question, graph_context, vector_context)

    def _assess_confidence(self, retrieval: RetrievalResult, answer: str) -> str:
        """Heuristic confidence scoring."""
        score = 0

        # More sources = higher confidence
        if retrieval.metadata.get("vector_count", 0) >= 3:
            score += 2
        elif retrieval.metadata.get("vector_count", 0) >= 1:
            score += 1

        if retrieval.metadata.get("graph_count", 0) >= 3:
            score += 2
        elif retrieval.metadata.get("graph_count", 0) >= 1:
            score += 1

        # Hedging language in answer = lower confidence
        hedging_phrases = ["i don't know", "unclear", "not enough", "cannot", "insufficient"]
        if any(p in answer.lower() for p in hedging_phrases):
            score -= 2

        if score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        return "low"


# Singleton
answer_generator = AnswerGenerator()

__all__ = ["answer_generator", "AnswerGenerator", "Answer"]
