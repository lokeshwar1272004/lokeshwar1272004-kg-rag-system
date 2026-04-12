"""
query/classifier.py
Classifies incoming user queries into:
  - GRAPH: relationship/connectivity questions
  - VECTOR: semantic/content questions  
  - HYBRID: requires both

Uses LLM for classification with keyword-based fallback.
"""

import re
from enum import Enum
from typing import Tuple

from utils.logger import logger
from utils.cache import cached
from llm.prompts import SYSTEM_QUERY_CLASSIFIER, build_query_classification_prompt


class QueryType(str, Enum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"
    HYBRID = "HYBRID"


# ── Keyword patterns for fast classification without LLM ──────────────────────

GRAPH_KEYWORDS = [
    r"\bwho\s+(?:works?|employed|leads?|founded|runs?|manages?)\b",
    r"\bhow\s+(?:are|is|were)\s+.+\s+(?:connected|related|linked)\b",
    r"\bwhat\s+(?:connects?|links?|relates?)\b",
    r"\brelationship\s+between\b",
    r"\bpath\s+(?:between|from|to)\b",
    r"\bconnected\s+to\b",
    r"\bworks?\s+(?:at|for|with)\b",
    r"\bpartner(?:s|ed)?\s+with\b",
    r"\bsubsidiary\s+of\b",
    r"\bheadquartered\s+in\b",
    r"\bfounded\s+by\b",
    r"\bowned\s+by\b",
    r"\bCEO\s+of\b",
    r"\bdirector\s+of\b",
]

VECTOR_KEYWORDS = [
    r"\bwhat\s+(?:does|did|is)\s+.+\s+(?:say|mean|state|describe|explain|define)\b",
    r"\bexplain\b",
    r"\bsummariz(?:e|ing)\b",
    r"\bwhat\s+(?:are|is)\s+the\s+(?:main|key|important|core)\b",
    r"\baccording\s+to\b",
    r"\bdescribe\b",
    r"\btell\s+me\s+about\b",
    r"\bwhat\s+information\b",
    r"\bdetails?\s+about\b",
    r"\boverview\b",
    r"\bbackground\b",
]

HYBRID_KEYWORDS = [
    r"\beverything\s+about\b",
    r"\bfull\s+(?:details?|profile|context)\b",
    r"\bcomprehensive\b",
    r"\ball\s+(?:information|details?)\b",
    r"\bwho\s+is\b",   # Needs both identity (graph) and description (vector)
    r"\bwhat\s+(?:kind|type)\s+of\s+(?:company|person|organization)\b",
]


class QueryClassifier:
    """
    Classifies queries into GRAPH, VECTOR, or HYBRID.
    
    Strategy:
    1. LLM-based classification (most accurate)
    2. Keyword-pattern fallback (fast, no API needed)
    3. Default to HYBRID (covers all bases)
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from llm.llm_router import llm_router
            self._llm = llm_router
        return self._llm

    def classify(self, question: str, use_llm: bool = True) -> Tuple[QueryType, float]:
        """
        Classify a question.
        Returns (QueryType, confidence_score).
        """
        if not question or not question.strip():
            return QueryType.VECTOR, 0.5

        # Try LLM classification
        if use_llm:
            try:
                query_type, confidence = self._classify_with_llm(question)
                logger.info(f"Query classified [{query_type}] (LLM, conf={confidence:.2f}): {question[:60]}")
                return query_type, confidence
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}. Using keywords.")

        # Keyword fallback
        query_type, confidence = self._classify_with_keywords(question)
        logger.info(f"Query classified [{query_type}] (keywords, conf={confidence:.2f}): {question[:60]}")
        return query_type, confidence

    @cached("query_class", ttl=300)
    def _classify_with_llm(self, question: str) -> Tuple[QueryType, float]:
        """Use LLM for classification."""
        prompt = build_query_classification_prompt(question)
        llm = self._get_llm()
        response = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_QUERY_CLASSIFIER,
            max_tokens=10,
            temperature=0.0,
        )

        if not response:
            raise ValueError("LLM returned empty response")

        response_clean = response.strip().upper()

        # Extract the classification
        for qt in [QueryType.HYBRID, QueryType.GRAPH, QueryType.VECTOR]:
            if qt.value in response_clean:
                return qt, 0.9

        raise ValueError(f"Could not parse classification: {response_clean}")

    def _classify_with_keywords(self, question: str) -> Tuple[QueryType, float]:
        """Keyword-based classification."""
        q = question.lower()

        # Count matches for each type
        graph_matches = sum(1 for p in GRAPH_KEYWORDS if re.search(p, q, re.IGNORECASE))
        vector_matches = sum(1 for p in VECTOR_KEYWORDS if re.search(p, q, re.IGNORECASE))
        hybrid_matches = sum(1 for p in HYBRID_KEYWORDS if re.search(p, q, re.IGNORECASE))

        # Determine winner
        if hybrid_matches > 0:
            return QueryType.HYBRID, 0.7

        if graph_matches > vector_matches:
            conf = min(0.8, 0.5 + graph_matches * 0.1)
            return QueryType.GRAPH, conf

        if vector_matches > graph_matches:
            conf = min(0.8, 0.5 + vector_matches * 0.1)
            return QueryType.VECTOR, conf

        # Default: HYBRID covers all bases
        return QueryType.HYBRID, 0.5


# Singleton
query_classifier = QueryClassifier()

__all__ = ["query_classifier", "QueryClassifier", "QueryType"]
