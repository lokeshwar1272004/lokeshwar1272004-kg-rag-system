"""
llm/prompts.py
Centralized prompt templates for all LLM operations.
Optimized for low token usage and high accuracy.
"""


# ── System Prompts ─────────────────────────────────────────────────────────────

SYSTEM_ANSWER_GENERATION = """You are an expert knowledge assistant. Answer questions accurately using the provided context.
Rules:
- Use ONLY the provided context to answer
- If context is insufficient, say so clearly
- Be concise and structured
- Cite key facts from the context"""

SYSTEM_CYPHER_GENERATION = """You are a Neo4j Cypher expert. Convert natural language to Cypher queries.
Rules:
- Return ONLY the Cypher query, no explanation
- Use MATCH, WHERE, RETURN patterns
- Always LIMIT results to 10 unless asked otherwise
- Use case-insensitive matching with toLower()"""

SYSTEM_QUERY_CLASSIFIER = """You are a query routing expert. Classify queries into types.
Respond with ONLY one of: GRAPH, VECTOR, HYBRID
- GRAPH: relationship/connection questions (who works at, what connects, how are X and Y related)
- VECTOR: semantic/content questions (what does X say about, explain, summarize)
- HYBRID: needs both graph relationships and content (everything about X, full context of Y)"""

SYSTEM_RELATION_EXTRACTOR = """You are a knowledge graph expert. Extract entity relationships from text.
Return ONLY valid JSON. No explanation."""


# ── Query Templates ────────────────────────────────────────────────────────────

def build_answer_prompt(question: str, context: str) -> str:
    return f"""CONTEXT:
{context}

QUESTION: {question}

Answer the question using only the context above. Be concise and accurate."""


def build_cypher_prompt(question: str, schema: str = "") -> str:
    schema_section = f"\nGRAPH SCHEMA:\n{schema}\n" if schema else ""
    return f"""Convert this question to a Neo4j Cypher query.{schema_section}

QUESTION: {question}

Return ONLY the Cypher query. Use toLower() for name matching. LIMIT 10."""


def build_query_classification_prompt(question: str) -> str:
    return f"""Classify this query:

QUERY: "{question}"

Respond with exactly one word: GRAPH, VECTOR, or HYBRID"""


def build_graph_summary_prompt(triples: list, question: str) -> str:
    triples_str = "\n".join([
        f"- {t.get('subject', '')} --[{t.get('predicate', '')}]--> {t.get('object', '')}"
        for t in triples[:20]  # Limit graph context
    ])
    return f"""KNOWLEDGE GRAPH FACTS:
{triples_str}

QUESTION: {question}

Answer using the graph relationships above."""


def build_hybrid_prompt(question: str, graph_context: str, vector_context: str) -> str:
    return f"""KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context}

DOCUMENT CONTEXT:
{vector_context}

QUESTION: {question}

Synthesize both sources to give a comprehensive answer."""


__all__ = [
    "SYSTEM_ANSWER_GENERATION",
    "SYSTEM_CYPHER_GENERATION",
    "SYSTEM_QUERY_CLASSIFIER",
    "SYSTEM_RELATION_EXTRACTOR",
    "build_answer_prompt",
    "build_cypher_prompt",
    "build_query_classification_prompt",
    "build_graph_summary_prompt",
    "build_hybrid_prompt",
]
