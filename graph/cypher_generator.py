"""
graph/cypher_generator.py
Converts natural language questions to Neo4j Cypher queries using LLM.
Includes safety validation and fallback patterns.
"""

import re
from typing import Optional, List

from utils.logger import logger
from utils.cache import cached
from llm.prompts import SYSTEM_CYPHER_GENERATION, build_cypher_prompt


# Safe Cypher patterns for common question types
CYPHER_TEMPLATES = {
    "who_works_for": """
        MATCH (p:Person)-[:WORKS_FOR]->(o:Organization)
        WHERE toLower(o.name) CONTAINS toLower($entity)
        RETURN p.display_name AS person, o.display_name AS organization
        LIMIT 10
    """,
    "find_entity": """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($entity)
           OR toLower(n.display_name) CONTAINS toLower($entity)
        RETURN n.display_name AS name, labels(n)[0] AS type
        LIMIT 10
    """,
    "relations_of": """
        MATCH (s)-[r]->(o)
        WHERE toLower(s.name) CONTAINS toLower($entity)
           OR toLower(o.name) CONTAINS toLower($entity)
        RETURN s.display_name AS subject, type(r) AS relation, o.display_name AS object
        LIMIT 15
    """,
    "path_between": """
        MATCH path = shortestPath(
            (a)-[*..5]-(b)
        )
        WHERE toLower(a.name) CONTAINS toLower($entity1)
          AND toLower(b.name) CONTAINS toLower($entity2)
        RETURN [n IN nodes(path) | n.display_name] AS path_nodes
        LIMIT 3
    """,
}

# Dangerous Cypher operations that should be blocked
CYPHER_BLACKLIST = [
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bDROP\b",
    r"\bCREATE\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bMERGE\b",
    r"\bCALL\s+db\.schema\.writable",
]


class CypherGenerator:
    """
    Generates Cypher queries from natural language.
    
    Strategy:
    1. Try LLM-generated Cypher (most flexible)
    2. Fall back to template-based queries (most reliable)
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from llm.llm_router import llm_router
            self._llm = llm_router
        return self._llm

    def generate(
        self,
        question: str,
        schema: str = "",
        use_llm: bool = True,
    ) -> Optional[str]:
        """
        Generate a safe Cypher query for a natural language question.
        """
        cypher = None

        if use_llm:
            try:
                cypher = self._generate_with_llm(question, schema)
            except Exception as e:
                logger.warning(f"LLM Cypher generation failed: {e}")

        if not cypher:
            cypher = self._generate_from_template(question)

        return cypher

    @cached("cypher_gen", ttl=600)
    def _generate_with_llm(self, question: str, schema: str) -> Optional[str]:
        """Use LLM to generate Cypher."""
        prompt = build_cypher_prompt(question, schema)
        llm = self._get_llm()
        response = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_CYPHER_GENERATION,
            max_tokens=300,
            temperature=0.0,
        )

        if not response:
            return None

        # Extract Cypher from response (handle markdown code blocks)
        cypher = self._extract_cypher(response)

        if cypher and self._is_safe(cypher):
            logger.debug(f"LLM Cypher: {cypher[:100]}...")
            return cypher

        return None

    def _extract_cypher(self, text: str) -> Optional[str]:
        """Extract Cypher query from LLM response."""
        # Handle markdown code blocks
        code_match = re.search(r'```(?:cypher)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()

        # If it starts with MATCH, return directly
        match_pattern = re.search(r'(MATCH\s+.*?)(?:\n\n|$)', text, re.DOTALL | re.IGNORECASE)
        if match_pattern:
            return match_pattern.group(1).strip()

        # Return cleaned response
        cleaned = text.strip()
        if "MATCH" in cleaned.upper():
            return cleaned

        return None

    def _is_safe(self, cypher: str) -> bool:
        """Validate Cypher is read-only."""
        cypher_upper = cypher.upper()
        for pattern in CYPHER_BLACKLIST:
            if re.search(pattern, cypher_upper):
                logger.warning(f"Unsafe Cypher blocked: {pattern}")
                return False
        return True

    def _generate_from_template(self, question: str) -> Optional[str]:
        """
        Pattern match question to template.
        More reliable but less flexible than LLM.
        """
        q_lower = question.lower()

        # Extract entity mentions
        entities = self._extract_entities_from_question(question)

        if "work" in q_lower or "employ" in q_lower or "company" in q_lower:
            if entities:
                return CYPHER_TEMPLATES["who_works_for"].replace("$entity", f"'{entities[0]}'")

        if "relation" in q_lower or "connect" in q_lower or "know" in q_lower:
            if entities:
                return CYPHER_TEMPLATES["relations_of"].replace("$entity", f"'{entities[0]}'")

        if "path" in q_lower or "between" in q_lower or "link" in q_lower:
            if len(entities) >= 2:
                return (CYPHER_TEMPLATES["path_between"]
                        .replace("$entity1", f"'{entities[0]}'")
                        .replace("$entity2", f"'{entities[1]}'"))

        # Generic entity search
        if entities:
            return CYPHER_TEMPLATES["relations_of"].replace("$entity", f"'{entities[0]}'")

        # Last resort: return all triples
        return "MATCH (s)-[r]->(o) RETURN s.display_name AS subject, type(r) AS relation, o.display_name AS object LIMIT 10"

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        Quick entity extraction from question text.
        Just looks for capitalized words/phrases.
        """
        # Find quoted entities
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
        entities = [q[0] or q[1] for q in quoted]

        # Find capitalized sequences
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        stop_words = {"Who", "What", "Where", "When", "How", "Why", "Is", "Are", "Does", "Do"}
        entities.extend([w for w in capitalized if w not in stop_words])

        return entities[:3]  # Return top 3


# Singleton
cypher_generator = CypherGenerator()

__all__ = ["cypher_generator", "CypherGenerator"]
