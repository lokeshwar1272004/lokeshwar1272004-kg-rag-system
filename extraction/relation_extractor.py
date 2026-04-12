"""
extraction/relation_extractor.py
Relation extraction between entities.
Primary: LLM-based extraction (Groq/Ollama)
Fallback: Rule-based heuristics

Extracts triples: (Subject Entity) --[RELATION]--> (Object Entity)
These become EDGES in the Knowledge Graph.
"""

import re
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass

from extraction.ner import Entity
from utils.logger import logger
from utils.text_utils import truncate_to_token_limit


@dataclass
class Relation:
    """A directed relationship between two entities."""
    subject: str        # Entity text
    subject_type: str   # Entity label
    predicate: str      # Relation type (e.g., "WORKS_FOR", "LOCATED_IN")
    object: str         # Entity text
    object_type: str    # Entity label
    confidence: float = 1.0
    source_text: str = ""


# Common relation patterns for rule-based fallback
RELATION_PATTERNS = [
    # Employment
    (r"(\w[\w\s]+)\s+(?:works?(?:\s+at)?|is\s+employed(?:\s+(?:at|by))?|joined)\s+([A-Z][\w\s]+)", "WORKS_FOR"),
    # Leadership
    (r"([A-Z][\w\s]+)\s+is\s+(?:the\s+)?(?:CEO|president|director|head|founder|co-founder)\s+of\s+([A-Z][\w\s]+)", "LEADS"),
    # Location
    (r"([A-Z][\w\s]+)\s+(?:is\s+)?(?:located|based|headquartered)\s+in\s+([A-Z][\w\s]+)", "LOCATED_IN"),
    # Ownership
    (r"([A-Z][\w\s]+)\s+(?:acquired|bought|purchased|owns?)\s+([A-Z][\w\s]+)", "OWNS"),
    # Partnership
    (r"([A-Z][\w\s]+)\s+(?:partnered?|collaborated?)\s+with\s+([A-Z][\w\s]+)", "PARTNERS_WITH"),
    # Product
    (r"([A-Z][\w\s]+)\s+(?:developed|created|built|launched)\s+([A-Z][\w\s]+)", "CREATED"),
    # Education
    (r"([A-Z][\w\s]+)\s+(?:studied|graduated from|attended)\s+([A-Z][\w\s]+)", "STUDIED_AT"),
]


class RelationExtractor:
    """
    Extracts entity relations from text.
    Uses LLM as primary method, falls back to rules.
    """

    def __init__(self):
        self._llm = None  # Lazy load

    def _get_llm(self):
        """Lazy-load LLM client."""
        if self._llm is None:
            from llm.llm_router import llm_router
            self._llm = llm_router
        return self._llm

    def extract(
        self,
        text: str,
        entities: List[Entity],
        use_llm: bool = True,
    ) -> List[Relation]:
        """
        Extract relations from text given known entities.
        Tries LLM first, then rule-based fallback.
        """
        if len(entities) < 2:
            return []

        relations = []

        if use_llm:
            try:
                llm_relations = self._extract_with_llm(text, entities)
                if llm_relations:
                    return llm_relations
            except Exception as e:
                logger.warning(f"LLM relation extraction failed: {e}. Using rules.")

        # Rule-based fallback
        rule_relations = self._extract_with_rules(text, entities)
        relations.extend(rule_relations)

        # Co-occurrence fallback: entities in same sentence are related
        if not relations:
            relations = self._extract_cooccurrence(text, entities)

        return relations

    def _extract_with_llm(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Use LLM to extract structured relations."""
        entity_list = [f"{e.text} ({e.label})" for e in entities[:15]]  # Limit
        entity_str = ", ".join(entity_list)

        truncated_text = truncate_to_token_limit(text, max_tokens=800)

        prompt = f"""Extract relationships between entities from the text.

ENTITIES: {entity_str}

TEXT: {truncated_text}

Return a JSON array of relationships. Each item must have:
- "subject": entity name (from list above)
- "predicate": relationship type in UPPER_CASE (e.g., WORKS_FOR, LOCATED_IN, OWNS, LEADS, CREATED, STUDIED_AT, PARTNERS_WITH)
- "object": entity name (from list above)

Return ONLY valid JSON array. No explanation. Example:
[{{"subject": "Elon Musk", "predicate": "LEADS", "object": "Tesla"}}]

If no clear relationships exist, return: []"""

        llm = self._get_llm()
        response = llm.generate(prompt, max_tokens=500)

        if not response:
            return []

        # Parse JSON from response
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not json_match:
                return []

            raw_relations = json.loads(json_match.group(0))
            entity_map = {e.text.lower(): e for e in entities}
            relations = []

            for r in raw_relations:
                subj_text = r.get("subject", "").strip()
                pred = r.get("predicate", "").strip().upper().replace(" ", "_")
                obj_text = r.get("object", "").strip()

                if not (subj_text and pred and obj_text):
                    continue

                subj_entity = entity_map.get(subj_text.lower())
                obj_entity = entity_map.get(obj_text.lower())

                relations.append(Relation(
                    subject=subj_text,
                    subject_type=subj_entity.label if subj_entity else "Entity",
                    predicate=pred,
                    object=obj_text,
                    object_type=obj_entity.label if obj_entity else "Entity",
                    confidence=0.8,
                    source_text=text[:200],
                ))

            logger.info(f"LLM extracted {len(relations)} relations")
            return relations

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM relations: {e}")
            return []

    def _extract_with_rules(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Rule-based relation extraction using regex patterns."""
        relations = []
        entity_texts = {e.text.lower(): e for e in entities}

        for pattern, predicate in RELATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj_text = match.group(1).strip()
                obj_text = match.group(2).strip()

                subj_entity = entity_texts.get(subj_text.lower())
                obj_entity = entity_texts.get(obj_text.lower())

                if subj_text and obj_text:
                    relations.append(Relation(
                        subject=subj_text,
                        subject_type=subj_entity.label if subj_entity else "Entity",
                        predicate=predicate,
                        object=obj_text,
                        object_type=obj_entity.label if obj_entity else "Entity",
                        confidence=0.6,
                        source_text=match.group(0),
                    ))

        return relations

    def _extract_cooccurrence(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Fallback: entities in the same sentence are weakly related.
        Creates RELATED_TO edges as a last resort.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relations = []
        seen_pairs = set()

        for sentence in sentences:
            sent_entities = [e for e in entities if e.text.lower() in sentence.lower()]
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    e1, e2 = sent_entities[i], sent_entities[j]
                    pair = tuple(sorted([e1.normalized, e2.normalized]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    relations.append(Relation(
                        subject=e1.text,
                        subject_type=e1.label,
                        predicate="RELATED_TO",
                        object=e2.text,
                        object_type=e2.label,
                        confidence=0.3,
                        source_text=sentence[:200],
                    ))

        return relations


# Singleton
relation_extractor = RelationExtractor()

__all__ = ["relation_extractor", "RelationExtractor", "Relation"]
