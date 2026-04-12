"""
extraction/ner.py
Named Entity Recognition (NER) using spaCy.
Extracts entities (Person, Org, Location, etc.) from text chunks.
These entities become NODES in the Knowledge Graph.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from utils.logger import logger


@dataclass
class Entity:
    """Represents a named entity extracted from text."""
    text: str            # The entity surface form
    label: str           # Entity type (PERSON, ORG, GPE, etc.)
    start: int           # Char start in source text
    end: int             # Char end in source text
    normalized: str = "" # Lowercase normalized form


# Map spaCy labels to our graph node types
ENTITY_TYPE_MAP = {
    "PERSON": "Person",
    "PER": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "GPE": "Location",      # Geo-political entity
    "LOC": "Location",
    "LOCATION": "Location",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "WorkOfArt",
    "LAW": "Law",
    "LANGUAGE": "Language",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Money",
    "NORP": "Group",        # Nationalities, religious/political groups
    "FAC": "Facility",
    "CARDINAL": None,       # Skip pure numbers
    "ORDINAL": None,
    "PERCENT": None,
    "QUANTITY": None,
}


class NERExtractor:
    """
    Extracts named entities using spaCy.
    Falls back to regex-based extraction if spaCy is unavailable.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load spaCy model with graceful fallback."""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"spaCy model loaded: {self.model_name}")
        except OSError:
            logger.warning(
                f"spaCy model '{self.model_name}' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self.nlp = None
        except ImportError:
            logger.warning("spaCy not installed. NER will use fallback.")
            self.nlp = None

    def extract(self, text: str) -> List[Entity]:
        """Extract entities from a text string."""
        if not text or not text.strip():
            return []

        if self.nlp is not None:
            return self._extract_spacy(text)
        else:
            return self._extract_fallback(text)

    def _extract_spacy(self, text: str) -> List[Entity]:
        """spaCy-based NER."""
        try:
            doc = self.nlp(text)
            entities = []
            seen = set()

            for ent in doc.ents:
                node_type = ENTITY_TYPE_MAP.get(ent.label_)
                if node_type is None:
                    continue

                normalized = ent.text.lower().strip()
                if normalized in seen or len(normalized) < 2:
                    continue
                seen.add(normalized)

                entities.append(Entity(
                    text=ent.text.strip(),
                    label=node_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    normalized=normalized,
                ))

            return entities

        except Exception as e:
            logger.error(f"spaCy NER failed: {e}")
            return self._extract_fallback(text)

    def _extract_fallback(self, text: str) -> List[Entity]:
        """
        Basic regex fallback: captures capitalized noun phrases.
        Less accurate but functional without spaCy.
        """
        import re
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = []
        seen = set()

        for match in re.finditer(pattern, text):
            surface = match.group(0).strip()
            normalized = surface.lower()

            stop_phrases = {"the", "a", "an", "this", "that", "these", "those"}
            if normalized in stop_phrases or len(surface) < 3:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)

            entities.append(Entity(
                text=surface,
                label="Entity",
                start=match.start(),
                end=match.end(),
                normalized=normalized,
            ))

        return entities

    def extract_batch(self, texts: List[str]) -> List[List[Entity]]:
        """Extract entities from multiple texts."""
        if self.nlp and hasattr(self.nlp, 'pipe'):
            # Efficient batch processing with spaCy
            results = []
            docs = list(self.nlp.pipe(texts, batch_size=32))
            for doc in docs:
                entities = []
                seen = set()
                for ent in doc.ents:
                    node_type = ENTITY_TYPE_MAP.get(ent.label_)
                    if node_type is None:
                        continue
                    normalized = ent.text.lower().strip()
                    if normalized in seen or len(normalized) < 2:
                        continue
                    seen.add(normalized)
                    entities.append(Entity(
                        text=ent.text.strip(),
                        label=node_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        normalized=normalized,
                    ))
                results.append(entities)
            return results
        else:
            return [self.extract(t) for t in texts]


# Singleton
ner_extractor = NERExtractor()

__all__ = ["ner_extractor", "NERExtractor", "Entity"]
