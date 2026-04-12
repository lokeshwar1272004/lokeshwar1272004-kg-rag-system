"""
utils/text_utils.py
Text preprocessing, cleaning, and token estimation utilities.
"""

import re
from typing import List, Optional
from config.settings import settings


def clean_text(text: str) -> str:
    """Remove noise from raw extracted text."""
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove special chars (keep punctuation)
    text = re.sub(r'[^\w\s\.,;:!?\-\'\"()\[\]{}/]', '', text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style)."""
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated for context limit ...]"


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[str]:
    """
    Split text into overlapping chunks for RAG indexing.
    Uses character-based chunking with word boundary respect.
    """
    chunk_size = chunk_size or settings.retrieval.chunk_size
    chunk_overlap = chunk_overlap or settings.retrieval.chunk_overlap

    if not text.strip():
        return []

    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            # Keep overlap: retain last few sentences
            overlap_text = " ".join(current_chunk)
            overlap_start = max(0, len(overlap_text) - chunk_overlap)
            overlap_portion = overlap_text[overlap_start:]
            current_chunk = [overlap_portion] if overlap_portion else []
            current_len = len(overlap_portion)

        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Simple keyword extraction using frequency analysis.
    Filters out stop words.
    """
    stop_words = {
        "the", "a", "an", "is", "in", "of", "and", "or", "to",
        "for", "on", "at", "by", "from", "with", "this", "that",
        "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "it", "its", "as", "up", "so", "but",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq = {}
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def format_context_for_llm(
    vector_results: List[dict],
    graph_results: Optional[List[dict]] = None,
    max_tokens: int = None,
) -> str:
    """
    Format retrieved context into a clean prompt-ready string.
    Prioritizes graph results, then vector results.
    """
    max_tokens = max_tokens or settings.retrieval.max_context_tokens
    parts = []

    if graph_results:
        parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
        for i, r in enumerate(graph_results[:5], 1):
            parts.append(f"[Graph {i}] {r}")

    if vector_results:
        parts.append("\n=== SEMANTIC SEARCH CONTEXT ===")
        for i, r in enumerate(vector_results, 1):
            content = r.get("content", r.get("text", str(r)))
            parts.append(f"[Doc {i}] {content}")

    combined = "\n".join(parts)
    return truncate_to_token_limit(combined, max_tokens)


__all__ = [
    "clean_text",
    "estimate_tokens",
    "truncate_to_token_limit",
    "chunk_text",
    "extract_keywords",
    "format_context_for_llm",
]
