"""
chunker.py - Splits long text into smaller overlapping chunks.

LLMs have token limits, so we split website content into manageable
chunks that can be embedded and retrieved individually.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config


def split_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks using LangChain's splitter.

    The splitter tries to keep paragraphs/sentences intact by
    recursively splitting on separators: ["\n\n", "\n", ". ", " ", ""].

    Args:
        text: Raw text to split (e.g., scraped website content)

    Returns:
        List of text chunks (str)
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    # Filter out empty/whitespace-only chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    return chunks


def get_chunk_stats(chunks: list[str]) -> dict:
    """
    Return statistics about the chunks (useful for debugging/UI).

    Args:
        chunks: List of text chunks

    Returns:
        Dictionary with count, total chars, avg/min/max chunk size.
    """
    if not chunks:
        return {
            "count": 0,
            "total_chars": 0,
            "avg_size": 0,
            "min_size": 0,
            "max_size": 0,
        }

    sizes = [len(c) for c in chunks]

    return {
        "count": len(chunks),
        "total_chars": sum(sizes),
        "avg_size": sum(sizes) // len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
    }