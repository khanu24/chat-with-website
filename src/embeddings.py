"""
embeddings.py - Creates embeddings and manages the vector store.

Uses HuggingFace's free sentence-transformers model to convert text
chunks into vector representations, stores them in a local ChromaDB
vector database, and provides similarity search.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model.

    Uses 'all-MiniLM-L6-v2' — a small, fast, free model that runs
    locally on CPU. Ideal for beginners and low-resource machines.

    Returns:
        HuggingFaceEmbeddings instance
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def build_vectorstore(chunks: list[str]) -> Chroma:
    """
    Create a ChromaDB vector store from text chunks.

    Each chunk is embedded and stored in a local persistent directory
    so it can be queried later.

    Args:
        chunks: List of text chunks to embed and store

    Returns:
        Chroma vector store instance
    """
    if not chunks:
        raise ValueError("Cannot build vector store: chunks list is empty")

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=config.VECTORSTORE_DIR,
    )

    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Returns:
        Chroma vector store instance
    """
    embeddings = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

    return vectorstore


def search_similar_chunks(vectorstore: Chroma, query: str, k: int = None) -> list[str]:
    """
    Find the top-k most similar chunks for a given query.

    Args:
        vectorstore: Chroma vector store to search
        query: User's question
        k: Number of results (defaults to config.RETRIEVAL_K)

    Returns:
        List of relevant text chunks
    """
    if k is None:
        k = config.RETRIEVAL_K

    results = vectorstore.similarity_search(query, k=k)

    return [doc.page_content for doc in results]