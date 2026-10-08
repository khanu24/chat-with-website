"""
rag.py - Full RAG (Retrieval Augmented Generation) pipeline.

This module orchestrates the entire flow:
    URL → Scraper → Chunker → Embeddings → Vector Store → Retrieval → LLM

It exposes two main functions:
    - build_rag_from_url(url): processes a website and returns a vectorstore
    - ask_question(vectorstore, question): answers questions using the RAG system
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src import config
from src.scraper import fetch_website_text, get_website_title
from src.chunker import split_text
from src.embeddings import build_vectorstore, search_similar_chunks


# ============================================================
# LLM Setup
# ============================================================
def get_llm() -> ChatGroq:
    """
    Initialize the Groq LLM client.

    Returns:
        ChatGroq instance configured from config.py
    """
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
    )
    return llm


# ============================================================
# Prompt Template
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
- If the answer is not in the context, say "I couldn't find that information in the website content."
- Be concise and accurate.
- Quote specific details from the context when helpful.

Context from website:
{context}

Question: {question}

Answer:"""
)


# ============================================================
# Build RAG Pipeline
# ============================================================
def build_rag_from_url(url: str) -> dict:
    """
    Full pipeline: scrape URL → chunk → embed → store in vector DB.

    Args:
        url: Website URL to process

    Returns:
        Dictionary containing:
            - vectorstore: Chroma vector store instance
            - title: Website title
            - num_chunks: Number of chunks created
            - num_chars: Number of characters scraped
    """
    print(f"🌐 Processing: {url}")

    # Step 1: Scrape website
    print("  └─ Scraping website...")
    text = fetch_website_text(url)
    if not text or not text.strip():
        raise ValueError(f"No content extracted from: {url}")
    print(f"     ✅ Extracted {len(text):,} characters")

    # Step 2: Get website title
    title = get_website_title(url)

    # Step 3: Split into chunks
    print("  └─ Splitting into chunks...")
    chunks = split_text(text)
    if not chunks:
        raise ValueError(f"Could not create chunks from: {url}")
    print(f"     ✅ Created {len(chunks)} chunks")

    # Step 4: Build vector store
    print("  └─ Building embeddings & vector store (may take a minute)...")
    vectorstore = build_vectorstore(chunks)
    print(f"     ✅ Vector store ready")

    return {
        "vectorstore": vectorstore,
        "title": title,
        "num_chunks": len(chunks),
        "num_chars": len(text),
    }


# ============================================================
# Ask a Question
# ============================================================
def ask_question(vectorstore, question: str) -> str:
    """
    Answer a user's question using the RAG pipeline.

    Args:
        vectorstore: Chroma vector store built from build_rag_from_url()
        question: User's question

    Returns:
        LLM's answer (str)
    """
    if not question or not question.strip():
        return "Please provide a valid question."

    # Step 1: Retrieve relevant chunks
    relevant_chunks = search_similar_chunks(vectorstore, question, k=config.RETRIEVAL_K)

    if not relevant_chunks:
        return "I couldn't find relevant information in the website content."

    # Step 2: Combine chunks into context
    context = "\n\n---\n\n".join(relevant_chunks)

    # Step 3: Build prompt
    prompt = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    # Step 4: Get LLM response
    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content


# ============================================================
# Get Website Summary
# ============================================================
def get_website_summary(vectorstore) -> str:
    """
    Generate a short summary of the website.

    Uses more chunks than a normal query to capture the overall topic.

    Args:
        vectorstore: Chroma vector store built from build_rag_from_url()

    Returns:
        Summary text (str)
    """
    # Use more chunks for summary (15 instead of 4)
    chunks = search_similar_chunks(
        vectorstore,
        "What is this website about? What services or topics does it cover?",
        k=15,
    )

    if not chunks:
        return "Could not generate a summary."

    context = "\n\n---\n\n".join(chunks)

    summary_prompt = ChatPromptTemplate.from_template(
        """Based on the following excerpts from a website, write a concise 2-3 sentence summary
of what this website is about and what it offers.

Excerpts:
{context}

Summary:"""
    )

    prompt = summary_prompt.format_messages(context=context)
    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content