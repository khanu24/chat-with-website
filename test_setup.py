import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("SETUP TEST (Groq)")
print("=" * 50)

# 1. API Key check
api_key = os.getenv("GROQ_API_KEY")
if api_key and api_key.startswith("gsk_"):
    print(f"✅ Groq API Key loaded: {api_key[:12]}...")
else:
    print("❌ Groq API Key NOT found in .env")

# 2. Libraries check
try:
    import langchain
    import langchain_groq
    import bs4
    import requests
    import streamlit
    import chromadb
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✅ All libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")

# 3. Groq LLM test
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    response = llm.invoke("Say 'Hello from Groq!' in one line")
    print(f"✅ Groq says: {response.content}")
except Exception as e:
    print(f"❌ Groq error: {e}")

print("=" * 50)