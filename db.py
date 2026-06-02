"""
Shared database and embedding resources.

Both the vector search and keyword search tools need access to ChromaDB
and the embedding model. This module provides lazy-loaded singletons so
the heavy initialization only happens once per process.
"""

import os
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "lexi_judgments"
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
