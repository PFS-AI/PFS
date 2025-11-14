# File Version: 1.1.0
# /backend/kb_manager.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
Manages the application's internal Knowledge Base (KB).

This module is responsible for setting up and providing access to a vector store
containing the application's own documentation (e.g., help files, feature
explanations). This allows the AI search feature to answer questions about how
to use the application itself.

Key functionalities:
- `load_and_index_knowledge_base()`: Called once at startup, this function
  loads a markdown document, splits it into manageable chunks, generates
  embeddings using the HuggingFace model specified in the user's config,
  and indexes them into an in-memory Qdrant vector store.
- `get_kb_retriever()`: Provides a LangChain `VectorStoreRetriever` object,
  which the AI search module can use to fetch relevant document chunks based on
  a user's query.
- `search_knowledge_base()`: A direct search function for retrieving text chunks
  from the KB.
"""

# 1. IMPORTS ####################################################################################################
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from qdrant_client import QdrantClient, models
from typing import List, Optional

from .config_manager import get_config

# Block Version: 1.1.0
from .rag_pipeline import get_torch_device, get_and_prepare_model_path

# 2. SETUP & GLOBALS ############################################################################################
logger = logging.getLogger(__name__)

_db: Optional[QdrantVectorStore] = None
_embeddings = None

# 3. CORE FUNCTIONALITY #########################################################################################
def load_and_index_knowledge_base():
    """
    Loads, chunks, and indexes the application's knowledge base into an
    in-memory Qdrant vector store at application startup using the configured model.
    """
    global _db, _embeddings

    if _db is not None:
        logger.info("Knowledge base is already initialized.")
        return

    try:
        logger.info("Initializing Application Knowledge Base...")

        embedding_config = get_config("embedding_model")
        model_name = embedding_config.get("model_name")

        if not model_name:
            logger.warning("No embedding model name found in config. Application Knowledge Base (Q&A) will be disabled.")
            return

        kb_path = "static/kb/kb.md"
        logger.debug(f"Loading knowledge base document from: {kb_path}")
        loader = TextLoader(kb_path, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunked_docs = text_splitter.split_documents(documents)
        logger.debug(f"Split knowledge base into {len(chunked_docs)} chunks.")

        device = get_torch_device(embedding_config.get("device", "auto"))

        # Block Version: 1.1.0
        local_model_path = get_and_prepare_model_path(model_name, "Embedding Model (for KB)")
        logger.debug(f"Initializing KB embedding model from local path: {local_model_path} on device: {device}")

        _embeddings = HuggingFaceEmbeddings(
            model_name=local_model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        embedding_dim = len(_embeddings.embed_query("test"))
        logger.debug(f"Knowledge Base embedding dimension: {embedding_dim}")

        logger.debug("Initializing in-memory Qdrant client for KB.")
        client = QdrantClient(":memory:")
        collection_name = "app_knowledge_base"

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
        )
        logger.debug(f"In-memory collection '{collection_name}' created successfully.")

        _db = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=_embeddings
        )

        _db.add_documents(chunked_docs)

        logger.info("Application Knowledge Base initialized successfully.")

    except Exception:
        logger.critical("Could not initialize Application Knowledge Base. Q&A feature will be disabled.", exc_info=True)
        _db = None

# 4. ACCESSOR FUNCTIONS #########################################################################################
def get_kb_retriever() -> Optional[VectorStoreRetriever]:
    """
    Returns a retriever for the initialized knowledge base.
    Returns None if the knowledge base is not available.
    """
    if _db is None:
        return None
    return _db.as_retriever(search_kwargs={"k": 3})

def search_knowledge_base(query: str, k: int = 3) -> List[str]:
    """
    Searches the in-memory knowledge base for the most relevant chunks.
    """
    retriever = get_kb_retriever()
    if not retriever:
        logger.warning("Attempted to search knowledge base, but it is not available.")
        return ["The application's knowledge base is not available."]

    try:
        logger.debug(f"Searching knowledge base with k={k} for query: '{query}'")
        results = retriever.invoke(query)
        return [doc.page_content for doc in results]
    except Exception as e:
        logger.exception("An error occurred while searching the knowledge base.")
        return [f"An error occurred while searching the knowledge base: {e}"]
