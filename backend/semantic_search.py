# File Version: 1.2.0
# /backend/semantic_search.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
Manages the semantic search capabilities of the application.

This module is responsible for the entire lifecycle of semantic search, including
indexing files, storing document data, and performing retrieval. It leverages a
Retrieval-Augmented Generation (RAG) pipeline with embeddings, a vector database
(Qdrant), and an optional reranker.

Key functionalities include:
- **Indexing (`run_indexing_task`):** Scans a specified directory, loads supported
  documents. It uses PyMuPDF for fast, self-contained PDF parsing and falls back
  to `unstructured` for other file types. It then splits documents into parent/child
  chunks, generates embeddings, and upserts them into a Qdrant vector store.
- **Document Storage:** Implements a persistent document store using an SQLite
  database (`docstore.db`). This stores the larger parent chunks, which are
  retrieved after a smaller child chunk is found via vector search, providing
  richer context to the LLM.
- **Searching (`perform_semantic_search`):** Takes a user query, embeds it,
  performs a similarity search against the Qdrant index to find relevant child
  chunks, retrieves their corresponding parent documents from the docstore, and
  then optionally uses a cross-encoder model to rerank the results for improved
  relevance.
- **RAG Component Initialization:** Handles the loading and initialization of the
  embedding model and the reranker model based on application settings, ensuring
  they are ready for use.
- **State Management:** Manages the status of the indexing process through a
  JSON file, allowing the frontend to monitor progress.
- **File Change Tracking:** Tracks file modification times to avoid re-indexing
  unchanged files.
"""

# 1. IMPORTS & SETUP ############################################################################################
import json
import uuid
import sqlite3
import os
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple

import fitz
from langchain_core.documents import Document

from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from .rag_pipeline import (
    initialize_embedding_model,
    initialize_reranker_model,
    rerank_retrieved_documents,
    get_torch_device
)
from .config_manager import get_config, DATA_FOLDER
from .security_utils import validate_and_resolve_path


# 2. CONFIGURATION & GLOBAL STATE ###############################################################################
logger = logging.getLogger(__name__)

STATUS_FILE = os.path.join(DATA_FOLDER, "semantic_status.json")
DOCSTORE_DB_FILE = os.path.join(DATA_FOLDER, "docstore.db")

EMBEDDING_CONFIG = get_config("embedding_model")
RERANKER_CONFIG = get_config("reranker_model")
RETRIEVAL_CONFIG = get_config("retrieval_params")
QDRANT_CONFIG = get_config("vectordb").get("qdrant", {})


EMBEDDING_DEVICE = get_torch_device(EMBEDDING_CONFIG.get("device", "auto"))
RERANKER_DEVICE = get_torch_device(RERANKER_CONFIG.get("device", "auto"))
RERANKER_ENABLED = RETRIEVAL_CONFIG.get("enable_reranker", False)

QDRANT_CLIENT: QdrantClient = None

# 3. RAG PIPELINE INITIALIZATION ################################################################################
EMBEDDINGS = initialize_embedding_model(EMBEDDING_CONFIG["model_name"], EMBEDDING_DEVICE)
RERANKER_COMPONENTS = None
if RERANKER_ENABLED:
    RERANKER_COMPONENTS = initialize_reranker_model(RERANKER_CONFIG["model_name"], RERANKER_DEVICE)
else:
    logger.info("Reranker is disabled by default in config. Skipping model load.")

if EMBEDDINGS:
    EMBEDDING_DIMENSION = len(EMBEDDINGS.embed_query("test query"))
    logger.info(f"Detected embedding dimension: {EMBEDDING_DIMENSION}")
else:
    EMBEDDING_DIMENSION = 0
    logger.warning("Embedding dimension set to 0 as no embedding model is loaded.")

# 4. CLIENT & DOCSTORE MANAGEMENT ###############################################################################
def get_qdrant_client() -> QdrantClient:
    """Initializes and returns the Qdrant client using a singleton pattern."""
    global QDRANT_CLIENT
    if QDRANT_CLIENT is None:
        logger.info("Initializing Qdrant client for the first time...")
        qdrant_cfg = get_config("vectordb").get("qdrant", {})
        try:
            if qdrant_cfg.get("mode") == "local_on_disk":
                path = os.path.join(DATA_FOLDER, qdrant_cfg["storage_path"])
                QDRANT_CLIENT = QdrantClient(path=path)
                logger.info(f"Qdrant client initialized in local mode. Storage: '{path}'")
            else:
                host = qdrant_cfg.get("host", "localhost")
                port = qdrant_cfg.get("port", 6333)
                QDRANT_CLIENT = QdrantClient(host=host, port=port)
                logger.info(f"Qdrant client initialized in server mode. Host: {host}")
        except Exception:
            logger.critical("Could not connect to Qdrant.", exc_info=True)
            raise
    return QDRANT_CLIENT

def init_docstore_db():
    """Initializes the SQLite database and table for the document store."""
    try:
        with sqlite3.connect(DOCSTORE_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parent_documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_path TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_state (
                    file_path TEXT PRIMARY KEY,
                    modified_time REAL NOT NULL
                )
            ''')
            conn.commit()
        logger.debug("Docstore database initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize the docstore SQLite database.")


def _save_docstore_to_db(parent_docs_to_save: List[Tuple[str, str, str]]):
    """Saves the provided parent documents to the SQLite database, replacing old ones."""
    try:
        with sqlite3.connect(DOCSTORE_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM parent_documents")
            if parent_docs_to_save:
                cursor.executemany("INSERT INTO parent_documents (id, content, source_path) VALUES (?, ?, ?)", parent_docs_to_save)
            conn.commit()
        logger.info(f"Successfully saved {len(parent_docs_to_save)} parent documents to the persistent cache.")
    except Exception:
        logger.exception("Failed to save documents to the docstore database.")

def _save_file_state(file_states: Dict[str, float]):
    """Saves file paths and their modification times to the SQLite database."""
    try:
        with sqlite3.connect(DOCSTORE_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM file_state")
            if file_states:
                cursor.executemany(
                    "INSERT INTO file_state (file_path, modified_time) VALUES (?, ?)",
                    [(path, mtime) for path, mtime in file_states.items()]
                )
            conn.commit()
        logger.info(f"Saved {len(file_states)} file states to the database.")
    except Exception:
        logger.exception("Failed to save file states to the database.")

def _get_file_state() -> Dict[str, float]:
    """Retrieves file paths and their modification times from the SQLite database."""
    try:
        with sqlite3.connect(DOCSTORE_DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, modified_time FROM file_state")
            return {row[0]: row[1] for row in cursor.fetchall()}
    except Exception:
        logger.exception("Failed to retrieve file states from the database.")
        return {}

def _get_parent_chunks_from_db(parent_ids: List[str]) -> Dict[str, str]:
    """
    Retrieves specific parent document chunks from the SQLite DB by their IDs.
    """
    if not parent_ids:
        return {}

    try:
        with sqlite3.connect(DOCSTORE_DB_FILE) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in parent_ids)
            query = f"SELECT id, content FROM parent_documents WHERE id IN ({placeholders})"
            cursor.execute(query, parent_ids)
            return {row[0]: row[1] for row in cursor.fetchall()}
    except Exception:
        logger.exception("Failed to retrieve documents from the docstore database.")
        return {}


# 5. CORE INDEXING LOGIC ########################################################################################
def is_index_ready():
    """Checks if the Qdrant vector store and the docstore DB are populated."""
    try:
        if not EMBEDDINGS:
            return False

        qdrant_cfg = get_config("vectordb").get("qdrant", {})
        storage_path = Path(os.path.join(DATA_FOLDER, qdrant_cfg.get("storage_path", "qds")))

        docstore_has_content = False
        if Path(DOCSTORE_DB_FILE).exists():
            with sqlite3.connect(DOCSTORE_DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM parent_documents LIMIT 1")
                if cursor.fetchone():
                    docstore_has_content = True

        return storage_path.exists() and any(storage_path.iterdir()) and docstore_has_content
    except Exception:
        return False

def _update_status(status: str, progress: int, total: int, current_file: str = ""):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({"status": status, "progress": progress, "total": total, "current_file": current_file}, f)
    except Exception:
        logger.warning("Failed to update semantic status file.", exc_info=True)

def run_indexing_task(search_path: str, excluded_folders: Set[str], file_extensions: List[str], include_dot_folders: bool):
    """Scans, splits, and indexes documents into a persistent Qdrant vector store."""
    try:
        validated_path = validate_and_resolve_path(search_path)
        logger.info(f"Starting semantic indexing task for validated path: {validated_path}")
    except ValueError as e:
        error_message = f"Invalid search path provided for indexing. {e}"
        _update_status("error", 0, 0, error_message)
        logger.error(error_message)
        return

    client = get_qdrant_client()
    if EMBEDDINGS is None or client is None:
        err_msg = "Cannot index: Embedding model is not configured or Qdrant client failed."
        _update_status("error", 0, 0, err_msg)
        logger.error(err_msg)
        return

    _update_status("running", 0, 0, "Starting scan...")
    collection_name = QDRANT_CONFIG["collection_name"]
    try:
        logger.info("Phase 1: Discovering files...")
        root_path = Path(validated_path)
        allowed_extensions = {ext.lower() for ext in file_extensions}

        previous_file_states = _get_file_state()

        all_filepaths = [
            p for p in root_path.rglob('*')
            if p.is_file() and p.suffix.lower() in allowed_extensions and
            not any(excluded in p.parts for excluded in excluded_folders) and
            (include_dot_folders or not any(part.startswith('.') for part in p.parts))
        ]

        files_to_process = []
        for file_path in all_filepaths:
            current_mtime = file_path.stat().st_mtime
            previous_mtime = previous_file_states.get(str(file_path))
            if previous_mtime is None or current_mtime != previous_mtime:
                files_to_process.append(file_path)

        files_to_index = len(files_to_process)
        if files_to_index == 0:
            logger.info("No new or modified files to index.")
            _update_status("complete", len(all_filepaths), len(all_filepaths), "No changes detected.")
            return

        logger.info(f"Phase 2: Loading and processing {files_to_index} new/modified documents...")
        _update_status("running", 0, files_to_index, f"Found {files_to_index} changed files, starting processing...")
        docs = []
        processed_file_states = {}

        for i, file_path in enumerate(files_to_process):
            _update_status("running", i + 1, files_to_index, f"Loading: {file_path.name}")
            try:
                loaded_content = False
                if file_path.suffix.lower() == ".pdf":
                    with fitz.open(file_path) as pdf_doc:
                        full_text = "".join(page.get_text() for page in pdf_doc)

                    if full_text.strip():
                        metadata = {'source': str(file_path)}
                        doc = Document(page_content=full_text, metadata=metadata)
                        docs.append(doc)
                        loaded_content = True
                    else:
                        logger.info(
                            f"Skipping PDF file '{file_path.name}' as it contains no extractable digital text. "
                            "It might be a scanned or image-only document."
                        )

                else:
                    # Block Version: 1.2.0
                    loader_kwargs = {"strategy": "fast", "languages": ["eng"]}
                    loader = UnstructuredLoader(str(file_path), mode="elements", loader_kwargs=loader_kwargs)
                    loaded_docs = loader.load()
                    if loaded_docs:
                        for doc in loaded_docs:
                            doc.metadata['source'] = str(file_path)
                        docs.extend(loaded_docs)
                        loaded_content = True

                if loaded_content:
                    processed_file_states[str(file_path)] = file_path.stat().st_mtime

            except Exception as e:
                logger.warning(f"Could not load file '{file_path}'. Error: {e}", exc_info=True)

        if not docs:
            _update_status("complete", files_to_index, files_to_index, "No processable documents found.")
            logger.warning("File discovery found files, but none could be processed by the loader.")
            return

        logger.info("Phase 3: Splitting documents into parent/child chunks...")
        _update_status("running", files_to_index, files_to_index, "Splitting documents...")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        parent_docs = parent_splitter.split_documents(docs)
        child_docs_to_index = []
        parent_docs_to_save = []

        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            source_path = parent_doc.metadata.get('source', '')
            parent_docs_to_save.append((parent_id, parent_doc.page_content, source_path))

            child_chunks = child_splitter.split_documents([parent_doc])
            for chunk in child_chunks:
                chunk.metadata = parent_doc.metadata.copy()
                chunk.metadata["parent_id"] = parent_id
                child_docs_to_index.append(chunk)
        logger.info(f"Created {len(parent_docs)} parent and {len(child_docs_to_index)} child chunks.")

        if not child_docs_to_index:
            _update_status("complete", files_to_index, files_to_index, "Could not split documents into chunks.")
            logger.warning("Documents were loaded but could not be split into indexable chunks.")
            return

        logger.info(f"Phase 4: Indexing {len(child_docs_to_index)} child chunks into Qdrant collection '{collection_name}'.")
        _update_status("running", files_to_index, files_to_index, "Setting up vector collection...")

        collection_exists = any(
            coll.name == collection_name for coll in client.get_collections().collections
        )
        if not collection_exists:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE)
            )

        _update_status("running", files_to_index, files_to_index, f"Embedding {len(child_docs_to_index)} chunks...")
        contents_to_embed = [doc.page_content for doc in child_docs_to_index]
        vectors = EMBEDDINGS.embed_documents(contents_to_embed)

        _update_status("running", files_to_index, files_to_index, "Uploading to vector database...")
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=doc.metadata)
                for doc, vector in zip(child_docs_to_index, vectors)
            ],
            wait=True
        )

        logger.info("Phase 5: Persisting document store and file states to disk...")
        _save_docstore_to_db(parent_docs_to_save)
        master_file_state = _get_file_state()
        master_file_state.update(processed_file_states)
        _save_file_state(master_file_state)

        _update_status("complete", files_to_index, files_to_index, "Indexing complete.")
        logger.info("Qdrant indexing task finished successfully.")

    except Exception:
        error_message = "An unexpected error occurred during Qdrant indexing."
        _update_status("error", 0, 0, error_message)
        logger.exception(error_message)

# 6. CORE SEARCH LOGIC ##########################################################################################
def perform_semantic_search(
    query: str,
    k_initial: int,
    vector_score_threshold: float,
    vector_top_n: int,
    enable_reranker: bool,
    rerank_top_n: int,
    score_threshold: float
) -> List[Dict[str, Any]]:
    """Performs vector search and optional reranking to find relevant documents."""
    client = get_qdrant_client()
    if not all([EMBEDDINGS, client]):
        raise RuntimeError("Cannot search: Embedding model is not configured or Qdrant client is not available.")

    effective_reranker = enable_reranker and RERANKER_COMPONENTS is not None
    if enable_reranker and RERANKER_COMPONENTS is None:
        logger.warning("Reranking was requested but model is not loaded. Falling back to vector search only.")

    logger.info(f"Performing semantic search for query: '{query[:100]}...'")
    collection_name = QDRANT_CONFIG["collection_name"]
    query_vector = EMBEDDINGS.embed_query(query)

    logger.debug(f"Searching Qdrant collection '{collection_name}' with k={k_initial} and threshold={vector_score_threshold}.")
    child_chunk_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k_initial,
        score_threshold=vector_score_threshold,
        with_payload=True
    )
    if not child_chunk_results:
        logger.info("Vector search returned no results.")
        return []

    logger.debug(f"Vector search returned {len(child_chunk_results)} initial child chunks. Retrieving parent documents.")

    parent_ids_to_fetch = list(set(
        point.payload.get("parent_id") for point in child_chunk_results if point.payload.get("parent_id")
    ))
    retrieved_parent_chunks = _get_parent_chunks_from_db(parent_ids_to_fetch)

    if not retrieved_parent_chunks:
        logger.warning("Vector search found chunks, but could not retrieve any parent documents from the docstore.")
        return []

    parent_docs_to_process = {}
    for point in child_chunk_results:
        parent_id = point.payload.get("parent_id")
        if parent_id and parent_id in retrieved_parent_chunks:
            if parent_id not in parent_docs_to_process or point.score > parent_docs_to_process[parent_id]["vector_score"]:
                 parent_docs_to_process[parent_id] = {
                    "path": point.payload.get("source", "Unknown"),
                    "chunk": retrieved_parent_chunks[parent_id],
                    "vector_score": float(point.score)
                }

    docs_to_process = list(parent_docs_to_process.values())
    docs_to_process.sort(key=lambda x: x["vector_score"], reverse=True)
    logger.info(f"Retrieved {len(docs_to_process)} unique parent documents for processing.")

    if effective_reranker:
        logger.info("Applying reranker to the retrieved documents.")
        reranked_results = rerank_retrieved_documents(query, docs_to_process, RERANKER_COMPONENTS)
        final_results = [doc for doc in reranked_results if doc["rerank_score"] >= score_threshold]
        logger.info(f"Reranking complete. Returning top {min(len(final_results), rerank_top_n)} results.")
        return final_results[:rerank_top_n]
    else:
        logger.info(f"Reranker not used. Returning top {min(len(docs_to_process), vector_top_n)} results from vector search.")
        return docs_to_process[:vector_top_n]

# 7. MODULE INITIALIZATION CALLS ################################################################################
init_docstore_db()
