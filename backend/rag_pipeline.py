# File Version: 1.3.0
# /backend/rag_pipeline.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
Manages the core components of the Retrieval-Augmented Generation (RAG) pipeline.

This module is responsible for initializing and managing the machine learning
models required for advanced semantic search features, specifically the embedding
model and the reranker (cross-encoder) model. It abstracts away the complexities
of loading these models onto the appropriate hardware (CPU, CUDA, MPS).

Key functionalities include:
- `initialize_embedding_model`: Loads a sentence-transformer model from
  HuggingFace, which is used to convert text documents and queries into
  numerical vectors for semantic comparison.
- `initialize_reranker_model`: Loads a cross-encoder model and its tokenizer,
  used to perform a more computationally intensive but accurate relevance
  scoring of the top documents retrieved by the initial vector search.
- `rerank_retrieved_documents`: Takes a query and a list of documents and uses
  the loaded cross-encoder to re-order them based on semantic relevance.
- `get_torch_device`: A utility function that intelligently detects and selects
  the best available PyTorch device (CUDA, MPS, or CPU) for running the models,
  while respecting user configuration.
"""

# 1. IMPORTS ####################################################################################################
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import snapshot_download

# Block Version: 1.3.0
from .config_manager import MODELS_FOLDER

# 2. SETUP & UTILITIES ##########################################################################################
logger = logging.getLogger(__name__)

IS_OFFLINE = os.environ.get("HF_HUB_OFFLINE") == "1"


def get_torch_device(configured_device: str = "auto") -> str:
    """Determines the optimal torch device (CUDA, MPS, CPU) for model loading."""
    configured_device = configured_device.lower().strip()
    logger.debug(f"Requesting device: '{configured_device}'")

    if configured_device == "cuda" and torch.cuda.is_available():
        logger.info("CUDA device selected by user and is available.")
        return "cuda"
    if configured_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS device selected by user and is available.")
        return "mps"
    if configured_device == "cpu":
        logger.info("CPU device selected by user.")
        return "cpu"

    if torch.cuda.is_available():
        logger.info("Auto-detected and selected CUDA device.")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-detected and selected MPS device.")
        return "mps"

    logger.info("No GPU detected, falling back to CPU.")
    return "cpu"

# Block Version: 1.3.0
def get_and_prepare_model_path(model_name: str, model_type: str) -> str:
    """
    Ensures a model is available locally, downloading if necessary.
    Returns the absolute local path to the model for offline-safe loading.
    """
    if not model_name or not model_name.strip():
        raise ValueError(f"{model_type} name cannot be empty.")

    safe_model_name = model_name.replace("/", "__")
    local_model_path = os.path.join(MODELS_FOLDER, safe_model_name)

    if os.path.isdir(local_model_path):
        logger.info(f"Found existing local model for '{model_name}' at '{local_model_path}'.")
        return local_model_path

    if IS_OFFLINE:
        logger.error(f"Offline mode: Model '{model_name}' not found at '{local_model_path}'.")
        raise RuntimeError(
            f"Offline mode is enabled, but the required {model_type} '{model_name}' was not "
            "found locally. Please run the application with an internet connection once to "
            "download the necessary models."
        )

    logger.info(f"Model '{model_name}' not found locally. Downloading to '{local_model_path}'...")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Successfully downloaded {model_type} '{model_name}'.")
        return local_model_path
    except Exception as e:
        logger.critical(
            f"Failed to download {model_type} '{model_name}'. "
            f"Please check your internet connection and the model name. Error: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Could not acquire required model: {model_name}") from e


# 3. INITIALIZATION FUNCTIONS ###################################################################################
def initialize_embedding_model(model_name: str, device: str) -> Optional[HuggingFaceEmbeddings]:
    """Initializes and returns the sentence-transformer model for embeddings."""
    if not model_name:
        logger.warning("No embedding model name provided in config. Semantic-based features will be disabled.")
        return None

    try:
        local_path = get_and_prepare_model_path(model_name, "Embedding Model")

        logger.info(f"Loading embedding model from local path: {local_path} on device: {device}")
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name=local_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        logger.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.critical(f"Failed to load embedding model '{model_name}'. Semantic features will be unavailable.", exc_info=True)
        raise e

def initialize_reranker_model(model_name: str, device: str) -> Optional[Tuple]:
    """Initializes and returns the cross-encoder model and tokenizer for reranking."""
    if not model_name:
        logger.warning("No reranker model name provided in config. Reranker will be disabled.")
        return None

    try:
        local_path = get_and_prepare_model_path(model_name, "Reranker Model")

        logger.info(f"Loading reranker model from local path: {local_path} on device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path).to(device)
        model.eval()
        logger.info("Reranker model loaded successfully.")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Failed to load reranker model '{model_name}'. Reranking will be unavailable.", exc_info=True)
        return None

# 4. RERANKING LOGIC ############################################################################################
def rerank_retrieved_documents(query: str, documents: List[Dict[str, Any]], reranker_components: tuple) -> List[Dict[str, Any]]:
    """
    Reranks documents based on relevance to a query using a cross-encoder.
    """
    if not reranker_components:
        logger.warning("Rerank called but no reranker components available. Returning original document order.")
        return documents

    tokenizer, model, device = reranker_components
    logger.debug(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

    pairs = [[query, doc["chunk"]] for doc in documents]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        scores_sigmoid = torch.sigmoid(scores).cpu().numpy()

    for doc, score in zip(documents, scores_sigmoid):
        doc["rerank_score"] = float(score)

    documents.sort(key=lambda x: x["rerank_score"], reverse=True)

    logger.debug("Reranking complete.")
    return documents
