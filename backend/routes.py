# File Version: 1.2.0
# /backend/routes.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
Defines the API layer for the backend using FastAPI.

This module sets up all the HTTP and WebSocket endpoints that the frontend
application interacts with. It uses a FastAPI `APIRouter` to organize the
routes and Pydantic models to validate incoming request data.

The routes are organized by feature:
- **Classifier:** Endpoints for starting a classification task, checking its
  status, retrieving results, and performing file operations (move/copy) based
  on classification tags.
- **Trainer:** Endpoints to trigger the retraining of the document classifier model.
- **Semantic Search:** Endpoints for building the semantic index, checking
  indexing status, and performing vector-based searches.
- **AI Search:** A high-level endpoint that orchestrates the entire AI-powered
  search pipeline, from intent detection to summarization.
- **General & Config:** Utility endpoints for opening files/folders on the
  server's host system, and for managing application settings (get, set, reset).
- **WebSocket:** A dedicated endpoint for classic (non-AI) searches, which
  streams file results and progress updates in real-time to the client.

This module acts as the primary interface, delegating the actual business logic
to other specialized modules like `app_logic`, `ai_search`, and `semantic_search`.
"""

# 1. IMPORTS & SETUP ############################################################################################
import os
import shutil
import sys
import subprocess
import uuid
import asyncio
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import APIRouter, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from backend import app_logic
from .classifier_trainer import run_training_task
from .config_manager import get_config, set_config, reset_to_defaults
from .semantic_search import (
    run_indexing_task,
    perform_semantic_search,
    is_index_ready,
    STATUS_FILE as SEMANTIC_STATUS_FILE,
    RETRIEVAL_CONFIG
)
from .ai_search import run_ai_search
from .file_engine import FileSearchEngine
from .ai_search import summarize_results_with_llm

logger = logging.getLogger(__name__)

router = APIRouter()

# 2. PYDANTIC MODELS ############################################################################################
class SearchType(str, Enum):
    FILE_CONTENT = "file_content"
    FILE_NAME = "file_name"
    FOLDER_NAME = "folder_name"
    FILE_CATEGORY = "file_category"

class SearchRequest(BaseModel):
    search_path: str
    keywords: List[str]
    search_type: SearchType
    excluded_folders: Optional[List[str]] = None
    file_extensions: Optional[List[str]] = None
    include_dot_folders: bool = False
    case_sensitive: bool = False
    use_regex: bool = False
    file_category: Optional[str] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None

class OpenRequest(BaseModel):
    path: str
    action: str

class ConfigRequest(BaseModel):
    excluded_folders: List[str]
    file_extensions: List[str]
    enable_reranker: bool
    llm_config: Optional[Dict[str, Any]] = None
    embedding_model: Optional[Dict[str, Any]] = None
    reranker_model: Optional[Dict[str, Any]] = None

class ClassifyRequest(BaseModel):
    search_path: str

class SemanticSearchRequest(BaseModel):
    query: str
    k_fetch_initial: int = RETRIEVAL_CONFIG['k_fetch_initial']
    vector_score_threshold: float = RETRIEVAL_CONFIG.get('vector_score_threshold', 0.3)
    vector_top_n: int = RETRIEVAL_CONFIG.get('vector_top_n', 10)
    enable_reranker: bool = RETRIEVAL_CONFIG.get('enable_reranker', False)
    rerank_top_n: int = RETRIEVAL_CONFIG['rerank_top_n']
    rerank_score_threshold: float = RETRIEVAL_CONFIG['rerank_score_threshold']

class IndexRequest(BaseModel):
    search_path: str
    include_dot_folders: bool = False

class FileAction(str, Enum):
    COPY = "copy"
    MOVE = "move"

class FileOperationRequest(BaseModel):
    tag: str
    action: FileAction
    destination_path: str

class TrainerRequest(BaseModel):
    data_path: str
    test_size: float
    n_estimators: int

class AISearchRequest(BaseModel):
    query: str
    temperature: float = app_logic.AI_SEARCH_PARAMS.get("default_temperature", 0.2)
    max_tokens: int = app_logic.AI_SEARCH_PARAMS.get("default_max_tokens", 4096)
    k_fetch_initial: Optional[int] = None
    vector_score_threshold: Optional[float] = None
    vector_top_n: Optional[int] = None
    enable_reranker: Optional[bool] = None
    rerank_top_n: Optional[int] = None
    rerank_score_threshold: Optional[float] = None

class DeleteTagRequest(BaseModel):
    tag: str

class OrganizeAllRequest(BaseModel):
    base_destination_path: str

class SummarizeRequest(BaseModel):
    query: str
    search_results: List[Dict[str, Any]]
    temperature: float = 0.2
    max_tokens: int = 4096


# 3. CLASSIFIER API ENDPOINTS ###################################################################################
@router.post("/api/classifier/start")
async def start_classification(req: ClassifyRequest, background_tasks: BackgroundTasks):
    if app_logic.classifier_status['status'] == 'running':
        raise HTTPException(status_code=409, detail="A classification task is already in progress.")
    if not os.path.isdir(req.search_path):
        raise HTTPException(status_code=404, detail="The specified directory does not exist.")
    if app_logic.CLASSIFIER_MODEL is None:
        raise HTTPException(status_code=503, detail="Classifier model is not loaded. Cannot start task.")
    background_tasks.add_task(app_logic.run_classification_task, req.search_path)
    return JSONResponse(content={"status": "success", "message": "Classification task started."})

@router.get("/api/classifier/status")
async def get_classification_status():
    return JSONResponse(content=app_logic.classifier_status)

@router.get("/api/classifier/results")
async def get_classification_results():
    try:
        conn = sqlite3.connect(app_logic.DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT tag, path FROM classified_files ORDER BY tag")
        results = cursor.fetchall()
        conn.close()
        grouped_results = {}
        for tag, path in results:
            grouped_results.setdefault(tag, []).append(path)
        return JSONResponse(content=grouped_results)
    except Exception:
        logger.exception("Failed to retrieve classification results from database.")
        raise HTTPException(status_code=500, detail="Could not retrieve classification results.")

@router.post("/api/classifier/file-operation")
async def handle_file_operation(req: FileOperationRequest):
    dest_dir = Path(req.destination_path)
    if not dest_dir.is_dir():
        raise HTTPException(status_code=404, detail="Destination directory does not exist.")

    conn = sqlite3.connect(app_logic.DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM classified_files WHERE tag = ?", (req.tag,))
    source_paths = [Path(row[0]) for row in cursor.fetchall()]
    conn.close()

    if not source_paths:
        raise HTTPException(status_code=404, detail=f"No files found for tag '{req.tag}'.")

    logger.info(f"Performing '{req.action}' operation for {len(source_paths)} files with tag '{req.tag}' to '{dest_dir}'.")
    success_count, failures = 0, []
    for src_path in source_paths:
        if src_path.exists():
            try:
                if req.action == FileAction.COPY:
                    shutil.copy2(src_path, dest_dir / src_path.name)
                elif req.action == FileAction.MOVE:
                    shutil.move(str(src_path), str(dest_dir / src_path.name))
                success_count += 1
            except Exception as e:
                failure_msg = f"Failed to process {src_path.name}: {e}"
                failures.append(failure_msg)
                logger.error(f"File operation error for tag '{req.tag}': {failure_msg}")
        else:
            failures.append(f"Source file not found: {src_path}")

    if req.action == FileAction.MOVE and success_count > 0:
        logger.info(f"Moving files completed. Deleting tag '{req.tag}' from classifier database.")
        with sqlite3.connect(app_logic.DB_FILE) as conn:
            conn.cursor().execute("DELETE FROM classified_files WHERE tag = ?", (req.tag,))
            conn.commit()

    op_word = "Copied" if req.action == FileAction.COPY else "Moved"
    message = f"{op_word} {success_count} of {len(source_paths)} files to '{dest_dir}'."
    if failures:
        message += f" Failures: {len(failures)}. See server logs for details."
    return JSONResponse(content={"status": "success", "message": message})

@router.post("/api/classifier/organize-all")
async def handle_organize_all(req: OrganizeAllRequest):
    base_dest_dir = Path(req.base_destination_path)
    if not base_dest_dir.is_dir():
        raise HTTPException(status_code=404, detail="Base destination directory does not exist.")

    try:
        conn = sqlite3.connect(app_logic.DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT path, tag FROM classified_files")
        all_files = cursor.fetchall()
        conn.close()

        if not all_files:
            raise HTTPException(status_code=404, detail="No classified files found to organize.")

        logger.info(f"Starting 'Auto Organize' operation for {len(all_files)} files to base path: {base_dest_dir}")
        success_count, failures = 0, []

        for file_path_str, tag in all_files:
            src_path = Path(file_path_str)
            target_dir = base_dest_dir / tag
            target_dir.mkdir(exist_ok=True)

            if src_path.exists():
                try:
                    shutil.move(str(src_path), str(target_dir / src_path.name))
                    success_count += 1
                except Exception as e:
                    failures.append(f"Failed to move {src_path.name}: {e}")
            else:
                failures.append(f"Source file not found: {src_path}")

        with sqlite3.connect(app_logic.DB_FILE) as conn:
            conn.cursor().execute("DELETE FROM classified_files")
            conn.commit()

        message = f"Successfully organized {success_count} of {len(all_files)} files into categorized subfolders under '{base_dest_dir}'."
        if failures:
            message += f" Failures: {len(failures)}. See server logs for details."
            logger.error("Auto Organize failures:\n" + "\n".join(failures))

        return JSONResponse(content={"status": "success", "message": message})
    except Exception as e:
        logger.exception("A critical error occurred during the 'Auto Organize' operation.")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/classifier/delete-tag")
async def delete_classification_tag(req: DeleteTagRequest):
    try:
        with sqlite3.connect(app_logic.DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM classified_files WHERE tag = ?", (req.tag,))
            if not cursor.fetchone():
                 raise HTTPException(status_code=404, detail=f"No classification results found for tag '{req.tag}'.")

            logger.info(f"Deleting all classification entries for tag: '{req.tag}'")
            cursor.execute("DELETE FROM classified_files WHERE tag = ?", (req.tag,))
            conn.commit()
        return JSONResponse(content={"status": "success", "message": f"Classification results for tag '{req.tag}' have been deleted."})
    except sqlite3.Error:
        logger.exception(f"Database error while deleting tag '{req.tag}'.")
        raise HTTPException(status_code=500, detail="A database error occurred.")
    except Exception:
        logger.exception(f"Unexpected error while deleting tag '{req.tag}'.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# 4. TRAINER API ENDPOINTS ######################################################################################
@router.post("/api/trainer/start")
async def start_training(req: TrainerRequest, background_tasks: BackgroundTasks):
    if app_logic.trainer_status['status'] == 'running':
        raise HTTPException(status_code=409, detail="A training task is already in progress.")
    if not os.path.isdir(req.data_path):
        raise HTTPException(status_code=404, detail="Training data directory does not exist.")
    app_logic.trainer_status = {"status": "starting", "log": [], "accuracy": None}
    logger.info(f"Starting model training with data from: {req.data_path}")
    background_tasks.add_task(run_training_task, app_logic.trainer_status, req.data_path, req.test_size, req.n_estimators, app_logic.DEFAULT_EXCLUDED_FOLDERS)

    return JSONResponse(content={"status": "success", "message": "Model training started."})

@router.get("/api/trainer/status")
async def get_training_status():
    return JSONResponse(content=app_logic.trainer_status)

# 5. SEMANTIC SEARCH API ENDPOINTS ##############################################################################
@router.post("/api/semantic/start-indexing")
async def start_indexing(req: IndexRequest, background_tasks: BackgroundTasks):
    try:
        if Path(SEMANTIC_STATUS_FILE).exists():
            with open(SEMANTIC_STATUS_FILE, "r") as f:
                if json.load(f).get("status") == "running":
                    raise HTTPException(status_code=409, detail="Indexing is already in progress.")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if not os.path.isdir(req.search_path):
        raise HTTPException(status_code=404, detail="The specified directory does not exist.")

    logger.info(f"Starting semantic indexing for path: {req.search_path}")
    background_tasks.add_task(run_indexing_task, req.search_path, app_logic.DEFAULT_EXCLUDED_FOLDERS, app_logic.DEFAULT_FILE_EXTENSIONS, req.include_dot_folders)
    return JSONResponse(content={"status": "success", "message": "Semantic indexing started."})

@router.get("/api/semantic/status")
async def get_indexing_status():
    try:
        with open(SEMANTIC_STATUS_FILE, "r") as f:
            status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"status": "idle", "progress": 0, "total": 0, "current_file": ""}

    status["index_ready"] = is_index_ready()
    return JSONResponse(content=status)

@router.post("/api/semantic/search")
async def semantic_search(req: SemanticSearchRequest):
    if not is_index_ready():
        raise HTTPException(status_code=503, detail="Semantic index is not ready. Please build it first.")

    try:
        results = perform_semantic_search(
            query=req.query, k_initial=req.k_fetch_initial, vector_score_threshold=req.vector_score_threshold,
            vector_top_n=req.vector_top_n, enable_reranker=req.enable_reranker, rerank_top_n=req.rerank_top_n,
            score_threshold=req.rerank_score_threshold
        )
        return JSONResponse(content=results)
    except RuntimeError as e:
        if "reranker model is not loaded" in str(e):
            raise HTTPException(status_code=409, detail="Reranker is not available. Please enable it in Settings and restart.")
        logger.error(f"Runtime error during semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("An unexpected error occurred during semantic search.")
        raise HTTPException(status_code=500, detail="An error occurred during semantic search.")

# 6. AI SEARCH API ENDPOINT #####################################################################################
@router.post("/api/ai/search")
async def ai_search_endpoint(req: AISearchRequest):
    llm_config = get_config("llm_config")
    if not llm_config or not llm_config.get("api_key") or "YOUR_LLM_API_KEY_HERE" in llm_config.get("api_key"):
        error_data = {
            "summary": "### AI Search is Not Configured\n\nTo use this feature, you must provide a valid API Key for your Large Language Model in the **Settings** tab.",
            "relevant_files": []
        }
        raise HTTPException(
            status_code=409,
            detail={"status": "error", "data": error_data}
        )

    try:
        ai_response = await run_ai_search(
            query=req.query,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            k_fetch_initial=req.k_fetch_initial,
            vector_score_threshold=req.vector_score_threshold,
            vector_top_n=req.vector_top_n,
            enable_reranker=req.enable_reranker,
            rerank_top_n=req.rerank_top_n,
            rerank_score_threshold=req.rerank_score_threshold
        )
        return JSONResponse(content={"status": "success", "data": ai_response})
    except Exception as e:
        logger.exception("A critical error occurred in the AI search endpoint.")
        error_data = {
            "summary": f"### A critical server error occurred:\n\n```\n{str(e)}\n```",
            "relevant_files": []
        }
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "data": error_data}
        )

@router.post("/api/ai/summarize-results")
async def ai_summarize_endpoint(req: SummarizeRequest):
    """
    Takes existing search results and a new query, and uses an LLM to generate a summary.
    This bypasses the search and routing steps for a more direct Q&A experience.
    """
    llm_config = get_config("llm_config")
    if not llm_config or not llm_config.get("api_key") or "YOUR_LLM_API_KEY_HERE" in llm_config.get("api_key"):
        error_data = {
            "summary": "### AI Search is Not Configured\n\nTo use this feature, you must provide a valid API Key for your Large Language Model in the **Settings** tab.",
            "relevant_files": []
        }
        raise HTTPException(status_code=409, detail={"status": "error", "data": error_data})

    try:
        strategy = "semantic_content"

        summary_response = summarize_results_with_llm(
            user_query=req.query,
            search_results=req.search_results,
            strategy=strategy,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        return JSONResponse(content={"status": "success", "data": summary_response})
    except Exception as e:
        logger.exception("A critical error occurred in the AI summarization endpoint.")
        error_data = {
            "summary": f"### A critical server error occurred:\n\n```\n{str(e)}\n```",
            "relevant_files": []
        }
        raise HTTPException(status_code=500, detail={"status": "error", "data": error_data})

# 7. GENERAL & CONFIG API ENDPOINTS #############################################################################
@router.post("/api/open")
async def open_path(req: OpenRequest):
    path_to_open = os.path.dirname(req.path) if req.action == 'folder' else req.path
    if not os.path.exists(path_to_open):
        raise HTTPException(status_code=404, detail="Path not found")
    try:
        if sys.platform == "win32":
            os.startfile(path_to_open)
        elif sys.platform == "darwin":
            subprocess.run(["open", path_to_open], check=True)
        else:
            subprocess.run(["xdg-open", path_to_open], check=True)
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Failed to open path '{path_to_open}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to open path: {e}")

@router.post("/api/config")
async def save_config(config_request: ConfigRequest):
    try:
        set_config("excluded_folders", config_request.excluded_folders)
        set_config("file_extensions", config_request.file_extensions)

        retrieval_params = get_config("retrieval_params")
        retrieval_params["enable_reranker"] = config_request.enable_reranker
        set_config("retrieval_params", retrieval_params)

        if config_request.llm_config:
            set_config("llm_config", config_request.llm_config)

        if config_request.embedding_model:
            emb_config = get_config("embedding_model")
            emb_config.update(config_request.embedding_model)
            set_config("embedding_model", emb_config)

        if config_request.reranker_model:
            rer_config = get_config("reranker_model")
            rer_config.update(config_request.reranker_model)
            set_config("reranker_model", rer_config)

        app_logic.DEFAULT_EXCLUDED_FOLDERS = set(config_request.excluded_folders)
        app_logic.DEFAULT_FILE_EXTENSIONS = config_request.file_extensions
        RETRIEVAL_CONFIG['enable_reranker'] = config_request.enable_reranker

        logger.info("Configuration saved successfully.")
        return JSONResponse(content={"status": "success", "message": "Configuration saved."})
    except Exception as e:
        logger.exception("Failed to save configuration.")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {e}")

@router.post("/api/config/reset")
async def reset_config():
    try:
        reset_to_defaults()
        app_logic.DEFAULT_EXCLUDED_FOLDERS = set(get_config("excluded_folders"))
        app_logic.DEFAULT_FILE_EXTENSIONS = get_config("file_extensions")
        RETRIEVAL_CONFIG.update(get_config("retrieval_params"))

        logger.info("Configuration has been reset to defaults.")
        return JSONResponse(content={"status": "success", "message": "Configuration reset to defaults."})
    except Exception as e:
        logger.exception("Failed to reset configuration.")
        raise HTTPException(status_code=500, detail=f"Failed to reset configuration: {e}")

# 8. WEBSOCKET ENDPOINT #########################################################################################
@router.websocket("/ws/search")
async def websocket_search_endpoint(websocket: WebSocket):
    await websocket.accept()
    search_task = None
    try:
        temp_engine = FileSearchEngine(app_logic.DEFAULT_EXCLUDED_FOLDERS, app_logic.DEFAULT_FILE_EXTENSIONS)

        await websocket.send_json({
            "type": "config",
            "defaults": {
                "excluded_folders": get_config("excluded_folders"),
                "file_extensions": get_config("file_extensions"),
                "retrieval_params": get_config("retrieval_params"),
                "ai_search_params": get_config("ai_search_params"),
                "llm_config": get_config("llm_config"),
                "embedding_model": get_config("embedding_model"),
                "reranker_model": get_config("reranker_model"),
                "vectordb": get_config("vectordb"),
            },
            "file_categories": list(temp_engine.file_categories.keys())
        })
        data = await websocket.receive_json()
        if data.get("type") == "start_search":
            req = SearchRequest(**data['payload'])
            search_id = str(uuid.uuid4())
            logger.info(f"Starting classic search (ID: {search_id}) for path: {req.search_path}")
            await websocket.send_json({"type": "scan_start", "task_id": search_id})
            search_task = asyncio.create_task(app_logic.FILE_SEARCH_ENGINE.run_search(websocket, search_id, req))
            await search_task
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
        if search_task and not search_task.done():
            search_task.cancel()
    except Exception:
        logger.exception("An exception occurred in the WebSocket endpoint.")
    finally:
        if not websocket.client_state.name == 'DISCONNECTED':
            try:
                await websocket.close()
                logger.info("WebSocket connection closed gracefully from server.")
            except RuntimeError as e:
                logger.warning(f"Ignoring runtime error during WebSocket close: {e}")
