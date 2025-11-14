# File Version: 1.2.0
# /main.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
The main entry point for the Precision File Search (PFS) application.

This script is responsible for initializing and running the FastAPI web server that
powers the entire backend. Its key responsibilities include:

- **Argument Parsing:** Uses `argparse` to allow setting the logging level via
  command-line arguments (e.g., `python main.py --debug`).
- **Logging Setup:** Initializes the application-wide logging configuration
  from `backend.logging_config` at the very beginning of execution.
- **Database Initialization:** Ensures that the SQLite database for the document
  classifier feature is created with the necessary tables.
- **FastAPI Application Lifecycle:**
  - Creates the main `FastAPI` app instance with a `lifespan` context manager.
  - On startup, it triggers the loading of the application's internal knowledge base.
- **Routing and Static Files:**
  - Mounts the `static` directory and includes API endpoints from `backend.routes`.
  - Defines root endpoints to serve the primary HTML pages.
- **Server Execution:**
  - When run as the main script, it starts a `uvicorn` server in a separate thread.
  - It waits for server startup before automatically opening the user's web browser.
  - It manages graceful shutdown on KeyboardInterrupt (Ctrl+C).
"""

# 1. IMPORTS & SETUP ############################################################################################
import threading
import webbrowser
import sqlite3
import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import logging

from backend.kb_manager import load_and_index_knowledge_base
from backend.routes import router as api_router
from backend.app_logic import DB_FILE
from backend.logging_config import setup_logging, LOG_LEVELS, DEFAULT_LOG_LEVEL

load_dotenv()
logger = None 

# 2. APPLICATION INITIALIZATION HELPERS #########################################################################
def init_db():
    """Initializes the classifier's SQLite database and table if they don't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classified_files (
                path TEXT PRIMARY KEY,
                tag TEXT NOT NULL,
                modified_time REAL NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        if logger:
            logger.debug("Classifier database initialized successfully.")
    except Exception:
        if logger:
            logger.exception("Failed to initialize the classifier database.")

def initialize_ai_models():
    """
    Initializes AI models based on the current configuration.

    - On first run (with empty config), this function will do nothing and finish quickly.
    - After the user configures models and restarts, this function will handle the
      download and loading process, providing feedback in the console.
    """
    if not logger:
        print("Logger not available for AI model initialization.")
        return

    logger.info("Verifying AI model configuration...")
    try:
        from backend import semantic_search

        if semantic_search.EMBEDDINGS is None:
            if semantic_search.EMBEDDING_CONFIG.get("model_name"):
                raise RuntimeError("Embedding model is configured but failed to initialize.")
            else:
                logger.info("No embedding model configured. Skipping initialization.")

        if semantic_search.RERANKER_ENABLED:
            if semantic_search.RERANKER_COMPONENTS is None:
                if semantic_search.RERANKER_CONFIG.get("model_name"):
                    logger.warning("Reranker is enabled, but the model failed to load.")
                else:
                    logger.info("Reranker is enabled, but no model is configured. Skipping.")

        logger.info("AI model verification complete.")
    except Exception as e:
        logger.critical(f"A critical error occurred during AI model initialization: {e}", exc_info=True)

# Block Version: 1.2.0
def warm_up_unstructured():
    """
    Warms up the 'unstructured' library by running a trivial partition.
    This pays the one-time model loading cost at startup instead of during
    the first classic content search, dramatically improving search performance.
    """
    if not logger:
        print("Logger not available for unstructured warm-up.")
        return

    logger.info("Warming up text extraction engine (unstructured)...")
    try:
        from unstructured.partition.text import partition_text
        partition_text(text="warm-up")
        logger.info("Text extraction engine is ready.")
    except Exception as e:
        logger.warning(f"An error occurred during unstructured warm-up: {e}", exc_info=True)


def open_browser():
    """Opens the default web browser to the application's URL."""
    try:
        webbrowser.open("http://127.0.0.1:9090")
    except Exception:
        if logger:
            logger.error("Failed to open web browser automatically.", exc_info=True)


# 3. FASTAPI LIFESPAN MANAGER & APP CREATION ####################################################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events for FastAPI."""
    if logger:
        logger.info("FastAPI application startup...")

    load_and_index_knowledge_base()

    if hasattr(app.state, "startup_event"):
        app.state.startup_event.set()

    yield

    if logger:
        logger.info("FastAPI application shutdown.")

app = FastAPI(
    title="Precision File Search (PFS)",
    description="A local file search and classification application with an advanced RAG retrieval engine.",
    version="1.0.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router)

# 4. STATIC HTML ENDPOINTS ######################################################################################
@app.get("/", response_class=HTMLResponse)
async def get_root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)

@app.get("/documentation", response_class=HTMLResponse)
async def get_help():
    try:
        with open("static/doc.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: doc.html not found.</h1>", status_code=404)

@app.get("/license", response_class=HTMLResponse)
async def get_license():
    try:
        with open("static/license.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: license.html not found.</h1>", status_code=404)

# 5. MAIN APPLICATION EXECUTION BLOCK ###########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Precision File Search application server.")
    parser.add_argument(
        "--level", type=str.upper, default=DEFAULT_LOG_LEVEL, choices=LOG_LEVELS.keys(),
        help=f"Set the application-wide logging level. Defaults to {DEFAULT_LOG_LEVEL}."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Enable DEBUG logging level. A shorthand for --level DEBUG."
    )
    args = parser.parse_args()
    log_level_to_use = "DEBUG" if args.debug else args.level

    setup_logging(level=log_level_to_use)
    logger = logging.getLogger(__name__)

    init_db()

    logger.info("="*60)
    logger.info("Starting Precision File Search (PFS)...")
    logger.info(f"Log level set to: {log_level_to_use}")

    model_init_thread = threading.Thread(target=initialize_ai_models, daemon=True)
    model_init_thread.start()

    unstructured_warmup_thread = threading.Thread(target=warm_up_unstructured, daemon=True)
    unstructured_warmup_thread.start()


    startup_event = threading.Event()
    app.state.startup_event = startup_event
    config = uvicorn.Config(app, host="127.0.0.1", port=9090, log_config=None)
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    logger.info("Waiting for web server to start...")
    startup_event.wait()

    logger.info("Waiting for AI model verification to complete...")
    model_init_thread.join()

    logger.info("Application startup complete. Launching browser...")
    open_browser()
    logger.info("Your browser should open automatically to http://127.0.0.1:9090")
    logger.info("Press Ctrl+C in this terminal to shut down the server.")
    logger.info("="*60)

    try:
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Shutdown signal (Ctrl+C) received. Asking server to exit.")
        server.should_exit = True
        server_thread.join(timeout=5.0)
        logger.info("Server shut down gracefully.")
