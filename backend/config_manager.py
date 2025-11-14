# File Version: 1.4.0
# /backend/config_manager.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
Manages the application's configuration using a persistent SQLite database.

This module provides a centralized and robust way to handle all application
settings. On first run, it creates an SQLite database (`app_config.db`) and
populates it with a comprehensive set of default configurations defined in the
`DEFAULT_CONFIG` dictionary.

This approach offers several advantages over a simple JSON or INI file:
-   **Persistence:** Settings are saved between application restarts.
-   **Type Safety:** Storing values as JSON strings preserves data structures
    like lists and dictionaries.
-   **Centralization:** Provides a single, consistent interface for accessing
    and modifying settings throughout the application.
-   **Resilience:** Includes functions to retrieve individual settings, update
    them, or reset the entire configuration back to its default state without
    deleting the database file itself, avoiding potential file lock issues.

Key Functions:
-   `get_config(key)`: Retrieves a configuration value for a given key.
-   `set_config(key, value)`: Sets or updates a configuration value.
-   `reset_to_defaults()`: Resets all settings in the database to their
    original default values.
"""

# 1. IMPORTS ####################################################################################################
import sqlite3
import json
import os
import logging
from typing import Any

# 2. CONSTANTS & DEFAULTS #######################################################################################
logger = logging.getLogger(__name__)

def get_user_data_dir() -> str:
    """
    Returns a writable directory for user-specific application data.

    This is crucial for distributable applications, as they should not write
    to their installation directory (e.g., Program Files).

    - On Windows, this typically resolves to: C:\\Users\\<Username>\\AppData\\Roaming\\PrecisionFileSearch
    - On macOS/Linux, it resolves to: /home/<username>/.config/PrecisionFileSearch
    """
    if os.name == 'nt':
        app_data_path = os.getenv('APPDATA')
        if app_data_path:
            user_dir = os.path.join(app_data_path, "PrecisionFileSearch")
        else:
            user_dir = os.path.join(os.path.expanduser("~"), ".PrecisionFileSearch")
    else:
        xdg_config_home = os.getenv('XDG_CONFIG_HOME')
        if xdg_config_home:
            user_dir = os.path.join(xdg_config_home, "PrecisionFileSearch")
        else:
            user_dir = os.path.join(os.path.expanduser("~"), ".config", "PrecisionFileSearch")

    try:
        os.makedirs(user_dir, exist_ok=True)
    except OSError:
        logger.error(f"Could not create user data directory at {user_dir}. Falling back to a local folder.")
        user_dir = "user_data"
        os.makedirs(user_dir, exist_ok=True)

    return user_dir

DATA_FOLDER = get_user_data_dir()
CONFIG_DB = os.path.join(DATA_FOLDER, "app_config.db")

MODELS_FOLDER = os.path.join(DATA_FOLDER, "models")
os.makedirs(MODELS_FOLDER, exist_ok=True)


DEFAULT_CONFIG = {
    "excluded_folders": [
        "$RECYCLE.BIN", "System Volume Information", ".Trash", ".Trashes", "AppData",
        "Application Data", "Library", ".cache", "cache", "logs", "tmp", "temp",
        "node_modules", ".git", ".svn", ".hg", "dist", "build", "out", "target",
        "bin", "obj", "venv", ".venv", "env", ".env", "__pycache__", ".vscode",
        ".idea", ".vs", "vendor", "bower_components", ".npm", ".nuget", ".gradle",
        ".project", ".settings", ".classpath", ".pytest_cache"
    ],
    "file_extensions": [
        ".txt", ".md", ".pdf", ".docx", ".doc", ".rtf", ".log", ".csv", ".json",
        ".xml", ".eml", ".msg", ".epub", ".html", ".htm", ".rst", ".odt", ".wpd",
        ".pages", ".xlsx", ".xls", ".pptx", ".ppt", ".ods", ".odp", ".tsv", ".tex",
        ".py", ".js", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rb",
        ".rs", ".swift", ".kt", ".scala", ".php", ".pl", ".vb", ".css", ".scss",
        ".sass", ".less", ".svg", ".jsx", ".tsx", ".yaml", ".yml", ".ini", ".toml",
        ".sql", ".conf", ".cfg", ".env", ".properties", ".sh", ".bash", ".ps1", ".bat"
    ],
    "ai_search_params": {
        "default_temperature": 0.2,
        "default_max_tokens": 4096
    },
    "llm_config": {
        "api_key": "YOUR_LLM_API_KEY_HERE",
        "model_name": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "base_url": "https://api.groq.com/openai/v1"
    },
    "retrieval_params": {
        "enable_reranker": True,
        "k_fetch_initial": 50,
        "vector_score_threshold": 0.3,
        "vector_top_n": 10,
        "rerank_top_n": 10,
        "rerank_score_threshold": 0.5
    },
    # Block Version: 1.1.0
    "embedding_model": {
        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "device": "auto"
    },
    "reranker_model": {
        "model_name": "",
        "device": "auto"
    },
    "vectordb": {
        "provider": "qdrant",
        "qdrant": {
            "mode": "local_on_disk",
            "storage_path": "qds",
            "collection_name": "QKB"
        }
    }
}

# 3. DATABASE INITIALIZATION ####################################################################################
def _initialize_database():
    """
    Internal function to create and populate the config DB.
    """
    try:
        with sqlite3.connect(CONFIG_DB) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            for key, value in DEFAULT_CONFIG.items():
                cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)", (key, json.dumps(value)))
            conn.commit()
        logger.debug(f"Configuration database initialized successfully at: {CONFIG_DB}")
    except Exception:
        logger.exception("Failed to initialize the configuration database.")
        raise

_initialize_database()

# 4. CORE CONFIG FUNCTIONS ######################################################################################
def get_config(key: str) -> Any:
    """
    Retrieves a configuration value by key from the SQLite database.
    """
    try:
        with sqlite3.connect(CONFIG_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.warning(f"Could not read key '{key}' from config DB, using default. Error: {e}")

    logger.debug(f"Key '{key}' not found in DB or read failed; returning default value.")
    return DEFAULT_CONFIG.get(key)

def set_config(key: str, value: Any):
    """
    Sets or updates a configuration value in the SQLite database.
    """
    try:
        with sqlite3.connect(CONFIG_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, json.dumps(value)))
            conn.commit()
        logger.debug(f"Configuration key '{key}' has been set/updated.")
    except Exception:
        logger.exception(f"Failed to set configuration for key '{key}'.")
        raise

def reset_to_defaults():
    """
    Resets the configuration by clearing the config table and re-populating it with default values.
    """
    try:
        with sqlite3.connect(CONFIG_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM config")

            for key, value in DEFAULT_CONFIG.items():
                cursor.execute("INSERT INTO config (key, value) VALUES (?, ?)", (key, json.dumps(value)))

            conn.commit()
        logger.info("Configuration database has been reset to defaults.")
    except Exception:
        logger.exception("Error resetting configuration database.")
        raise
