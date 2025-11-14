# File Version: 1.3.0
# /backend/file_engine.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
# Precision File Search - High-Performance File Engine
# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

High-performance, concurrent file engine module.

This module implements the principles of the High-Performance Concurrent
File-Search Checklist to provide maximum search speed and efficiency. It uses a
producer-consumer model with a ThreadPoolExecutor to parallelize file system
I/O and content processing.

Key Features:
- Concurrent file discovery and processing.
- Hybrid content searching: fast, chunked reading for plain text and robust
  extraction for complex formats like .docx and .pdf.
- Single-pass multi-keyword searching using compiled regex.
- Intelligent pre-filtering of files by metadata before processing content.
- Robust error handling for individual files to ensure the pipeline continues.
- Detailed logging for observability.
"""

# 1. IMPORTS ####################################################################################################
import os
import re
import asyncio
import time
import logging
from pathlib import Path
from typing import List, Set, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from unstructured.partition.auto import partition

# 2. SETUP & CONSTANTS ##########################################################################################
logger = logging.getLogger(__name__)

CONTENT_CHUNK_SIZE = 32 * 1024  # 32 KB

PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".log", ".csv", ".json", ".xml", ".html", ".htm", ".rst",
    ".tsv", ".tex", ".py", ".js", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rb", ".rs", ".swift", ".kt", ".scala", ".php", ".pl", ".vb",
    ".css", ".scss", ".sass", ".less", ".svg", ".jsx", ".tsx", ".yaml",
    ".yml", ".ini", ".toml", ".sql", ".conf", ".cfg", ".env", ".properties",
    ".asc", ".text", ".xhtml", ".cxx", ".cc", ".hxx", ".php3", ".php4",
    ".phtml", ".sh", ".bash", ".ps1", ".bat", ".m", ".mm", ".erb", ".jspes"
}


# 3. CORE HELPER FUNCTIONS ######################################################################################

# Block Version: 1.3.0
def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts text content from a file using 'unstructured' with a fallback.
    Uses the "fast" strategy and specifies the language to avoid slow detection.
    """
    try:
        if not file_path.is_file():
            return ""
        elements = partition(filename=str(file_path), strategy="fast", languages=['eng'])
        return "\n".join([str(el) for el in elements])
    except Exception as e:
        logger.warning(f"Unstructured failed on {file_path.name}: {e}. Falling back to text read.")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as fe:
            logger.error(f"Fallback text read also failed for {file_path.name}: {fe}")
            return ""

def _is_match_content_chunked_reader(file_path: Path, pattern: re.Pattern) -> bool:
    """
    Reads a plain text file in chunks and checks if any chunk matches the regex pattern.
    This is highly memory-efficient for large text files.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            overlap = 1024
            buffer = ""
            while chunk := f.read(CONTENT_CHUNK_SIZE):
                if pattern.search(buffer + chunk):
                    return True
                buffer = chunk[-overlap:]
    except Exception as e:
        logger.debug(f"Could not read content from {file_path} using chunked reader: {e}")
    return False

def _is_match_content_full_extraction(file_path: Path, pattern: re.Pattern) -> bool:
    """
    Extracts all text content from complex file types (like .docx, .pdf)
    using the 'unstructured' library and then checks for a match.
    """
    try:
        content = extract_text_from_file(file_path)
        if content and pattern.search(content):
            return True
    except Exception as e:
        logger.debug(f"Could not extract or search content from {file_path}: {e}")
    return False

def _is_match_content(file_path: Path, pattern: re.Pattern) -> bool:
    """
    Orchestrates content matching by selecting the appropriate method.
    Uses a fast chunked reader for plain text files and full extraction for others.
    """
    if file_path.suffix.lower() in PLAIN_TEXT_EXTENSIONS:
        return _is_match_content_chunked_reader(file_path, pattern)
    else:
        return _is_match_content_full_extraction(file_path, pattern)

def _translate_wildcard_to_regex(pattern: str) -> str:
    """
    Translates a file system wildcard pattern (*, ?) to a valid regex pattern.
    This makes the "Use Regex" feature more intuitive for file searches.
    """
    escaped_pattern = re.escape(pattern)
    regex_pattern = escaped_pattern.replace(r'\*', '.*')
    regex_pattern = regex_pattern.replace(r'\?', '.')
    return regex_pattern

# 4. FILE SEARCH ENGINE CLASS ###################################################################################

class FileSearchEngine:
    """
    Core file search engine that handles concurrent file/folder searches.
    """
    def __init__(self, default_excluded_folders: Set[str], default_file_extensions: List[str]):
        self.default_excluded_folders = default_excluded_folders
        self.default_file_extensions = default_file_extensions
        self.file_categories = {
            "Documents": [".pdf", ".doc", ".docx", ".txt", ".md", ".odt", ".xls", ".xlsx", ".ppt", ".pptx", ".rtf"],
            "Pictures": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg", ".webp", ".heic"],
            "Videos": [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"],
            "Audio": [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"],
            "Compressed": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
            "Executable": [".exe", ".msi", ".bat", ".sh", ".app", ".jar", ".com"]
        }
        cpu_cores = os.cpu_count() or 1
        self.max_workers = min(32, cpu_cores + 4)
        logger.info(f"FileSearchEngine initialized with max_workers={self.max_workers}")

    def _compile_search_pattern(self, keywords: List[str], use_regex: bool, case_sensitive: bool) -> re.Pattern:
        """
        Compiles a single, efficient regex pattern. If 'use_regex' is true,
        it translates common wildcards (*, ?) to their regex equivalents.
        """
        flags = 0 if case_sensitive else re.IGNORECASE

        if use_regex:
            # Translate each keyword from wildcard to regex before joining
            translated_keywords = [_translate_wildcard_to_regex(k) for k in keywords]
            pattern_str = '|'.join(translated_keywords)
        else:
            # Just escape for literal search
            pattern_str = '|'.join(map(re.escape, keywords))

        return re.compile(pattern_str, flags)

    def _get_file_extensions(self, req: Any) -> tuple:
        """Get file extensions tuple based on search request."""
        if req.search_type.value == "file_category":
            return tuple(self.file_categories.get(req.file_category, []))
        return tuple(req.file_extensions or self.default_file_extensions)

    def _process_file(self, file_path_str: str, req: Any, pattern: re.Pattern, extensions: tuple) -> Dict[str, Any]:
        """
        Self-contained worker function to process a single file in a thread pool.
        """
        file_path = Path(file_path_str)

        try:
            if req.search_type.value in ["file_content", "file_category"] and extensions:
                if file_path.suffix.lower() not in extensions:
                    return {"status": "skipped_extension"}

            stat_info = file_path.stat()
            if (req.min_size is not None and stat_info.st_size < req.min_size) or \
               (req.max_size is not None and stat_info.st_size > req.max_size):
                return {"status": "skipped_size"}

            match_found = False
            if req.search_type.value in ["file_name", "file_category"]:
                if pattern.search(file_path.name):
                    match_found = True
            elif req.search_type.value == "file_content":
                if _is_match_content(file_path, pattern):
                    match_found = True

            if match_found:
                return {"status": "found", "path": file_path_str, "mtime": stat_info.st_mtime}

            return {"status": "no_match"}
        except Exception:
            return {"status": "error_processing"}

    async def run_search(self, websocket: Any, search_id: str, req: Any):
        """
        Performs a high-performance, concurrent file search and streams results.
        """
        start_time = time.monotonic()
        found_items = []
        items_scanned = 0

        try:
            pattern = self._compile_search_pattern(req.keywords, req.use_regex, req.case_sensitive)
        except re.error as e:
            await websocket.send_json({"type": "error", "message": f"Invalid Regex: {e}"})
            return

        extensions = self._get_file_extensions(req)
        excluded = set(req.excluded_folders or self.default_excluded_folders)

        loop = asyncio.get_running_loop()
        process_func = partial(self._process_file, req=req, pattern=pattern, extensions=extensions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if req.search_type.value == "folder_name":
                for dirpath, dirnames, _ in os.walk(req.search_path, topdown=True):
                    dirnames[:] = [d for d in dirnames if d not in excluded and (req.include_dot_folders or not d.startswith('.'))]
                    for dirname in dirnames:
                        items_scanned += 1
                        if pattern.search(dirname):
                            try:
                                path = os.path.join(dirpath, dirname)
                                found_items.append({"path": path, "mtime": os.path.getmtime(path)})
                                await websocket.send_json({"type": "item_found", "path": path})
                            except OSError:
                                continue
                    await websocket.send_json({"type": "scan_progress", "progress": {"scanned": items_scanned, "found": len(found_items)}})
                    await asyncio.sleep(0)
            else:
                futures = []
                for dirpath, dirnames, filenames in os.walk(req.search_path, topdown=True):
                    dirnames[:] = [d for d in dirnames if d not in excluded and (req.include_dot_folders or not d.startswith('.'))]
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        futures.append(loop.run_in_executor(executor, process_func, file_path))

                    if len(futures) >= self.max_workers * 5:
                        for future in asyncio.as_completed(futures):
                            result = await future
                            items_scanned += 1
                            if result["status"] == "found":
                                found_items.append(result)
                                await websocket.send_json({"type": "item_found", "path": result["path"]})
                        futures.clear()
                        await websocket.send_json({"type": "scan_progress", "progress": {"scanned": items_scanned, "found": len(found_items)}})

                for future in asyncio.as_completed(futures):
                    result = await future
                    items_scanned += 1
                    if result["status"] == "found":
                        found_items.append(result)
                        await websocket.send_json({"type": "item_found", "path": result["path"]})

        found_items.sort(key=lambda x: x["mtime"], reverse=True)
        sorted_paths = [item["path"] for item in found_items]

        duration = time.monotonic() - start_time
        item_type = "folders" if req.search_type.value == "folder_name" else "files"
        summary = f"Search complete in {duration:.2f} seconds. Scanned {items_scanned} items and found {len(found_items)} matching {item_type}."
        logger.info(summary)

        await websocket.send_json({
            "type": "scan_complete", "task_id": search_id, "results": sorted_paths, "summary": summary
        })
