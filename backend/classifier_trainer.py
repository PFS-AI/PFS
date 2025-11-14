# File Version: 1.2.0
# /backend/classifier_trainer.py

# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

"""
# Precision File Search
# Copyright (c) 2025 Ali Kazemi
# Licensed under MPL 2.0
# This file is part of a derivative work and must retain this notice.

Handles the machine learning model training for the document classifier.

This module is responsible for the end-to-end process of creating, training,
evaluating, and saving a document classification model. It is designed to be
run as a background task, reporting its progress and results through a shared
status dictionary.

The main function, `run_training_task`, orchestrates the following steps:
1.  **Data Loading:** Scans a specified directory for training data, which is
    expected to be organized into subdirectories where each subdirectory's name
    represents a class label (e.g., 'Invoices', 'Contracts').
2.  **Text Extraction:** Uses the `unstructured` library to robustly extract
    text content from a wide variety of file formats found within the data
    directories. It includes a fallback for plain text reading if `unstructured`
    fails.
3.  **Data Splitting:** Divides the loaded documents and their corresponding
    labels into training and testing sets using scikit-learn's `train_test_split`.
4.  **Model Training:** Defines and trains a scikit-learn `Pipeline`. This pipeline
    first converts the text data into numerical features using `TfidfVectorizer`
    and then trains a `RandomForestClassifier` on these features.
5.  **Evaluation:** Assesses the trained model's performance by making predictions
    on the test set and calculating the accuracy score.
6.  **Model Saving:** Serializes the trained pipeline object to a file using
    `ml`, making it available for the main application to load and use for
    inference.
"""

# 1. IMPORTS ####################################################################################################
import joblib as ml
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os
from typing import Set
from unstructured.partition.auto import partition
from .config_manager import DATA_FOLDER

# 2. CONSTANTS & SETUP ##########################################################################################
logger = logging.getLogger(__name__)

CLASSIFIER_MODEL_PATH = os.path.join(DATA_FOLDER, "document_classifier.ml")

# 3. HELPER FUNCTIONS ###########################################################################################
# Block Version: 1.2.0
def _extract_text(file_path: Path) -> str:
    """
    Extracts text from a file, using the 'fast' strategy for performance.
    """
    try:
        if not file_path.is_file():
            return ""
        # FIX: Explicitly set languages to bypass auto-detection.
        elements = partition(filename=str(file_path), strategy="fast", languages=['eng'])
        return "\n".join([str(el) for el in elements])
    except Exception as e:
        logger.warning(f"Unstructured failed on {file_path.name} during training: {e}. Falling back to text read.")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as fe:
            logger.error(f"Fallback text read also failed for {file_path.name} during training: {fe}")
            return ""


# 4. MAIN TRAINING FUNCTION #####################################################################################
def run_training_task(status_dict: dict, data_path: str, test_size: float, n_estimators: int, excluded_folders: Set[str]):
    """
    Main function to run the training process and report status.
    Now respects the provided set of excluded folder names.
    """
    try:
        status_dict['status'] = 'running'
        status_dict['log'] = ["Training process started..."]
        status_dict['accuracy'] = None
        logger.info(f"Starting classifier training with data from '{data_path}'.")

        data_dir = Path(data_path)
        if not data_dir.is_dir():
            raise FileNotFoundError("The specified training data directory does not exist.")

        texts = []
        labels = []
        log_entry = "Phase 1: Loading and extracting text from documents."
        status_dict['log'].append(log_entry)
        logger.info(log_entry)

        category_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name not in excluded_folders]

        if not category_dirs:
             raise ValueError(f"No valid category directories found in '{data_path}'. All subdirectories were either excluded or non-existent.")

        for i, category_dir in enumerate(category_dirs):
            category = category_dir.name
            log_msg = f"-> Loading category '{category}' ({i+1}/{len(category_dirs)})..."
            status_dict['log'].append(log_msg)
            logger.debug(log_msg)

            for file_path in category_dir.rglob("*"):
                if file_path.is_file():
                    content = _extract_text(file_path)
                    if content:
                        texts.append(content)
                        labels.append(category)

        if len(texts) < 10 or len(set(labels)) < 2:
             raise ValueError("Insufficient data. Need at least 10 documents and 2 different categories.")

        log_entry = f"\nLoaded {len(texts)} documents from {len(set(labels))} categories."
        status_dict['log'].append(log_entry)
        logger.info(log_entry.strip())
        time.sleep(1)

        log_entry = "\nPhase 2: Splitting data for training and testing."
        status_dict['log'].append(log_entry)
        logger.info(log_entry.strip())

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        status_dict['log'].append(f"-> Training set: {len(X_train)} documents.")
        status_dict['log'].append(f"-> Testing set: {len(X_test)} documents.")
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test.")
        time.sleep(1)

        log_entry = f"\nPhase 3: Training the model (n_estimators={n_estimators}). This is the slowest step."
        status_dict['log'].append(log_entry)
        logger.info(log_entry.strip())

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=2)),
            ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1))
        ])
        pipeline.fit(X_train, y_train)

        log_entry = "-> Model training complete."
        status_dict['log'].append(log_entry)
        logger.info(log_entry)
        time.sleep(1)

        log_entry = "\nPhase 4: Evaluating model performance."
        status_dict['log'].append(log_entry)
        logger.info(log_entry.strip())

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        status_dict['accuracy'] = f"{accuracy * 100:.2f}%"

        log_entry = f"-> Final Model Accuracy: {status_dict['accuracy']}"
        status_dict['log'].append(log_entry)
        logger.info(log_entry)
        time.sleep(1)

        log_entry = "\nPhase 5: Saving the new model to 'document_classifier.ml'."
        status_dict['log'].append(log_entry)
        logger.info(log_entry.strip())

        ml.dump(pipeline, CLASSIFIER_MODEL_PATH)

        log_entry = "-> Model saved successfully. It will be used on the next app restart."
        status_dict['log'].append(log_entry)
        logger.info(log_entry)

        status_dict['status'] = 'complete'

    except Exception as e:
        error_message = f"An error occurred during training: {e}"
        logger.exception("Classifier training task failed.")
        status_dict['log'].append(f"\nERROR: {error_message}")
        status_dict['status'] = 'error'
