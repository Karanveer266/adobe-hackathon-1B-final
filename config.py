"""
Configuration settings for PDF heading detection system
"""

import os
from pathlib import Path

class Config:
    # Project directories
    PROJECT_ROOT = Path(__file__).parent
    MODELS_DIR = PROJECT_ROOT / "models"
    TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
    
    # Model file paths
    BINARY_MODEL_PATH = MODELS_DIR / "binary_classifier.joblib"
    HIERARCHICAL_MODEL_PATH = MODELS_DIR / "hierarchical_classifier.joblib"
    FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
    
    # Training data paths
    FEATURES_FILE = TRAINING_DATA_DIR / "pdf_synthetic_features.csv"
    LABELS_FILE = TRAINING_DATA_DIR / "pdf_synthetic_labels.csv"
    
    # Feature extraction settings
    MIN_FONT_SIZE = 6
    MAX_FONT_SIZE = 72
    MIN_WORDS_FOR_HEADING = 1
    MAX_WORDS_FOR_HEADING = 25
    MIN_CHARS_FOR_HEADING = 2
    MAX_CHARS_FOR_HEADING = 200
    
    # Classification thresholds
    BINARY_CONFIDENCE_THRESHOLD = 0.75
    HIERARCHICAL_CONFIDENCE_THRESHOLD = 0.50
    ENSEMBLE_WEIGHT_BINARY = 0.6
    ENSEMBLE_WEIGHT_HIERARCHICAL = 0.4
    
    # Font size percentiles for hierarchy determination
    TITLE_PERCENTILE = 90
    H1_PERCENTILE = 75
    H2_PERCENTILE = 60
    H3_PERCENTILE = 45
    
    # Heading classification labels
    HEADING_TYPES = ["title", "h1", "h2", "h3"]
    ALL_TYPES = HEADING_TYPES + ["non-heading"]
    
    # Synthetic data generation settings
    DEFAULT_SAMPLES_PER_CLASS = 15000
    MIN_SAMPLES_PER_CLASS = 5000
    MAX_SAMPLES_PER_CLASS = 50000
    
    # Output settings
    OUTPUT_FORMAT = "json"
    INCLUDE_CONFIDENCE = True
    INCLUDE_FONT_INFO = True
    INCLUDE_PROCESSING_STATS = True
    
    # Performance optimization
    MAX_FEATURES_FOR_TRAINING = 50
    ENABLE_FEATURE_SELECTION = True
    CROSS_VALIDATION_FOLDS = 5
    
    # Model hyperparameters
    BINARY_CLASSIFIER_PARAMS = {
        'n_estimators': 150,
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    HIERARCHICAL_CLASSIFIER_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }

# Create directories if they don't exist
for directory in [Config.MODELS_DIR, Config.TRAINING_DATA_DIR]:
    directory.mkdir(exist_ok=True, parents=True)
