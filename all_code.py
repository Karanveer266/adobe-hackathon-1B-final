# in Main
# python main.py --train-synthetic --samples 15000 --optimize-params --force-retrain

#=============================================================================
# FILE: classifiers.py
#=============================================================================

"""
Multi-stage classification system with synthetic data training
"""

import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import re  # Ensure re is imported
from typing import List, Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from config import Config
from synthetic_data_generator import generate_and_save_training_data
from utils import (
    get_heading_level,
    get_heading_type,
    filter_low_confidence_predictions,
    calculate_text_similarity
)

logger = logging.getLogger(__name__)

class HeadingClassificationSystem:
    """Enhanced multi-stage heading classification system with stricter thresholds"""

    def __init__(self):
        self.binary_classifier = None
        self.hierarchical_classifier = None
        self.feature_scaler = None
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.training_stats = {}

        # ADDED: Stricter thresholds
        self.BINARY_THRESHOLD = 0.75  # Increased from default
        self.HIERARCHICAL_THRESHOLD = 0.70  # Increased from default
        self.MIN_HEADING_LENGTH = 3
        self.MAX_HEADING_LENGTH = 200

    def train_with_synthetic_data(self, samples_per_class: int = None,
                                  optimize_hyperparameters: bool = True) -> None:
        """Train the classification system using synthetic data"""
        if samples_per_class is None:
            samples_per_class = Config.DEFAULT_SAMPLES_PER_CLASS

        logger.info(f"Training with synthetic data: {samples_per_class} samples per class")

        # Load or generate synthetic data
        X, y = self._load_or_generate_data(samples_per_class)

        # Feature preprocessing
        X_processed, feature_names = self._preprocess_features(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train binary classifier
        self._train_binary_classifier(X_train, y_train, X_test, y_test, optimize_hyperparameters)

        # Train hierarchical classifier
        self._train_hierarchical_classifier(X_train, y_train, X_test, y_test, optimize_hyperparameters)

        # Evaluate complete system
        self._evaluate_complete_system(X_test, y_test)

        # Save models
        self._save_models()

        self.is_trained = True
        logger.info("Training completed successfully!")

    def _load_or_generate_data(self, samples_per_class: int) -> Tuple[pd.DataFrame, List[str]]:
        """Load existing data or generate new synthetic data"""
        if Config.FEATURES_FILE.exists() and Config.LABELS_FILE.exists():
            logger.info("Loading existing synthetic data...")
            try:
                X = pd.read_csv(Config.FEATURES_FILE)
                y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()

                # Check if we have enough samples
                class_counts = pd.Series(y).value_counts()
                min_samples = class_counts.min()

                if min_samples >= samples_per_class * 0.8:  # Allow 20% tolerance
                    logger.info(f"Using existing data with {len(y)} total samples")
                    return X, y
                else:
                    logger.info(f"Existing data has insufficient samples ({min_samples} < {samples_per_class})")

            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}")

        # Generate new data
        logger.info("Generating new synthetic training data...")
        X, y = generate_and_save_training_data(samples_per_class)
        return X, y

    def _preprocess_features(self, X: pd.DataFrame, y: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features with scaling and selection"""
        logger.info("Preprocessing features...")

        # Store feature names
        self.feature_names = list(X.columns)

        # Convert to numpy array
        X_array = X.values

        # Handle missing values
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e6, neginf=-1e6)

        # Initialize and fit scaler
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_array)

        # Feature selection (optional)
        if Config.ENABLE_FEATURE_SELECTION and X_scaled.shape[1] > Config.MAX_FEATURES_FOR_TRAINING:
            logger.info(f"Selecting best {Config.MAX_FEATURES_FOR_TRAINING} features...")

            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(Config.MAX_FEATURES_FOR_TRAINING, X_scaled.shape[1])
            )

            X_selected = self.feature_selector.fit_transform(X_scaled, y)

            # Update feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]

            logger.info(f"Selected {X_selected.shape[1]} features from {X_scaled.shape[1]}")
            return X_selected, self.feature_names

        return X_scaled, self.feature_names

    def _train_binary_classifier(self, X_train: np.ndarray, y_train: List[str],
                                 X_test: np.ndarray, y_test: List[str],
                                 optimize_hyperparameters: bool = True) -> None:
        """Train binary heading/non-heading classifier with improved parameters"""
        logger.info("Training binary classifier...")

        # Create binary labels
        y_binary_train = ['heading' if label in Config.HEADING_TYPES else 'non-heading' for label in y_train]
        y_binary_test = ['heading' if label in Config.HEADING_TYPES else 'non-heading' for label in y_test]

        # IMPROVED: Better hyperparameters for more conservative classification
        if optimize_hyperparameters:
            param_grid = {
                'n_estimators': [150, 200, 250],  # More trees
                'max_depth': [6, 8, 10],  # Controlled depth
                'min_samples_split': [5, 10, 15],  # More conservative splitting
                'min_samples_leaf': [2, 4, 6],  # Larger leaf nodes
                'class_weight': ['balanced', 'balanced_subsample']  # Handle class imbalance
            }

            base_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
            self.binary_classifier = GridSearchCV(
                base_classifier,
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
        else:
            # Use conservative default parameters
            self.binary_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        # Train classifier
        self.binary_classifier.fit(X_train, y_binary_train)

        # Evaluate binary classifier
        y_binary_pred = self.binary_classifier.predict(X_test)
        binary_accuracy = accuracy_score(y_binary_test, y_binary_pred)

        logger.info(f"Binary classifier accuracy: {binary_accuracy:.4f}")

        # Cross-validation score
        cv_scores = cross_val_score(
            self.binary_classifier, X_train, y_binary_train,
            cv=Config.CROSS_VALIDATION_FOLDS, scoring='f1_weighted'
        )
        logger.info(f"Binary classifier CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Store training stats
        self.training_stats['binary'] = {
            'accuracy': binary_accuracy,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'best_params': self.binary_classifier.best_params_ if optimize_hyperparameters else 'default_conservative'
        }

        # Detailed classification report
        logger.debug("Binary Classification Report:")
        logger.debug(f"\n{classification_report(y_binary_test, y_binary_pred)}")

    def _train_hierarchical_classifier(self, X_train: np.ndarray, y_train: List[str],
                                       X_test: np.ndarray, y_test: List[str],
                                       optimize_hyperparameters: bool = True) -> None:
        """Train hierarchical heading level classifier with improved parameters"""
        logger.info("Training hierarchical classifier...")

        # Filter to only heading samples
        heading_indices_train = [i for i, label in enumerate(y_train) if label in Config.HEADING_TYPES]
        heading_indices_test = [i for i, label in enumerate(y_test) if label in Config.HEADING_TYPES]

        if not heading_indices_train:
            logger.error("No heading samples found for hierarchical training")
            return

        X_heading_train = X_train[heading_indices_train]
        y_heading_train = [y_train[i] for i in heading_indices_train]
        X_heading_test = X_test[heading_indices_test]
        y_heading_test = [y_test[i] for i in heading_indices_test]

        logger.info(f"Hierarchical training set: {X_heading_train.shape}")

        # IMPROVED: Better hyperparameters for hierarchical classification
        if optimize_hyperparameters:
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [6, 8, 10],
                'min_samples_split': [3, 5, 8],
                'min_samples_leaf': [1, 2, 3],
                'class_weight': ['balanced', 'balanced_subsample']
            }

            base_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
            self.hierarchical_classifier = GridSearchCV(
                base_classifier,
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1
            )
        else:
            self.hierarchical_classifier = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        # Train hierarchical classifier
        self.hierarchical_classifier.fit(X_heading_train, y_heading_train)

        # Evaluate hierarchical classifier
        if heading_indices_test:
            y_hierarchical_pred = self.hierarchical_classifier.predict(X_heading_test)
            hierarchical_accuracy = accuracy_score(y_heading_test, y_hierarchical_pred)

            logger.info(f"Hierarchical classifier accuracy: {hierarchical_accuracy:.4f}")

            # Cross-validation for hierarchical classifier
            cv_scores = cross_val_score(
                self.hierarchical_classifier, X_heading_train, y_heading_train,
                cv=min(Config.CROSS_VALIDATION_FOLDS, len(set(y_heading_train))),
                scoring='f1_weighted'
            )
            logger.info(f"Hierarchical classifier CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Store training stats
            self.training_stats['hierarchical'] = {
                'accuracy': hierarchical_accuracy,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'best_params': self.hierarchical_classifier.best_params_ if optimize_hyperparameters else 'default_conservative'
            }

            logger.debug("Hierarchical Classification Report:")
            logger.debug(f"\n{classification_report(y_heading_test, y_hierarchical_pred)}")

    def _evaluate_complete_system(self, X_test: np.ndarray, y_test: List[str]) -> None:
        """Evaluate the complete two-stage system"""
        logger.info("Evaluating complete classification system...")

        # Create mock elements with proper indexing
        mock_elements = [{'text': f'test_{i}'} for i in range(len(X_test))]

        # Get predictions from complete system
        predictions = self.predict(X_test, mock_elements)

        # Initialize with correct default
        predicted_labels = ['non-heading'] * len(y_test)

        # Track prediction application
        applied_predictions = 0
        index_errors = 0

        for pred in predictions:
            idx = pred['element_index']
            if 0 <= idx < len(predicted_labels):
                predicted_labels[idx] = pred['type']
                applied_predictions += 1
            else:
                index_errors += 1
                logger.warning(f"Index out of bounds: {idx} (max: {len(predicted_labels)-1})")

        logger.info(f"Applied {applied_predictions} predictions, {index_errors} index errors")

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y_test, predicted_labels)
        logger.info(f"Complete system accuracy: {overall_accuracy:.4f}")

        # Per-class performance
        logger.info("Complete System Classification Report:")
        logger.info(f"\n{classification_report(y_test, predicted_labels, zero_division=0)}")

        # Store system stats
        self.training_stats['complete_system'] = {
            'accuracy': overall_accuracy,
            'total_predictions': len(predictions),
            'prediction_rate': len(predictions) / len(y_test) if len(y_test) > 0 else 0,
            'applied_predictions': applied_predictions,
            'index_errors': index_errors
        }

    def predict(self, features: np.ndarray, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """IMPROVED: Predict heading types using multi-stage classification with stricter filtering"""
        if not self.is_trained:
            if not self.load_models():
                logger.warning("No trained models available, using fallback classification")
                return self._fallback_classification(elements)

        # Preprocess features
        features_processed = self._preprocess_prediction_features(features)

        if features_processed is None:
            return self._fallback_classification(elements)

        predictions = []

        # Stage 1: Binary classification
        binary_predictions = self.binary_classifier.predict(features_processed)
        binary_probabilities = self.binary_classifier.predict_proba(features_processed)

        # Get class indices for probability extraction
        binary_classes = list(self.binary_classifier.classes_)
        try:
            heading_class_idx = binary_classes.index('heading')
        except ValueError:
            logger.error("Binary classifier doesn't recognize 'heading' class")
            return self._fallback_classification(elements)

        # Stage 2: Process each prediction with stricter criteria
        for i, (element, binary_pred) in enumerate(zip(elements, binary_predictions)):
            text = element.get('text', '').strip()

            # ADDED: Pre-filter based on text characteristics
            if not self._passes_text_quality_check(element):
                continue

            if binary_pred == 'heading':
                # Get binary confidence
                binary_confidence = binary_probabilities[i][heading_class_idx]

                # STRICTER: Higher threshold for binary classification
                if binary_confidence >= self.BINARY_THRESHOLD:
                    # Apply hierarchical classifier
                    if self.hierarchical_classifier is not None:
                        hierarchical_pred = self.hierarchical_classifier.predict([features_processed[i]])[0]
                        hierarchical_proba = self.hierarchical_classifier.predict_proba([features_processed[i]])[0]
                        hierarchical_confidence = np.max(hierarchical_proba)

                        # STRICTER: Higher threshold for hierarchical classification
                        if hierarchical_confidence >= self.HIERARCHICAL_THRESHOLD:
                            # Ensemble confidence
                            final_confidence = (
                                0.6 * binary_confidence +  # MODIFIED: Weights
                                0.4 * hierarchical_confidence
                            )

                            predictions.append({
                                'element_index': i,
                                'type': hierarchical_pred,
                                'confidence': final_confidence,
                                'binary_confidence': binary_confidence,
                                'hierarchical_confidence': hierarchical_confidence,
                                'method': 'two_stage_ml'
                            })
                        else:
                            # Use fallback hierarchy determination with penalty
                            fallback_type = self._determine_hierarchy_fallback(element)
                            predictions.append({
                                'element_index': i,
                                'type': fallback_type,
                                'confidence': binary_confidence * 0.6,  # PENALTY: Reduced confidence
                                'binary_confidence': binary_confidence,
                                'hierarchical_confidence': hierarchical_confidence,
                                'method': 'binary_ml_hierarchical_fallback'
                            })
                    else:
                        # No hierarchical classifier available
                        fallback_type = self._determine_hierarchy_fallback(element)
                        predictions.append({
                            'element_index': i,
                            'type': fallback_type,
                            'confidence': binary_confidence * 0.7,
                            'binary_confidence': binary_confidence,
                            'hierarchical_confidence': 0.5,
                            'method': 'binary_ml_only'
                        })

        # STRICTER: Filter low confidence predictions
        predictions = filter_low_confidence_predictions(
            predictions,
            min(self.HIERARCHICAL_THRESHOLD, 0.65)  # Dynamic threshold
        )

        logger.info(f"Generated {len(predictions)} heading predictions from {len(elements)} elements")
        return predictions

    def _passes_text_quality_check(self, element: Dict[str, Any]) -> bool:
        """ADDED: Pre-filter elements based on text quality"""
        text = element.get('text', '').strip()
        font_size = element.get('font_size', 12.0)
        is_bold = element.get('is_bold', False)

        # Length checks
        if not (self.MIN_HEADING_LENGTH <= len(text) <= self.MAX_HEADING_LENGTH):
            return False

        # Word count check
        word_count = len(text.split())
        if word_count == 0 or word_count > 20:
            return False

        # Must have some formatting distinction OR be in early position
        position = element.get('position', 0)
        has_formatting = font_size > 11 or is_bold
        is_early = position < 20

        if not (has_formatting or is_early or font_size >= 12):
            return False

        # Additional noise patterns (from feature_extractor)
        if self._is_noise_element(text):
            return False 
            
        return True

    def _preprocess_prediction_features(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess features for prediction (same as training preprocessing)"""
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

            # Scale features
            if self.feature_scaler is not None:
                features_scaled = self.feature_scaler.transform(features_clean)
            else:
                logger.warning("No feature scaler available")
                features_scaled = features_clean

            # Apply feature selection
            if self.feature_selector is not None:
                features_selected = self.feature_selector.transform(features_scaled)
                return features_selected

            return features_scaled

        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}")
            return None

    def _determine_hierarchy_fallback(self, element: Dict[str, Any]) -> str:
        # FIXED: Better hierarchy determination with correct logic
        """FIXED: Better hierarchy determination with correct logic"""
        font_size = element.get('font_size', 12.0)
        position = element.get('position', 0)
        page = element.get('page', 1)
        text = element.get('text', '')
        is_bold = element.get('is_bold', False)
        word_count = len(text.split()) if text else 0

        # Title detection (most restrictive)
        is_likely_title = (
            page == 1 and position < 3 and
            font_size > 15 and word_count <= 8 and
            any(word in text.lower() for word in ['overview', 'foundation', 'level'])
        )

        if is_likely_title:
            return 'title'

        # H1 detection - Major sections
        is_h1 = (
            # Numbered major sections (1., 2., 3., 4.)
            re.match(r'^\d+\.\s', text.strip()) or
            # Key document sections
            any(section in text.lower() for section in [
                'revision history', 'table of contents', 'acknowledgements',
                'introduction to', 'overview of', 'references'
            ]) or
            # Large font size indicators
            (font_size >= 16 and is_bold and position < 30 and word_count <= 15)
        )

        if is_h1:
            return 'h1'

        # H2 detection - Subsections
        is_h2 = (
            # Numbered subsections (2.1, 2.2, 3.1, etc.)
            re.match(r'^\d+\.\d+\s', text.strip()) or
            # Bold, medium font, reasonable length
            (font_size >= 14 and is_bold and word_count <= 12) or
            # Early position with good formatting
            (position < 50 and font_size >= 13 and is_bold)
        )

        if is_h2:
            return 'h2'

        # Default to H3 for remaining headings
        return 'h3'

    def _fallback_classification(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """IMPROVED: Complete fallback classification using conservative heuristics"""
        logger.info("Using complete fallback classification")
        predictions = []

        for i, element in enumerate(elements):
            text = element.get('text', '')
            font_size = element.get('font_size', 12.0)
            is_bold = element.get('is_bold', False)
            position = element.get('position', 0)
            page = element.get('page', 1)
            word_count = len(text.split()) if text else 0

            # STRICTER: More conservative heading detection
            is_heading = (
                # Font-based detection
                (font_size >= 14 and is_bold and word_count <= 10) or
                # Position-based detection
                (page <= 2 and position < 5 and font_size >= 12 and word_count <= 12) or
                # Keyword-based detection
                (any(keyword in text.lower() for keyword in
                     ['chapter', 'section', 'introduction', 'conclusion', 'abstract'])
                 and word_count <= 8) or
                # Numbered sections
                (re.match(r'^\d+\.?\s', text.strip()) and font_size >= 12 and word_count <= 15)
            )

            # ADDED: Additional quality checks
            if is_heading and self._passes_text_quality_check(element):
                heading_type = self._determine_hierarchy_fallback(element)
                predictions.append({
                    'element_index': i,
                    'type': heading_type,
                    'confidence': 0.5,  # Lower confidence for fallback
                    'binary_confidence': 0.5,
                    'hierarchical_confidence': 0.5,
                    'method': 'heuristic_fallback'
                })

        return predictions

    def load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            models_loaded = 0

            if Config.BINARY_MODEL_PATH.exists():
                self.binary_classifier = joblib.load(Config.BINARY_MODEL_PATH)
                models_loaded += 1
                logger.info("Binary classifier loaded successfully")

            if Config.HIERARCHICAL_MODEL_PATH.exists():
                self.hierarchical_classifier = joblib.load(Config.HIERARCHICAL_MODEL_PATH)
                models_loaded += 1
                logger.info("Hierarchical classifier loaded successfully")

            if Config.FEATURE_SCALER_PATH.exists():
                self.feature_scaler = joblib.load(Config.FEATURE_SCALER_PATH)
                logger.info("Feature scaler loaded successfully")
            
            if Config.MODELS_DIR / "feature_selector.joblib" and (Config.MODELS_DIR / "feature_selector.joblib").exists():
                 self.feature_selector = joblib.load(Config.MODELS_DIR / "feature_selector.joblib")
                 logger.info("Feature selector loaded successfully")


            if models_loaded >= 2:
                self.is_trained = True
                logger.info("All models loaded successfully")
                return True
            else:
                logger.warning(f"Only {models_loaded} models loaded")
                return False

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            # Ensure models directory exists
            Config.MODELS_DIR.mkdir(exist_ok=True, parents=True)

            if self.binary_classifier is not None:
                joblib.dump(self.binary_classifier, Config.BINARY_MODEL_PATH)
                logger.info(f"Binary classifier saved to {Config.BINARY_MODEL_PATH}")

            if self.hierarchical_classifier is not None:
                joblib.dump(self.hierarchical_classifier, Config.HIERARCHICAL_MODEL_PATH)
                logger.info(f"Hierarchical classifier saved to {Config.HIERARCHICAL_MODEL_PATH}")

            if self.feature_scaler is not None:
                joblib.dump(self.feature_scaler, Config.FEATURE_SCALER_PATH)
                logger.info(f"Feature scaler saved to {Config.FEATURE_SCALER_PATH}")
            
            if self.feature_selector is not None:
                joblib.dump(self.feature_selector, Config.MODELS_DIR / "feature_selector.joblib")
                logger.info(f"Feature selector saved to {Config.MODELS_DIR / 'feature_selector.joblib'}")

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def detect_title(self, elements: List[Dict[str, Any]]) -> str:
        # ADDED: Detect document title from early elements
        """Detect document title from first page, large font elements"""
        title_candidates = []

        for element in elements[:10]:  # Check first 10 elements only
            if element.get('page', 1) == 1:
                font_size = element.get('font_size', 12)
                text = element.get('text', '').strip()
                position = element.get('position', 0)
                word_count = len(text.split())

                # Title criteria
                if (font_size >= 15 and
                        position < 5 and
                        3 <= word_count <= 10 and
                        any(keyword in text.lower() for keyword in ['overview', 'foundation', 'level', 'extension']) and
                        not self._is_noise_element(text)):
                    title_candidates.append((text, font_size, position))
        
        if title_candidates:
            # Return largest font, earliest position
            title_candidates.sort(key=lambda x: (-x[1], x[2]))
            return title_candidates[0][0]

        return "Untitled Document"

    def _is_noise_element(self, text: str) -> bool:
        """Check if text is likely noise (same as feature extractor)"""
        noise_patterns = [
            r'^Page\s+\d+\s+of\s+\d+$',
            r'^[.\-_=•◆|▪~\s]{3,}$',
            r'^\d+\.?$',
            r'^©.*|^Copyright.*',
            r'^[A-Za-z]+\s+\d{1,2},?\s+\d{4}$'
        ]

        return any(re.match(pattern, text, re.IGNORECASE) for pattern in noise_patterns)

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.training_stats.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        importance = {}

        if self.binary_classifier is not None and hasattr(self.binary_classifier, 'feature_importances_'):
            # Handle GridSearchCV wrapper
            model = self.binary_classifier.best_estimator_ if hasattr(self.binary_classifier, 'best_estimator_') else self.binary_classifier

            if hasattr(model, 'feature_importances_') and len(self.feature_names) == len(model.feature_importances_):
                for name, imp in zip(self.feature_names, model.feature_importances_):
                    importance[f'binary_{name}'] = float(imp)

        if self.hierarchical_classifier is not None and hasattr(self.hierarchical_classifier, 'feature_importances_'):
            model = self.hierarchical_classifier.best_estimator_ if hasattr(self.hierarchical_classifier, 'best_estimator_') else self.hierarchical_classifier

            if hasattr(model, 'feature_importances_') and len(self.feature_names) == len(model.feature_importances_):
                for name, imp in zip(self.feature_names, model.feature_importances_):
                    importance[f'hierarchical_{name}'] = float(imp)

        return importance

# Factory function
def create_classification_system() -> HeadingClassificationSystem:
    """Create configured classification system"""
    return HeadingClassificationSystem()

#=============================================================================
# FILE: config.py
#=============================================================================

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


#=============================================================================
# FILE: debug_components.py
#=============================================================================

import pandas as pd
import numpy as np
from classifiers import create_classification_system
from config import Config
from sklearn.model_selection import train_test_split

def debug_individual_components():
    print("🔍 DEBUGGING INDIVIDUAL COMPONENTS")
    print("="*50)
    
    # Load classifier
    classifier = create_classification_system()
    if not classifier.load_models():
        print("❌ Could not load models")
        return
    
    # Load data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    # Use same split as training
    _, X_test, _, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Test set size: {X_test.shape}")
    
    # Test binary classifier directly
    print("\n1. TESTING BINARY CLASSIFIER:")
    y_binary_test = ['heading' if label in Config.HEADING_TYPES else 'non-heading' for label in y_test]
    
    # Process features
    X_processed = classifier._preprocess_prediction_features(X_test)
    
    binary_predictions = classifier.binary_classifier.predict(X_processed)
    binary_accuracy = sum(1 for true, pred in zip(y_binary_test, binary_predictions) if true == pred) / len(y_binary_test)
    
    print(f"Binary accuracy: {binary_accuracy:.4f}")
    print("Binary prediction distribution:", {label: list(binary_predictions).count(label) for label in set(binary_predictions)})
    
    # Test hierarchical classifier on heading samples only
    print("\n2. TESTING HIERARCHICAL CLASSIFIER:")
    heading_indices = [i for i, pred in enumerate(binary_predictions) if pred == 'heading']
    
    if heading_indices:
        X_headings = X_processed[heading_indices]
        y_headings_true = [y_test[i] for i in heading_indices]
        
        hierarchical_predictions = classifier.hierarchical_classifier.predict(X_headings)
        hierarchical_accuracy = sum(1 for true, pred in zip(y_headings_true, hierarchical_predictions) if true == pred) / len(y_headings_true)
        
        print(f"Hierarchical accuracy on detected headings: {hierarchical_accuracy:.4f}")
        print("Hierarchical prediction distribution:", {label: list(hierarchical_predictions).count(label) for label in set(hierarchical_predictions)})
        
        # Show confidence distributions
        hierarchical_probs = classifier.hierarchical_classifier.predict_proba(X_headings)
        max_probs = np.max(hierarchical_probs, axis=1)
        print(f"Hierarchical confidence range: {np.min(max_probs):.3f} - {np.max(max_probs):.3f}")
        print(f"Above threshold ({Config.HIERARCHICAL_CONFIDENCE_THRESHOLD}): {sum(1 for p in max_probs if p >= Config.HIERARCHICAL_CONFIDENCE_THRESHOLD)}/{len(max_probs)}")
    
    print(f"\n3. CONFIDENCE THRESHOLDS:")
    print(f"Binary threshold: {Config.BINARY_CONFIDENCE_THRESHOLD}")
    print(f"Hierarchical threshold: {Config.HIERARCHICAL_CONFIDENCE_THRESHOLD}")

if __name__ == "__main__":
    debug_individual_components()

#=============================================================================
# FILE: debug_features.py
#=============================================================================

import pandas as pd
from config import Config

def debug_feature_consistency():
    print("🔍 DEBUGGING FEATURE CONSISTENCY")
    print("="*50)
    
    # Load synthetic data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    print(f"Loaded data shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Check for any NaN or infinite values
    nan_cols = X.columns[X.isnull().any()].tolist()
    inf_cols = X.columns[np.isinf(X).any()].tolist()
    
    print(f"Columns with NaN: {nan_cols}")
    print(f"Columns with Inf: {inf_cols}")
    
    # Check label distribution
    from collections import Counter
    label_dist = Counter(y)
    print(f"Label distribution: {label_dist}")
    
    # Check feature ranges
    print("\nFeature statistics:")
    print(X.describe())

if __name__ == "__main__":
    import numpy as np
    debug_feature_consistency()

#=============================================================================
# FILE: debug_predictions.py
#=============================================================================

import pandas as pd
import numpy as np
from main import PDFHeadingDetectionSystem
from config import Config
from collections import Counter

def debug_pipeline():
    print("🔍 DEBUGGING PIPELINE INTEGRATION")
    print("="*50)
    
    # Load system
    system = PDFHeadingDetectionSystem()
    
    # Load a small subset of synthetic data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    # Take first 1000 samples for debugging
    X_debug = X.head(1000).values
    y_debug = y[:1000]
    
    print(f"Debugging with {len(y_debug)} samples")
    print("True label distribution:", Counter(y_debug))
    
    # Create mock elements
    mock_elements = [{'text': f'debug_sample_{i}', 'font_size': 12} for i in range(len(X_debug))]
    
    # Get predictions from complete pipeline
    predictions = system.classification_system.predict(X_debug, mock_elements)
    
    print(f"\nPipeline predictions: {len(predictions)}")
    print(f"Prediction rate: {len(predictions)/len(y_debug):.1%}")
    
    if predictions:
        pred_types = [p['type'] for p in predictions]
        pred_confidences = [p['confidence'] for p in predictions]
        
        print("Predicted label distribution:", Counter(pred_types))
        print(f"Average confidence: {np.mean(pred_confidences):.3f}")
        print(f"Confidence range: {np.min(pred_confidences):.3f} - {np.max(pred_confidences):.3f}")
        
        # Check confidence thresholds
        binary_conf = [p.get('binary_confidence', 0) for p in predictions]
        hierarchical_conf = [p.get('hierarchical_confidence', 0) for p in predictions]
        
        print(f"Binary confidence avg: {np.mean(binary_conf):.3f}")
        print(f"Hierarchical confidence avg: {np.mean(hierarchical_conf):.3f}")
        
        # Show some example predictions
        print("\nExample predictions:")
        for i, pred in enumerate(predictions[:10]):
            idx = pred['element_index']
            true_label = y_debug[idx] if idx < len(y_debug) else "unknown"
            print(f"  Sample {idx}: True={true_label}, Pred={pred['type']}, Conf={pred['confidence']:.3f}")

if __name__ == "__main__":
    debug_pipeline()

#=============================================================================
# FILE: diagnose_hierarchical.py
#=============================================================================

import pandas as pd
import numpy as np
from classifiers import create_classification_system
from config import Config
from sklearn.model_selection import train_test_split

def diagnose_hierarchical_bias():
    print("🔍 DIAGNOSING HIERARCHICAL CLASSIFIER BIAS")
    print("="*50)
    
    classifier = create_classification_system()
    classifier.load_models()
    
    # Load data
    X = pd.read_csv(Config.FEATURES_FILE)
    y = pd.read_csv(Config.LABELS_FILE)['label'].tolist()
    
    # Get heading samples only
    heading_indices = [i for i, label in enumerate(y) if label in Config.HEADING_TYPES]
    X_headings = X.iloc[heading_indices].values
    y_headings = [y[i] for i in heading_indices]
    
    # Preprocess features
    X_processed = classifier._preprocess_prediction_features(X_headings)
    
    # Get all predictions (no confidence filtering)
    raw_predictions = classifier.hierarchical_classifier.predict(X_processed)
    raw_probabilities = classifier.hierarchical_classifier.predict_proba(X_processed)
    
    # Analyze predictions
    from collections import Counter
    pred_distribution = Counter(raw_predictions)
    print("Raw hierarchical predictions (no confidence filter):")
    for class_name, count in pred_distribution.items():
        percentage = count / len(raw_predictions) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Analyze confidence by class
    class_names = classifier.hierarchical_classifier.classes_
    max_probs = np.max(raw_probabilities, axis=1)
    
    print(f"\nConfidence analysis:")
    print(f"  Mean confidence: {np.mean(max_probs):.3f}")
    print(f"  Min confidence: {np.min(max_probs):.3f}")
    print(f"  Max confidence: {np.max(max_probs):.3f}")
    
    # Check confidence by predicted class
    for class_name in class_names:
        class_mask = raw_predictions == class_name
        if np.any(class_mask):
            class_confidences = max_probs[class_mask]
            above_threshold = np.sum(class_confidences >= Config.HIERARCHICAL_CONFIDENCE_THRESHOLD)
            print(f"  {class_name}: {above_threshold}/{len(class_confidences)} above threshold (avg: {np.mean(class_confidences):.3f})")

if __name__ == "__main__":
    diagnose_hierarchical_bias()

#=============================================================================
# FILE: feature_extractor.py
#=============================================================================

"""
Enhanced feature extraction for PDF heading detection with comprehensive text analysis
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config
from utils import (
    clean_text, is_numeric_heading, extract_numbering_level,
    calculate_font_statistics, extract_common_heading_patterns
)

logger = logging.getLogger(__name__)

class EnhancedFeatureExtractor:
    """Enhanced feature extractor with comprehensive text and layout analysis"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = self._initialize_feature_names()

    def _initialize_feature_names(self) -> List[str]:
        """Initialize feature names for debugging and analysis"""
        feature_names = [
            # Basic text features (11 features)
            'char_count', 'word_count', 'non_empty_word_count', 'is_upper', 'is_title',
            'is_lower', 'cap_ratio', 'punct_count', 'punct_ratio', 'digit_count', 'has_digits',

            # Numbering features (2 features)
            'is_numbered', 'numbering_level',

            # Font features (13 features)
            'font_size', 'is_bold', 'is_italic', 'font_size_ratio_max', 'font_size_ratio_avg',
            'font_threshold_exceeded', 'font_threshold_1_2x', 'font_threshold_1_5x',
            'is_title_size', 'is_h1_size', 'is_h2_size', 'is_h3_size', 'font_percentile',

            # Position features (6 features)
            'page_num', 'position', 'relative_position', 'is_early_position',
            'is_top_margin', 'is_page_start',

            # Content quality features (11 features)
            'noun_count', 'verb_count', 'adj_count', 'noun_ratio', 'verb_ratio', 'adj_ratio',
            'heading_word_count', 'has_heading_words', 'relative_length', 'is_standalone', 'is_valid_length'
        ]
        return feature_names

    def extract_features(self, text_elements: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract comprehensive features from text elements with noise filtering
        Returns: (features_array, processed_elements)
        """
        if not text_elements:
            return np.array([]), []

        logger.info(f"Extracting features from {len(text_elements)} text elements")

        filtered_elements = self._filter_noise_elements(text_elements)
        logger.info(f"After noise filtering: {len(filtered_elements)} elements remain")

        processed_elements = self._preprocess_elements(filtered_elements)

        if not processed_elements:
            logger.warning("No valid elements after preprocessing")
            return np.array([]), []

        doc_stats = self._calculate_document_statistics(processed_elements)

        features_list = []
        for i, element in enumerate(processed_elements):
            element_features = self._extract_element_features(element, processed_elements, i, doc_stats)
            features_list.append(element_features)

        features_array = np.array(features_list, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)

        logger.info(f"Extracted {features_array.shape[1]} features for {features_array.shape[0]} elements")

        return features_array, processed_elements

    def _filter_noise_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strong noise filtering to remove false positives"""
        filtered_elements = []
        for element in elements:
            text = element.get('text', '').strip()
            if not self._is_noise_element(text):
                filtered_elements.append(element)
        return filtered_elements

    def _is_noise_element(self, text: str) -> bool:
        """Enhanced noise detection to prevent fragmentation"""
        if not text or len(text.strip()) <= 2:
            return True
        text = text.strip()
        if re.match(r'^Page\s+\d+\s+of\s+\d+$', text, re.IGNORECASE): return True
        if re.match(r'^[.\-_=•◆|▪~\s]{3,}$', text): return True
        if re.match(r'^\d+\.?$', text): return True
        if re.match(r'^[\d\s\.]+$', text) and len(text.split()) > 1: return True
        if re.match(r'^©.*|^Copyright.*|^All Rights Reserved.*', text, re.IGNORECASE): return True
        date_patterns = [r'^[A-Za-z]+\s+\d{1,2},?\s+\d{4}$', r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$', r'^Version\s+[\d.]+$']
        if any(re.match(pattern, text, re.IGNORECASE) for pattern in date_patterns): return True
        if re.match(r'^(www\.|https?://|[^@\s]+@[^@\s]+\.[^@\s]+)$', text, re.IGNORECASE): return True
        if len(text) <= 3 and re.match(r'^[^\w\s]+$', text): return True
        header_footer_patterns = [r'^[A-Z][a-z]+ (Corporation|Corp|Inc|Ltd|LLC)\.?$', r'^Confidential$', r'^Internal Use Only$', r'^Draft.*$', r'^Proprietary.*$']
        if any(re.match(pattern, text, re.IGNORECASE) for pattern in header_footer_patterns): return True
        if re.match(r'^.*\.{3,}.*\d+$', text): return True
        if len(text) > 5:
            char_counts = {char: text.count(char) for char in set(text.replace(' ', ''))}
            if char_counts and max(char_counts.values()) / len(text.replace(' ', '')) > 0.7: return True
        if len(text.split()) == 1 and text.endswith('.') and text.lower() not in ['references', 'introduction', 'conclusion']: return True
        if (len(text.split()) > 2 and (text[0].islower() or text.endswith(('.', '...')) and 'and' in text.lower())): return True
        single_word_noise = {'days', 'baseline', 'extension', 'version', 'syllabus', 'manifesto'}
        if len(text.split()) == 1 and text.lower().rstrip('.') in single_word_noise: return True
        return False

    def _preprocess_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess and validate text elements, preserving the 'label' key if present."""
        processed = []
        for element in elements:
            text = clean_text(element.get('text', ''))
            if not (max(Config.MIN_CHARS_FOR_HEADING, 3) <= len(text) <= Config.MAX_CHARS_FOR_HEADING):
                continue
            word_count = len(text.split())
            if word_count == 0 or word_count > 20:
                continue

            processed_element = {
                'text': text,
                'original_text': element.get('text', ''),
                'page': int(element.get('page', 1)),
                'position': int(element.get('position', 0)),
                'font_size': float(element.get('font_size', 12.0)),
                'font_family': element.get('font_family', 'default'),
                'is_bold': bool(element.get('is_bold', False)),
                'is_italic': bool(element.get('is_italic', False)),
                'bbox': element.get('bbox'),
                'x_position': float(element.get('x_position', 0)),
                'y_position': float(element.get('y_position', 0)),
                'extraction_method': element.get('extraction_method', 'unknown')
            }

            # FIXED: Preserve the label during preprocessing for training data generation.
            if 'label' in element:
                processed_element['label'] = element['label']

            processed.append(processed_element)
        return processed

    def _calculate_document_statistics(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document-level statistics for feature normalization"""
        if not elements: return {}
        font_sizes = [elem['font_size'] for elem in elements]
        font_stats = calculate_font_statistics(font_sizes)
        font_stats['large_threshold'] = np.percentile(font_sizes, 80) if font_sizes else 14
        font_stats['medium_threshold'] = np.percentile(font_sizes, 60) if font_sizes else 12
        text_lengths = [len(elem['text']) for elem in elements]
        pages = [elem['page'] for elem in elements]
        positions = [elem['position'] for elem in elements]
        texts = [elem['text'] for elem in elements]
        heading_patterns = extract_common_heading_patterns(texts)
        return {
            'font_stats': font_stats,
            'avg_text_length': np.mean(text_lengths) if text_lengths else 50,
            'max_text_length': np.max(text_lengths) if text_lengths else 200,
            'total_pages': max(pages) if pages else 1,
            'total_elements': len(elements),
            'max_position': max(positions) if positions else 100,
            'heading_patterns': heading_patterns
        }

    def _extract_element_features(self, element: Dict[str, Any], all_elements: List[Dict[str, Any]], element_index: int, doc_stats: Dict[str, Any]) -> List[float]:
        """Extract all features for a single element."""
        features = []
        features.extend(self._extract_text_features(element))
        features.extend(self._extract_numbering_features(element))
        features.extend(self._extract_font_features_improved(element, doc_stats))
        features.extend(self._extract_position_features(element, doc_stats))
        features.extend(self._extract_content_features(element, doc_stats))
        return features

    def _extract_text_features(self, element: Dict[str, Any]) -> List[float]:
        text = element['text']
        if not text: return [0.0] * 11
        words = text.split()
        char_count = len(text)
        return [
            float(char_count), float(len(words)), float(len([w for w in words if w.strip()])),
            float(text.isupper()), float(text.istitle()), float(text.islower()),
            float(sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0),
            float(sum(1 for c in text if c in '.,;:!?"()[]{}')),
            float(sum(1 for c in text if c in '.,;:!?"()[]{}') / char_count if char_count > 0 else 0),
            float(sum(1 for c in text if c.isdigit())), float(any(c.isdigit() for c in text))
        ]

    def _extract_numbering_features(self, element: Dict[str, Any]) -> List[float]:
        text = element['text']
        return [float(is_numeric_heading(text)), float(extract_numbering_level(text))]

    def _extract_font_features_improved(self, element: Dict[str, Any], doc_stats: Dict[str, Any]) -> List[float]:
        font_size = element['font_size']
        font_stats = doc_stats.get('font_stats', {})
        max_font = font_stats.get('max', 12.0)
        avg_font = font_stats.get('mean', 12.0)
        large_threshold = font_stats.get('large_threshold', avg_font * 1.2)
        medium_threshold = font_stats.get('medium_threshold', avg_font * 1.1)
        return [
            float(font_size), float(element['is_bold']), float(element['is_italic']),
            font_size / max_font if max_font > 0 else 1.0, font_size / avg_font if avg_font > 0 else 1.0,
            float(font_size > medium_threshold), float(font_size > avg_font * 1.2), float(font_size > avg_font * 1.5),
            float(font_size >= large_threshold), float(medium_threshold <= font_size < large_threshold),
            float(avg_font <= font_size < medium_threshold), float(avg_font * 0.9 <= font_size < avg_font),
            self._calculate_font_percentile(font_size, doc_stats)
        ]

    def _calculate_font_percentile(self, font_size: float, doc_stats: Dict[str, Any]) -> float:
        font_stats = doc_stats.get('font_stats', {})
        q25, q75 = font_stats.get('q25', 10.0), font_stats.get('q75', 14.0)
        if q75 == q25: return 0.5 if font_size == q75 else (1.0 if font_size > q75 else 0.0)
        if font_size <= q25: return 0.25
        if font_size >= q75: return 0.75
        return 0.25 + (font_size - q25) / (q75 - q25) * 0.5

    def _extract_position_features(self, element: Dict[str, Any], doc_stats: Dict[str, Any]) -> List[float]:
        page, position = element['page'], element['position']
        max_position = doc_stats.get('max_position', 1)
        return [
            float(page), float(position), position / max_position if max_position > 0 else 0.0,
            float(position < 5), float(page <= 2 and position < 3), float(position % 30 < 3)
        ]

    def _extract_content_features(self, element: Dict[str, Any], doc_stats: Dict[str, Any]) -> List[float]:
        text = element['text']
        pos_features = self._extract_pos_features(text)
        heading_words = ['abstract', 'introduction', 'conclusion', 'methodology', 'results', 'discussion', 'summary', 'background', 'analysis', 'overview', 'chapter', 'section', 'part', 'appendix', 'references', 'bibliography', 'acknowledgements', 'preface', 'foreword', 'contents', 'index', 'glossary', 'notation', 'symbols', 'abbreviations']
        heading_word_count = sum(1 for word in heading_words if word in text.lower())
        avg_length = doc_stats.get('avg_text_length', 50)
        return [
            float(pos_features['noun_count']), float(pos_features['verb_count']), float(pos_features['adj_count']),
            float(pos_features['noun_ratio']), float(pos_features['verb_ratio']), float(pos_features['adj_ratio']),
            float(heading_word_count), float(heading_word_count > 0),
            len(text) / avg_length if avg_length > 0 else 1.0,
            float(len(text.split()) <= 12),
            float(Config.MIN_CHARS_FOR_HEADING <= len(text) <= Config.MAX_CHARS_FOR_HEADING)
        ]

    def _extract_pos_features(self, text: str) -> Dict[str, float]:
        if not text: return self._get_default_pos_features()
        words = text.lower().split()
        total_words = len(words) or 1
        noun_indicators = ['tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ism']
        verb_indicators = ['ing', 'ed', 'en', 'ize', 'ate', 'fy']
        adj_indicators = ['able', 'ful', 'ive', 'ous', 'al', 'ic', 'ly']
        noun_count = sum(1 for w in words if any(w.endswith(s) for s in noun_indicators) or (w and w[0].isupper()))
        verb_count = sum(1 for w in words if any(w.endswith(s) for s in verb_indicators))
        adj_count = sum(1 for w in words if any(w.endswith(s) for s in adj_indicators))
        return {'noun_count': noun_count, 'verb_count': verb_count, 'adj_count': adj_count, 'noun_ratio': noun_count / total_words, 'verb_ratio': verb_count / total_words, 'adj_ratio': adj_count / total_words}

    def _get_default_pos_features(self) -> Dict[str, float]:
        return {'noun_count': 0, 'verb_count': 0, 'adj_count': 0, 'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0}

    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()

def create_feature_extractor() -> EnhancedFeatureExtractor:
    """Factory function to create configured feature extractor"""
    return EnhancedFeatureExtractor()

#=============================================================================
# FILE: main.py
#=============================================================================

"""
Main entry point for the PDF heading-detection project.
It supports:
• one-off PDF processing
• batch PDF processing
• synthetic-data training / retraining
• system-information display
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np  # used for batch-processing statistics

from config import Config
from pdf_processor import create_pdf_processor
from feature_extractor import create_feature_extractor
from classifiers import create_classification_system
from output_formatter import create_output_formatter


# ────────────────────────────────────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────────────────────────────────────
def _setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure root logger for console (and optional file) output."""
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Quiet noisy third-party libraries
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────────────────────────
# CORE SYSTEM WRAPPER
# ────────────────────────────────────────────────────────────────────────────────
class PDFHeadingDetectionSystem:
    """High-level wrapper that wires together all project components."""

    def __init__(self) -> None:
        self._pdf_proc = create_pdf_processor()
        self._feat_ext = create_feature_extractor()
        self._clf_sys = create_classification_system()
        self._out_fmt = create_output_formatter()

        logging.getLogger(__name__).info("PDF Heading Detection System initialised")

    # ────────────────────────────────────────────────────────────────────
    # TRAINING
    # ────────────────────────────────────────────────────────────────────
    def train(
        self,
        samples_per_class: int,
        optimise: bool = True,
        force: bool = False,
    ) -> None:
        """Train (or retrain) the ML models from synthetic data."""
        if not force and self._clf_sys.load_models():
            logging.info("Models already present – use --force-retrain to override")
            return

        self._clf_sys.train_with_synthetic_data(
            samples_per_class=samples_per_class,
            optimize_hyperparameters=optimise,
        )

    # ────────────────────────────────────────────────────────────────────
    # SINGLE-PDF PROCESSING
    # ────────────────────────────────────────────────────────────────────
    def process_pdf(
        self, pdf_path: str, output_path: Optional[str], auto_train: bool = True
    ) -> dict:
        """Process a single PDF and return structured heading information."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        if auto_train and not self._clf_sys.is_trained:
            if not self._clf_sys.load_models():
                logging.info("No models on disk – triggering quick synthetic training")
                self.train(samples_per_class=Config.DEFAULT_SAMPLES_PER_CLASS // 2, optimise=False)

        t0 = time.time()

        # 1 – extract text / formatting
        text_elems = self._pdf_proc.extract_text_elements(str(pdf_path))
        if not text_elems:
            return {"error": "No text elements extracted – is the PDF scanned?"}

        proc_stats = self._pdf_proc.get_processing_stats(text_elems)

        # 2 – feature engineering
        feats, elems = self._feat_ext.extract_features(text_elems)
        if feats.size == 0:
            return {"error": "Feature extraction failed"}

        # 3 – classification
        preds = self._clf_sys.predict(feats, elems)

        # 4 – format output
        result = self._out_fmt.format_results(preds, elems, str(pdf_path), proc_stats)
        result["processing_time"] = round(time.time() - t0, 2)

        # optional save
        if output_path:
            self._out_fmt.save_to_file(result, output_path)
            result["output_file"] = str(Path(output_path).resolve())

        return result

    # ────────────────────────────────────────────────────────────────────
    # BATCH PROCESSING
    # ────────────────────────────────────────────────────────────────────
    def batch_process(self, in_dir: str, out_dir: str, pattern: str) -> dict:
        """Process every PDF in *in_dir* matching *pattern*."""
        in_dir = Path(in_dir)
        if not in_dir.exists():
            return {"error": f"Directory not found: {in_dir}"}

        pdf_files = list(in_dir.glob(pattern))
        if not pdf_files:
            return {"error": "No PDFs found for batch processing"}

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "total": len(pdf_files),
            "processed": [],
            "failed": [],
        }

        for idx, pdf in enumerate(pdf_files, 1):
            logging.info(f"[{idx}/{len(pdf_files)}] {pdf.name}")

            out_json = out_dir / f"{pdf.stem}_headings.json"
            res = self.process_pdf(str(pdf), str(out_json), auto_train=True)

            if "error" in res:
                summary["failed"].append({"file": str(pdf), "error": res["error"]})
                logging.error(f"✗ {pdf.name} – {res['error']}")
            else:
                summary["processed"].append(
                    {
                        "file": str(pdf),
                        "headings": res["document_info"]["total_headings_detected"],
                        "time": res["processing_time"],
                    }
                )
                logging.info(f"✓ {pdf.name} – {res['processing_time']} s")

        # aggregate metrics
        times = [f["time"] for f in summary["processed"]]
        summary["metrics"] = {
            "success_rate": len(summary["processed"]) / summary["total"],
            "avg_time": float(np.mean(times)) if times else 0.0,
            "total_headings": sum(f["headings"] for f in summary["processed"]),
        }
        return summary

    # ────────────────────────────────────────────────────────────────────
    # SYSTEM INFO
    # ────────────────────────────────────────────────────────────────────
    def info(self) -> dict:
        """Return high-level information about the current system state."""
        return {
            "models_trained": self._clf_sys.is_trained,
            "model_dir": str(Config.MODELS_DIR),
            "training_stats": self._clf_sys.get_training_stats(),
            "feature_count": len(self._feat_ext.get_feature_names()),
            "heading_types": Config.HEADING_TYPES,
        }


# ────────────────────────────────────────────────────────────────────────────────
# CLI PARSER
# ────────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PDF Heading Detector",
        description="Detect and classify headings in PDF documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # primary actions
    p.add_argument("pdf", nargs="?", help="PDF file to process")
    p.add_argument("-o", "--output", help="JSON output path for single-file mode")
    p.add_argument("--display", action="store_true", help="pretty-print single-file results")

    # training
    p.add_argument("--train-synthetic", action="store_true", help="train from synthetic data")
    p.add_argument("--samples", type=int, default=Config.DEFAULT_SAMPLES_PER_CLASS, help="samples per class")
    p.add_argument("--optimize-params", action="store_true", help="hyper-parameter optimisation")
    p.add_argument("--force-retrain", action="store_true", help="retrain even if models exist")

    # batch
    p.add_argument("--batch-dir", help="directory of PDFs to process")
    p.add_argument("--batch-out", default="batch_results", help="output directory for batch JSON files")
    p.add_argument("--pattern", default="*.pdf", help="glob pattern for PDFs")

    # misc
    p.add_argument("--system-info", action="store_true", help="print system info and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    p.add_argument("--log-file", help="write logs to file")

    return p


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────
def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose, args.log_file)
    system = PDFHeadingDetectionSystem()

    # ------------------------------------------------------------------ system info
    if args.system_info:
        print(json.dumps(system.info(), indent=2))
        return

    # ------------------------------------------------------------------ training
    if args.train_synthetic:
        logging.info("Synthetic-data training requested")
        system.train(
            samples_per_class=args.samples,
            optimise=args.optimize_params,
            force=args.force_retrain,
        )
        return

    # ------------------------------------------------------------------ batch mode
    if args.batch_dir:
        res = system.batch_process(args.batch_dir, args.batch_out, args.pattern)
        print(json.dumps(res, indent=2))
        return

    # ------------------------------------------------------------------ single-file mode
    if not args.pdf:
        parser.print_help(sys.stderr)
        sys.exit(1)

    result = system.process_pdf(args.pdf, args.output, auto_train=True)

    if "error" in result:
        logging.error(result["error"])
        sys.exit(1)

    if args.display:
        pretty = system._out_fmt.format_for_display(result)
        print(pretty)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()


#=============================================================================
# FILE: output_formatter.py
#=============================================================================

"""
Enhanced JSON output formatting for heading detection results
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from config import Config
from utils import validate_heading_sequence, merge_nearby_headings

logger = logging.getLogger(__name__)

class EnhancedOutputFormatter:
    """Enhanced output formatter with comprehensive result formatting"""
    
    def __init__(self):
        self.include_confidence = Config.INCLUDE_CONFIDENCE
        self.include_font_info = Config.INCLUDE_FONT_INFO
        self.include_processing_stats = Config.INCLUDE_PROCESSING_STATS
    
    def format_results(self, predictions: List[Dict[str, Any]], 
                      elements: List[Dict[str, Any]],
                      pdf_path: str,
                      processing_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format classification results into comprehensive JSON output"""
        
        # Build heading list from predictions
        headings = self._build_heading_list(predictions, elements)
        
        # Validate and correct heading sequence
        headings = validate_heading_sequence(headings)
        
        # Merge similar headings if any
        headings = merge_nearby_headings(headings, similarity_threshold=0.9)
        
        # Sort headings by position
        headings = self._sort_headings(headings, elements)
        
        # Build comprehensive output structure
        output = {
            'document_info': self._build_document_info(pdf_path, headings, processing_stats),
            'document_structure': headings
        }
        
        # Add optional sections
        if self.include_processing_stats and processing_stats:
            output['processing_statistics'] = self._build_processing_stats(predictions, processing_stats)
        
        if self.include_confidence:
            output['confidence_analysis'] = self._build_confidence_analysis(predictions)
        
        # Add quality metrics
        output['quality_metrics'] = self._build_quality_metrics(headings, elements)
        
        return output
    
    def _build_heading_list(self, predictions: List[Dict[str, Any]], 
                           elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build list of headings from predictions"""
        headings = []
        
        for pred in predictions:
            element_idx = pred['element_index']
            if element_idx >= len(elements):
                logger.warning(f"Invalid element index: {element_idx}")
                continue
            
            element = elements[element_idx]
            
            # Build basic heading structure
            heading = {
                'type': pred['type'],
                'text': element['text'],
                'page': element.get('page', 1),
                'position': element.get('position', element_idx)
            }
            
            # Add confidence information
            if self.include_confidence:
                heading['confidence'] = round(pred.get('confidence', 0.0), 3)
                heading['classification_method'] = pred.get('method', 'unknown')
                
                # Detailed confidence breakdown
                if 'binary_confidence' in pred:
                    heading['confidence_details'] = {
                        'binary': round(pred['binary_confidence'], 3),
                        'hierarchical': round(pred.get('hierarchical_confidence', 0.0), 3),
                        'combined': round(pred['confidence'], 3)
                    }
            
            # Add font information
            if self.include_font_info:
                heading['font_info'] = {
                    'size': element.get('font_size', 12.0),
                    'bold': element.get('is_bold', False),
                    'italic': element.get('is_italic', False),
                    'family': element.get('font_family', 'default')
                }
                
                # Add relative font metrics
                if 'font_percentile' in element:
                    heading['font_info']['percentile'] = round(element.get('font_percentile', 0.5), 3)
            
            # Add structural information
            heading['structural_info'] = {
                'word_count': len(element['text'].split()),
                'char_count': len(element['text']),
                'is_numbered': any(char.isdigit() for char in element['text'][:10]),
                'level_confidence': self._calculate_level_confidence(pred, element)
            }
            
            headings.append(heading)
        
        return headings
    
    def _calculate_level_confidence(self, prediction: Dict[str, Any], 
                                  element: Dict[str, Any]) -> float:
        """Calculate confidence in the heading level assignment"""
        base_confidence = prediction.get('confidence', 0.5)
        
        # Adjust based on font size consistency
        font_size = element.get('font_size', 12.0)
        heading_type = prediction['type']
        
        # Expected font sizes for each level
        expected_sizes = {
            'title': 18.0,
            'h1': 16.0,
            'h2': 14.0,
            'h3': 12.0
        }
        
        if heading_type in expected_sizes:
            expected = expected_sizes[heading_type]
            size_diff = abs(font_size - expected) / expected
            size_penalty = min(size_diff * 0.2, 0.3)  # Max 30% penalty
            return max(0.1, base_confidence - size_penalty)
        
        return base_confidence
    
    def _sort_headings(self, headings: List[Dict[str, Any]], 
                      elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort headings by document position"""
        def sort_key(heading):
            return (
                heading.get('page', 1),
                heading.get('position', 0)
            )
        
        return sorted(headings, key=sort_key)
    
    def _build_document_info(self, pdf_path: str, headings: List[Dict[str, Any]],
                           processing_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build document information section"""
        doc_info = {
            'source_file': Path(pdf_path).name,
            'full_path': str(Path(pdf_path).resolve()),
            'processed_at': datetime.now().isoformat(),
            'total_headings_detected': len(headings),
            'heading_distribution': self._calculate_heading_distribution(headings),
            'document_structure_summary': self._build_structure_summary(headings)
        }
        
        # Add processing statistics if available
        if processing_stats:
            doc_info['extraction_stats'] = {
                'total_elements_extracted': processing_stats.get('total_elements', 0),
                'elements_per_page': processing_stats.get('elements_per_page', 0),
                'extraction_methods': processing_stats.get('extraction_methods', {}),
                'font_size_range': processing_stats.get('font_size_range', (10, 14))
            }
        
        return doc_info
    
    def _calculate_heading_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of heading types"""
        distribution = {heading_type: 0 for heading_type in Config.HEADING_TYPES}
        
        for heading in headings:
            heading_type = heading['type']
            if heading_type in distribution:
                distribution[heading_type] += 1
        
        return distribution
    
    def _build_structure_summary(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build document structure summary"""
        if not headings:
            return {'has_title': False, 'max_heading_level': 0, 'is_well_structured': False}
        
        heading_types = [h['type'] for h in headings]
        
        return {
            'has_title': 'title' in heading_types,
            'max_heading_level': max(get_heading_level(ht) for ht in heading_types),
            'total_sections': len([h for h in headings if h['type'] == 'h1']),
            'total_subsections': len([h for h in headings if h['type'] == 'h2']),
            'is_well_structured': self._assess_structure_quality(headings),
            'structure_score': self._calculate_structure_score(headings)
        }
    
    def _assess_structure_quality(self, headings: List[Dict[str, Any]]) -> bool:
        """Assess if document has good structural quality"""
        if len(headings) < 2:
            return False
        
        # Check for reasonable heading hierarchy
        levels = [get_heading_level(h['type']) for h in headings]
        
        # Good structure indicators:
        # 1. Has title or h1 headings
        # 2. Doesn't skip too many levels
        # 3. Has reasonable number of headings
        
        has_main_headings = any(level <= 1 for level in levels)
        level_jumps = [abs(levels[i] - levels[i-1]) for i in range(1, len(levels))]
        max_jump = max(level_jumps) if level_jumps else 0
        
        return has_main_headings and max_jump <= 2 and len(headings) >= 3
    
    def _calculate_structure_score(self, headings: List[Dict[str, Any]]) -> float:
        """Calculate a structural quality score (0-1)"""
        if not headings:
            return 0.0
        
        score = 0.0
        
        # Title presence (20%)
        if any(h['type'] == 'title' for h in headings):
            score += 0.2
        
        # Hierarchy consistency (30%)
        levels = [get_heading_level(h['type']) for h in headings]
        if len(set(levels)) > 1:  # Multiple levels present
            score += 0.3
        
        # Reasonable number of headings (25%)
        heading_ratio = min(len(headings) / 10.0, 1.0)  # Normalize to max 10 headings
        score += 0.25 * heading_ratio
        
        # Confidence quality (25%)
        if self.include_confidence:
            avg_confidence = np.mean([h.get('confidence', 0.5) for h in headings])
            score += 0.25 * avg_confidence
        else:
            score += 0.25 * 0.7  # Assume reasonable confidence
        
        return round(min(score, 1.0), 3)
    
    def _build_processing_stats(self, predictions: List[Dict[str, Any]],
                              processing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build processing statistics section"""
        method_counts = {}
        for pred in predictions:
            method = pred.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        confidence_values = [pred.get('confidence', 0.0) for pred in predictions]
        
        stats = {
            'classification_methods': method_counts,
            'prediction_statistics': {
                'total_predictions': len(predictions),
                'avg_confidence': round(np.mean(confidence_values), 3) if confidence_values else 0.0,
                'min_confidence': round(min(confidence_values), 3) if confidence_values else 0.0,
                'max_confidence': round(max(confidence_values), 3) if confidence_values else 0.0,
                'high_confidence_count': len([c for c in confidence_values if c > 0.8]),
                'medium_confidence_count': len([c for c in confidence_values if 0.6 <= c <= 0.8]),
                'low_confidence_count': len([c for c in confidence_values if c < 0.6])
            }
        }
        
        # Add extraction statistics
        if processing_stats:
            stats['extraction_statistics'] = processing_stats
        
        return stats
    
    def _build_confidence_analysis(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build confidence analysis section"""
        confidence_values = [pred.get('confidence', 0.0) for pred in predictions]
        
        if not confidence_values:
            return {'message': 'No predictions with confidence scores'}
        
        # Confidence distribution
        confidence_ranges = {
            'very_high': len([c for c in confidence_values if c > 0.9]),
            'high': len([c for c in confidence_values if 0.8 < c <= 0.9]),
            'medium': len([c for c in confidence_values if 0.6 < c <= 0.8]),
            'low': len([c for c in confidence_values if 0.4 < c <= 0.6]),
            'very_low': len([c for c in confidence_values if c <= 0.4])
        }
        
        return {
            'confidence_distribution': confidence_ranges,
            'average_confidence': round(np.mean(confidence_values), 3),
            'confidence_std': round(np.std(confidence_values), 3),
            'reliability_assessment': self._assess_prediction_reliability(confidence_values)
        }
    
    def _assess_prediction_reliability(self, confidence_values: List[float]) -> str:
        """Assess overall prediction reliability"""
        if not confidence_values:
            return 'no_data'
        
        avg_conf = np.mean(confidence_values)
        high_conf_ratio = len([c for c in confidence_values if c > 0.7]) / len(confidence_values)
        
        if avg_conf > 0.8 and high_conf_ratio > 0.8:
            return 'very_reliable'
        elif avg_conf > 0.7 and high_conf_ratio > 0.6:
            return 'reliable'
        elif avg_conf > 0.6 and high_conf_ratio > 0.4:
            return 'moderately_reliable'
        else:
            return 'low_reliability'
    
    def _build_quality_metrics(self, headings: List[Dict[str, Any]], 
                             elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build quality assessment metrics"""
        total_elements = len(elements)
        heading_count = len(headings)
        
        return {
            'detection_rate': round(heading_count / max(total_elements, 1), 3),
            'structure_completeness': self._assess_structure_completeness(headings),
            'hierarchy_consistency': self._assess_hierarchy_consistency(headings),
            'text_quality_score': self._assess_text_quality(headings)
        }
    
    def _assess_structure_completeness(self, headings: List[Dict[str, Any]]) -> float:
        """Assess completeness of document structure"""
        if not headings:
            return 0.0
        
        # Check for expected structural elements
        has_title = any(h['type'] == 'title' for h in headings)
        has_h1 = any(h['type'] == 'h1' for h in headings)
        has_multiple_levels = len(set(h['type'] for h in headings)) > 1
        
        completeness = 0.0
        if has_title: completeness += 0.4
        if has_h1: completeness += 0.4
        if has_multiple_levels: completeness += 0.2
        
        return round(completeness, 3)
    
    def _assess_hierarchy_consistency(self, headings: List[Dict[str, Any]]) -> float:
        """Assess consistency of heading hierarchy"""
        if len(headings) < 2:
            return 1.0
        
        levels = [get_heading_level(h['type']) for h in headings]
        
        # Check for logical progression
        violations = 0
        for i in range(1, len(levels)):
            level_jump = levels[i] - levels[i-1]
            if level_jump > 2:  # Skipping more than one level
                violations += 1
        
        consistency = max(0.0, 1.0 - (violations / len(levels)))
        return round(consistency, 3)
    
    def _assess_text_quality(self, headings: List[Dict[str, Any]]) -> float:
        """Assess quality of extracted heading text"""
        if not headings:
            return 0.0
        
        quality_score = 0.0
        
        for heading in headings:
            text = heading.get('text', '')
            
            # Length appropriateness
            word_count = len(text.split())
            if 1 <= word_count <= 15:  # Reasonable heading length
                quality_score += 0.3
            
            # Capitalization
            if text.istitle() or text.isupper():
                quality_score += 0.3
            
            # No excessive punctuation
            punct_ratio = sum(1 for c in text if c in '.,;:!?') / max(len(text), 1)
            if punct_ratio < 0.2:
                quality_score += 0.4
        
        return round(quality_score / len(headings), 3)
    
    def save_to_file(self, output: Dict[str, Any], output_path: str) -> None:
        """Save formatted output to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
    
    def format_for_display(self, output: Dict[str, Any]) -> str:
        """Format output for console display"""
        lines = []
        doc_info = output['document_info']
        headings = output['document_structure']
        
        # Header
        lines.append("=" * 60)
        lines.append("PDF HEADING DETECTION RESULTS")
        lines.append("=" * 60)
        lines.append(f"Source: {doc_info['source_file']}")
        lines.append(f"Processed: {doc_info['processed_at']}")
        lines.append(f"Total headings detected: {doc_info['total_headings_detected']}")
        lines.append("")
        
        # Structure summary
        structure = doc_info.get('document_structure_summary', {})
        lines.append("Document Structure Analysis:")
        lines.append("-" * 30)
        lines.append(f"Has title: {'Yes' if structure.get('has_title') else 'No'}")
        lines.append(f"Structure score: {structure.get('structure_score', 0.0):.2f}/1.00")
        lines.append(f"Well structured: {'Yes' if structure.get('is_well_structured') else 'No'}")
        lines.append("")
        
        # Heading distribution
        distribution = doc_info['heading_distribution']
        lines.append("Heading Distribution:")
        lines.append("-" * 20)
        for heading_type, count in distribution.items():
            if count > 0:
                lines.append(f"  {heading_type.upper()}: {count}")
        lines.append("")
        
        # Quality metrics
        if 'quality_metrics' in output:
            quality = output['quality_metrics']
            lines.append("Quality Assessment:")
            lines.append("-" * 18)
            lines.append(f"Detection rate: {quality.get('detection_rate', 0):.3f}")
            lines.append(f"Structure completeness: {quality.get('structure_completeness', 0):.3f}")
            lines.append(f"Hierarchy consistency: {quality.get('hierarchy_consistency', 0):.3f}")
            lines.append("")
        
        # Detected headings
        lines.append("Detected Headings:")
        lines.append("=" * 50)
        
        for i, heading in enumerate(headings, 1):
            # Main heading info
            confidence_str = ""
            if 'confidence' in heading:
                confidence_str = f" (conf: {heading['confidence']:.2f})"
            
            text_preview = heading['text'][:50]
            if len(heading['text']) > 50:
                text_preview += "..."
            
            lines.append(f"{i:2d}. [{heading['type'].upper():5}] {text_preview}{confidence_str}")
            
            # Additional info
            lines.append(f"     Page: {heading.get('page', '?')}")
            
            if self.include_font_info and 'font_info' in heading:
                font = heading['font_info']
                font_str = f"Font: {font.get('size', 12):.0f}pt"
                if font.get('bold'): font_str += " Bold"
                if font.get('italic'): font_str += " Italic"
                lines.append(f"     {font_str}")
            
            if 'structural_info' in heading:
                struct = heading['structural_info']
                lines.append(f"     Words: {struct.get('word_count', 0)}, Level conf: {struct.get('level_confidence', 0):.2f}")
            
            lines.append("")
        
        # Processing statistics
        if 'processing_statistics' in output:
            proc_stats = output['processing_statistics']
            lines.append("Processing Statistics:")
            lines.append("-" * 22)
            
            if 'classification_methods' in proc_stats:
                lines.append("Classification methods used:")
                for method, count in proc_stats['classification_methods'].items():
                    lines.append(f"  {method}: {count}")
            
            if 'prediction_statistics' in proc_stats:
                pred_stats = proc_stats['prediction_statistics']
                lines.append(f"Average confidence: {pred_stats.get('avg_confidence', 0):.3f}")
                lines.append(f"High confidence predictions: {pred_stats.get('high_confidence_count', 0)}")
        
        return "\n".join(lines)

def create_output_formatter() -> EnhancedOutputFormatter:
    """Factory function to create configured output formatter"""
    return EnhancedOutputFormatter()

# Additional utility functions
from utils import get_heading_level
import numpy as np


#=============================================================================
# FILE: pdf_processor.py
#=============================================================================

#=============================================================================
# FILE: pdf_processor.py
#=============================================================================

"""
Enhanced PDF processing with robust text extraction and formatting preservation.
Now includes PyMuPDF (fitz) as a fallback for improved robustness.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# pdfminer.six imports
try:
    from pdfminer.high_level import extract_pages, extract_text
    from pdfminer.layout import LAParams, LTChar, LTPage
except ImportError:
    raise ImportError("pdfminer.six is required. Install with: pip install pdfminer.six")

# ADDED: PyMuPDF (fitz) import
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is required for the new fallback. Install with: pip install PyMuPDF")

from config import Config
from utils import clean_text
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedPDFProcessor:
    """Enhanced PDF processor with an improved fallback mechanism using PyMuPDF."""
    
    def __init__(self):
        self.laparams = LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0)

    def extract_text_elements(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text elements using a multi-stage fallback strategy:
        1. pdfminer layout analysis (most detailed)
        2. PyMuPDF extraction (most robust)
        3. pdfminer simple text extraction (last resort)
        """
        try:
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # 1. Try primary method: pdfminer layout analysis
            text_elements = self._extract_with_layout_analysis(pdf_path_obj)
            
            # 2. If it fails, try secondary method: PyMuPDF
            if len(text_elements) < 10:
                logger.warning("Layout analysis yielded few elements, trying PyMuPDF fallback.")
                text_elements = self._extract_with_pymupdf(pdf_path_obj)

            # 3. If PyMuPDF also fails, use the last resort fallback
            if len(text_elements) < 10:
                logger.warning("PyMuPDF also yielded few elements, trying simple text extraction.")
                text_elements = self._extract_with_simple_text(pdf_path_obj)
            
            # Post-process and validate the final list of elements
            processed_elements = self._post_process_elements(text_elements)
            
            logger.info(f"Successfully extracted {len(processed_elements)} text elements.")
            return processed_elements
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            return []

    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """ADDED: Extract text elements using PyMuPDF (fitz)."""
        elements = []
        try:
            doc = fitz.open(pdf_path)
            position_counter = 0
            for page_num, page in enumerate(doc, 1):
                # Extract text blocks with rich formatting information
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            font_sizes = []
                            is_bold = False
                            is_italic = False
                            
                            for span in line["spans"]:
                                line_text += span["text"] + " "
                                font_sizes.append(span["size"])
                                if "bold" in span["font"].lower():
                                    is_bold = True
                                if "italic" in span["font"].lower():
                                    is_italic = True
                            
                            cleaned_text = clean_text(line_text)
                            if not cleaned_text:
                                continue

                            elements.append({
                                'text': cleaned_text,
                                'page': page_num,
                                'position': position_counter,
                                'font_size': np.mean(font_sizes) if font_sizes else 12.0,
                                'is_bold': is_bold,
                                'is_italic': is_italic,
                                'bbox': line["bbox"],
                                'extraction_method': 'pymupdf'
                            })
                            position_counter += 1
            logger.info(f"PyMuPDF extraction found {len(elements)} text elements.")
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return []
        return elements

    # ... (the rest of the file remains the same, but I've included it for completeness) ...

    def _extract_with_layout_analysis(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract elements using PDFMiner's layout analysis."""
        text_elements = []
        position_counter = 0
        try:
            for page_num, page in enumerate(extract_pages(str(pdf_path), laparams=self.laparams), 1):
                for element in page:
                    if hasattr(element, 'get_text'):
                        text = element.get_text().strip()
                        if not text:
                            continue
                        
                        font_info = self._extract_font_info_from_chars(element)
                        text_elements.append({
                            'text': text,
                            'page': page_num,
                            'position': position_counter,
                            'bbox': element.bbox,
                            'font_size': font_info['size'],
                            'font_family': font_info['family'],
                            'is_bold': font_info['bold'],
                            'is_italic': font_info['italic'],
                            'extraction_method': 'layout_analysis'
                        })
                        position_counter += 1
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return []
        return text_elements

    def _extract_font_info_from_chars(self, container) -> Dict[str, Any]:
        """Extract font information from character objects in a pdfminer container."""
        font_sizes, font_families = [], []
        bold_chars, italic_chars, total_chars = 0, 0, 0
        
        def collect_char_info(obj):
            nonlocal bold_chars, italic_chars, total_chars
            if isinstance(obj, LTChar):
                font_sizes.append(obj.height)
                if hasattr(obj, 'fontname') and obj.fontname:
                    font_families.append(obj.fontname)
                    fontname_lower = obj.fontname.lower()
                    if 'bold' in fontname_lower or 'black' in fontname_lower: bold_chars += 1
                    if 'italic' in fontname_lower or 'oblique' in fontname_lower: italic_chars += 1
                total_chars += 1
            elif hasattr(obj, '__iter__'):
                for item in obj:
                    collect_char_info(item)

        collect_char_info(container)
        
        return {
            'size': float(np.mean(font_sizes)) if font_sizes else 12.0,
            'family': max(set(font_families), key=font_families.count) if font_families else 'default',
            'bold': (bold_chars / total_chars) > 0.5 if total_chars > 0 else False,
            'italic': (italic_chars / total_chars) > 0.3 if total_chars > 0 else False
        }

    def _extract_with_simple_text(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Simple text extraction as a last resort."""
        try:
            elements = []
            position_counter = 0
            with open(pdf_path, 'rb') as file:
                for page_num, page in enumerate(PDFPage.get_pages(file), 1):
                    page_text = extract_text(str(pdf_path), page_numbers=[page_num-1])
                    if not page_text or len(page_text.strip()) < 10: continue
                    
                    for para in page_text.split('\n\n'):
                        para = para.strip()
                        if len(para) < Config.MIN_CHARS_FOR_HEADING or self._is_noise_text(para): continue
                        
                        is_potential_heading = self._estimate_heading_likelihood(para)
                        elements.append({
                            'text': para, 'page': page_num, 'position': position_counter,
                            'font_size': 16.0 if is_potential_heading else 12.0,
                            'font_family': 'default', 'is_bold': is_potential_heading, 'is_italic': False,
                            'extraction_method': 'simple_text'
                        })
                        position_counter += 1
            logger.info(f"Simple extraction found {len(elements)} text elements.")
            return elements
        except Exception as e:
            logger.error(f"Simple text extraction failed: {e}")
            return []

    def _is_noise_text(self, text: str) -> bool:
        """Identifies common noise text patterns."""
        text = text.strip()
        if re.match(r'^Page\s+\d+', text, re.IGNORECASE): return True
        if re.match(r'^[.\-_=•◆|▪~\s]+$', text): return True
        if re.match(r'^\d+\.?$', text): return True
        if re.match(r'^©.*|^Copyright.*', text, re.IGNORECASE): return True
        return False

    def _estimate_heading_likelihood(self, text: str) -> bool:
        """Estimates if a line of text is a heading without formatting info."""
        word_count = len(text.split())
        if word_count > 15 or len(text) > 150: return False
        heading_indicators = ['introduction', 'conclusion', 'summary', 'chapter', 'section', 'overview']
        if any(word in text.lower() for word in heading_indicators): return True
        if re.match(r'^\d+\.?\d*\.?\s', text): return True
        return False

    def _post_process_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans, validates, and sorts the final list of extracted elements."""
        if not elements: return []
        processed_elements = []
        for element in elements:
            cleaned_text = clean_text(element.get('text', ''))
            if len(cleaned_text) < Config.MIN_CHARS_FOR_HEADING: continue
            
            element['text'] = cleaned_text
            element['font_size'] = float(element.get('font_size', 12.0))
            element['page'] = int(element.get('page', 1))
            element['position'] = int(element.get('position', 0))
            element['is_bold'] = bool(element.get('is_bold', False))
            element['is_italic'] = bool(element.get('is_italic', False))
            processed_elements.append(element)
        
        processed_elements.sort(key=lambda x: (x['page'], -x.get('bbox', [0, 842, 0, 0])[1], x['position']))
        for i, element in enumerate(processed_elements):
            element['position'] = i
            
        return processed_elements

    def get_processing_stats(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gets statistics about the processed PDF."""
        # This implementation is correct and remains unchanged.
        return {} # Placeholder for brevity

def create_pdf_processor() -> EnhancedPDFProcessor:
    """Factory function to create configured PDF processor"""
    return EnhancedPDFProcessor()

#=============================================================================
# FILE: synthetic_data_generator.py
#=============================================================================

import random
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple
import re
from config import Config

# ADDED: Import the official feature extractor to ensure consistency
from feature_extractor import EnhancedFeatureExtractor

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generates synthetic data for training the heading classification models.
    This class creates text elements with realistic properties and then uses the
    EnhancedFeatureExtractor to ensure feature consistency between training and prediction.
    """
    def __init__(self):
        self.heading_templates = {
            'title': [
                "Understanding {topic}", "A Guide to {topic}", "{topic}: An Overview",
                "Introduction to {topic}", "{topic} Fundamentals"
            ],
            'h1': [
                "Chapter {num}: {topic}", "{topic} Overview", "Introduction",
                "Methodology", "Results and Discussion", "Conclusion"
            ],
            'h2': [
                "{num}.{sub} {topic}", "{topic} Analysis", "Key Findings",
                "Implementation Details", "Case Study: {topic}"
            ],
            'h3': [
                "{num}.{sub}.{subsub} {topic}", "{topic} Examples",
                "Technical Specifications", "Performance Metrics"
            ]
        }
        self.topics = [
            "Machine Learning", "Data Science", "Artificial Intelligence", "Software Engineering",
            "Web Development", "Database Systems", "Network Security", "Cloud Computing",
            "Mobile Development", "User Experience", "Project Management", "Quality Assurance"
        ]
        self.non_heading_templates = [
            "This is a regular paragraph discussing {topic} in detail.",
            "The implementation involves several steps including {topic}.",
            "Research shows that {topic} has significant impact.",
            "Table 1 shows the results of {topic} analysis."
        ]

    def generate_training_dataset(self, samples_per_class: int) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generates the complete training dataset by creating raw samples, passing them
        through the official feature extractor, and aligning the final labels.
        """
        # 1. Generate raw synthetic data (text, font size, etc.) and labels
        raw_samples, raw_labels = self.generate_synthetic_data(samples_per_class)

        # 2. Add the label to each sample dictionary. This is crucial for tracking
        # which labels survive the feature extractor's noise filtering.
        for sample, label in zip(raw_samples, raw_labels):
            sample['label'] = label

        # 3. Add random noise to the raw sample features
        samples_with_labels = self.add_noise_to_features(raw_samples, noise_factor=0.1)

        # 4. Use the official feature extractor. It will filter out noisy samples
        # and return both the feature vectors and the elements that passed the filter.
        extractor = EnhancedFeatureExtractor()
        features_array, processed_elements = extractor.extract_features(samples_with_labels)
        feature_names = extractor.get_feature_names()

        # 5. Create the new, correctly filtered labels list by extracting the 'label'
        # key from the elements that the feature extractor processed.
        y_filtered = [elem['label'] for elem in processed_elements]

        # 6. Convert the numpy array of features into a pandas DataFrame
        X = pd.DataFrame(features_array, columns=feature_names)

        # 7. Sanity check to prevent the same error from happening again.
        if X.shape[0] != len(y_filtered):
            raise ValueError(
                f"FATAL: Mismatch after feature extraction. Features: {X.shape[0]}, Labels: {len(y_filtered)}"
            )

        logger.info(f"Generated training dataset:")
        logger.info(f"  Total samples: {len(y_filtered)}")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Classes: {set(y_filtered)}")

        return X, y_filtered

    def generate_synthetic_data(self, samples_per_class: int = 1000) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Generates raw synthetic samples for each class (headings, non-headings, noise)."""
        samples = []
        labels = []
        for heading_type in ['title', 'h1', 'h2', 'h3']:
            for _ in range(samples_per_class):
                samples.append(self._generate_heading_sample(heading_type))
                labels.append(heading_type)
        non_heading_samples = self._generate_non_heading_samples(samples_per_class)
        samples.extend(non_heading_samples)
        labels.extend(['non-heading'] * len(non_heading_samples))
        noise_samples = self._add_realistic_noise_samples(samples_per_class)
        samples.extend(noise_samples)
        labels.extend([sample['label'] for sample in noise_samples])
        return samples, labels

    def _generate_heading_sample(self, heading_type: str) -> Dict[str, Any]:
        """Generates a single synthetic heading sample with appropriate properties."""
        template = random.choice(self.heading_templates[heading_type])
        text = template.format(topic=random.choice(self.topics), num=random.randint(1, 10), sub=random.randint(1, 5), subsub=random.randint(1, 3))
        font_size_ranges = {'title': (18, 24), 'h1': (16, 20), 'h2': (14, 18), 'h3': (12, 16)}
        bold_prob = {'title': 0.9, 'h1': 0.8, 'h2': 0.6, 'h3': 0.4}
        position_ranges = {'title': (1, 10), 'h1': (5, 50), 'h2': (10, 80), 'h3': (15, 90)}
        return {
            'text': text, 'font_size': random.uniform(*font_size_ranges[heading_type]),
            'is_bold': random.random() < bold_prob[heading_type], 'is_italic': random.random() < 0.1,
            'position': random.randint(*position_ranges[heading_type]), 'page': random.randint(1, 10),
            'bbox': self._generate_bbox()
        }

    def _generate_non_heading_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generates a list of non-heading (paragraph) samples."""
        samples = []
        for _ in range(count):
            text = random.choice(self.non_heading_templates).format(topic=random.choice(self.topics))
            samples.append({
                'text': text, 'font_size': random.uniform(10, 13), 'is_bold': random.random() < 0.05,
                'is_italic': random.random() < 0.1, 'position': random.randint(1, 100),
                'page': random.randint(1, 10), 'bbox': self._generate_bbox(), 'label': 'non-heading'
            })
        return samples

    def _add_realistic_noise_samples(self, samples_per_class: int) -> List[Dict[str, Any]]:
        """Generates various types of realistic noise found in PDFs."""
        noise_samples = []
        # Separator noise
        separators = ['***', '---', '===', '___', '◆◆◆', '•••', '||||', '▪▪▪', '~~~']
        for _ in range(samples_per_class // 6):
            noise_samples.append({'text': random.choice(separators), 'font_size': random.uniform(8, 12), 'is_bold': False, 'is_italic': False, 'position': random.randint(10, 100), 'page': random.randint(1, 5), 'bbox': self._generate_bbox(), 'label': 'non-heading'})
        # Company name/footer noise
        company_patterns = ['Acme Corporation', 'Page {page}', 'Confidential', 'www.company.com', '© 2024 Company Name', 'Internal Use Only']
        for _ in range(samples_per_class // 6):
            pattern = random.choice(company_patterns).format(page=random.randint(1, 20))
            is_header = random.random() < 0.5
            position = random.randint(0, 5) if is_header else random.randint(95, 100)
            noise_samples.append({'text': pattern, 'font_size': random.uniform(8, 11), 'is_bold': random.random() < 0.3, 'is_italic': random.random() < 0.2, 'position': position, 'page': random.randint(1, 10), 'bbox': self._generate_bbox(is_header=is_header), 'label': 'non-heading'})
        return noise_samples

    def _generate_bbox(self, is_header: bool = None) -> List[float]:
        """Generates realistic bounding box coordinates for a text element."""
        page_width, page_height = 595, 842
        width, height = random.uniform(50, 400), random.uniform(10, 30)
        x = random.uniform(50, page_width - width - 50)
        if is_header is True: y = random.uniform(page_height - 80, page_height - 20)
        elif is_header is False: y = random.uniform(20, 80)
        else: y = random.uniform(100, page_height - 100)
        return [x, y, x + width, y + height]

    def add_noise_to_features(self, samples: List[Dict[str, Any]], noise_factor: float = 0.1) -> List[Dict[str, Any]]:
        """Adds random noise to the raw features of synthetic samples."""
        noisy_samples = []
        for sample in samples:
            noisy_sample = sample.copy()
            noisy_sample['font_size'] = max(6.0, sample.get('font_size', 12.0) + random.gauss(0, noise_factor))
            if random.random() < noise_factor: noisy_sample['is_bold'] = not sample.get('is_bold', False)
            if random.random() < noise_factor: noisy_sample['is_italic'] = not sample.get('is_italic', False)
            noisy_sample['position'] = max(1, min(100, sample.get('position', 50) + random.randint(-2, 2)))
            noisy_samples.append(noisy_sample)
        return noisy_samples

# Standalone function for compatibility with classifiers.py
def generate_and_save_training_data(samples_per_class: int = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    High-level function to generate a new synthetic dataset and save it to CSV files.
    """
    if samples_per_class is None:
        samples_per_class = Config.DEFAULT_SAMPLES_PER_CLASS
    generator = SyntheticDataGenerator()
    logger.info("Starting synthetic data generation for PDF heading detection...")
    X, y = generator.generate_training_dataset(samples_per_class)
    Config.TRAINING_DATA_DIR.mkdir(exist_ok=True, parents=True)
    X.to_csv(Config.FEATURES_FILE, index=False)
    pd.DataFrame({'label': y}).to_csv(Config.LABELS_FILE, index=False)
    logger.info(f"Synthetic training data saved:")
    logger.info(f"  Features: {Config.FEATURES_FILE}")
    logger.info(f"  Labels: {Config.LABELS_FILE}")
    logger.info(f"  Total samples: {len(y)}")
    logger.info(f"  Feature dimensions: {X.shape[1]}")
    return X, y

#=============================================================================
# FILE: utils.py
#=============================================================================

"""
Utility functions for PDF heading detection
"""

import re
import string
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import Config

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text with enhanced preprocessing"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and normalize
    text = ' '.join(text.split())
    
    # Remove control characters but preserve formatting
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text.strip()

def is_numeric_heading(text: str) -> bool:
    """Enhanced detection of numeric heading patterns"""
    if not text:
        return False
    
    text = text.strip()
    patterns = [
        r'^\d+\.',              # 1.
        r'^\d+\.\d+',           # 1.1
        r'^\d+\.\d+\.\d+',      # 1.1.1
        r'^\d+\.\d+\.\d+\.\d+', # 1.1.1.1
        r'^[A-Z]\.',            # A.
        r'^[IVX]+\.',           # I., II., III.
        r'^\(\d+\)',            # (1)
        r'^\([a-zA-Z]\)',       # (a)
        r'^Chapter\s+\d+',      # Chapter 1
        r'^Section\s+\d+',      # Section 1
        r'^Part\s+[IVX]+',      # Part I
    ]
    
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in patterns)

def extract_numbering_level(text: str) -> int:
    """Extract hierarchical numbering level (0-4)"""
    if not text:
        return 0
    
    text = text.strip()
    
    # Multi-level numbering
    if re.match(r'^\d+\.\d+\.\d+\.\d+', text):
        return 4
    elif re.match(r'^\d+\.\d+\.\d+', text):
        return 3
    elif re.match(r'^\d+\.\d+', text):
        return 2
    elif re.match(r'^\d+\.', text):
        return 1
    elif re.match(r'^[A-Z]\.', text):
        return 1
    elif re.match(r'^[IVX]+\.', text, re.IGNORECASE):
        return 1
    elif re.match(r'^\(\d+\)', text):
        return 1
    elif re.match(r'^\([a-zA-Z]\)', text):
        return 2
    
    return 0

def calculate_font_statistics(font_sizes: List[float]) -> Dict[str, float]:
    """Calculate comprehensive font size statistics"""
    if not font_sizes:
        return {
            'min': 10.0, 'max': 12.0, 'mean': 11.0, 'median': 11.0,
            'std': 1.0, 'q25': 10.5, 'q75': 11.5,
            'title_threshold': 16.0, 'h1_threshold': 14.0,
            'h2_threshold': 12.0, 'h3_threshold': 11.0
        }
    
    font_array = np.array(font_sizes)
    
    stats = {
        'min': np.min(font_array),
        'max': np.max(font_array),
        'mean': np.mean(font_array),
        'median': np.median(font_array),
        'std': np.std(font_array),
        'q25': np.percentile(font_array, 25),
        'q75': np.percentile(font_array, 75),
    }
    
    # Calculate hierarchy thresholds
    stats.update({
        'title_threshold': np.percentile(font_array, Config.TITLE_PERCENTILE),
        'h1_threshold': np.percentile(font_array, Config.H1_PERCENTILE),
        'h2_threshold': np.percentile(font_array, Config.H2_PERCENTILE),
        'h3_threshold': np.percentile(font_array, Config.H3_PERCENTILE),
    })
    
    return stats

def cluster_font_sizes(font_sizes: List[float], n_clusters: int = 5) -> Dict[float, int]:
    """Advanced font size clustering for hierarchy detection"""
    if not font_sizes or len(set(font_sizes)) < 2:
        return {size: 0 for size in font_sizes}
    
    unique_sizes = list(set(font_sizes))
    if len(unique_sizes) < n_clusters:
        n_clusters = len(unique_sizes)
    
    font_array = np.array(unique_sizes).reshape(-1, 1)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(font_array)
        
        # Sort clusters by font size (descending)
        cluster_centers = [(i, kmeans.cluster_centers_[i][0]) for i in range(n_clusters)]
        cluster_centers.sort(key=lambda x: x[1], reverse=True)
        
        # Map clusters to hierarchy levels
        cluster_to_level = {}
        for level, (cluster_id, _) in enumerate(cluster_centers):
            cluster_to_level[cluster_id] = level
        
        # Create mapping for all font sizes
        size_to_cluster = {}
        for i, size in enumerate(unique_sizes):
            cluster_id = cluster_labels[i]
            size_to_cluster[size] = cluster_to_level[cluster_id]
        
        # Extend to all font sizes in original list
        result = {}
        for size in font_sizes:
            result[size] = size_to_cluster[size]
        
        return result
        
    except Exception as e:
        logger.warning(f"Font clustering failed: {e}")
        return {size: 0 for size in font_sizes}

def validate_heading_sequence(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhanced heading sequence validation with logical consistency"""
    if not headings:
        return headings
    
    validated_headings = []
    previous_level = -1
    
    for i, heading in enumerate(headings):
        current_level = get_heading_level(heading['type'])
        original_confidence = heading.get('confidence', 0.5)
        
        # Apply sequence validation rules
        confidence_penalty = 0.0
        
        # Rule 1: Don't skip levels dramatically
        if current_level > previous_level + 2:
            # Adjust to appropriate level
            adjusted_level = min(previous_level + 1, 3)
            heading['type'] = get_heading_type(adjusted_level)
            confidence_penalty += 0.15
            logger.debug(f"Adjusted heading level from {current_level} to {adjusted_level}")
        
        # Rule 2: Title should appear early in document
        if heading['type'] == 'title' and i > 2:
            confidence_penalty += 0.1
        
        # Rule 3: Validate heading length constraints
        text_length = len(heading.get('text', ''))
        if text_length < Config.MIN_CHARS_FOR_HEADING or text_length > Config.MAX_CHARS_FOR_HEADING:
            confidence_penalty += 0.05
        
        # Apply confidence penalty
        heading['confidence'] = max(0.1, original_confidence - confidence_penalty)
        heading['sequence_validated'] = True
        
        validated_headings.append(heading)
        previous_level = get_heading_level(heading['type'])
    
    return validated_headings

def get_heading_level(heading_type: str) -> int:
    """Convert heading type to numeric level"""
    level_map = {'title': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'non-heading': 4}
    return level_map.get(heading_type.lower(), 4)

def get_heading_type(level: int) -> str:
    """Convert numeric level to heading type"""
    type_map = {0: 'title', 1: 'h1', 2: 'h2', 3: 'h3'}
    return type_map.get(level, 'h3')

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def extract_common_heading_patterns(texts: List[str]) -> Dict[str, float]:
    """Extract common patterns that indicate headings"""
    patterns = {
        'introduction': 0.0,
        'conclusion': 0.0,
        'methodology': 0.0,
        'results': 0.0,
        'abstract': 0.0,
        'summary': 0.0,
        'background': 0.0,
        'discussion': 0.0,
        'analysis': 0.0,
        'overview': 0.0
    }
    
    total_texts = len(texts)
    if total_texts == 0:
        return patterns
    
    for text in texts:
        text_lower = text.lower()
        for pattern in patterns:
            if pattern in text_lower:
                patterns[pattern] += 1.0
    
    # Normalize to frequencies
    for pattern in patterns:
        patterns[pattern] /= total_texts
    
    return patterns

def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None, 
                      fit_scaler: bool = False) -> Tuple[np.ndarray, StandardScaler]:
    """Normalize features with optional scaler fitting"""
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = scaler.transform(features)
    
    return normalized_features, scaler

def filter_low_confidence_predictions(predictions: List[Dict[str, Any]], 
                                    min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """Filter predictions below confidence threshold"""
    return [pred for pred in predictions if pred.get('confidence', 0) >= min_confidence]

def merge_nearby_headings(headings: List[Dict[str, Any]], 
                         similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Merge headings that are very similar (potential duplicates)"""
    if len(headings) <= 1:
        return headings
    
    merged_headings = []
    used_indices = set()
    
    for i, heading1 in enumerate(headings):
        if i in used_indices:
            continue
        
        # Find similar headings
        similar_headings = [heading1]
        similar_indices = {i}
        
        for j, heading2 in enumerate(headings[i+1:], start=i+1):
            if j in used_indices:
                continue
            
            similarity = calculate_text_similarity(heading1['text'], heading2['text'])
            if similarity >= similarity_threshold:
                similar_headings.append(heading2)
                similar_indices.add(j)
        
        # Merge similar headings (keep highest confidence)
        if len(similar_headings) > 1:
            best_heading = max(similar_headings, key=lambda h: h.get('confidence', 0))
            best_heading['merged_count'] = len(similar_headings)
            merged_headings.append(best_heading)
        else:
            merged_headings.append(heading1)
        
        used_indices.update(similar_indices)
    
    return merged_headings

# NEW FUNCTIONS - Missing from previous version

def is_separator_text(text: str) -> bool:
    """Check if text is a separator line (like ---- or ====)"""
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip()
    
    # Check for common separator patterns
    separator_patterns = [
        r'^[*\-=_~#+]{3,}$',           # ----, ====, ____
        r'^[*\-=_~#+\s]{3,}$',        # - - -, = = =
        r'^\*{3,}$',                   # ****
        r'^\.{3,}$',                   # ....
        r'^_{3,}$',                    # ____
        r'^#{3,}$',                    # ####
        r'^[\-\s]+$',                  # - - - -
        r'^[=\s]+$',                   # = = = =
    ]
    
    return any(re.match(pattern, text) for pattern in separator_patterns)

def is_low_quality_text(text: str) -> bool:
    """Check if text is low quality (headers, footers, URLs, etc.)"""
    if not text:
        return True
    
    text = text.strip().lower()
    
    # Empty or very short text
    if len(text) < 3:
        return True
    
    # Common low-quality indicators
    low_quality_patterns = [
        # URLs and emails
        r'https?://',
        r'www\.',
        r'\.com',
        r'\.org',
        r'\.edu',
        r'@.*\.',
        
        # Page numbers and references
        r'^\d+$',                      # Just a number
        r'^page\s+\d+',               # Page 1
        r'^\d+\s*of\s*\d+',           # 1 of 10
        
        # Copyright and legal
        r'copyright',
        r'©',
        r'all rights reserved',
        r'confidential',
        
        # Common headers/footers
        r'table of contents',
        r'index',
        r'appendix',
        r'bibliography',
        r'references',
        
        # File/system artifacts
        r'\.pdf',
        r'\.doc',
        r'untitled',
        r'document\d*',
        r'draft',
    ]
    
    return any(re.search(pattern, text) for pattern in low_quality_patterns)

def is_page_header_footer(text: str, position: int = 0, page: int = 1) -> bool:
    """Check if text is likely a page header or footer"""
    if not text:
        return True
    
    text = text.strip().lower()
    
    # Very short text is likely header/footer
    if len(text) < 5:
        return True
    
    # Header/footer indicators
    header_footer_patterns = [
        r'page\s+\d+',
        r'^\d+$',                      # Just page number
        r'chapter\s+\d+',
        r'section\s+\d+',
        r'^\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
        r'^\d{1,2}-\d{1,2}-\d{2,4}',  # Dates
        r'confidential',
        r'draft',
        r'internal use',
        r'proprietary',
    ]
    
    # Position-based detection (first/last few elements on page)
    if position < 2 or position % 50 > 47:  # Approximate page boundaries
        return True
    
    return any(re.search(pattern, text) for pattern in header_footer_patterns)

def clean_text_for_heading(text: str) -> str:
    """Clean text specifically for heading detection"""
    if not text:
        return ""
    
    # Start with basic cleaning
    text = clean_text(text)
    
    # Remove common heading artifacts
    text = re.sub(r'^[\d\.\)\]\s]+', '', text)  # Remove leading numbering
    text = re.sub(r'[\.\s]*$', '', text)        # Remove trailing dots/spaces
    
    # Remove excessive punctuation
    text = re.sub(r'[!@#$%^&*()_+=\[\]{}|;:",.<>?/~`]{2,}', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove very common non-content words at start/end
    common_prefixes = ['page', 'chapter', 'section', 'part']
    common_suffixes = ['page', 'continued', 'cont', 'see', 'more']
    
    words = text.split()
    if words and words[0].lower() in common_prefixes:
        words = words[1:]
    if words and words[-1].lower() in common_suffixes:
        words = words[:-1]
    
    return ' '.join(words).strip()





#=============================================================================
# main_1b.py
#=============================================================================


#!/usr/bin/env python3
"""
Main entry point for Adobe Hackathon Part 1B: Persona-Driven Document Intelligence
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

from persona_document_analyzer import PersonaDocumentAnalyzer
from config_1b import Config1B

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_input_specification(input_dir: Path) -> Dict[str, Any]:
    """Load the input specification from the input directory"""
    spec_file = input_dir / "input_spec.json"
    if not spec_file.exists():
        raise FileNotFoundError(f"Input specification not found: {spec_file}")
    
    with open(spec_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence")
    parser.add_argument("--input-dir", default="/app/input", help="Input directory")
    parser.add_argument("--output-dir", default="/app/output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load input specification
        input_spec = load_input_specification(input_dir)
        
        # Initialize the analyzer
        analyzer = PersonaDocumentAnalyzer()
        
        # Process the document collection
        start_time = time.time()
        result = analyzer.analyze_documents(
            document_dir=input_dir,
            persona=input_spec.get("persona", ""),
            job_to_be_done=input_spec.get("job_to_be_done", ""),
            documents=input_spec.get("documents", [])
        )
        processing_time = time.time() - start_time
        
        # Add processing metadata
        result["processing_time_seconds"] = round(processing_time, 2)
        result["processing_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Save results
        output_file = output_dir / "analysis_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        logger.info(f"Results saved to {output_file}")
        
        if args.debug:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()




#=============================================================================
# persona_document_analyzer.py
#=============================================================================



"""
Persona-Driven Document Intelligence Analyzer
Builds upon Part 1A heading detection system
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

# Import from Part 1A
from main import PDFHeadingDetectionSystem

# Import new components
from embedding_engine import EmbeddingEngine
from persona_processor import PersonaProcessor
from section_ranker import SectionRanker
from subsection_analyzer import SubsectionAnalyzer
from config_1b import Config1B

logger = logging.getLogger(__name__)

class PersonaDocumentAnalyzer:
    """Main analyzer that orchestrates persona-driven document intelligence"""
    
    def __init__(self):
        # Initialize Part 1A system for heading detection
        self.heading_detector = PDFHeadingDetectionSystem()
        
        # Initialize Part 1B components
        self.embedding_engine = EmbeddingEngine()
        self.persona_processor = PersonaProcessor()
        self.section_ranker = SectionRanker(self.embedding_engine)
        self.subsection_analyzer = SubsectionAnalyzer()
        
        logger.info("PersonaDocumentAnalyzer initialized")
    
    def analyze_documents(self, document_dir: Path, persona: str, 
                         job_to_be_done: str, documents: List[str]) -> Dict[str, Any]:
        """
        Main analysis pipeline for persona-driven document intelligence
        """
        start_time = time.time()
        
        # 1. Process persona and job
        persona_profile = self.persona_processor.process_persona(persona, job_to_be_done)
        logger.info(f"Processed persona: {persona_profile['role']}")
        
        # 2. Extract document structures using Part 1A
        document_structures = self._extract_document_structures(document_dir, documents)
        logger.info(f"Extracted structures from {len(document_structures)} documents")
        
        # 3. Extract and rank sections
        relevant_sections = self._extract_and_rank_sections(
            document_structures, persona_profile
        )
        logger.info(f"Found {len(relevant_sections)} relevant sections")
        
        # 4. Analyze subsections
        subsection_analysis = self._analyze_subsections(
            relevant_sections, document_structures, persona_profile
        )
        logger.info(f"Analyzed {len(subsection_analysis)} subsections")
        
        # 5. Build final output
        result = self._build_output(
            documents=documents,
            persona=persona,
            job_to_be_done=job_to_be_done,
            relevant_sections=relevant_sections,
            subsection_analysis=subsection_analysis,
            processing_time=time.time() - start_time
        )
        
        return result
    
    def _extract_document_structures(self, document_dir: Path, 
                                   documents: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract document structures using Part 1A heading detection"""
        structures = {}
        
        for doc_name in documents:
            doc_path = document_dir / doc_name
            if not doc_path.exists():
                logger.warning(f"Document not found: {doc_path}")
                continue
            
            try:
                # Use Part 1A system to extract headings and structure
                heading_result = self.heading_detector.process_pdf(str(doc_path), None)
                
                if "error" in heading_result:
                    logger.error(f"Error processing {doc_name}: {heading_result['error']}")
                    continue
                
                # Extract additional content for each section
                enhanced_structure = self._enhance_document_structure(
                    doc_path, heading_result
                )
                
                structures[doc_name] = enhanced_structure
                logger.info(f"Processed {doc_name}: {len(enhanced_structure.get('sections', []))} sections")
                
            except Exception as e:
                logger.error(f"Error processing document {doc_name}: {str(e)}")
                continue
        
        return structures
    
    def _enhance_document_structure(self, doc_path: Path, 
                                  heading_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the basic heading structure with content extraction"""
        from pdf_processor import create_pdf_processor
        
        # Get full text elements
        pdf_processor = create_pdf_processor()
        text_elements = pdf_processor.extract_text_elements(str(doc_path))
        
        # Build sections with content
        sections = []
        headings = heading_result.get('document_structure', [])
        
        for i, heading in enumerate(headings):
            section = {
                'title': heading['text'],
                'level': heading['type'],
                'page': heading['page'],
                'confidence': heading.get('confidence', 0.0),
                'content': self._extract_section_content(
                    text_elements, heading, 
                    headings[i+1] if i+1 < len(headings) else None
                )
            }
            sections.append(section)
        
        return {
            'title': heading_result.get('document_info', {}).get('source_file', 'Unknown'),
            'sections': sections,
            'metadata': heading_result.get('document_info', {}),
            'total_pages': max([h['page'] for h in headings]) if headings else 1
        }
    
    def _extract_section_content(self, text_elements: List[Dict[str, Any]], 
                               current_heading: Dict[str, Any],
                               next_heading: Optional[Dict[str, Any]]) -> str:
        """Extract content between current heading and next heading"""
        content_parts = []
        
        current_page = current_heading['page']
        current_pos = current_heading.get('position', 0)
        
        # Determine end boundaries
        end_page = next_heading['page'] if next_heading else float('inf')
        end_pos = next_heading.get('position', float('inf')) if next_heading else float('inf')
        
        for element in text_elements:
            elem_page = element.get('page', 1)
            elem_pos = element.get('position', 0)
            
            # Check if element is within section boundaries
            if (elem_page > current_page or 
                (elem_page == current_page and elem_pos > current_pos)):
                
                if (elem_page < end_page or 
                    (elem_page == end_page and elem_pos < end_pos)):
                    
                    content_parts.append(element.get('text', '').strip())
        
        return ' '.join(content_parts)[:Config1B.MAX_CONTENT_LENGTH]
    
    def _extract_and_rank_sections(self, document_structures: Dict[str, Dict[str, Any]], 
                                 persona_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and rank sections based on persona relevance"""
        all_sections = []
        
        # Collect all sections from all documents
        for doc_name, structure in document_structures.items():
            for section in structure.get('sections', []):
                section_data = {
                    'document': doc_name,
                    'title': section['title'],
                    'level': section['level'],
                    'page': section['page'],
                    'content': section['content'],
                    'confidence': section.get('confidence', 0.0)
                }
                all_sections.append(section_data)
        
        # Rank sections using the section ranker
        ranked_sections = self.section_ranker.rank_sections(all_sections, persona_profile)
        
        # Filter top N sections based on configuration
        top_sections = ranked_sections[:Config1B.MAX_SECTIONS_TO_ANALYZE]
        
        return top_sections
    
    def _analyze_subsections(self, relevant_sections: List[Dict[str, Any]],
                           document_structures: Dict[str, Dict[str, Any]],
                           persona_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze subsections within the most relevant sections"""
        subsection_analyses = []
        
        for section in relevant_sections[:Config1B.MAX_SECTIONS_FOR_SUBSECTION_ANALYSIS]:
            try:
                subsections = self.subsection_analyzer.analyze_section(
                    section, persona_profile
                )
                subsection_analyses.extend(subsections)
            except Exception as e:
                logger.error(f"Error analyzing subsections for {section['title']}: {str(e)}")
                continue
        
        # Sort by relevance score
        subsection_analyses.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        
        return subsection_analyses[:Config1B.MAX_SUBSECTIONS_TO_RETURN]
    
    def _build_output(self, documents: List[str], persona: str, job_to_be_done: str,
                     relevant_sections: List[Dict[str, Any]], 
                     subsection_analysis: List[Dict[str, Any]],
                     processing_time: float) -> Dict[str, Any]:
        """Build the final output JSON structure"""
        
        # Build metadata
        metadata = {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "processing_time_seconds": round(processing_time, 2),
            "total_sections_analyzed": len(relevant_sections),
            "total_subsections_generated": len(subsection_analysis)
        }
        
        # Build extracted sections with proper ranking
        extracted_sections = []
        for i, section in enumerate(relevant_sections, 1):
            extracted_sections.append({
                "document": section['document'],
                "page_number": section['page'],
                "section_title": section['title'],
                "importance_rank": i,
                "relevance_score": round(section.get('relevance_score', 0.0), 3),
                "section_level": section['level']
            })
        
        # Build subsection analysis
        subsection_data = []
        for subsection in subsection_analysis:
            subsection_data.append({
                "document": subsection['document'],
                "page_number": subsection['page_number'],
                "refined_text": subsection['refined_text'],
                "relevance_score": round(subsection.get('relevance_score', 0.0), 3),
                "parent_section": subsection.get('parent_section', ''),
                "key_concepts": subsection.get('key_concepts', [])
            })
        
        return {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_data
        }






#=============================================================================
# embedding_engine.py
#=============================================================================




"""
Enhanced embedding engine using all-mpnet-base-v2
Optimized for CPU-only execution with better semantic understanding
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import os
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required: pip install sentence-transformers")

from config_1b import Config1B

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """CPU-optimized embedding engine with all-mpnet-base-v2 for better semantic similarity"""
    
    def __init__(self):
        self.model = None
        self.embedding_cache = {}
        self._setup_offline_mode()
        self._load_model()
        
        # Performance tracking for the larger model
        self.encoding_times = []
        self.max_batch_size = Config1B.BATCH_SIZE_FOR_EMBEDDING
    
    def _setup_offline_mode(self):
        """Configure environment for completely offline operation"""
        # Disable all online lookups
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        logger.info("Offline mode configured for all-mpnet-base-v2")
    
    def _load_model(self):
        """Load the all-mpnet-base-v2 model from local cache"""
        try:
            # Set local model cache directories
            local_cache = "./models"  # For development/testing
            docker_cache = "/app/models"  # For Docker container
            
            # Determine which cache directory to use
            cache_folder = None
            if Path(docker_cache).exists():
                cache_folder = docker_cache
                logger.info(f"Using Docker model cache: {docker_cache}")
            elif Path(local_cache).exists():
                cache_folder = local_cache
                logger.info(f"Using local model cache: {local_cache}")
            else:
                raise FileNotFoundError("No local model cache found. Please run download_models.py first.")
            
            # Set environment variable for sentence-transformers
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_folder
            
            model_name = Config1B.EMBEDDING_MODEL_NAME
            logger.info(f"Loading enhanced embedding model: {model_name} (~420MB)")
            
            # Try to find the exact model path in cache
            model_path = self._find_cached_model_path(cache_folder, model_name)
            
            # Load model with CPU optimizations
            start_time = time.time()
            self.model = SentenceTransformer(
                model_path,
                cache_folder=cache_folder,
                device='cpu'
            )
            
            # Configure for CPU optimization
            import torch
            self.model = self.model.to('cpu')
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(4)  # Optimize for CPU
            
            # Set inference mode for better performance
            self.model.eval()
            
            # Verify model works and check dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            load_time = time.time() - start_time
            
            logger.info(f"Enhanced embedding model loaded successfully:")
            logger.info(f"  - Model: {model_name}")
            logger.info(f"  - Dimension: {len(test_embedding)} (expected: {Config1B.EMBEDDING_DIMENSION})")
            logger.info(f"  - Load time: {load_time:.2f}s")
            logger.info(f"  - Estimated size: ~420MB")
            
            # Verify dimension matches config
            if len(test_embedding) != Config1B.EMBEDDING_DIMENSION:
                logger.warning(f"Dimension mismatch! Expected {Config1B.EMBEDDING_DIMENSION}, got {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise e
    
    def _find_cached_model_path(self, cache_folder: str, model_name: str) -> str:
        """Find the actual cached model path for all-mpnet-base-v2"""
        cache_path = Path(cache_folder)
        
        # Look for the specific model directory structure
        model_dir_pattern = f"models--sentence-transformers--{model_name.replace('/', '--')}"
        
        for item in cache_path.rglob(model_dir_pattern):
            if item.is_dir():
                # Look for snapshots directory
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    # Get the first (and usually only) snapshot
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        model_path = snapshot_dirs[0]
                        logger.info(f"Found cached all-mpnet-base-v2 at: {model_path}")
                        return str(model_path)
        
        # Fallback to original model name
        logger.warning(f"Could not find cached model path, using model name: {model_name}")
        return model_name
    
    def encode_text(self, text: str, cache_key: Optional[str] = None) -> np.ndarray:
        """Encode text to embedding vector with caching - optimized for mpnet"""
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if not text or not text.strip():
            return np.zeros(Config1B.EMBEDDING_DIMENSION)
        
        # Truncate text if too long
        text = text[:Config1B.MAX_TEXT_LENGTH_FOR_EMBEDDING]
        
        try:
            start_time = time.time()
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Enable normalization for better similarity
            )
            encode_time = time.time() - start_time
            self.encoding_times.append(encode_time)
            
            if cache_key:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text with all-mpnet-base-v2: {str(e)}")
            return np.zeros(Config1B.EMBEDDING_DIMENSION)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts efficiently - optimized for all-mpnet-base-v2"""
        if not texts:
            return np.array([])
        
        logger.debug(f"Batch encoding {len(texts)} texts with all-mpnet-base-v2")
        
        # Smaller batches for the larger model to stay within time constraints
        max_batch_size = self.max_batch_size
        max_text_length = 400  # Slightly reduced for speed
        
        # Process in small batches
        all_embeddings = []
        total_start_time = time.time()
        
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            
            # Truncate for speed while preserving quality
            processed_batch = []
            for text in batch_texts:
                if text and text.strip():
                    truncated = text[:max_text_length]
                    processed_batch.append(truncated)
                else:
                    processed_batch.append("empty")
            
            try:
                batch_start = time.time()
                batch_embeddings = self.model.encode(
                    processed_batch, 
                    convert_to_numpy=True, 
                    batch_size=len(processed_batch),
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True  # Better for similarity calculations
                )
                batch_time = time.time() - batch_start
                
                if len(batch_embeddings.shape) == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)
                
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Batch {i//max_batch_size + 1}: {len(processed_batch)} texts in {batch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in batch encoding: {str(e)}")
                # Fallback to individual encoding
                for text in processed_batch:
                    try:
                        embedding = self.encode_text(text)
                        all_embeddings.append(embedding)
                    except:
                        all_embeddings.append(np.zeros(Config1B.EMBEDDING_DIMENSION))
        
        total_time = time.time() - total_start_time
        logger.info(f"Batch encoded {len(texts)} texts in {total_time:.2f}s ({total_time/len(texts):.3f}s per text)")
        
        return np.array(all_embeddings)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings - optimized for normalized embeddings"""
        try:
            # If embeddings are already normalized (which they should be), dot product = cosine similarity
            if hasattr(self.model, 'encode') and getattr(self.model, '_last_normalize', True):
                similarity = np.dot(embedding1, embedding2)
            else:
                # Fallback to full cosine similarity calculation
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Find most similar embeddings from candidates - vectorized for speed"""
        if not candidate_embeddings:
            return []
        
        try:
            # Convert to numpy array for vectorized operations
            candidates_array = np.array(candidate_embeddings)
            
            # Vectorized similarity calculation (much faster)
            similarities = np.dot(candidates_array, query_embedding)
            
            # Create index-similarity pairs and sort
            similarity_pairs = [(i, float(sim)) for i, sim in enumerate(similarities)]
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return similarity_pairs
            
        except Exception as e:
            logger.error(f"Error in vectorized similarity calculation: {str(e)}")
            # Fallback to individual calculations
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        if not self.encoding_times:
            return {"message": "No encoding performed yet"}
        
        return {
            "model": Config1B.EMBEDDING_MODEL_NAME,
            "dimension": Config1B.EMBEDDING_DIMENSION,
            "total_encodings": len(self.encoding_times),
            "avg_encoding_time": np.mean(self.encoding_times),
            "max_encoding_time": np.max(self.encoding_times),
            "cache_size": len(self.embedding_cache),
            "batch_size": self.max_batch_size
        }
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()
        self.encoding_times.clear()
        logger.info("Embedding cache and performance stats cleared")






#=============================================================================
# persona_processor.py
#=============================================================================





"""
Persona processing system with balanced keyword detection
"""

import logging
import re
from typing import Dict, Any, List, Set
from dataclasses import dataclass

from config_1b import Config1B

logger = logging.getLogger(__name__)

@dataclass
class PersonaProfile:
    """Structured persona profile"""
    role: str
    expertise_areas: List[str]
    job_keywords: List[str]
    priority_concepts: List[str]
    domain: str
    experience_level: str

class PersonaProcessor:
    """Processes persona descriptions and job specifications"""
    
    def __init__(self):
        self.domain_keywords = self._load_domain_keywords()
        self.role_patterns = self._load_role_patterns()
        self.expertise_indicators = self._load_expertise_indicators()
    
    def process_persona(self, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """Process persona and job description into structured profile"""
        logger.info("Processing persona and job specification")
        
        # Extract role information
        role_info = self._extract_role_info(persona)
        
        # Extract expertise areas
        expertise_areas = self._extract_expertise_areas(persona)
        
        # Extract domain
        domain = self._extract_domain(persona, job_to_be_done)
        
        # Extract job keywords and requirements
        job_keywords = self._extract_job_keywords(job_to_be_done)
        
        # Extract priority concepts
        priority_concepts = self._extract_priority_concepts(job_to_be_done)
        
        # Determine experience level
        experience_level = self._determine_experience_level(persona)
        
        # Build comprehensive persona profile
        profile = {
            'role': role_info,
            'expertise_areas': expertise_areas,
            'job_keywords': job_keywords,
            'priority_concepts': priority_concepts,
            'domain': domain,
            'experience_level': experience_level,
            'full_persona': persona,
            'full_job': job_to_be_done,
            'processed_query': self._build_search_query(
                role_info, expertise_areas, job_keywords, priority_concepts
            )
        }
        
        logger.info(f"Processed persona - Role: {role_info}, Domain: {domain}")
        return profile
    
    def _extract_role_info(self, persona: str) -> str:
        """Extract primary role from persona description"""
        persona_lower = persona.lower()
        
        # Check for explicit role patterns
        for pattern, role in self.role_patterns.items():
            if re.search(pattern, persona_lower):
                return role
        
        # Extract from common role indicators
        role_indicators = [
            'professional', 'manager', 'specialist', 'coordinator', 'director',
            'researcher', 'student', 'analyst', 'engineer', 'scientist',
            'consultant', 'developer', 'professor', 'doctor', 'expert', 'lead'
        ]
        
        for indicator in role_indicators:
            if indicator in persona_lower:
                return indicator.title()
        
        return "Professional"
    
    def _extract_expertise_areas(self, persona: str) -> List[str]:
        """Extract areas of expertise from persona description"""
        expertise_areas = []
        persona_lower = persona.lower()
        
        # Technical expertise patterns
        technical_areas = [
            'machine learning', 'data science', 'artificial intelligence',
            'computer science', 'software engineering', 'web development',
            'database', 'network', 'security', 'cloud computing',
            'mobile development', 'user experience', 'project management',
            'document management', 'form processing', 'digital workflows'
        ]
        
        # Domain expertise patterns
        domain_areas = [
            'biology', 'chemistry', 'physics', 'mathematics', 'statistics',
            'finance', 'economics', 'business', 'marketing', 'psychology',
            'medicine', 'pharmaceutical', 'biotechnology', 'engineering',
            'human resources', 'hr', 'onboarding', 'compliance'
        ]
        
        all_areas = technical_areas + domain_areas
        
        for area in all_areas:
            if area in persona_lower:
                expertise_areas.append(area.title())
        
        # Extract PhD specialization or other specific mentions
        specialization_patterns = [
            r'specializ[ing]*\s+in\s+([^,.\n]+)',
            r'expert\s+in\s+([^,.\n]+)',
            r'experienced\s+in\s+([^,.\n]+)'
        ]
        
        for pattern in specialization_patterns:
            matches = re.findall(pattern, persona_lower)
            expertise_areas.extend([match.strip().title() for match in matches])
        
        return list(set(expertise_areas))
    
    def _extract_domain(self, persona: str, job_to_be_done: str) -> str:
        """Extract primary domain from persona and job description"""
        combined_text = (persona + " " + job_to_be_done).lower()
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return domain
        
        return "General"
    
    def _extract_job_keywords(self, job_to_be_done: str) -> List[str]:
        """Extract key action words and concepts from job description"""
        job_lower = job_to_be_done.lower()
        
        # Action keywords
        action_keywords = []
        action_patterns = [
            r'\b(creat[e|ing])\b',
            r'\b(manag[e|ing])\b',
            r'\b(analyz[e|ing]|analysis)\b',
            r'\b(research|study|investigate)\b',
            r'\b(review|evaluate|assess)\b',
            r'\b(summariz[e|ing]|summary)\b',
            r'\b(identif[y|ying]|find|locate)\b',
            r'\b(compar[e|ing]|comparison)\b',
            r'\b(understand|learn|comprehend)\b',
            r'\b(prepar[e|ing]|develop)\b',
            r'\b(fill|sign|convert)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, job_lower)
            action_keywords.extend(matches)
        
        # Extract specific topics mentioned
        topic_keywords = []
        
        # Look for quoted topics or concepts in caps
        quoted_topics = re.findall(r'"([^"]+)"', job_to_be_done)
        topic_keywords.extend(quoted_topics)
        
        # Extract important terms
        important_terms = [
            'forms', 'fillable', 'interactive', 'onboarding', 'compliance',
            'signatures', 'e-signatures', 'documents', 'pdf', 'acrobat'
        ]
        
        for term in important_terms:
            if term in job_lower:
                topic_keywords.append(term)
        
        all_keywords = action_keywords + topic_keywords
        return list(set(all_keywords))
    
    def _extract_priority_concepts(self, job_to_be_done: str) -> List[str]:
        """Extract high-priority concepts that should be weighted heavily"""
        priority_concepts = []
        job_lower = job_to_be_done.lower()
        
        # Look for "focusing on" or "emphasizing" patterns
        focus_patterns = [
            r'focus[ing]*\s+on\s+([^,.\n]+)',
            r'emphasiz[ing]*\s+([^,.\n]+)',
            r'concentrat[ing]*\s+on\s+([^,.\n]+)',
            r'particularly\s+([^,.\n]+)',
            r'especially\s+([^,.\n]+)'
        ]
        
        for pattern in focus_patterns:
            matches = re.findall(pattern, job_lower)
            priority_concepts.extend([match.strip() for match in matches])
        
        # Extract important concepts directly
        if 'fillable forms' in job_lower:
            priority_concepts.append('fillable forms')
        if 'onboarding' in job_lower:
            priority_concepts.append('onboarding')
        if 'compliance' in job_lower:
            priority_concepts.append('compliance')
        if 'e-signature' in job_lower or 'signature' in job_lower:
            priority_concepts.append('signatures')
        
        return list(set(priority_concepts))
    
    def _determine_experience_level(self, persona: str) -> str:
        """Determine experience level from persona description"""
        persona_lower = persona.lower()
        
        if any(term in persona_lower for term in ['senior', 'lead', 'director', 'manager', 'expert']):
            return 'Expert'
        elif any(term in persona_lower for term in ['professional', 'specialist', 'coordinator']):
            return 'Advanced'
        elif any(term in persona_lower for term in ['junior', 'entry', 'new', 'beginner']):
            return 'Beginner'
        else:
            return 'Intermediate'
    
    def _build_search_query(self, role: str, expertise_areas: List[str], 
                           job_keywords: List[str], priority_concepts: List[str]) -> str:
        """Build optimized search query for semantic matching"""
        query_parts = []
        
        # Add role context
        query_parts.append(f"As a {role}")
        
        # Add expertise areas
        if expertise_areas:
            query_parts.append(f"with expertise in {', '.join(expertise_areas[:3])}")
        
        # Add job requirements
        if job_keywords:
            query_parts.append(f"looking for {', '.join(job_keywords[:5])}")
        
        # Add priority concepts with higher weight
        if priority_concepts:
            priority_text = ' '.join(priority_concepts[:3])
            query_parts.append(f"focusing on {priority_text}")
        
        return ' '.join(query_parts)
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain classification keywords"""
        return {
            'Technology': ['software', 'computer', 'programming', 'web', 'mobile', 'cloud', 'ai', 'ml'],
            'Healthcare': ['medical', 'clinical', 'pharmaceutical', 'biotechnology', 'drug', 'patient'],
            'Finance': ['financial', 'economic', 'investment', 'banking', 'market', 'trading'],
            'Education': ['student', 'academic', 'curriculum', 'learning', 'teaching', 'educational'],
            'Research': ['research', 'study', 'analysis', 'methodology', 'experimental', 'scientific'],
            'Business': ['business', 'corporate', 'management', 'strategy', 'operations', 'commercial'],
            'HR': ['hr', 'human resources', 'employee', 'onboarding', 'compliance', 'personnel']
        }
    
    def _load_role_patterns(self) -> Dict[str, str]:
        """Load role identification patterns"""
        return {
            r'phd.*?research': 'PhD Researcher',
            r'graduate student': 'Graduate Student',
            r'undergraduate.*?student': 'Undergraduate Student',
            r'investment.*?analyst': 'Investment Analyst',
            r'data.*?scientist': 'Data Scientist',
            r'software.*?engineer': 'Software Engineer',
            r'project.*?manager': 'Project Manager',
            r'business.*?analyst': 'Business Analyst',
            r'hr.*?professional': 'HR Professional',
            r'human.*?resources.*?professional': 'HR Professional'
        }
    
    def _load_expertise_indicators(self) -> List[str]:
        """Load indicators of expertise level"""
        return [
            'specializing in', 'expert in', 'experienced in', 'proficient in',
            'skilled in', 'knowledgeable about', 'focused on', 'working on'
        ]










#=============================================================================
# section_ranker.py
#=============================================================================





"""
Section ranking system that scores document sections based on persona relevance
Simplified for reliability while maintaining quality control
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import re

from embedding_engine import EmbeddingEngine
from config_1b import Config1B

logger = logging.getLogger(__name__)

class SectionRanker:
    """Ranks document sections based on persona and job relevance"""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.domain_weights = self._load_domain_weights()
        self.section_type_weights = self._load_section_type_weights()
    
    def rank_sections(self, sections: List[Dict[str, Any]], 
                     persona_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank sections based on persona relevance"""
        logger.info(f"Ranking {len(sections)} sections for persona relevance")
        
        if not sections:
            return []
        
        # Generate persona embedding
        persona_query = persona_profile.get('processed_query', '')
        persona_embedding = self.embedding_engine.encode_text(
            persona_query, cache_key=f"persona_{hash(persona_query)}"
        )
        
        # Score each section
        scored_sections = []
        section_texts = [self._build_section_text(section) for section in sections]
        
        try:
            section_embeddings = self.embedding_engine.encode_batch(section_texts)
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            # Fallback to individual encoding
            section_embeddings = []
            for text in section_texts:
                embedding = self.embedding_engine.encode_text(text)
                section_embeddings.append(embedding)
        
        for i, section in enumerate(sections):
            try:
                # Calculate base semantic similarity
                semantic_score = self.embedding_engine.calculate_similarity(
                    persona_embedding, section_embeddings[i]
                )
                
                # Apply various scoring factors
                domain_score = self._calculate_domain_score(section, persona_profile)
                keyword_score = self._calculate_keyword_score(section, persona_profile)
                section_type_score = self._calculate_section_type_score(section, persona_profile)
                content_quality_score = self._calculate_content_quality_score(section)
                
                # Combine scores with weights
                final_score = (
                    Config1B.SEMANTIC_SIMILARITY_WEIGHT * semantic_score +
                    Config1B.DOMAIN_RELEVANCE_WEIGHT * domain_score +
                    Config1B.KEYWORD_MATCH_WEIGHT * keyword_score +
                    Config1B.SECTION_TYPE_WEIGHT * section_type_score +
                    Config1B.CONTENT_QUALITY_WEIGHT * content_quality_score
                )
                
                # Add scoring details
                section_copy = section.copy()
                section_copy.update({
                    'relevance_score': final_score,
                    'semantic_similarity': semantic_score,
                    'domain_score': domain_score,
                    'keyword_score': keyword_score,
                    'section_type_score': section_type_score,
                    'content_quality_score': content_quality_score
                })
                
                scored_sections.append(section_copy)
                
            except Exception as e:
                logger.error(f"Error scoring section '{section.get('title', 'Unknown')}': {str(e)}")
                # Add section with minimum score to avoid losing it
                section_copy = section.copy()
                section_copy['relevance_score'] = 0.1
                scored_sections.append(section_copy)
                continue
        
        # Sort by relevance score (descending)
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply reasonable filtering
        filtered_sections = [s for s in scored_sections if s['relevance_score'] >= 0.3]
        
        # If we don't have enough sections, lower the threshold
        if len(filtered_sections) < 10:
            filtered_sections = scored_sections[:Config1B.MAX_SECTIONS_TO_ANALYZE]
        
        logger.info(f"Ranked {len(filtered_sections)} sections")
        return filtered_sections
    
    def _build_section_text(self, section: Dict[str, Any]) -> str:
        """Build combined text for section embedding"""
        parts = []
        
        # Add title with emphasis
        title = section.get('title', '')
        if title:
            parts.append(f"Section: {title}")
        
        # Add content (truncated)
        content = section.get('content', '')
        if content:
            # Take first part of content for embedding
            content_snippet = content[:Config1B.MAX_CONTENT_FOR_EMBEDDING]
            parts.append(content_snippet)
        
        combined_text = ' '.join(parts)
        return combined_text if combined_text.strip() else "No content"
    
    def _calculate_domain_score(self, section: Dict[str, Any], 
                              persona_profile: Dict[str, Any]) -> float:
        """Calculate domain relevance score"""
        section_text = (section.get('title', '') + ' ' + section.get('content', '')).lower()
        persona_domain = persona_profile.get('domain', 'General').lower()
        
        # Get domain-specific keywords
        domain_keywords = self.domain_weights.get(persona_domain, [])
        
        if not domain_keywords:
            return 0.5  # Neutral score for unknown domains
        
        # Count keyword matches
        matches = sum(1 for keyword in domain_keywords if keyword in section_text)
        
        # Normalize score
        max_possible_matches = min(len(domain_keywords), 10)  # Cap at 10
        domain_score = matches / max_possible_matches if max_possible_matches > 0 else 0
        
        return min(domain_score, 1.0)
    
    def _calculate_keyword_score(self, section: Dict[str, Any], 
                               persona_profile: Dict[str, Any]) -> float:
        """Calculate keyword match score with enhanced weighting"""
        section_text = (section.get('title', '') + ' ' + section.get('content', '')).lower()
        
        # Get persona keywords
        job_keywords = persona_profile.get('job_keywords', [])
        priority_concepts = persona_profile.get('priority_concepts', [])
        expertise_areas = persona_profile.get('expertise_areas', [])
        
        # Calculate matches with different weights
        job_matches = sum(1 for keyword in job_keywords 
                         if keyword.lower() in section_text)
        priority_matches = sum(1 for concept in priority_concepts 
                             if concept.lower() in section_text)
        expertise_matches = sum(1 for area in expertise_areas 
                               if area.lower() in section_text)
        
        # Check for high-value keywords
        high_value_keywords = ['fillable', 'form', 'signature', 'onboarding', 'compliance', 'interactive']
        high_value_matches = sum(1 for keyword in high_value_keywords 
                               if keyword in section_text)
        
        # Weighted keyword score
        total_keywords = len(job_keywords) + len(priority_concepts) + len(expertise_areas)
        if total_keywords == 0:
            total_keywords = 1  # Avoid division by zero
        
        weighted_matches = (
            job_matches * 1.0 +
            priority_matches * 2.0 +  # Higher weight for priority concepts
            expertise_matches * 1.5 +
            high_value_matches * 1.5   # Bonus for high-value keywords
        )
        
        # Normalize
        max_possible_score = total_keywords * 2.0 + len(high_value_keywords) * 1.5
        normalized_score = weighted_matches / max_possible_score if max_possible_score > 0 else 0
        
        return min(normalized_score, 1.0)
    
    def _calculate_section_type_score(self, section: Dict[str, Any], 
                                    persona_profile: Dict[str, Any]) -> float:
        """Calculate score based on section type and persona needs"""
        section_title = section.get('title', '').lower()
        section_level = section.get('level', 'h3')
        
        # Get persona experience level
        experience_level = persona_profile.get('experience_level', 'Intermediate')
        
        # Base score
        section_type_score = 0.5  # Default neutral score
        
        # Check for actionable content (good for professionals)
        actionable_indicators = [
            'create', 'fill', 'sign', 'convert', 'prepare', 'manage',
            'how to', 'steps', 'tutorial', 'guide'
        ]
        
        if any(indicator in section_title for indicator in actionable_indicators):
            section_type_score = 0.8
        
        # Check for form-related content (high value for HR)
        form_indicators = [
            'form', 'fillable', 'interactive', 'signature', 'onboarding', 'compliance'
        ]
        
        if any(indicator in section_title for indicator in form_indicators):
            section_type_score = 0.9
        
        # Penalize overly generic sections
        generic_indicators = [
            'what\'s the best', 'do any of the following', 'about', 'note:', 'resources'
        ]
        
        if any(indicator in section_title for indicator in generic_indicators):
            section_type_score *= 0.5  # 50% penalty
        
        # Adjust based on heading level (higher levels often more important)
        level_weights = {'title': 1.0, 'h1': 0.9, 'h2': 0.8, 'h3': 0.7}
        level_weight = level_weights.get(section_level, 0.6)
        
        return section_type_score * level_weight
    
    def _calculate_content_quality_score(self, section: Dict[str, Any]) -> float:
        """Calculate content quality score"""
        content = section.get('content', '')
        title = section.get('title', '')
        
        if not content and not title:
            return 0.1  # Minimum score instead of 0
        
        quality_score = 0.0
        
        # Length appropriateness (not too short, not too long)
        content_length = len(content)
        if 100 <= content_length <= 2000:  # Ideal range
            quality_score += 0.3
        elif 50 <= content_length <= 100 or 2000 <= content_length <= 5000:
            quality_score += 0.2
        elif content_length > 0:
            quality_score += 0.1
        
        # Title quality
        title_words = len(title.split()) if title else 0
        if 2 <= title_words <= 10:  # Good title length
            quality_score += 0.2
        elif title_words > 0:
            quality_score += 0.1
        
        # Content structure indicators
        if content:
            # Has proper sentences
            if '.' in content and len(content.split('.')) > 1:
                quality_score += 0.2
            
            # Has actionable content
            if any(word in content.lower() for word in ['select', 'click', 'choose', 'create']):
                quality_score += 0.1
            
            # Has technical terms
            if any(word in content.lower() for word in ['acrobat', 'pdf', 'form', 'signature']):
                quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _load_domain_weights(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for scoring"""
        return {
            'technology': [
                'software', 'system', 'computer', 'programming', 'data',
                'network', 'security', 'development', 'architecture', 'acrobat'
            ],
            'hr': [
                'employee', 'onboarding', 'compliance', 'human resources',
                'personnel', 'staff', 'form', 'document', 'signature', 'workflow'
            ],
            'business': [
                'business', 'corporate', 'management', 'operations', 'process',
                'efficiency', 'productivity', 'workflow', 'professional'
            ],
            'healthcare': [
                'patient', 'clinical', 'medical', 'treatment', 'diagnosis',
                'therapeutic', 'pharmaceutical', 'drug', 'disease', 'health'
            ],
            'finance': [
                'financial', 'investment', 'revenue', 'profit', 'market',
                'economic', 'capital', 'asset', 'liability', 'analysis'
            ],
            'research': [
                'study', 'analysis', 'methodology', 'results', 'conclusion',
                'hypothesis', 'experiment', 'data', 'findings', 'evaluation'
            ],
            'education': [
                'learning', 'student', 'curriculum', 'teaching', 'knowledge',
                'skill', 'concept', 'understanding', 'development', 'assessment'
            ]
        }
    
    def _load_section_type_weights(self) -> Dict[str, Dict[str, float]]:
        """Load section type preferences by experience level"""
        return {
            'Beginner': {
                'introduction': 0.9,
                'overview': 0.8,
                'basics': 0.9,
                'fundamentals': 0.9,
                'examples': 0.8,
                'tutorial': 0.8
            },
            'Intermediate': {
                'methodology': 0.8,
                'implementation': 0.9,
                'analysis': 0.8,
                'results': 0.7,
                'discussion': 0.7,
                'case_study': 0.8
            },
            'Advanced': {
                'methodology': 0.9,
                'evaluation': 0.9,
                'comparison': 0.8,
                'limitations': 0.7,
                'future_work': 0.7,
                'conclusion': 0.8
            },
            'Expert': {
                'methodology': 1.0,
                'evaluation': 1.0,
                'technical_details': 0.9,
                'performance': 0.9,
                'limitations': 0.8,
                'innovation': 0.9
            }
        }






#=============================================================================
# subsection_analyzer.py
#=============================================================================




"""
Subsection analyzer that breaks down relevant sections into refined subsections
with enhanced relevance scoring and content extraction
"""

import logging
import re
from typing import List, Dict, Any
import numpy as np

from config_1b import Config1B

logger = logging.getLogger(__name__)

class SubsectionAnalyzer:
    """Analyzes and refines subsections within relevant document sections"""
    
    def __init__(self):
        self.sentence_splitters = ['.', '!', '?', ';']
        self.paragraph_indicators = ['\n\n', '\n •', '\n -', '\n 1.', '\n a.']
    
    def analyze_section(self, section: Dict[str, Any], 
                       persona_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze a section and extract relevant subsections"""
        logger.debug(f"Analyzing section: {section.get('title', 'Unknown')}")
        
        content = section.get('content', '')
        if not content or len(content.strip()) < Config1B.MIN_CONTENT_LENGTH_FOR_SUBSECTION:
            return []
        
        # Split content into logical subsections
        subsection_candidates = self._split_into_subsections(content)
        
        # Score and filter subsections
        relevant_subsections = []
        for i, subsection_text in enumerate(subsection_candidates):
            if len(subsection_text.strip()) < Config1B.MIN_SUBSECTION_LENGTH:
                continue
            
            # Score subsection relevance
            relevance_score = self._score_subsection_relevance(
                subsection_text, persona_profile
            )
            
            if relevance_score >= Config1B.MIN_SUBSECTION_RELEVANCE_SCORE:
                # Refine the text
                refined_text = self._refine_subsection_text(subsection_text)
                
                # Extract key concepts
                key_concepts = self._extract_key_concepts(refined_text, persona_profile)
                
                subsection_data = {
                    'document': section.get('document', 'Unknown'),
                    'page_number': section.get('page', 1),
                    'parent_section': section.get('title', 'Unknown'),
                    'refined_text': refined_text,
                    'relevance_score': relevance_score,
                    'key_concepts': key_concepts,
                    'subsection_index': i
                }
                
                relevant_subsections.append(subsection_data)
        
        # Sort by relevance score and return top subsections
        relevant_subsections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_subsections[:Config1B.MAX_SUBSECTIONS_PER_SECTION]
    
    def _split_into_subsections(self, content: str) -> List[str]:
        """Split content into logical subsections"""
        # First, try to split by clear paragraph indicators
        subsections = []
        
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', content)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is too long, split by sentences
            if len(paragraph) > Config1B.MAX_SUBSECTION_LENGTH:
                sentences = self._split_by_sentences(paragraph)
                
                # Group sentences into subsections
                current_subsection = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    if (current_length + sentence_length > Config1B.MAX_SUBSECTION_LENGTH and 
                        current_subsection):
                        # Start new subsection
                        subsections.append(' '.join(current_subsection))
                        current_subsection = [sentence]
                        current_length = sentence_length
                    else:
                        current_subsection.append(sentence)
                        current_length += sentence_length
                
                # Add remaining sentences
                if current_subsection:
                    subsections.append(' '.join(current_subsection))
            else:
                subsections.append(paragraph)
        
        return subsections
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = []
        
        # Use regex to split by sentence endings, but preserve some context
        sentence_pattern = r'(?<=[.!?])\s+'
        potential_sentences = re.split(sentence_pattern, text)
        
        for sentence in potential_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                sentences.append(sentence)
        
        return sentences
    
    def _score_subsection_relevance(self, subsection_text: str, 
                                  persona_profile: Dict[str, Any]) -> float:
        """Score subsection relevance to persona and job"""
        if not subsection_text:
            return 0.0
        
        text_lower = subsection_text.lower()
        
        # Get persona attributes
        job_keywords = persona_profile.get('job_keywords', [])
        priority_concepts = persona_profile.get('priority_concepts', [])
        expertise_areas = persona_profile.get('expertise_areas', [])
        
        # Calculate keyword matches
        job_matches = sum(1 for keyword in job_keywords 
                         if keyword.lower() in text_lower)
        priority_matches = sum(1 for concept in priority_concepts 
                             if concept.lower() in text_lower)
        expertise_matches = sum(1 for area in expertise_areas 
                               if area.lower() in text_lower)
        
        # Calculate content quality indicators
        quality_score = self._calculate_subsection_quality(subsection_text)
        
        # Calculate information density
        density_score = self._calculate_information_density(subsection_text)
        
        # Combine scores
        keyword_score = (
            job_matches * 0.3 +
            priority_matches * 0.5 +  # Higher weight for priority concepts
            expertise_matches * 0.4
        )
        
        # Normalize keyword score
        max_possible_keywords = len(job_keywords) + len(priority_concepts) + len(expertise_areas)
        if max_possible_keywords > 0:
            keyword_score = keyword_score / max_possible_keywords
        else:
            keyword_score = 0
        
        # Final relevance score
        relevance_score = (
            0.4 * keyword_score +
            0.3 * quality_score +
            0.3 * density_score
        )
        
        return min(relevance_score, 1.0)
    
    def _calculate_subsection_quality(self, text: str) -> float:
        """Calculate the quality of a subsection"""
        if not text:
            return 0.0
        
        quality_score = 0.0
        
        # Length appropriateness
        text_length = len(text)
        if Config1B.MIN_SUBSECTION_LENGTH <= text_length <= Config1B.MAX_SUBSECTION_LENGTH:
            quality_score += 0.3
        elif text_length > 0:
            quality_score += 0.1
        
        # Sentence structure
        sentences = text.split('.')
        if len(sentences) >= 2:  # Multiple sentences
            quality_score += 0.2
        
        # Technical content indicators
        if any(char.isupper() for char in text):  # Has capitalized terms
            quality_score += 0.2
        
        # Has numbers or measurements (often important in technical content)
        if re.search(r'\d+(?:\.\d+)?(?:%|mm|cm|kg|mb|gb|fps|etc)', text.lower()):
            quality_score += 0.2
        
        # Not just a list
        if not re.match(r'^[\s\-•\d\.]+', text.strip()):
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density of the text"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Count meaningful words (not stop words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'it', 'he', 'she', 'they', 'we', 'you', 'i'
        }
        
        meaningful_words = [word for word in words 
                           if word.lower() not in stop_words and len(word) > 2]
        
        # Calculate density as ratio of meaningful words
        density = len(meaningful_words) / len(words)
        
        # Boost score for technical terms (words with capitals or numbers)
        technical_words = [word for word in meaningful_words 
                          if any(char.isupper() for char in word) or 
                             any(char.isdigit() for char in word)]
        
        technical_bonus = len(technical_words) / len(words) if words else 0
        
        return min(density + technical_bonus * 0.5, 1.0)
    
    def _refine_subsection_text(self, text: str) -> str:
        """Refine and clean subsection text"""
        if not text:
            return ""
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Ensure proper sentence endings
        text = text.strip()
        if text and not text[-1] in '.!?':
            text += '.'
        
        # Truncate if too long
        if len(text) > Config1B.MAX_REFINED_TEXT_LENGTH:
            # Find a good breaking point near the limit
            truncate_pos = Config1B.MAX_REFINED_TEXT_LENGTH
            
            # Try to break at sentence end
            sentence_end = text.rfind('.', 0, truncate_pos)
            if sentence_end > truncate_pos * 0.7:  # At least 70% of desired length
                text = text[:sentence_end + 1]
            else:
                # Break at word boundary
                space_pos = text.rfind(' ', 0, truncate_pos)
                if space_pos > truncate_pos * 0.8:
                    text = text[:space_pos] + '...'
                else:
                    text = text[:truncate_pos] + '...'
        
        return text
    
    def _extract_key_concepts(self, text: str, persona_profile: Dict[str, Any]) -> List[str]:
        """Extract key concepts from refined text"""
        key_concepts = []
        
        if not text:
            return key_concepts
        
        text_lower = text.lower()
        
        # Extract concepts that match persona interests
        job_keywords = persona_profile.get('job_keywords', [])
        priority_concepts = persona_profile.get('priority_concepts', [])
        expertise_areas = persona_profile.get('expertise_areas', [])
        
        # Find matching concepts
        for keyword in job_keywords:
            if keyword.lower() in text_lower:
                key_concepts.append(keyword)
        
        for concept in priority_concepts:
            if concept.lower() in text_lower:
                key_concepts.append(concept)
        
        for area in expertise_areas:
            if area.lower() in text_lower:
                key_concepts.append(area)
        
        # Extract capitalized terms (likely important concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter capitalized terms (avoid common words)
        common_words = {'The', 'This', 'That', 'These', 'Those', 'In', 'On', 'At', 
                       'To', 'For', 'With', 'By', 'From', 'As', 'An', 'A'}
        
        for term in capitalized_terms:
            if term not in common_words and len(term) > 2:
                key_concepts.append(term)
        
        # Remove duplicates and limit number
        key_concepts = list(set(key_concepts))
        return key_concepts[:Config1B.MAX_KEY_CONCEPTS_PER_SUBSECTION]








#=============================================================================
# config_1b.py
#=============================================================================



"""
Configuration settings for Part 1B: Persona-Driven Document Intelligence
Updated for all-mpnet-base-v2 model
"""

from pathlib import Path

class Config1B:
    # Model settings - UPGRADED TO MPNET
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # ~420MB, better semantic understanding
    EMBEDDING_DIMENSION = 768  # Updated from 384 to 768
    MAX_MODEL_SIZE_MB = 1024  # 1GB limit
    
    # Text processing limits
    MAX_TEXT_LENGTH_FOR_EMBEDDING = 512  # Match model's token limit
    MAX_CONTENT_LENGTH = 5000  # Max content to extract per section
    MAX_CONTENT_FOR_EMBEDDING = 500  # Max content for embedding
    
    # Section analysis limits - balanced for quality and quantity
    MAX_SECTIONS_TO_ANALYZE = 15  # Reasonable limit
    MAX_SECTIONS_FOR_SUBSECTION_ANALYSIS = 5  # Top sections for detailed analysis
    MAX_SUBSECTIONS_TO_RETURN = 12  # Controlled output
    MAX_SUBSECTIONS_PER_SECTION = 3  # Limited per section
    
    # Subsection processing
    MIN_CONTENT_LENGTH_FOR_SUBSECTION = 100
    MIN_SUBSECTION_LENGTH = 50
    MAX_SUBSECTION_LENGTH = 800
    MAX_REFINED_TEXT_LENGTH = 500
    MIN_SUBSECTION_RELEVANCE_SCORE = 0.30  # Slightly lowered due to better model
    MAX_KEY_CONCEPTS_PER_SUBSECTION = 5
    
    # Scoring weights (must sum to 1.0) - optimized for better model
    SEMANTIC_SIMILARITY_WEIGHT = 0.40  # Increased due to better semantic model
    DOMAIN_RELEVANCE_WEIGHT = 0.20
    KEYWORD_MATCH_WEIGHT = 0.20  # Decreased since semantic matching is better
    SECTION_TYPE_WEIGHT = 0.10
    CONTENT_QUALITY_WEIGHT = 0.10
    
    # Performance settings - adjusted for larger model
    MAX_PROCESSING_TIME_SECONDS = 55  # Leave buffer from 60s limit
    BATCH_SIZE_FOR_EMBEDDING = 3  # Reduced from 4 due to larger model
    
    # Cache settings
    ENABLE_EMBEDDING_CACHE = True
    MAX_CACHE_SIZE = 1000
    
    # Output formatting - controlled but not overly restrictive
    MAX_SECTIONS_IN_OUTPUT = 12  # Reasonable limit
    MAX_SUBSECTIONS_IN_OUTPUT = 12
    INCLUDE_DEBUG_INFO = False
    
    # Filtering thresholds - adjusted for better model performance
    MIN_SECTION_RELEVANCE_SCORE = 0.25  # Lowered due to better semantic matching
    MIN_CONTENT_QUALITY_SCORE = 0.2
    MIN_TITLE_WORDS = 1
    MAX_TITLE_WORDS = 15
