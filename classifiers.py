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
        self.BINARY_THRESHOLD = 0.50 # Increased from default
        self.HIERARCHICAL_THRESHOLD = 0.50  # Increased from default
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