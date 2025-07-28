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