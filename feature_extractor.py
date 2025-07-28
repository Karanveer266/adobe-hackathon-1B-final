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