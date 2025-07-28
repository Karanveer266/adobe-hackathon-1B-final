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
