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
