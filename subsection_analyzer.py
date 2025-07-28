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
                    'refined_text': refined_text,
                    'relevance_score': relevance_score,
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
