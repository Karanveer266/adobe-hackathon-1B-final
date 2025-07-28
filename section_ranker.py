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
