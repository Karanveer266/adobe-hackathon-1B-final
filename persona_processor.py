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
