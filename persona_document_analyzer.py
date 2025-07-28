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
        }
        
        # Build extracted sections with proper ranking
        extracted_sections = []
        for i, section in enumerate(relevant_sections[:5], 1):
            extracted_sections.append({
                "document": section['document'],
                "section_title": section['title'],
                "importance_rank": i,
                "page_number" : section['page']
            })
        
        # Build subsection analysis
        subsection_data = []
        for subsection in subsection_analysis[:5]:
            subsection_data.append({
                "document": subsection['document'],
                "page_number": subsection['page_number'],
                "refined_text": subsection['refined_text'],
            })
        
        return {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_data
        }


        