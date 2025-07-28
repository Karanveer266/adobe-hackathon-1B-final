#=============================================================================
# FILE: main_1b.py
#=============================================================================
import json
import logging
from pathlib import Path
import argparse  # Ensure argparse is imported
import time

from persona_document_analyzer import PersonaDocumentAnalyzer
from config_1b import Config1B

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_input_specification(spec_file: Path) -> dict[str, any]:
    """Load the input specification from the provided JSON file path."""
    if not spec_file.exists():
        raise FileNotFoundError(f"Input specification not found: {spec_file}")
    
    with open(spec_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence")
    parser.add_argument("--input-dir", required=True, help="Path to the collection directory (e.g., 'Challenge_1b/Collection 2')")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # The input dir is the collection directory itself.
    collection_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # FIXED: Look for the JSON file directly in the collection directory.
        input_spec_path = collection_dir / "challenge1b_input.json"
        input_spec = load_input_specification(input_spec_path)
        
        # Define the path to the PDFs subfolder
        pdf_dir = collection_dir / "PDFs"
        
        # Initialize the analyzer
        analyzer = PersonaDocumentAnalyzer()
        
        # Process the document collection
        start_time = time.time()
        result = analyzer.analyze_documents(
            document_dir=pdf_dir,
            persona=input_spec["persona"]["role"],
            job_to_be_done=input_spec["job_to_be_done"]["task"],
            documents=[doc["filename"] for doc in input_spec["documents"]]
        )
        processing_time = time.time() - start_time
        
        # Add processing metadata to the final output
        result["metadata"]["processing_time_seconds"] = round(processing_time, 2)
        
        # Save results
        output_file = output_dir / "challenge1b_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        logger.info(f"Results saved to {output_file}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()