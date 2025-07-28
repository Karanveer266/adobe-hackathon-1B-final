#!/usr/bin/env python3
"""
Model download script for Adobe Hackathon Part 1B
Downloads all-mpnet-base-v2 for offline use in Docker container
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_offline_environment():
    """Setup environment variables for offline operation"""
    offline_vars = {
        'HF_HUB_OFFLINE': '0',  # Allow download during build
        'TRANSFORMERS_OFFLINE': '0',
        'HF_DATASETS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_DISABLE_PROGRESS_BARS': '0',  # Show progress during download
        'HF_HUB_DISABLE_SYMLINKS_WARNING': '1'
    }
    
    for var, value in offline_vars.items():
        os.environ[var] = value
        logger.info(f"Set {var}={value}")

def download_models():
    """Download all required models for the hackathon"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    
    # Setup cache directory
    models_dir = Path("/app/models")  # Docker path
    if not models_dir.exists():
        models_dir = Path("./models")  # Local development path
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Set cache directory
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(models_dir)
    
    logger.info(f"Using models directory: {models_dir}")
    
    # Model to download
    model_name = "all-mpnet-base-v2"
    
    logger.info(f"Starting download of {model_name} (~420MB)")
    logger.info("This may take a few minutes depending on your internet connection...")
    
    try:
        # Download and cache the model
        model = SentenceTransformer(
            model_name,
            cache_folder=str(models_dir),
            device='cpu'
        )
        
        # Test the model to ensure it works
        test_text = "This is a test sentence to verify the model works correctly."
        test_embedding = model.encode(test_text)
        
        logger.info(f"✅ Successfully downloaded {model_name}")
        logger.info(f"   - Embedding dimension: {len(test_embedding)}")
        logger.info(f"   - Cache location: {models_dir}")
        logger.info(f"   - Estimated size: ~420MB")
        
        # Verify model files exist
        model_files = list(models_dir.rglob("*"))
        logger.info(f"   - Total files downloaded: {len(model_files)}")
        
        # Set environment for offline use after download
        offline_post_download = {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1'
        }
        
        for var, value in offline_post_download.items():
            os.environ[var] = value
        
        logger.info("🔒 Offline mode enabled for production use")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {str(e)}")
        return False

def verify_download():
    """Verify that the model was downloaded correctly"""
    models_dir = Path("/app/models") if Path("/app/models").exists() else Path("./models")
    
    # Look for model files
    model_pattern = "*all-mpnet-base-v2*"
    model_files = list(models_dir.rglob(model_pattern))
    
    if model_files:
        logger.info(f"✅ Model files found: {len(model_files)} files")
        
        # Check for key model files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        found_files = []
        
        for model_file in model_files:
            if model_file.name in required_files:
                found_files.append(model_file.name)
        
        logger.info(f"   - Key files found: {found_files}")
        
        if len(found_files) >= 2:  # At least config and model files
            logger.info("✅ Model appears to be downloaded correctly")
            return True
        else:
            logger.warning("⚠️  Some model files may be missing")
            return False
    else:
        logger.error("❌ No model files found")
        return False

def main():
    """Main function"""
    logger.info("🚀 Adobe Hackathon Model Download Script")
    logger.info("Downloading all-mpnet-base-v2 for enhanced semantic similarity")
    
    # Setup environment
    setup_offline_environment()
    
    # Download models
    success = download_models()
    
    if success:
        # Verify download
        verify_success = verify_download()
        
        if verify_success:
            logger.info("🎉 Model download completed successfully!")
            logger.info("Your system is ready for offline operation.")
            logger.info("You can now build your Docker container.")
        else:
            logger.error("❌ Model verification failed")
            sys.exit(1)
    else:
        logger.error("❌ Model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
