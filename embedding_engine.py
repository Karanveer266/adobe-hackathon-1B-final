"""
Enhanced embedding engine using all-mpnet-base-v2
Optimized for CPU-only execution with better semantic understanding
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import os
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required: pip install sentence-transformers")

from config_1b import Config1B

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """CPU-optimized embedding engine with all-mpnet-base-v2 for better semantic similarity"""
    
    def __init__(self):
        self.model = None
        self.embedding_cache = {}
        self._setup_offline_mode()
        self._load_model()
        
        # Performance tracking for the larger model
        self.encoding_times = []
        self.max_batch_size = Config1B.BATCH_SIZE_FOR_EMBEDDING
    
    def _setup_offline_mode(self):
        """Configure environment for completely offline operation"""
        # Disable all online lookups
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        logger.info("Offline mode configured for all-mpnet-base-v2")
    
    def _load_model(self):
        """Load the all-mpnet-base-v2 model from local cache"""
        try:
            # Set local model cache directories
            local_cache = "./models"  # For development/testing
            docker_cache = "/app/models"  # For Docker container
            
            # Determine which cache directory to use
            cache_folder = None
            if Path(docker_cache).exists():
                cache_folder = docker_cache
                logger.info(f"Using Docker model cache: {docker_cache}")
            elif Path(local_cache).exists():
                cache_folder = local_cache
                logger.info(f"Using local model cache: {local_cache}")
            else:
                raise FileNotFoundError("No local model cache found. Please run download_models.py first.")
            
            # Set environment variable for sentence-transformers
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_folder
            
            model_name = Config1B.EMBEDDING_MODEL_NAME
            logger.info(f"Loading enhanced embedding model: {model_name} (~420MB)")
            
            # Try to find the exact model path in cache
            model_path = self._find_cached_model_path(cache_folder, model_name)
            
            # Load model with CPU optimizations
            start_time = time.time()
            self.model = SentenceTransformer(
                model_path,
                cache_folder=cache_folder,
                device='cpu'
            )
            
            # Configure for CPU optimization
            import torch
            self.model = self.model.to('cpu')
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(4)  # Optimize for CPU
            
            # Set inference mode for better performance
            self.model.eval()
            
            # Verify model works and check dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            load_time = time.time() - start_time
            
            logger.info(f"Enhanced embedding model loaded successfully:")
            logger.info(f"  - Model: {model_name}")
            logger.info(f"  - Dimension: {len(test_embedding)} (expected: {Config1B.EMBEDDING_DIMENSION})")
            logger.info(f"  - Load time: {load_time:.2f}s")
            logger.info(f"  - Estimated size: ~420MB")
            
            # Verify dimension matches config
            if len(test_embedding) != Config1B.EMBEDDING_DIMENSION:
                logger.warning(f"Dimension mismatch! Expected {Config1B.EMBEDDING_DIMENSION}, got {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise e
    
    def _find_cached_model_path(self, cache_folder: str, model_name: str) -> str:
        """Find the actual cached model path for all-mpnet-base-v2"""
        cache_path = Path(cache_folder)
        
        # Look for the specific model directory structure
        model_dir_pattern = f"models--sentence-transformers--{model_name.replace('/', '--')}"
        
        for item in cache_path.rglob(model_dir_pattern):
            if item.is_dir():
                # Look for snapshots directory
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    # Get the first (and usually only) snapshot
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        model_path = snapshot_dirs[0]
                        logger.info(f"Found cached all-mpnet-base-v2 at: {model_path}")
                        return str(model_path)
        
        # Fallback to original model name
        logger.warning(f"Could not find cached model path, using model name: {model_name}")
        return model_name
    
    def encode_text(self, text: str, cache_key: Optional[str] = None) -> np.ndarray:
        """Encode text to embedding vector with caching - optimized for mpnet"""
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if not text or not text.strip():
            return np.zeros(Config1B.EMBEDDING_DIMENSION)
        
        # Truncate text if too long
        text = text[:Config1B.MAX_TEXT_LENGTH_FOR_EMBEDDING]
        
        try:
            start_time = time.time()
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Enable normalization for better similarity
            )
            encode_time = time.time() - start_time
            self.encoding_times.append(encode_time)
            
            if cache_key:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text with all-mpnet-base-v2: {str(e)}")
            return np.zeros(Config1B.EMBEDDING_DIMENSION)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts efficiently - optimized for all-mpnet-base-v2"""
        if not texts:
            return np.array([])
        
        logger.debug(f"Batch encoding {len(texts)} texts with all-mpnet-base-v2")
        
        # Smaller batches for the larger model to stay within time constraints
        max_batch_size = self.max_batch_size
        max_text_length = 400  # Slightly reduced for speed
        
        # Process in small batches
        all_embeddings = []
        total_start_time = time.time()
        
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            
            # Truncate for speed while preserving quality
            processed_batch = []
            for text in batch_texts:
                if text and text.strip():
                    truncated = text[:max_text_length]
                    processed_batch.append(truncated)
                else:
                    processed_batch.append("empty")
            
            try:
                batch_start = time.time()
                batch_embeddings = self.model.encode(
                    processed_batch, 
                    convert_to_numpy=True, 
                    batch_size=len(processed_batch),
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True  # Better for similarity calculations
                )
                batch_time = time.time() - batch_start
                
                if len(batch_embeddings.shape) == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)
                
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Batch {i//max_batch_size + 1}: {len(processed_batch)} texts in {batch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in batch encoding: {str(e)}")
                # Fallback to individual encoding
                for text in processed_batch:
                    try:
                        embedding = self.encode_text(text)
                        all_embeddings.append(embedding)
                    except:
                        all_embeddings.append(np.zeros(Config1B.EMBEDDING_DIMENSION))
        
        total_time = time.time() - total_start_time
        logger.info(f"Batch encoded {len(texts)} texts in {total_time:.2f}s ({total_time/len(texts):.3f}s per text)")
        
        return np.array(all_embeddings)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings - optimized for normalized embeddings"""
        try:
            # If embeddings are already normalized (which they should be), dot product = cosine similarity
            if hasattr(self.model, 'encode') and getattr(self.model, '_last_normalize', True):
                similarity = np.dot(embedding1, embedding2)
            else:
                # Fallback to full cosine similarity calculation
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Find most similar embeddings from candidates - vectorized for speed"""
        if not candidate_embeddings:
            return []
        
        try:
            # Convert to numpy array for vectorized operations
            candidates_array = np.array(candidate_embeddings)
            
            # Vectorized similarity calculation (much faster)
            similarities = np.dot(candidates_array, query_embedding)
            
            # Create index-similarity pairs and sort
            similarity_pairs = [(i, float(sim)) for i, sim in enumerate(similarities)]
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return similarity_pairs
            
        except Exception as e:
            logger.error(f"Error in vectorized similarity calculation: {str(e)}")
            # Fallback to individual calculations
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        if not self.encoding_times:
            return {"message": "No encoding performed yet"}
        
        return {
            "model": Config1B.EMBEDDING_MODEL_NAME,
            "dimension": Config1B.EMBEDDING_DIMENSION,
            "total_encodings": len(self.encoding_times),
            "avg_encoding_time": np.mean(self.encoding_times),
            "max_encoding_time": np.max(self.encoding_times),
            "cache_size": len(self.embedding_cache),
            "batch_size": self.max_batch_size
        }
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()
        self.encoding_times.clear()
        logger.info("Embedding cache and performance stats cleared")
