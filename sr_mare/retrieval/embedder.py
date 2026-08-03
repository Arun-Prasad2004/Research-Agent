"""
Embedder module for generating text embeddings using sentence-transformers.
"""

import logging
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2", base_url: str = None):
        """
        Initialize the embedder.
        
        Args:
            model: Name of the sentence-transformer model to use
            base_url: Ignored, kept for compatibility
        """
        self.model_name = model
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers to use LocalEmbedder: pip install sentence-transformers")
        
        logger.info(f"Loading embedding model {model}...")
        self.model = SentenceTransformer(model)
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise Exception(f"Embedding generation failed: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts...")
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise Exception(f"Batch embedding generation failed: {e}")
    
    def test_connection(self) -> bool:
        """
        Test if embedder is working.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            test_text = "test"
            self.embed_text(test_text)
            logger.info("✓ Embedder loaded successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Embedder failed: {e}")
            return False
