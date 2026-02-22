"""
Embedder module for generating text embeddings using Ollama's nomic-embed-text model.
"""

import requests
import logging
from typing import List, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Generate embeddings using Ollama's nomic-embed-text model."""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        Initialize the embedder.
        
        Args:
            model: Name of the embedding model to use
            base_url: Base URL for Ollama API
        """
        self.model = model
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embeddings"
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
            
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(self.embed_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            embedding = np.array(result["embedding"], dtype=np.float32)
            
            return embedding
            
        except requests.exceptions.RequestException as e:
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
        embeddings = []
        
        for i, text in enumerate(texts):
            logger.info(f"Embedding text {i+1}/{len(texts)}")
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def test_connection(self) -> bool:
        """
        Test if Ollama API is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_text = "test"
            self.embed_text(test_text)
            logger.info("✓ Embedder connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Embedder connection failed: {e}")
            return False
