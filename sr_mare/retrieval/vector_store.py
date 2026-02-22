"""
Vector store module using FAISS for similarity search.
"""

import faiss
import numpy as np
import logging
from typing import List, Tuple, Optional
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, dimension: int = 768):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadata: List[dict] = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: Optional[List[dict]] = None):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: Numpy array of embeddings (n_docs, dimension)
            metadata: Optional metadata for each document
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match store dimension {self.dimension}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (document, similarity_score, metadata)
        """
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Ensure query is 2D and normalized
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Convert L2 distance to similarity score (0-1)
            # After normalization, L2 distance ranges from 0 to 2
            similarity = 1 - (distance / 2.0)
            similarity = max(0.0, min(1.0, similarity))
            
            results.append((
                self.documents[idx],
                float(similarity),
                self.metadata[idx]
            ))
        
        return results
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the store
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path_obj / "index.faiss"))
        
        # Save documents and metadata
        with open(path_obj / "store_data.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str):
        """
        Load the vector store from disk.
        
        Args:
            path: Directory path containing the saved store
        """
        path_obj = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path_obj / "index.faiss"))
        
        # Load documents and metadata
        with open(path_obj / "store_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
        
        logger.info(f"Vector store loaded from {path}. Contains {len(self.documents)} documents")
    
    def clear(self):
        """Clear all documents from the store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        logger.info("Vector store cleared")
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_size": self.index.ntotal
        }
