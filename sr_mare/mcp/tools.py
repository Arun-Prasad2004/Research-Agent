"""
MCP Tool implementations for SR-MARE components.

This module wraps existing SR-MARE components as MCP tools.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from sr_mare.retrieval.embedder import OllamaEmbedder
from sr_mare.retrieval.vector_store import FAISSVectorStore
from sr_mare.evaluation.uncertainty import UncertaintyEstimator
from sr_mare.evaluation.metrics import ResearchMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPTools:
    """Wrapper class for SR-MARE components as MCP tools."""
    
    def __init__(
        self,
        embedder: OllamaEmbedder,
        vector_store: FAISSVectorStore,
        uncertainty_estimator: UncertaintyEstimator,
        metrics: ResearchMetrics
    ):
        """
        Initialize MCP tools.
        
        Args:
            embedder: Ollama embedder instance
            vector_store: FAISS vector store instance
            uncertainty_estimator: Uncertainty estimator instance
            metrics: Research metrics instance
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.uncertainty_estimator = uncertainty_estimator
        self.metrics = metrics
        
        logger.info("🔧 MCP Tools initialized")
    
    def retrieve_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        This tool wraps the embedder and vector store for retrieval.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        logger.info(f"🔍 MCP Tool: Retrieving context (k={k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=k)
            
            # Format results
            documents = []
            for doc, score, metadata in results:
                documents.append({
                    "text": doc,
                    "similarity_score": float(score),
                    "metadata": metadata
                })
            
            return {
                "query": query,
                "documents": documents,
                "num_retrieved": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def evaluate_confidence(
        self,
        answer: str,
        hypotheses: List[str],
        retrieved_docs: List[Dict[str, Any]],
        quality_score: float
    ) -> Dict[str, Any]:
        """
        Evaluate confidence in an answer.
        
        This tool wraps the uncertainty estimator.
        
        Args:
            answer: Generated answer
            hypotheses: Alternative hypotheses
            retrieved_docs: Retrieved documents (dict format)
            quality_score: Quality score from critic
            
        Returns:
            Dictionary containing confidence metrics
        """
        logger.info("📊 MCP Tool: Evaluating confidence")
        
        try:
            # Convert retrieved docs back to tuple format
            docs_tuples = [
                (doc["text"], doc["similarity_score"], doc.get("metadata", {}))
                for doc in retrieved_docs
            ]
            
            # Compute confidence
            confidence_metrics = self.uncertainty_estimator.compute_confidence_score(
                answer,
                hypotheses,
                docs_tuples,
                quality_score
            )
            
            return confidence_metrics
            
        except Exception as e:
            logger.error(f"Confidence evaluation failed: {e}")
            raise
    
    def store_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Store documents in vector store.
        
        This tool wraps document embedding and storage.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            Dictionary with storage results
        """
        logger.info(f"💾 MCP Tool: Storing {len(documents)} documents")
        
        try:
            # Generate embeddings
            embeddings = self.embedder.embed_batch(documents)
            
            # Add to vector store
            self.vector_store.add_documents(documents, embeddings, metadata)
            
            return {
                "num_stored": len(documents),
                "total_documents": len(self.vector_store.documents),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Document storage failed: {e}")
            raise
    
    def score_retrieval_quality(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Score quality of retrieved documents.
        
        This tool wraps retrieval metrics computation.
        
        Args:
            retrieved_docs: Retrieved documents (dict format)
            
        Returns:
            Dictionary containing retrieval metrics
        """
        logger.info("📈 MCP Tool: Scoring retrieval quality")
        
        try:
            # Convert to tuple format
            docs_tuples = [
                (doc["text"], doc["similarity_score"], doc.get("metadata", {}))
                for doc in retrieved_docs
            ]
            
            # Compute metrics
            metrics = self.metrics.compute_retrieval_metrics(docs_tuples)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Retrieval scoring failed: {e}")
            raise
    
    def compute_self_consistency(
        self,
        hypotheses: List[str]
    ) -> Dict[str, Any]:
        """
        Compute self-consistency score for hypotheses.
        
        Args:
            hypotheses: List of hypothesis strings
            
        Returns:
            Dictionary with consistency score
        """
        logger.info("🎯 MCP Tool: Computing self-consistency")
        
        try:
            score = self.uncertainty_estimator.compute_self_consistency_score(hypotheses)
            
            return {
                "self_consistency_score": float(score),
                "num_hypotheses": len(hypotheses)
            }
            
        except Exception as e:
            logger.error(f"Self-consistency computation failed: {e}")
            raise
    
    def compute_evidence_diversity(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute diversity of evidence sources.
        
        Args:
            retrieved_docs: Retrieved documents (dict format)
            
        Returns:
            Dictionary with diversity score
        """
        logger.info("🌈 MCP Tool: Computing evidence diversity")
        
        try:
            # Convert to tuple format
            docs_tuples = [
                (doc["text"], doc["similarity_score"], doc.get("metadata", {}))
                for doc in retrieved_docs
            ]
            
            score = self.uncertainty_estimator.compute_evidence_diversity_score(docs_tuples)
            
            return {
                "evidence_diversity_score": float(score),
                "num_documents": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Evidence diversity computation failed: {e}")
            raise
