"""
Uncertainty estimation module for assessing answer confidence.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Estimate uncertainty and confidence in generated answers."""
    
    def __init__(self):
        """Initialize the uncertainty estimator."""
        pass
    
    def compute_self_consistency_score(self, hypotheses: List[str]) -> float:
        """
        Compute agreement between multiple hypotheses using text similarity.
        
        Args:
            hypotheses: List of hypothesis strings
            
        Returns:
            Self-consistency score (0-1), higher means more agreement
        """
        if len(hypotheses) < 2:
            return 1.0
        
        # Use token-level overlap as a proxy for semantic similarity
        token_sets = [set(self._tokenize(h)) for h in hypotheses]
        
        # Compute pairwise Jaccard similarities
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Average similarity as consistency score
        consistency = np.mean(similarities)
        
        logger.debug(f"Self-consistency score: {consistency:.3f}")
        return float(consistency)
    
    def compute_evidence_diversity_score(
        self,
        retrieved_docs: List[Tuple[str, float, dict]]
    ) -> float:
        """
        Compute diversity of retrieved evidence.
        
        Args:
            retrieved_docs: List of (document, similarity_score, metadata) tuples
            
        Returns:
            Evidence diversity score (0-1), higher means more diverse sources
        """
        if len(retrieved_docs) == 0:
            return 0.0
        
        if len(retrieved_docs) == 1:
            return 0.3
        
        # Extract vocabulary from documents
        doc_tokens = [set(self._tokenize(doc)) for doc, _, _ in retrieved_docs]
        
        # Compute pairwise diversity (inverse of similarity)
        diversities = []
        for i in range(len(doc_tokens)):
            for j in range(i + 1, len(doc_tokens)):
                intersection = len(doc_tokens[i] & doc_tokens[j])
                union = len(doc_tokens[i] | doc_tokens[j])
                if union > 0:
                    # Diversity is inverse of similarity
                    diversity = 1.0 - (intersection / union)
                    diversities.append(diversity)
        
        if not diversities:
            return 0.5
        
        diversity_score = np.mean(diversities)
        
        logger.debug(f"Evidence diversity score: {diversity_score:.3f}")
        return float(diversity_score)
    
    def compute_token_entropy(self, text: str) -> float:
        """
        Compute token-level entropy as a proxy for model uncertainty.
        
        Args:
            text: Generated text
            
        Returns:
            Normalized entropy score (0-1)
        """
        tokens = self._tokenize(text)
        
        if len(tokens) == 0:
            return 0.5
        
        # Compute token frequency distribution
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Compute Shannon entropy
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total_tokens
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(token_counts)) if len(token_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        logger.debug(f"Token entropy: {normalized_entropy:.3f}")
        return float(normalized_entropy)
    
    def compute_confidence_score(
        self,
        answer: str,
        hypotheses: List[str],
        retrieved_docs: List[Tuple[str, float, dict]],
        critic_score: float
    ) -> Dict[str, float]:
        """
        Compute overall confidence score combining multiple uncertainty metrics.
        
        Args:
            answer: The generated answer
            hypotheses: Alternative hypotheses
            retrieved_docs: Retrieved evidence documents
            critic_score: Quality score from critic agent (0-1)
            
        Returns:
            Dictionary containing individual scores and final confidence
        """
        logger.info("📊 Computing confidence metrics...")
        
        # Compute individual metrics
        self_consistency = self.compute_self_consistency_score(hypotheses)
        evidence_diversity = self.compute_evidence_diversity_score(retrieved_docs)
        token_entropy = self.compute_token_entropy(answer)
        
        # Compute retrieval quality (average similarity of top docs)
        retrieval_quality = 0.5
        if retrieved_docs:
            top_scores = [score for _, score, _ in retrieved_docs[:3]]
            retrieval_quality = np.mean(top_scores) if top_scores else 0.5
        
        # Weighted combination for final confidence
        # Higher critic score = higher confidence
        # Higher consistency = higher confidence
        # Higher retrieval quality = higher confidence
        # Higher diversity = slightly lower confidence (more uncertainty)
        # Moderate entropy is good (too high = uncertain, too low = repetitive)
        
        entropy_factor = 1.0 - abs(token_entropy - 0.6)  # Penalize extreme entropy
        
        weights = {
            "critic_score": 0.35,
            "self_consistency": 0.25,
            "retrieval_quality": 0.20,
            "entropy_factor": 0.10,
            "evidence_diversity": 0.10
        }
        
        final_confidence = (
            weights["critic_score"] * critic_score +
            weights["self_consistency"] * self_consistency +
            weights["retrieval_quality"] * retrieval_quality +
            weights["entropy_factor"] * entropy_factor +
            weights["evidence_diversity"] * evidence_diversity
        )
        
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        metrics = {
            "final_confidence": float(final_confidence),
            "critic_quality_score": float(critic_score),
            "self_consistency_score": float(self_consistency),
            "evidence_diversity_score": float(evidence_diversity),
            "token_entropy": float(token_entropy),
            "retrieval_quality": float(retrieval_quality)
        }
        
        logger.info(f"✓ Final confidence score: {final_confidence:.3f}")
        
        return metrics
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for text analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter out very short tokens and common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'being', 'to', 'of', 'in', 'for', 'on', 'with'}
        tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
        
        return tokens
    
    def assess_answer_stability(self, answers: List[str]) -> float:
        """
        Assess stability across multiple answer generations.
        
        Args:
            answers: List of answer strings from multiple runs
            
        Returns:
            Stability score (0-1), higher means more stable
        """
        if len(answers) < 2:
            return 1.0
        
        consistency = self.compute_self_consistency_score(answers)
        return consistency
