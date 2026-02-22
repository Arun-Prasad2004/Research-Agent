"""
Metrics module for evaluating research system performance.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchMetrics:
    """Compute and track research-grade evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.experiment_history = []
    
    def compute_iteration_improvement(
        self,
        iteration_scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute improvement metrics across refinement iterations.
        
        Args:
            iteration_scores: List of quality scores from each iteration
            
        Returns:
            Dictionary containing improvement metrics
        """
        if len(iteration_scores) < 2:
            return {
                "total_improvement": 0.0,
                "avg_improvement_per_iteration": 0.0,
                "converged": False
            }
        
        improvements = []
        for i in range(1, len(iteration_scores)):
            improvement = iteration_scores[i] - iteration_scores[i-1]
            improvements.append(improvement)
        
        total_improvement = iteration_scores[-1] - iteration_scores[0]
        avg_improvement = np.mean(improvements)
        
        # Check if converged (improvement < threshold)
        converged = abs(improvements[-1]) < 0.05 if improvements else False
        
        return {
            "total_improvement": float(total_improvement),
            "avg_improvement_per_iteration": float(avg_improvement),
            "final_score": float(iteration_scores[-1]),
            "initial_score": float(iteration_scores[0]),
            "converged": converged,
            "num_iterations": len(iteration_scores)
        }
    
    def compute_retrieval_metrics(
        self,
        retrieved_docs: List[tuple],
        relevance_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compute retrieval quality metrics.
        
        Args:
            retrieved_docs: List of (document, similarity_score, metadata) tuples
            relevance_threshold: Threshold for considering a document relevant
            
        Returns:
            Dictionary containing retrieval metrics
        """
        if not retrieved_docs:
            return {
                "num_retrieved": 0,
                "num_relevant": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0
            }
        
        similarities = [score for _, score, _ in retrieved_docs]
        relevant_count = sum(1 for score in similarities if score >= relevance_threshold)
        
        return {
            "num_retrieved": len(retrieved_docs),
            "num_relevant": relevant_count,
            "relevance_ratio": relevant_count / len(retrieved_docs),
            "avg_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities))
        }
    
    def compute_confidence_calibration(
        self,
        confidence_score: float,
        actual_quality: float
    ) -> Dict[str, float]:
        """
        Assess calibration between predicted confidence and actual quality.
        
        Args:
            confidence_score: Predicted confidence (0-1)
            actual_quality: Actual quality score from critic (0-1)
            
        Returns:
            Calibration metrics
        """
        calibration_error = abs(confidence_score - actual_quality)
        
        # Check if confidence is well-calibrated (within 0.15)
        well_calibrated = calibration_error < 0.15
        
        # Check if overconfident or underconfident
        if confidence_score > actual_quality + 0.1:
            calibration_status = "overconfident"
        elif confidence_score < actual_quality - 0.1:
            calibration_status = "underconfident"
        else:
            calibration_status = "well-calibrated"
        
        return {
            "calibration_error": float(calibration_error),
            "well_calibrated": well_calibrated,
            "calibration_status": calibration_status,
            "confidence_score": float(confidence_score),
            "actual_quality": float(actual_quality)
        }
    
    def log_experiment(
        self,
        question: str,
        result: Dict[str, Any],
        confidence_metrics: Dict[str, float],
        iteration_metrics: Dict[str, Any]
    ):
        """
        Log experiment results for analysis.
        
        Args:
            question: Research question
            result: Full result dictionary
            confidence_metrics: Confidence computation results
            iteration_metrics: Iteration improvement metrics
        """
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "final_confidence": confidence_metrics.get("final_confidence", 0.0),
            "iterations": iteration_metrics.get("num_iterations", 0),
            "total_improvement": iteration_metrics.get("total_improvement", 0.0),
            "converged": iteration_metrics.get("converged", False),
            "critic_score": confidence_metrics.get("critic_quality_score", 0.0)
        }
        
        self.experiment_history.append(experiment)
        logger.info(f"📝 Experiment logged: confidence={experiment['final_confidence']:.3f}, iterations={experiment['iterations']}")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all logged experiments.
        
        Returns:
            Dictionary containing aggregate statistics
        """
        if not self.experiment_history:
            return {"message": "No experiments logged yet"}
        
        confidences = [exp["final_confidence"] for exp in self.experiment_history]
        iterations = [exp["iterations"] for exp in self.experiment_history]
        improvements = [exp["total_improvement"] for exp in self.experiment_history]
        
        return {
            "total_experiments": len(self.experiment_history),
            "avg_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "avg_iterations": float(np.mean(iterations)),
            "avg_improvement": float(np.mean([i for i in improvements if i is not None])),
            "convergence_rate": sum(exp["converged"] for exp in self.experiment_history) / len(self.experiment_history)
        }
    
    def compute_answer_length_metrics(self, answer: str) -> Dict[str, int]:
        """
        Compute basic length metrics for an answer.
        
        Args:
            answer: The answer text
            
        Returns:
            Dictionary with length metrics
        """
        words = answer.split()
        sentences = answer.split('.')
        
        return {
            "num_words": len(words),
            "num_sentences": len(sentences),
            "num_characters": len(answer),
            "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0
        }
