"""
Orchestrator for coordinating the multi-agent research pipeline.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from sr_mare.agents.planner import PlannerAgent
from sr_mare.agents.analyst import AnalystAgent
from sr_mare.agents.critic import CriticAgent
from sr_mare.agents.refiner import RefinerAgent
from sr_mare.retrieval.embedder import OllamaEmbedder
from sr_mare.retrieval.vector_store import FAISSVectorStore
from sr_mare.evaluation.uncertainty import UncertaintyEstimator
from sr_mare.evaluation.metrics import ResearchMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates the complete multi-agent research pipeline."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_iterations: int = 3,
        confidence_threshold: float = 0.75,
        vector_store_path: str = None
    ):
        """
        Initialize the research orchestrator.
        
        Args:
            base_url: Base URL for Ollama API
            max_iterations: Maximum refinement iterations
            confidence_threshold: Minimum confidence to stop refinement
            vector_store_path: Path to load existing vector store
        """
        logger.info("🚀 Initializing SR-MARE Research Orchestrator...")
        
        # Initialize agents
        self.planner = PlannerAgent(model="mistral", base_url=base_url)
        self.analyst = AnalystAgent(model="mistral", base_url=base_url)
        self.critic = CriticAgent(model="llama3.2", base_url=base_url)
        self.refiner = RefinerAgent(model="llama3.2", base_url=base_url)
        
        # Initialize retrieval components
        self.embedder = OllamaEmbedder(model="nomic-embed-text", base_url=base_url)
        self.vector_store = FAISSVectorStore(dimension=768)
        
        # Initialize evaluation components
        self.uncertainty_estimator = UncertaintyEstimator()
        self.metrics = ResearchMetrics()
        
        # Configuration
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
        # Load existing vector store if provided
        if vector_store_path and Path(vector_store_path).exists():
            self.vector_store.load(vector_store_path)
            logger.info(f"✓ Loaded vector store from {vector_store_path}")
        
        # Experiment log
        self.experiment_log = []
        
        logger.info("✓ Orchestrator initialized")
    
    def test_connections(self) -> bool:
        """
        Test all component connections.
        
        Returns:
            True if all connections successful
        """
        logger.info("🔌 Testing connections...")
        
        success = True
        success &= self.embedder.test_connection()
        success &= self.planner.test_connection()
        success &= self.critic.test_connection()
        
        if success:
            logger.info("✓ All connections successful")
        else:
            logger.error("✗ Some connections failed")
        
        return success
    
    def load_documents(self, documents: List[str]):
        """
        Load documents into the vector store.
        
        Args:
            documents: List of document texts to index
        """
        logger.info(f"📚 Loading {len(documents)} documents into vector store...")
        
        if len(documents) == 0:
            logger.warning("No documents to load")
            return
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(documents)
        
        # Add to vector store
        metadata = [{"doc_id": i, "loaded_at": datetime.now().isoformat()} 
                   for i in range(len(documents))]
        self.vector_store.add_documents(documents, embeddings, metadata)
        
        logger.info(f"✓ Loaded {len(documents)} documents")
    
    def retrieve_context(self, question: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Retrieve relevant context for a question.
        
        Args:
            question: The research question
            k: Number of documents to retrieve
            
        Returns:
            List of (document, similarity_score, metadata) tuples
        """
        logger.info(f"🔍 Retrieving top {k} relevant documents...")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(question)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        logger.info(f"✓ Retrieved {len(results)} documents")
        return results
    
    def research(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the complete research pipeline for a question.
        
        Args:
            question: The research question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing complete research results
        """
        logger.info("=" * 70)
        logger.info(f"🔬 STARTING RESEARCH: {question}")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Step 1: Planning
        logger.info("\n[STEP 1] Planning...")
        plan = self.planner.plan(question)
        
        # Step 2: Retrieval
        logger.info("\n[STEP 2] Retrieving relevant context...")
        retrieved_docs = self.retrieve_context(question, k=top_k)
        
        # Compute retrieval metrics
        retrieval_metrics = self.metrics.compute_retrieval_metrics(retrieved_docs)
        
        # Step 3: Initial analysis
        logger.info("\n[STEP 3] Generating initial analysis...")
        analysis = self.analyst.analyze_with_context(question, retrieved_docs, plan)
        
        current_answer = analysis["answer"]
        hypotheses = analysis["hypotheses"]
        
        # Iterative refinement loop
        iteration_history = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"\n[ITERATION {iteration}] Evaluating and refining...")
            
            # Step 4: Critique
            critique = self.critic.critique(question, current_answer, hypotheses)
            quality_score = critique.get("quality_score", 0.5)
            
            # Step 5: Compute uncertainty metrics
            confidence_metrics = self.uncertainty_estimator.compute_confidence_score(
                current_answer,
                hypotheses,
                retrieved_docs,
                quality_score
            )
            
            current_confidence = confidence_metrics["final_confidence"]
            
            # Log iteration
            iteration_history.append({
                "iteration": iteration,
                "quality_score": quality_score,
                "confidence_score": current_confidence,
                "critique": critique,
                "answer": current_answer
            })
            
            logger.info(f"  Quality: {quality_score:.3f} | Confidence: {current_confidence:.3f}")
            
            # Check if we should stop
            if current_confidence >= self.confidence_threshold:
                logger.info(f"✓ Confidence threshold reached ({current_confidence:.3f} >= {self.confidence_threshold})")
                break
            
            if iteration >= self.max_iterations:
                logger.info(f"⚠ Maximum iterations reached ({self.max_iterations})")
                break
            
            # Step 6: Refinement
            logger.info(f"  Refining answer (iteration {iteration})...")
            current_answer = self.refiner.iterative_refine(
                question,
                current_answer,
                critique,
                retrieved_docs,
                iteration
            )
        
        # Final evaluation
        final_critique = self.critic.critique(question, current_answer, hypotheses)
        final_confidence_metrics = self.uncertainty_estimator.compute_confidence_score(
            current_answer,
            hypotheses,
            retrieved_docs,
            final_critique.get("quality_score", 0.5)
        )
        
        # Compute iteration improvement metrics
        quality_scores = [iter_data["quality_score"] for iter_data in iteration_history]
        iteration_metrics = self.metrics.compute_iteration_improvement(quality_scores)
        
        # Compute calibration metrics
        calibration = self.metrics.compute_confidence_calibration(
            final_confidence_metrics["final_confidence"],
            final_critique.get("quality_score", 0.5)
        )
        
        # Compute answer metrics
        answer_metrics = self.metrics.compute_answer_length_metrics(current_answer)
        
        # Compile final result
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "question": question,
            "task_breakdown": plan,
            "final_answer": current_answer,
            "confidence_score": final_confidence_metrics["final_confidence"],
            "confidence_metrics": final_confidence_metrics,
            "critic_feedback": final_critique,
            "alternative_hypotheses": hypotheses,
            "iterations": iteration,
            "iteration_history": iteration_history,
            "iteration_metrics": iteration_metrics,
            "calibration_metrics": calibration,
            "retrieved_sources": [
                {
                    "text": doc[:200] + "..." if len(doc) > 200 else doc,
                    "similarity": float(score),
                    "metadata": meta
                }
                for doc, score, meta in retrieved_docs
            ],
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "duration_seconds": duration,
            "timestamp": end_time.isoformat()
        }
        
        # Log experiment
        self.metrics.log_experiment(question, result, final_confidence_metrics, iteration_metrics)
        self.experiment_log.append(result)
        
        logger.info("=" * 70)
        logger.info(f"✓ RESEARCH COMPLETE ({duration:.2f}s)")
        logger.info(f"  Final Confidence: {result['confidence_score']:.3f}")
        logger.info(f"  Iterations: {iteration}")
        logger.info("=" * 70)
        
        return result
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """
        Save research result to formatted text file.
        
        Args:
            result: Research result dictionary
            output_path: Path to save the result
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("🔬 SR-MARE RESEARCH REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("📝 QUESTION:\n")
            f.write(f"  {result['question']}\n\n")
            
            f.write("🎯 TASK BREAKDOWN:\n")
            plan = result['task_breakdown']
            if isinstance(plan, dict):
                if 'subtasks' in plan:
                    for i, task in enumerate(plan['subtasks'], 1):
                        f.write(f"  {i}. {task}\n")
                if 'key_concepts' in plan:
                    f.write(f"\n  Key Concepts: {', '.join(plan['key_concepts'])}\n")
            f.write("\n")
            
            f.write("💡 FINAL ANSWER:\n")
            f.write(f"  {result['final_answer']}\n\n")
            
            f.write("📊 CONFIDENCE METRICS:\n")
            conf = result['confidence_metrics']
            f.write(f"  Overall Confidence:       {conf['final_confidence']:.3f}\n")
            f.write(f"  Critic Quality Score:     {conf['critic_quality_score']:.3f}\n")
            f.write(f"  Self-Consistency:         {conf['self_consistency_score']:.3f}\n")
            f.write(f"  Evidence Diversity:       {conf['evidence_diversity_score']:.3f}\n")
            f.write(f"  Retrieval Quality:        {conf['retrieval_quality']:.3f}\n\n")
            
            f.write("🎯 CRITIC FEEDBACK:\n")
            critique = result['critic_feedback']
            if 'strengths' in critique and critique['strengths']:
                f.write("  Strengths:\n")
                for strength in critique['strengths']:
                    f.write(f"    ✓ {strength}\n")
            if 'weaknesses' in critique and critique['weaknesses']:
                f.write("  Areas for Improvement:\n")
                for weakness in critique['weaknesses']:
                    f.write(f"    • {weakness}\n")
            f.write(f"  Hallucination Risk:       {critique.get('hallucination_risk', 'unknown')}\n\n")
            
            f.write("🔄 ITERATION METRICS:\n")
            f.write(f"  Total Iterations:         {result['iterations']}\n")
            iter_metrics = result['iteration_metrics']
            f.write(f"  Total Improvement:        {iter_metrics.get('total_improvement', 0):.3f}\n")
            f.write(f"  Converged:                {iter_metrics.get('converged', False)}\n\n")
            
            f.write("📚 RETRIEVAL METRICS:\n")
            retr = result['retrieval_metrics']
            f.write(f"  Documents Retrieved:      {retr['num_retrieved']}\n")
            f.write(f"  Average Similarity:       {retr['avg_similarity']:.3f}\n")
            f.write(f"  Relevant Documents:       {retr['num_relevant']}\n\n")
            
            f.write("🔍 RETRIEVED SOURCES:\n")
            for i, source in enumerate(result['retrieved_sources'], 1):
                f.write(f"  [{i}] Similarity: {source['similarity']:.3f}\n")
                f.write(f"      {source['text'][:200]}...\n\n")
            
            f.write(f"⏱ PROCESSING TIME: {result['duration_seconds']:.2f} seconds\n\n")
            f.write("="*80 + "\n")
        
        logger.info(f"💾 Result saved to {output_path}")
    
    def save_vector_store(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the store
        """
        self.vector_store.save(path)
        logger.info(f"💾 Vector store saved to {path}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all research sessions.
        
        Returns:
            Dictionary containing aggregate statistics
        """
        return self.metrics.generate_summary_statistics()
