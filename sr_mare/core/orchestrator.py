"""
Orchestrator for coordinating the multi-agent research pipeline with MCP.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime

from sr_mare.agents.planner import PlannerAgent
from sr_mare.agents.analyst import AnalystAgent
from sr_mare.agents.critic import CriticAgent
from sr_mare.agents.refiner import RefinerAgent
from sr_mare.retrieval.embedder import OllamaEmbedder
from sr_mare.retrieval.vector_store import FAISSVectorStore
from sr_mare.evaluation.uncertainty import UncertaintyEstimator
from sr_mare.evaluation.metrics import ResearchMetrics

# Import MCP components
from sr_mare.mcp.server import MCPServer
from sr_mare.mcp.client import MCPClient
from sr_mare.mcp.tools import MCPTools
from sr_mare.mcp.schema import ToolCategory, ToolParameter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates the complete multi-agent research pipeline with MCP layer."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_iterations: int = 3,
        confidence_threshold: float = 0.75,
        vector_store_path: Optional[str] = None
    ):
        """
        Initialize the research orchestrator with MCP integration.
        
        Args:
            base_url: Base URL for Ollama API
            max_iterations: Maximum refinement iterations
            confidence_threshold: Minimum confidence to stop refinement
            vector_store_path: Path to load existing vector store
        """
        logger.info("🚀 Initializing SR-MARE Research Orchestrator with MCP...")
        
        # Initialize retrieval components
        self.embedder = OllamaEmbedder(model="nomic-embed-text", base_url=base_url)
        self.vector_store = FAISSVectorStore(dimension=768)
        
        # Initialize evaluation components
        self.uncertainty_estimator = UncertaintyEstimator()
        self.metrics = ResearchMetrics()
        
        # Initialize MCP Server
        logger.info("🔧 Initializing MCP Server...")
        self.mcp_server = MCPServer()
        
        # Initialize MCP Tools wrapper
        self.mcp_tools = MCPTools(
            embedder=self.embedder,
            vector_store=self.vector_store,
            uncertainty_estimator=self.uncertainty_estimator,
            metrics=self.metrics
        )
        
        # Register tools with MCP server
        self._register_mcp_tools()
        
        # Create MCP client for agents
        self.mcp_client = MCPClient(self.mcp_server)
        
        # Initialize agents with MCP client
        self.planner = PlannerAgent(
            model="mistral", 
            base_url=base_url,
            mcp_client=self.mcp_client
        )
        self.analyst = AnalystAgent(
            model="mistral", 
            base_url=base_url,
            mcp_client=self.mcp_client
        )
        self.critic = CriticAgent(
            model="llama3.2", 
            base_url=base_url,
            mcp_client=self.mcp_client
        )
        self.refiner = RefinerAgent(
            model="llama3.2", 
            base_url=base_url,
            mcp_client=self.mcp_client
        )
        
        # Configuration
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
        # Load existing vector store if provided
        if vector_store_path and Path(vector_store_path).exists():
            self.vector_store.load(vector_store_path)
            logger.info(f"✓ Loaded vector store from {vector_store_path}")
        
        # Experiment log
        self.experiment_log = []
        
        logger.info("✓ Orchestrator initialized with MCP layer")
        self._log_mcp_stats()
    
    def _register_mcp_tools(self):
        """Register all MCP tools with the server."""
        logger.info("📝 Registering MCP tools...")
        
        # Tool 1: Retrieve Context
        self.mcp_server.register_tool(
            name="retrieve_context",
            description="Retrieve relevant documents for a query using vector similarity search",
            category=ToolCategory.RETRIEVAL,
            implementation=self.mcp_tools.retrieve_context,
            parameters=[
                ToolParameter(
                    name="query",
                    type="str",
                    description="Query text to search for",
                    required=True
                ),
                ToolParameter(
                    name="k",
                    type="int",
                    description="Number of documents to retrieve",
                    required=False,
                    default=5
                )
            ],
            returns="Dictionary with retrieved documents and metadata"
        )
        
        # Tool 2: Evaluate Confidence
        self.mcp_server.register_tool(
            name="evaluate_confidence",
            description="Evaluate confidence in an answer using uncertainty metrics",
            category=ToolCategory.EVALUATION,
            implementation=self.mcp_tools.evaluate_confidence,
            parameters=[
                ToolParameter(
                    name="answer",
                    type="str",
                    description="Generated answer to evaluate",
                    required=True
                ),
                ToolParameter(
                    name="hypotheses",
                    type="list",
                    description="List of alternative hypotheses",
                    required=True
                ),
                ToolParameter(
                    name="retrieved_docs",
                    type="list",
                    description="List of retrieved documents (dict format)",
                    required=True
                ),
                ToolParameter(
                    name="quality_score",
                    type="float",
                    description="Quality score from critic",
                    required=True
                )
            ],
            returns="Dictionary with confidence metrics"
        )
        
        # Tool 3: Store Documents
        self.mcp_server.register_tool(
            name="store_documents",
            description="Store documents in the vector store",
            category=ToolCategory.MEMORY,
            implementation=self.mcp_tools.store_documents,
            parameters=[
                ToolParameter(
                    name="documents",
                    type="list",
                    description="List of document texts to store",
                    required=True
                ),
                ToolParameter(
                    name="metadata",
                    type="list",
                    description="Optional metadata for each document",
                    required=False,
                    default=None
                )
            ],
            returns="Dictionary with storage results"
        )
        
        # Tool 4: Score Retrieval Quality
        self.mcp_server.register_tool(
            name="score_retrieval_quality",
            description="Compute quality metrics for retrieved documents",
            category=ToolCategory.EVALUATION,
            implementation=self.mcp_tools.score_retrieval_quality,
            parameters=[
                ToolParameter(
                    name="retrieved_docs",
                    type="list",
                    description="List of retrieved documents (dict format)",
                    required=True
                )
            ],
            returns="Dictionary with retrieval metrics"
        )
        
        # Tool 5: Compute Self-Consistency
        self.mcp_server.register_tool(
            name="compute_self_consistency",
            description="Compute self-consistency score for multiple hypotheses",
            category=ToolCategory.COMPUTATION,
            implementation=self.mcp_tools.compute_self_consistency,
            parameters=[
                ToolParameter(
                    name="hypotheses",
                    type="list",
                    description="List of hypothesis strings",
                    required=True
                )
            ],
            returns="Dictionary with consistency score"
        )
        
        # Tool 6: Compute Evidence Diversity
        self.mcp_server.register_tool(
            name="compute_evidence_diversity",
            description="Compute diversity score for evidence sources",
            category=ToolCategory.COMPUTATION,
            implementation=self.mcp_tools.compute_evidence_diversity,
            parameters=[
                ToolParameter(
                    name="retrieved_docs",
                    type="list",
                    description="List of retrieved documents (dict format)",
                    required=True
                )
            ],
            returns="Dictionary with diversity score"
        )
        
        logger.info(f"✓ Registered {self.mcp_server.registry.get_tool_count()} MCP tools")
    
    def _log_mcp_stats(self):
        """Log MCP server statistics."""
        stats = self.mcp_server.get_server_stats()
        logger.info(f"📊 MCP Server Stats:")
        logger.info(f"   - Protocol Version: {stats['protocol_version']}")
        logger.info(f"   - Registered Tools: {stats['registered_tools']}")
        logger.info(f"   - Tools by Category: {stats['tools_by_category']}")
    
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
        Load documents into the vector store using MCP.
        
        Args:
            documents: List of document texts to index
        """
        logger.info(f"📚 Loading {len(documents)} documents into vector store via MCP...")
        
        if len(documents) == 0:
            logger.warning("No documents to load")
            return
        
        # Use MCP tool to store documents
        metadata = [{"doc_id": i, "loaded_at": datetime.now().isoformat()} 
                   for i in range(len(documents))]
        
        result = self.mcp_client.execute_tool(
            tool_name="store_documents",
            parameters={
                "documents": documents,
                "metadata": metadata
            }
        )
        
        logger.info(f"✓ Loaded {result['num_stored']} documents (Total: {result['total_documents']})")
    
    def retrieve_context(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a question using MCP.
        
        Args:
            question: The research question
            k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries with metadata
        """
        logger.info(f"🔍 Retrieving top {k} relevant documents via MCP...")
        
        result = self.mcp_client.execute_tool(
            tool_name="retrieve_context",
            parameters={
                "query": question,
                "k": k
            }
        )
        
        logger.info(f"✓ Retrieved {result['num_retrieved']} documents")
        return result["documents"]
    
    def research(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the complete research pipeline for a question using MCP.
        
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
        
        # Step 2: Retrieval via MCP
        logger.info("\n[STEP 2] Retrieving relevant context via MCP...")
        retrieved_docs = self.retrieve_context(question, k=top_k)
        
        # Compute retrieval metrics via MCP
        retrieval_metrics = self.mcp_client.execute_tool(
            tool_name="score_retrieval_quality",
            parameters={"retrieved_docs": retrieved_docs}
        )
        
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
            
            # Step 5: Compute confidence via MCP
            confidence_metrics = self.mcp_client.execute_tool(
                tool_name="evaluate_confidence",
                parameters={
                    "answer": current_answer,
                    "hypotheses": hypotheses,
                    "retrieved_docs": retrieved_docs,
                    "quality_score": quality_score
                }
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
        final_confidence_metrics = self.mcp_client.execute_tool(
            tool_name="evaluate_confidence",
            parameters={
                "answer": current_answer,
                "hypotheses": hypotheses,
                "retrieved_docs": retrieved_docs,
                "quality_score": final_critique.get("quality_score", 0.5)
            }
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
                    "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                    "similarity": doc["similarity_score"],
                    "metadata": doc.get("metadata", {})
                }
                for doc in retrieved_docs
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
