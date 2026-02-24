"""
Refiner agent for improving answers based on critic feedback.
"""

import requests
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinerAgent:
    """Agent responsible for refining answers based on feedback."""
    
    def __init__(
        self, 
        model: str = "llama3.2", 
        base_url: str = "http://localhost:11434",
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the refiner agent.
        
        Args:
            model: Name of the Ollama model to use (llama3.2 for refinement)
            base_url: Base URL for Ollama API
            mcp_client: MCP client for tool interaction
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.mcp_client = mcp_client
        
        if mcp_client:
            logger.info("🔌 Refiner agent connected to MCP")
        
    def _call_ollama(self, prompt: str, temperature: float = 0.5) -> str:
        """
        Call Ollama API with retry logic.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "options": {
                        "num_predict": 1500
                    }
                }
                
                response = requests.post(self.generate_url, json=payload, timeout=90)
                response.raise_for_status()
                
                result = response.json()
                return result["response"].strip()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to call Ollama after {max_retries} attempts: {e}")
        
        return ""
    
    def refine(
        self,
        question: str,
        original_answer: str,
        critique: Dict[str, Any],
        retrieved_context
    ) -> str:
        """
        Refine the answer based on critic feedback.
        
        Args:
            question: The original research question
            original_answer: The answer to be refined
            critique: Feedback from the critic agent
            retrieved_context: Retrieved documents (tuple or dict format)
            
        Returns:
            Refined answer
        """
        logger.info("✨ Refiner: Improving answer based on feedback...")
        
        # Handle both tuple and dict formats
        if retrieved_context and isinstance(retrieved_context[0], dict):
            # Dict format from MCP
            context_str = "\n\n".join([
                f"[Source {i+1}]\n{doc['text']}"
                for i, doc in enumerate(retrieved_context[:5])
            ])
        else:
            # Tuple format (legacy)
            context_str = "\n\n".join([
                f"[Source {i+1}]\n{doc}"
                for i, (doc, score, _) in enumerate(retrieved_context[:5])
            ])
        
        # Format critique feedback
        weaknesses = "\n".join([f"- {w}" for w in critique.get("weaknesses", [])])
        suggestions = critique.get("improvement_suggestions", "Address identified issues")
        
        refinement_prompt = f"""You are a research answer refiner. Your task is to improve an answer by addressing identified weaknesses and gaps.

Research Question: {question}

Original Answer:
{original_answer}

Context (for reference):
{context_str}

Identified Weaknesses:
{weaknesses}

Improvement Suggestions:
{suggestions}

Hallucination Risk: {critique.get("hallucination_risk", "unknown")}
Logical Gaps: {critique.get("logical_gaps", "none identified")}

Instructions:
1. Address ALL identified weaknesses
2. Maintain the strengths of the original answer
3. Ensure all claims are well-supported by evidence
4. Fix any logical gaps or inconsistencies
5. Make the answer more complete and accurate
6. Remove or qualify any unsupported claims

Provide the refined, improved answer:"""

        refined_answer = self._call_ollama(refinement_prompt, temperature=0.5)
        
        logger.info("✓ Refiner: Answer refined")
        return refined_answer
    
    def iterative_refine(
        self,
        question: str,
        answer: str,
        critique: Dict[str, Any],
        retrieved_context: List[Tuple[str, float, dict]],
        iteration: int
    ) -> str:
        """
        Perform iterative refinement with iteration context.
        
        Args:
            question: The research question
            answer: Current answer
            critique: Latest critique
            retrieved_context: Retrieved documents
            iteration: Current iteration number
            
        Returns:
            Refined answer
        """
        logger.info(f"🔄 Refiner: Iteration {iteration} - Quality score: {critique.get('quality_score', 0):.2f}")
        
        return self.refine(question, answer, critique, retrieved_context)
