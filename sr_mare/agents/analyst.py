"""
Analyst agent for generating initial answers and multiple hypotheses.
"""

import requests
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalystAgent:
    """Agent responsible for generating answers and hypotheses."""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize the analyst agent.
        
        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
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
    
    def generate_answer(
        self, 
        question: str, 
        retrieved_context: List[Tuple[str, float, dict]],
        plan: Dict[str, Any]
    ) -> str:
        """
        Generate an initial answer based on the question and retrieved context.
        
        Args:
            question: The research question
            retrieved_context: List of (document, similarity_score, metadata) tuples
            plan: Task breakdown from planner
            
        Returns:
            Generated answer
        """
        logger.info("🔍 Analyst: Generating initial answer...")
        
        # Format retrieved context
        context_str = "\n\n".join([
            f"[Source {i+1}] (Relevance: {score:.2f})\n{doc}"
            for i, (doc, score, _) in enumerate(retrieved_context[:5])
        ])
        
        if not context_str:
            context_str = "No relevant documents found in knowledge base."
        
        analysis_prompt = f"""You are a research analyst. Based on the provided context, answer the research question comprehensively.

Research Question: {question}

Retrieved Context:
{context_str}

Task Breakdown:
{plan.get('subtasks', [])}

Instructions:
1. Synthesize information from the retrieved sources
2. Address all subtasks identified in the plan
3. Provide a well-structured, evidence-based answer
4. Cite sources where appropriate
5. Be explicit about any limitations or uncertainties

Your comprehensive answer:"""

        answer = self._call_ollama(analysis_prompt, temperature=0.7)
        logger.info("✓ Analyst: Generated initial answer")
        return answer
    
    def generate_hypotheses(
        self,
        question: str,
        retrieved_context: List[Tuple[str, float, dict]],
        num_hypotheses: int = 3
    ) -> List[str]:
        """
        Generate multiple independent hypotheses using self-consistency sampling.
        
        Args:
            question: The research question
            retrieved_context: List of retrieved documents
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of hypothesis strings
        """
        logger.info(f"💡 Analyst: Generating {num_hypotheses} independent hypotheses...")
        
        # Format context
        context_str = "\n\n".join([
            f"[Source {i+1}]\n{doc}"
            for i, (doc, score, _) in enumerate(retrieved_context[:5])
        ])
        
        if not context_str:
            context_str = "No relevant documents found in knowledge base."
        
        hypothesis_prompt = f"""You are a research analyst. Generate a concise hypothesis to answer the research question based on the provided context.

Research Question: {question}

Context:
{context_str}

Generate ONE clear, evidence-based hypothesis that answers the question. Be concise but specific.

Your hypothesis:"""

        hypotheses = []
        
        for i in range(num_hypotheses):
            # Use temperature=0.8 for diversity
            hypothesis = self._call_ollama(hypothesis_prompt, temperature=0.8)
            hypotheses.append(hypothesis)
            logger.info(f"  Generated hypothesis {i+1}/{num_hypotheses}")
        
        logger.info(f"✓ Analyst: Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def analyze_with_context(
        self,
        question: str,
        retrieved_context: List[Tuple[str, float, dict]],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform complete analysis: generate answer and hypotheses.
        
        Args:
            question: The research question
            retrieved_context: Retrieved documents
            plan: Task breakdown from planner
            
        Returns:
            Dictionary containing answer and hypotheses
        """
        # Generate main answer
        answer = self.generate_answer(question, retrieved_context, plan)
        
        # Generate alternative hypotheses
        hypotheses = self.generate_hypotheses(question, retrieved_context, num_hypotheses=3)
        
        return {
            "answer": answer,
            "hypotheses": hypotheses,
            "sources_used": len(retrieved_context)
        }
