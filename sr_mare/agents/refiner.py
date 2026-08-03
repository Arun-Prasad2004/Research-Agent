"""
Refiner agent for improving answers based on critic feedback.
"""

import requests
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinerAgent:
    """Agent responsible for refining answers based on feedback."""
    
    def __init__(
        self, 
        model: str = "llama-3.1-8b-instant", 
        api_key: str = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the refiner agent.
        
        Args:
            model: Name of the Groq model to use
            api_key: Groq API key
            mcp_client: MCP client for tool interaction
        """
        self.model = model
        self.api_key = api_key
        self.generate_url = "https://api.groq.com/openai/v1/chat/completions"
        self.mcp_client = mcp_client
        
        if not self.api_key:
            logger.warning("No Groq API key provided. Agent will fail if key is required.")
            
        if mcp_client:
            logger.info("🔌 Refiner agent connected to MCP")
        
    def _call_llm(self, prompt: str, temperature: float = 0.5) -> str:
        """
        Call Groq API with retry logic.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        max_retries = 5
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        fallback_models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        current_model_idx = fallback_models.index(self.model) if self.model in fallback_models else 0
        
        for attempt in range(max_retries):
            current_model = fallback_models[current_model_idx % len(fallback_models)]
            try:
                payload = {
                    "model": current_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2000
                }
                
                response = requests.post(self.generate_url, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed on {current_model}: {e}")
                
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    wait_time = int(e.response.headers.get("Retry-After", (attempt + 1) * 3))
                    if wait_time > 15:
                        logger.info(f"Rate limited on {current_model} for {wait_time}s. Switching to fallback model...")
                        current_model_idx += 1
                        if current_model_idx >= len(fallback_models):
                            raise Exception(f"Groq API rate limit exceeded on ALL models. Please wait {wait_time} seconds.")
                        continue
                    
                    logger.info(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2 ** attempt)
                    
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to call API after {max_retries} attempts: {e}")
        
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

CRITICAL REQUIREMENT: Provide ONLY the final refined answer. Do NOT include any "thinking process", "internal monologues", or meta-commentary. Output the final markdown response directly.

Your refined answer:"""

        refined_answer = self._call_llm(refinement_prompt, temperature=0.5)
        
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
