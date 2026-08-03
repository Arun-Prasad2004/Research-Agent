"""
Analyst agent for generating initial answers and multiple hypotheses.
"""

import requests
import logging
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalystAgent:
    """Agent responsible for generating answers and hypotheses."""
    
    def __init__(
        self, 
        model: str = "llama-3.1-8b-instant", 
        api_key: str = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the analyst agent.
        
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
            logger.info("🔌 Analyst agent connected to MCP")
        
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
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
                    "max_tokens": 1500
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
    
    def generate_answer(
        self, 
        question: str, 
        retrieved_context,
        plan: Dict[str, Any]
    ) -> str:
        """
        Generate an initial answer based on the question and retrieved context.
        
        Args:
            question: The research question
            retrieved_context: Retrieved documents (tuple or dict format)
            plan: Task breakdown from planner
            
        Returns:
            Generated answer
        """
        logger.info("🔍 Analyst: Generating initial answer...")
        
        # Handle both tuple and dict formats
        if retrieved_context and isinstance(retrieved_context[0], dict):
            # Dict format from MCP
            context_str = "\n\n".join([
                f"[Source {i+1}] (Relevance: {doc['similarity_score']:.2f})\n{doc['text']}"
                for i, doc in enumerate(retrieved_context[:5])
            ])
        else:
            # Tuple format (legacy)
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

CRITICAL REQUIREMENT: Provide ONLY the final comprehensive answer. Do NOT include any "thinking process", "internal monologues", or meta-commentary. Output the final markdown response directly.

Your comprehensive answer:"""

        answer = self._call_llm(analysis_prompt, temperature=0.7)
        logger.info("✓ Analyst: Generated initial answer")
        return answer
    
    def generate_hypotheses(
        self,
        question: str,
        retrieved_context,
        num_hypotheses: int = 3,
        event_callback: Callable = None
    ) -> List[str]:
        """
        Generate multiple independent hypotheses using self-consistency sampling.
        
        Args:
            question: The research question
            retrieved_context: Retrieved documents (tuple or dict format)
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of hypothesis strings
        """
        logger.info(f"💡 Analyst: Generating {num_hypotheses} independent hypotheses...")
        
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
        
        if not context_str:
            context_str = "No relevant documents found in knowledge base."
        
        hypothesis_prompt = f"""You are a research analyst. Generate a concise hypothesis to answer the research question based on the provided context.

Research Question: {question}

Context:
{context_str}

Generate ONE clear, evidence-based hypothesis that answers the question. Be concise but specific.

Your hypothesis:"""

        hypotheses = [None] * num_hypotheses
        
        def _generate_single_hypothesis(i: int):
            if event_callback: event_callback({"type": "status", "message": f"Deploying Sub-Agent {i+1} for hypothesis generation..."})
            h = self._call_llm(hypothesis_prompt, temperature=0.8)
            logger.info(f"  Generated hypothesis {i+1}/{num_hypotheses}")
            if event_callback: event_callback({"type": "status", "message": f"Sub-Agent {i+1} completed hypothesis generation."})
            return h

        if event_callback: event_callback({"type": "status", "message": f"Spawning {num_hypotheses} sub-agents concurrently..."})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_hypotheses) as executor:
            future_to_idx = {executor.submit(_generate_single_hypothesis, i): i for i in range(num_hypotheses)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    hypotheses[idx] = future.result()
                except Exception as e:
                    logger.error(f"Sub-agent {idx+1} failed: {e}")
                    hypotheses[idx] = "Failed to generate hypothesis."
        
        logger.info(f"✓ Analyst: Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def analyze_with_context(
        self,
        question: str,
        retrieved_context: Union[List[Tuple[str, float, dict]], List[Dict[str, Any]]],
        plan: Dict[str, Any],
        event_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Perform complete analysis: generate answer and hypotheses.
        
        Args:
            question: The research question
            retrieved_context: Retrieved documents (tuple or dict format)
            plan: Task breakdown from planner
            
        Returns:
            Dictionary containing answer and hypotheses
        """
        # Generate main answer
        if event_callback: event_callback({"type": "status", "message": "Deploying Main Analyst Agent to generate primary answer..."})
        answer = self.generate_answer(question, retrieved_context, plan)
        
        # Generate alternative hypotheses
        hypotheses = self.generate_hypotheses(question, retrieved_context, num_hypotheses=3, event_callback=event_callback)
        
        if event_callback: event_callback({"type": "status", "message": "All Analyst sub-agents killed. Merging outputs..."})
        
        return {
            "answer": answer,
            "hypotheses": hypotheses,
            "sources_used": len(retrieved_context)
        }
