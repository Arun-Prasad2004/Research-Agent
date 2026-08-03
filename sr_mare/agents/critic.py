"""
Critic agent for evaluating answer quality and identifying issues.
"""

import requests
import json
import logging
import time
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriticAgent:
    """Agent responsible for critically evaluating answers."""
    
    def __init__(
        self, 
        model: str = "llama-3.1-8b-instant", 
        api_key: str = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the critic agent.
        
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
            logger.info("🔌 Critic agent connected to MCP")
        
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
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
                    "max_tokens": 1000
                }
                
                response = requests.post(self.generate_url, headers=headers, json=payload, timeout=60)
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
    
    def critique(self, question: str, answer: str, hypotheses: list) -> Dict[str, Any]:
        """
        Evaluate answer quality and identify potential issues.
        
        Args:
            question: The original research question
            answer: The generated answer to evaluate
            hypotheses: Alternative hypotheses for comparison
            
        Returns:
            Dictionary containing critique results and quality score
        """
        logger.info("🎯 Critic: Evaluating answer quality...")
        
        hypotheses_str = "\n".join([f"{i+1}. {h}" for i, h in enumerate(hypotheses)])
        
        critique_prompt = f"""You are a critical evaluator of research answers. Your task is to identify weaknesses, gaps, and potential issues in the provided answer.

Research Question: {question}

Proposed Answer:
{answer}

Alternative Hypotheses:
{hypotheses_str}

Evaluate the answer on these dimensions:
1. FACTUAL ACCURACY: Does it contain unsupported claims or potential hallucinations?
2. LOGICAL COHERENCE: Is the reasoning sound and well-structured?
3. COMPLETENESS: Does it address all aspects of the question?
4. EVIDENCE SUPPORT: Is it properly grounded in the provided context?
5. CONSISTENCY: How well does it align with the alternative hypotheses?

Provide your evaluation as a JSON object with these fields:
- "strengths": list of 2-3 strong points
- "weaknesses": list of 2-4 identified issues or gaps
- "hallucination_risk": assessment of unsupported claims (low/medium/high)
- "logical_gaps": specific logical issues identified
- "quality_score": overall score from 0.0 to 1.0
- "improvement_suggestions": specific suggestions for enhancement

Provide ONLY the JSON object.

Example format:
{{
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "hallucination_risk": "low",
  "logical_gaps": "identified gaps",
  "quality_score": 0.75,
  "improvement_suggestions": "specific suggestions"
}}

Your JSON evaluation:"""

        response = self._call_llm(critique_prompt, temperature=0.3)
        
        # Parse JSON response
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                critique = json.loads(json_str)
            else:
                critique = self._create_fallback_critique(response)
            
            # Ensure quality_score exists and is valid
            if "quality_score" not in critique or not isinstance(critique["quality_score"], (int, float)):
                critique["quality_score"] = 0.5
            
            critique["quality_score"] = max(0.0, min(1.0, float(critique["quality_score"])))
            
            logger.info(f"✓ Critic: Quality score = {critique['quality_score']:.2f}")
            return critique
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse critique JSON: {e}")
            return self._create_fallback_critique(response)
    
    def _create_fallback_critique(self, response: str) -> Dict[str, Any]:
        """Create a fallback critique if JSON parsing fails."""
        # Try to extract a quality score from text
        quality_score = 0.6
        
        if any(word in response.lower() for word in ["excellent", "strong", "good"]):
            quality_score = 0.75
        elif any(word in response.lower() for word in ["poor", "weak", "insufficient"]):
            quality_score = 0.4
        
        return {
            "strengths": ["Answer provided"],
            "weaknesses": ["Evaluation format was not structured"],
            "hallucination_risk": "medium",
            "logical_gaps": "Unable to fully assess due to parsing issues",
            "quality_score": quality_score,
            "improvement_suggestions": "Ensure claims are well-supported by evidence",
            "raw_response": response
        }
    
    def test_connection(self) -> bool:
        """
        Test if Ollama API is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self._call_llm("Say 'OK'")
            logger.info("✓ Critic connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Critic connection failed: {e}")
            return False
