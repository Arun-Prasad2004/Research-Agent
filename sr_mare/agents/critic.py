"""
Critic agent for evaluating answer quality and identifying issues.
"""

import requests
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriticAgent:
    """Agent responsible for critically evaluating answers."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the critic agent.
        
        Args:
            model: Name of the Ollama model to use (llama3.2 for criticism)
            base_url: Base URL for Ollama API
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        
    def _call_ollama(self, prompt: str, temperature: float = 0.3) -> str:
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
                        "num_predict": 1200
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

        response = self._call_ollama(critique_prompt, temperature=0.3)
        
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
            response = self._call_ollama("Say 'OK'")
            logger.info("✓ Critic connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Critic connection failed: {e}")
            return False
