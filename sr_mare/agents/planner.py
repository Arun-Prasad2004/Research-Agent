"""
Planner agent for breaking down complex research questions into subtasks.
"""

import requests
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlannerAgent:
    """Agent responsible for decomposing complex questions into subtasks."""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize the planner agent.
        
        Args:
            model: Name of the Ollama model to use
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
                        "num_predict": 1000
                    }
                }
                
                response = requests.post(self.generate_url, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                return result["response"].strip()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to call Ollama after {max_retries} attempts: {e}")
        
        return ""
    
    def plan(self, research_question: str) -> Dict[str, Any]:
        """
        Break down a research question into structured subtasks.
        
        Args:
            research_question: The main research question to analyze
            
        Returns:
            Dictionary containing task breakdown and analysis strategy
        """
        logger.info("🧠 Planner: Analyzing research question...")
        
        planning_prompt = f"""You are a research planner. Your task is to break down a complex research question into structured subtasks.

Research Question: {research_question}

Please analyze this question and provide:
1. Key concepts that need to be understood
2. Specific subtasks needed to answer it
3. Information retrieval strategy
4. Expected challenges

Format your response as a structured JSON object with these keys:
- "key_concepts": list of important concepts
- "subtasks": list of specific subtasks (3-5 tasks)
- "retrieval_strategy": description of what information to search for
- "challenges": potential difficulties in answering

Provide ONLY the JSON object, no additional text.

Example format:
{{
  "key_concepts": ["concept1", "concept2"],
  "subtasks": ["subtask1", "subtask2", "subtask3"],
  "retrieval_strategy": "Focus on...",
  "challenges": "May struggle with..."
}}

Your JSON response:"""

        response = self._call_ollama(planning_prompt, temperature=0.3)
        
        # Try to parse JSON from response
        try:
            # Extract JSON if embedded in text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                plan = json.loads(json_str)
            else:
                # Fallback: create structured plan from text
                plan = self._create_fallback_plan(research_question, response)
            
            logger.info(f"✓ Planner: Created plan with {len(plan.get('subtasks', []))} subtasks")
            return plan
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._create_fallback_plan(research_question, response)
    
    def _create_fallback_plan(self, question: str, response: str) -> Dict[str, Any]:
        """Create a fallback plan if JSON parsing fails."""
        return {
            "key_concepts": question.split()[:5],
            "subtasks": [
                "Understand the context and background",
                "Identify key components and relationships",
                "Analyze available evidence",
                "Synthesize findings into coherent answer"
            ],
            "retrieval_strategy": "Search for relevant documents related to: " + question,
            "challenges": "May require domain-specific knowledge",
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
            logger.info("✓ Planner connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Planner connection failed: {e}")
            return False
