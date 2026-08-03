"""
Planner agent for breaking down complex research questions into subtasks.
"""

import requests
import logging
import time
import json
from typing import List, Dict, Any, Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlannerAgent:
    """Agent responsible for decomposing complex questions into subtasks."""
    
    def __init__(
        self, 
        model: str = "llama-3.1-8b-instant", 
        api_key: str = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the planner agent.
        
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
            logger.info("🔌 Planner agent connected to MCP")
        
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
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
    
    def plan(self, research_question: str, event_callback: Callable = None, mcq_callback: Callable = None) -> Dict[str, Any]:
        """
        Break down a research question into structured subtasks.
        
        Args:
            research_question: The main research question to analyze
            event_callback: Optional callback for status events
            mcq_callback: Optional callback for user MCQ disambiguation
            
        Returns:
            Dictionary containing task breakdown and analysis strategy
        """
        logger.info("🧠 Planner: Analyzing research question...")
        
        # Disambiguation Check
        disambiguation_prompt = f"""You are a research analyst determining if a query is ambiguous.
Query: "{research_question}"

Is this query highly ambiguous in a way that would completely alter the research direction? (e.g. "How do transformers work" could mean AI transformers, electrical transformers, or fictional toys).
If it is NOT ambiguous, return exactly this JSON:
{{"is_ambiguous": false}}

If it IS ambiguous, return exactly this JSON with 3 to 4 multiple-choice options to clarify the user's intent:
{{"is_ambiguous": true, "question": "What kind of transformers do you mean?", "options": ["AI Transformers (Deep Learning)", "Electrical Transformers (Power grid)", "Transformers (Hasbro toys)"]}}
"""
        
        try:
            clarification_response = self._call_llm(disambiguation_prompt, temperature=0.1)
            # Find JSON block
            start_idx = clarification_response.find('{')
            end_idx = clarification_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = clarification_response[start_idx:end_idx]
                clarification_data = json.loads(json_str)
                
                if clarification_data.get("is_ambiguous") and mcq_callback and "options" in clarification_data:
                    logger.info("🧠 Planner: Detected ambiguity, requesting user clarification...")
                    if event_callback: event_callback({"type": "status", "message": "Query is ambiguous. Waiting for user clarification..."})
                    
                    mcq_payload = {
                        "type": "mcq",
                        "question": clarification_data.get("question", "Please clarify your query:"),
                        "options": clarification_data["options"]
                    }
                    user_choice = mcq_callback(mcq_payload)
                    
                    if user_choice:
                        logger.info(f"🧠 Planner: User clarified intent: {user_choice}")
                        research_question = f"{research_question} (Specifically focusing on: {user_choice})"
                        
        except Exception as e:
            logger.warning(f"Failed to run disambiguation check: {e}")
        
        if event_callback: event_callback({"type": "status", "message": "Generating structured research plan..."})

        
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

        response = self._call_llm(planning_prompt, temperature=0.3)
        
        # Try to parse JSON from response
        try:
            # Extract JSON if embedded in text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                plan = json.loads(json_str)
                plan["refined_question"] = research_question
            else:
                # Fallback: create structured plan from text
                plan = self._create_fallback_plan(research_question, response)
            
            logger.info(f"✓ Planner: Created plan with {len(plan.get('subtasks', []))} subtasks")
            return plan
            
        except json.JSONDecodeError as e:
            logger.warning(f"Planner response was not valid JSON, creating fallback plan. Error: {e}")
            return self._create_fallback_plan(research_question, response)
    
    def _create_fallback_plan(self, question: str, response: str) -> Dict[str, Any]:
        """Create a fallback plan if JSON parsing fails."""
        return {
            "refined_question": question,
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
            response = self._call_llm("Say 'OK'")
            logger.info("✓ Planner connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ Planner connection failed: {e}")
            return False
