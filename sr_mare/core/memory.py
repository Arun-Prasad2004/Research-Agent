import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryBus:
    """Handles Episodic and Failure Memory persistence for cross-session learning."""
    
    def __init__(self, data_dir: str = "sr_mare/data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.episodic_path = os.path.join(self.data_dir, "episodic_memory.json")
        self.failure_path = os.path.join(self.data_dir, "failure_memory.json")
        
        self.episodic_memory = self._load_memory(self.episodic_path)
        self.failure_memory = self._load_memory(self.failure_path)
        logger.info("🧠 Memory Bus initialized. Loaded episodic and failure memories.")
        
    def _load_memory(self, filepath: str) -> Dict[str, Any]:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory from {filepath}: {e}")
        return {"records": []}
        
    def _save_memory(self, filepath: str, data: Dict[str, Any]):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory to {filepath}: {e}")
            
    def store_episode(self, question: str, plan: Dict[str, Any], final_answer: str, confidence: float):
        """Store a successful reasoning trajectory."""
        self.episodic_memory["records"].append({
            "question": question,
            "plan": plan,
            "final_answer": final_answer,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory(self.episodic_path, self.episodic_memory)
        
    def store_failure(self, question: str, hypothesis: str, falsification_reason: str):
        """Store a failed hypothesis to prevent repeating dead ends."""
        self.failure_memory["records"].append({
            "question": question,
            "hypothesis": hypothesis,
            "falsification_reason": falsification_reason,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory(self.failure_path, self.failure_memory)
        
    def retrieve_relevant_failures(self, query: str) -> List[Dict[str, Any]]:
        """Naive keyword search for failures. Will be upgraded to semantic search."""
        words = set(query.lower().split())
        results = []
        for record in self.failure_memory["records"]:
            q_words = set(record["question"].lower().split())
            if len(words.intersection(q_words)) > 0:
                results.append(record)
        return results[-5:]  # Return most recent 5
