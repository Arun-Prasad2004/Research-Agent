import logging
from typing import List, Dict, Any, Tuple
import json

logger = logging.getLogger(__name__)

class ForagerAgent:
    """Agentic Search: Performs fractal retrieval loops to resolve information gaps."""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        logger.info("🕵️ Forager Agent initialized.")
        
    def forage(self, query: str, max_depth: int = 3) -> Tuple[List[Dict[str, Any]], str]:
        """
        Dynamically retrieve information, evaluate if there are gaps, and rewrite queries if needed.
        Returns the accumulated documents and a synthesis of the knowledge graph.
        """
        accumulated_docs = []
        excluded_ids = []
        current_query = query
        
        for depth in range(max_depth):
            logger.info(f"🕵️ Forager searching (Depth {depth+1}/{max_depth}): '{current_query}'")
            
            try:
                # Use MCP to search knowledge base
                result = self.mcp_client.execute_tool(
                    tool_name="retrieve_context",
                    parameters={"query": current_query, "k": 3}
                )
                new_docs = result.get("documents", [])
            except Exception as e:
                logger.error(f"Forager search failed: {e}")
                break
                
            # Filter out docs we've already seen (fallback in case vector_store exclusion missed some)
            for doc in new_docs:
                if doc.get("id") not in excluded_ids:
                    accumulated_docs.append(doc)
                    if "id" in doc and doc["id"] is not None:
                        excluded_ids.append(doc["id"])
                        
            # In a true LLM-based forager, we would ask the LLM: 
            # "Does this answer the query, or is there a gap? If gap, what is the next search query?"
            # For this MVP, we stop if we found enough docs, or mock a query shift.
            if len(accumulated_docs) >= 5:
                logger.info("🕵️ Forager found sufficient context.")
                break
                
            # Mocking query expansion
            current_query = f"{query} details and implications"
            
        # Synthesize a simple knowledge graph string
        kg_synthesis = "Synthesized Knowledge Graph Context:\n"
        for i, doc in enumerate(accumulated_docs):
            kg_synthesis += f"- Node {i+1}: {doc['text'][:100]}...\n"
            
        return accumulated_docs, kg_synthesis
