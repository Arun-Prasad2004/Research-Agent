"""
Example demonstration of SR-MARE with MCP integration.

This script shows how agents interact with tools through the MCP layer.
"""

import logging
from pathlib import Path
from sr_mare.core.orchestrator import ResearchOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run MCP-enabled SR-MARE research example."""
    
    # Initialize orchestrator with MCP
    logger.info("🚀 Initializing SR-MARE with MCP integration...")
    orchestrator = ResearchOrchestrator(
        base_url="http://localhost:11434",
        max_iterations=3,
        confidence_threshold=0.75
    )
    
    # Test connections
    if not orchestrator.test_connections():
        logger.error("⚠️ Connection tests failed. Please ensure Ollama is running.")
        return
    
    # Display MCP server stats
    mcp_stats = orchestrator.mcp_server.get_server_stats()
    logger.info("=" * 70)
    logger.info("📊 MCP Server Information")
    logger.info("=" * 70)
    logger.info(f"Protocol Version: {mcp_stats['protocol_version']}")
    logger.info(f"Registered Tools: {mcp_stats['registered_tools']}")
    logger.info(f"Tools by Category:")
    for category, count in mcp_stats['tools_by_category'].items():
        logger.info(f"  - {category}: {count} tools")
    logger.info("=" * 70)
    
    # List available tools via MCP client
    logger.info("\n🔧 Available MCP Tools:")
    tools = orchestrator.mcp_client.discover_tools()
    for tool in tools:
        logger.info(f"  - {tool.name} ({tool.category})")
        logger.info(f"    {tool.description}")
    
    # Load sample documents
    logger.info("\n📚 Loading knowledge base...")
    data_file = Path(__file__).parent / "sr_mare" / "data" / "documents.txt"
    
    if data_file.exists():
        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        documents = [
            chunk.strip() 
            for chunk in content.split("\n\n") 
            if chunk.strip() and len(chunk.strip()) > 50
        ]
        
        logger.info(f"Loading {len(documents)} documents via MCP...")
        orchestrator.load_documents(documents)
    else:
        logger.warning(f"⚠️ Data file not found: {data_file}")
        logger.info("Creating sample documents...")
        sample_docs = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
            "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
            "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward."
        ]
        orchestrator.load_documents(sample_docs)
    
    # Run research query
    logger.info("\n" + "=" * 70)
    logger.info("🔬 Starting Research with MCP-Enabled Agents")
    logger.info("=" * 70)
    
    research_question = "What is machine learning and how does it relate to artificial intelligence?"
    
    result = orchestrator.research(research_question, top_k=5)
    
    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("📊 RESEARCH RESULTS")
    logger.info("=" * 70)
    logger.info(f"\n📝 Question: {result['question']}")
    logger.info(f"\n💡 Answer:\n{result['final_answer']}")
    logger.info(f"\n📈 Confidence Score: {result['confidence_score']:.3f}")
    logger.info(f"🔄 Iterations: {result['iterations']}")
    logger.info(f"⏱️ Duration: {result['duration_seconds']:.2f}s")
    
    logger.info("\n📊 Confidence Breakdown:")
    conf = result['confidence_metrics']
    logger.info(f"  - Critic Quality: {conf['critic_quality_score']:.3f}")
    logger.info(f"  - Self-Consistency: {conf['self_consistency_score']:.3f}")
    logger.info(f"  - Evidence Diversity: {conf['evidence_diversity_score']:.3f}")
    logger.info(f"  - Retrieval Quality: {conf['retrieval_quality']:.3f}")
    
    # Display MCP execution stats
    logger.info("\n🔧 MCP Execution Statistics:")
    final_stats = orchestrator.mcp_server.get_server_stats()
    logger.info(f"  - Total Executions: {final_stats['total_executions']}")
    logger.info(f"  - Successful: {final_stats['successful_executions']}")
    logger.info(f"  - Failed: {final_stats['failed_executions']}")
    logger.info(f"  - Success Rate: {final_stats['success_rate']:.2%}")
    
    # Show recent tool executions
    recent_executions = orchestrator.mcp_server.get_execution_log(limit=10)
    logger.info("\n📋 Recent Tool Executions:")
    for exec_entry in recent_executions[-5:]:
        status = "✓" if exec_entry["success"] else "✗"
        logger.info(f"  {status} {exec_entry['tool_name']} ({exec_entry['execution_time']:.3f}s)")
    
    # Save result
    output_file = "research_result_mcp.txt"
    orchestrator.save_result(result, output_file)
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ MCP-Enabled Research Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
