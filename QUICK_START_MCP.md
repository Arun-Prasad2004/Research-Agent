# SR-MARE MCP Quick Start Guide

## 🚀 Getting Started with MCP-Enabled SR-MARE

This guide will get you up and running with the new MCP-integrated SR-MARE system in minutes.

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** running locally on port 11434
3. Required models pulled:
   ```bash
   ollama pull mistral
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Example 1: Simple Research Query

```python
from sr_mare.core.orchestrator import ResearchOrchestrator

# Initialize orchestrator (MCP is automatically set up)
orchestrator = ResearchOrchestrator()

# Load some documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "NLP helps computers understand human language."
]
orchestrator.load_documents(documents)

# Run research query
result = orchestrator.research("What is machine learning?")

print(f"Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence_score']:.3f}")
```

### Example 2: Using MCP Tools Directly

```python
# Access the MCP client
mcp_client = orchestrator.mcp_client

# Discover available tools
tools = mcp_client.discover_tools()
for tool in tools:
    print(f"- {tool.name}: {tool.description}")

# Execute a tool directly
result = mcp_client.execute_tool(
    tool_name="retrieve_context",
    parameters={"query": "machine learning", "k": 3}
)
print(f"Retrieved {result['num_retrieved']} documents")
```

### Example 3: Check MCP Statistics

```python
# Get MCP server statistics
stats = orchestrator.mcp_server.get_server_stats()
print(f"Registered Tools: {stats['registered_tools']}")
print(f"Total Executions: {stats['total_executions']}")
print(f"Success Rate: {stats['success_rate']:.2%}")

# View execution log
log = orchestrator.mcp_server.get_execution_log(limit=10)
for entry in log:
    status = "✓" if entry["success"] else "✗"
    print(f"{status} {entry['tool_name']} ({entry['execution_time']:.3f}s)")
```

## Run Examples

### Command Line

```bash
# Run the MCP example
python example_mcp.py

# Run with your own question
python main.py \
  --question "What is artificial intelligence?" \
  --documents sr_mare/data/documents.txt \
  --output results.txt
```

### Web Interface

```bash
python start_web.py
# Open http://localhost:5000 in your browser
```

## Understanding MCP Tools

### Available Tools

| Tool | Purpose | Example Usage |
|------|---------|---------------|
| `retrieve_context` | Find relevant documents | Research information lookup |
| `evaluate_confidence` | Assess answer quality | Quality assurance |
| `store_documents` | Index new documents | Knowledge base expansion |
| `score_retrieval_quality` | Evaluate retrieval | Performance monitoring |
| `compute_self_consistency` | Check hypothesis agreement | Validation |
| `compute_evidence_diversity` | Measure source variety | Evidence assessment |

### Tool Usage Pattern

```python
# 1. Discover tools by category
from sr_mare.mcp.schema import ToolCategory

retrieval_tools = mcp_client.discover_tools(category=ToolCategory.RETRIEVAL)

# 2. Check if tool is available
if mcp_client.check_tool_available("retrieve_context"):
    print("Tool is ready!")

# 3. Execute with error handling
try:
    result = mcp_client.execute_tool(
        tool_name="retrieve_context",
        parameters={"query": "AI", "k": 5}
    )
except RuntimeError as e:
    print(f"Tool execution failed: {e}")

# 4. Safe execution with default
result = mcp_client.execute_tool_safe(
    tool_name="retrieve_context",
    parameters={"query": "AI", "k": 5},
    default_on_error={"documents": [], "num_retrieved": 0}
)
```

## Adding Custom Tools

### Step 1: Create Tool Function

```python
def my_summarizer(text: str, max_length: int = 100) -> dict:
    """Summarize text to specified length."""
    summary = text[:max_length] + "..." if len(text) > max_length else text
    return {
        "original_length": len(text),
        "summary": summary,
        "summary_length": len(summary)
    }
```

### Step 2: Register with MCP

```python
from sr_mare.mcp.schema import ToolCategory, ToolParameter

orchestrator.mcp_server.register_tool(
    name="summarize_text",
    description="Summarize text to a specified maximum length",
    category=ToolCategory.UTILITY,
    implementation=my_summarizer,
    parameters=[
        ToolParameter(
            name="text",
            type="str",
            description="Text to summarize",
            required=True
        ),
        ToolParameter(
            name="max_length",
            type="int",
            description="Maximum summary length",
            required=False,
            default=100
        )
    ],
    returns="Dictionary with summary and metadata"
)
```

### Step 3: Use Your Tool

```python
result = mcp_client.execute_tool(
    tool_name="summarize_text",
    parameters={
        "text": "Long text here...",
        "max_length": 50
    }
)
print(result["summary"])
```

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Check Tool Status

```python
# List all registered tools
all_tools = orchestrator.mcp_server.list_all_tools()
for tool in all_tools:
    print(f"{tool.name} v{tool.version} - {tool.category}")

# Get specific tool info
tool_def = orchestrator.mcp_server.get_tool_definition("retrieve_context")
print(f"Parameters: {[p.name for p in tool_def.parameters]}")
```

### Inspect Execution Errors

```python
# Get recent executions
recent = orchestrator.mcp_server.get_execution_log(limit=20)

# Filter failed executions
failed = [e for e in recent if not e["success"]]
for failure in failed:
    print(f"Failed: {failure['tool_name']}")
    print(f"Error: {failure['error']}")
```

## Configuration

### Customize Orchestrator

```python
orchestrator = ResearchOrchestrator(
    base_url="http://localhost:11434",  # Ollama URL
    max_iterations=3,                    # Max refinement iterations
    confidence_threshold=0.75,           # Stop threshold
    vector_store_path="./vector_store"  # Optional: load existing store
)
```

### Customize Agent Behavior

Agents automatically receive the MCP client, but you can access them:

```python
# Access agents with MCP client
planner = orchestrator.planner
analyst = orchestrator.analyst
critic = orchestrator.critic
refiner = orchestrator.refiner

# Each agent has .mcp_client attribute
if analyst.mcp_client:
    print("Analyst is MCP-enabled!")
```

## Common Patterns

### Pattern 1: Research with Custom Knowledge Base

```python
# Load domain-specific documents
documents = load_documents_from_file("my_knowledge_base.txt")
orchestrator.load_documents(documents)

# Run research
result = orchestrator.research("My domain-specific question")
```

### Pattern 2: Batch Processing

```python
questions = [
    "What is machine learning?",
    "How does deep learning work?",
    "What is NLP?"
]

results = []
for question in questions:
    result = orchestrator.research(question)
    results.append(result)
    
# Analyze results
avg_confidence = sum(r["confidence_score"] for r in results) / len(results)
print(f"Average confidence: {avg_confidence:.3f}")
```

### Pattern 3: Progressive Knowledge Building

```python
# Start with base knowledge
base_docs = ["Basic AI concepts..."]
orchestrator.load_documents(base_docs)

# Research and get answer
result1 = orchestrator.research("What is AI?")

# Add more specialized knowledge
specialized_docs = ["Advanced ML techniques..."]
orchestrator.load_documents(specialized_docs)

# Research more specific question
result2 = orchestrator.research("How does gradient descent work?")
```

## Troubleshooting

### Issue: "Connection failed"
**Solution**: Ensure Ollama is running: `ollama serve`

### Issue: "Model not found"
**Solution**: Pull required models:
```bash
ollama pull mistral
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Issue: "No documents in vector store"
**Solution**: Load documents first:
```python
orchestrator.load_documents(["doc1", "doc2"])
```

### Issue: "Tool execution failed"
**Solution**: Check tool parameters match definition:
```python
tool_def = mcp_client.get_tool_info("tool_name")
print(tool_def.parameters)
```

## Next Steps

1. **Read Full Documentation**: See [MCP_DOCUMENTATION.md](MCP_DOCUMENTATION.md)
2. **Explore Examples**: Check [example_mcp.py](example_mcp.py)
3. **Add Custom Tools**: Follow the "Adding Custom Tools" section
4. **Integrate with Your System**: Use the orchestrator in your application

## Support

For issues or questions:
1. Check the documentation
2. Review example files
3. Inspect MCP execution logs
4. Enable debug logging

---

**Happy researching with MCP-enabled SR-MARE!** 🚀
