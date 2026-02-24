# SR-MARE with Model Context Protocol (MCP) Integration

## 🎯 Overview

SR-MARE (Self-Reflective Multi-Agent Research Engine) has been refactored to integrate a custom **Model Context Protocol (MCP)** layer. This architecture decouples agents from direct tool calls, providing a modular, extensible, and observable research pipeline.

## 🏗️ Architecture

### Before (Direct Tool Calls)
```
Agents → Direct Function Calls → Components (Retrieval, Evaluation, etc.)
```

### After (MCP Layer)
```
Agents → MCP Client → MCP Server → MCP Tools → Components
```

## 📁 New Structure

```
sr_mare/
├── mcp/
│   ├── __init__.py          # MCP module exports
│   ├── schema.py            # Pydantic models for MCP protocol
│   ├── registry.py          # Tool registration and management
│   ├── protocol.py          # Request/response handling
│   ├── server.py            # MCP server with tool execution
│   ├── client.py            # MCP client for agents
│   └── tools.py             # MCP tool implementations
├── agents/
│   ├── planner.py           # ✨ Now MCP-enabled
│   ├── analyst.py           # ✨ Now MCP-enabled
│   ├── critic.py            # ✨ Now MCP-enabled
│   └── refiner.py           # ✨ Now MCP-enabled
├── core/
│   └── orchestrator.py      # ✨ Updated to use MCP
├── retrieval/
│   ├── embedder.py
│   └── vector_store.py
└── evaluation/
    ├── uncertainty.py
    └── metrics.py
```

## 🔧 MCP Tools

The following tools are registered in the MCP layer:

| Tool Name | Category | Description |
|-----------|----------|-------------|
| `retrieve_context` | RETRIEVAL | Vector similarity search for relevant documents |
| `evaluate_confidence` | EVALUATION | Compute uncertainty and confidence metrics |
| `store_documents` | MEMORY | Index documents in vector store |
| `score_retrieval_quality` | EVALUATION | Compute retrieval metrics |
| `compute_self_consistency` | COMPUTATION | Measure hypothesis agreement |
| `compute_evidence_diversity` | COMPUTATION | Measure source diversity |

## 🚀 Usage

### Basic Example

```python
from sr_mare.core.orchestrator import ResearchOrchestrator

# Initialize with MCP
orchestrator = ResearchOrchestrator(
    base_url="http://localhost:11434",
    max_iterations=3,
    confidence_threshold=0.75
)

# Load documents (via MCP)
documents = ["doc1", "doc2", "doc3"]
orchestrator.load_documents(documents)

# Run research query
result = orchestrator.research(
    "What is machine learning?",
    top_k=5
)

print(f"Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence_score']:.3f}")
```

### Run Examples

```bash
# Run MCP-enabled example
python example_mcp.py

# Run original main script (now MCP-enabled)
python main.py --question "What is AI?" --documents sr_mare/data/documents.txt
```

## 🎨 Key Features

### 1. **Tool Discovery**
Agents can query available tools:

```python
# In agent code
tools = self.mcp_client.discover_tools(category=ToolCategory.RETRIEVAL)
for tool in tools:
    print(f"{tool.name}: {tool.description}")
```

### 2. **Structured Tool Execution**
```python
# Execute tool with parameters
result = self.mcp_client.execute_tool(
    tool_name="retrieve_context",
    parameters={"query": "machine learning", "k": 5}
)
```

### 3. **Async Support**
```python
# Async execution
result = await server.execute_tool_async(
    tool_name="evaluate_confidence",
    parameters={...}
)
```

### 4. **Execution Logging**
All tool executions are logged with metadata:

```python
stats = orchestrator.mcp_server.get_server_stats()
print(f"Total Executions: {stats['total_executions']}")
print(f"Success Rate: {stats['success_rate']:.2%}")
```

### 5. **Type Safety**
Pydantic models ensure type-safe communication:

```python
from sr_mare.mcp.schema import ToolExecutionRequest

request = ToolExecutionRequest(
    tool_name="retrieve_context",
    parameters={"query": "AI", "k": 5}
)
```

## 📊 Benefits

✅ **Modularity**: Tools are independent, pluggable modules  
✅ **Type Safety**: Pydantic validation ensures correctness  
✅ **Observability**: All executions logged with comprehensive metadata  
✅ **Extensibility**: Add new tools without modifying agents  
✅ **Testability**: Tools can be mocked/replaced for testing  
✅ **Async Support**: Tools can execute asynchronously  
✅ **Clean Architecture**: Clear separation of concerns  

## 🔌 Adding New Tools

Adding a new tool is straightforward:

```python
# 1. Create tool function
def my_custom_tool(param1: str, param2: int) -> dict:
    # Your tool logic
    return {"result": "success"}

# 2. Register with MCP
orchestrator.mcp_server.register_tool(
    name="my_custom_tool",
    description="Does something useful",
    category=ToolCategory.UTILITY,
    implementation=my_custom_tool,
    parameters=[
        ToolParameter(
            name="param1",
            type="str",
            description="First parameter",
            required=True
        ),
        ToolParameter(
            name="param2",
            type="int",
            description="Second parameter",
            required=False,
            default=10
        )
    ],
    returns="Description of what the tool returns"
)

# 3. Use in agents
result = mcp_client.execute_tool(
    tool_name="my_custom_tool",
    parameters={"param1": "value", "param2": 42}
)
```

## 📚 Documentation

- **[MCP_DOCUMENTATION.md](MCP_DOCUMENTATION.md)**: Detailed technical documentation
- **[example_mcp.py](example_mcp.py)**: Complete working example

## 🔍 What Was Changed

### Core Components

1. **orchestrator.py**:
   - Initializes MCP server
   - Registers all tools
   - Creates MCP client for agents
   - Uses MCP for retrieval and evaluation

2. **All Agents** (planner, analyst, critic, refiner):
   - Added `mcp_client` parameter
   - Can query and execute tools via MCP
   - Maintain backward compatibility

3. **New MCP Layer**:
   - Complete protocol implementation
   - Tool registry and management
   - Server and client architecture
   - Tool wrapper implementations

## 🔄 Maintained Functionality

✅ All existing features work unchanged:
- Iterative refinement loop
- Self-consistency sampling
- Confidence scoring
- Multi-agent collaboration
- Vector retrieval
- Uncertainty estimation
- Metrics tracking

## 🧪 Testing

```bash
# Verify setup
python verify_setup.py

# Run MCP example
python example_mcp.py

# Run with web interface
python start_web.py
```

## 📈 MCP Statistics

The MCP server tracks execution statistics:

```python
stats = orchestrator.mcp_server.get_server_stats()
# {
#   "protocol_version": "1.0.0",
#   "registered_tools": 6,
#   "tools_by_category": {...},
#   "total_executions": 42,
#   "successful_executions": 40,
#   "failed_executions": 2,
#   "success_rate": 0.95
# }
```

## 🛠️ Requirements

Same as original SR-MARE:
- Python 3.8+
- Ollama (running locally)
- Dependencies in `requirements.txt`

## 📝 License

Same license as original SR-MARE project.

## 🎓 Learn More

- Read the [MCP Documentation](MCP_DOCUMENTATION.md)
- Check out [example_mcp.py](example_mcp.py) for a complete example
- Explore the [mcp/](sr_mare/mcp/) folder for implementation details

---

**Refactored Architecture**: From direct tool calls to modular MCP-based tool execution! 🚀
