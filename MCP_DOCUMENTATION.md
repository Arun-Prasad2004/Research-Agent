"""
Technical documentation for SR-MARE MCP Integration.

This document explains the Model Context Protocol (MCP) layer implementation
and how it decouples agents from direct tool execution.
"""

# SR-MARE MCP Integration Guide

## Overview

The Model Context Protocol (MCP) layer provides a standardized interface for tool
registration, discovery, and execution in the SR-MARE research system.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Research Orchestrator              │
│  - Initializes MCP Server                       │
│  - Registers tools                              │
│  - Creates MCP Client for agents                │
└─────────────────────┬───────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐   ┌────────▼────────┐
│    MCP Server     │   │   MCP Client    │
│  - Tool Registry  │   │  - Tool Query   │
│  - Protocol       │   │  - Execution    │
│  - Execution      │   │  - Discovery    │
└─────────┬─────────┘   └────────┬────────┘
          │                      │
          │    ┌─────────────────┘
          │    │
┌─────────▼────▼──────────────────────────────────┐
│               MCP Tools Layer                    │
│  - retrieve_context                              │
│  - evaluate_confidence                           │
│  - store_documents                               │
│  - score_retrieval_quality                       │
│  - compute_self_consistency                      │
│  - compute_evidence_diversity                    │
└──────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────┐
│        Core Components                        │
│  - OllamaEmbedder                             │
│  - FAISSVectorStore                           │
│  - UncertaintyEstimator                       │
│  - ResearchMetrics                            │
└───────────────────────────────────────────────┘
```

## Components

### 1. MCP Schema (schema.py)
Defines Pydantic models for type-safe communication:
- `ToolDefinition`: Tool metadata and parameters
- `ToolExecutionRequest`: Execution request format
- `ToolExecutionResponse`: Execution result format
- `ToolCategory`: Tool categorization enum

### 2. Tool Registry (registry.py)
Manages tool registration and lookup:
- Register/unregister tools
- Validate parameters
- Filter by category
- Name pattern matching

### 3. MCP Protocol (protocol.py)
Handles request/response communication:
- Discovery request handling
- Execution request handling
- JSON serialization/deserialization
- Error handling

### 4. MCP Server (server.py)
Central tool execution server:
- Tool registration endpoint
- Discovery endpoint
- Synchronous/async execution
- Execution logging and stats

### 5. MCP Client (client.py)
Agent interface to MCP server:
- Tool discovery
- Safe execution with error handling
- Result extraction
- Tool availability checks

### 6. MCP Tools (tools.py)
Wrapper implementations:
- `retrieve_context`: Vector similarity search
- `evaluate_confidence`: Uncertainty metrics
- `store_documents`: Document indexing
- `score_retrieval_quality`: Retrieval metrics
- `compute_self_consistency`: Hypothesis agreement
- `compute_evidence_diversity`: Source diversity

## Agent Integration

Agents receive an MCP client instance and use it to:

```python
# Discover available tools
tools = agent.mcp_client.discover_tools(category=ToolCategory.RETRIEVAL)

# Execute tool
result = agent.mcp_client.execute_tool(
    tool_name="retrieve_context",
    parameters={"query": question, "k": 5}
)
```

## Benefits

1. **Modularity**: Tools are independent, pluggable modules
2. **Type Safety**: Pydantic validation ensures correctness
3. **Observability**: All executions are logged with metadata
4. **Extensibility**: Easy to add new tools without changing agents
5. **Testing**: Tools can be mocked/replaced for testing
6. **Async Support**: Tools can run asynchronously when needed

## Tool Execution Flow

```
1. Agent queries available tools via MCP Client
2. Client sends discovery request to MCP Server
3. Server returns matching tool definitions
4. Agent constructs execution request with parameters
5. Client sends execution request to Server
6. Server validates parameters via Registry
7. Server retrieves tool implementation
8. Tool executes and returns result
9. Server wraps result in ToolExecutionResponse
10. Server logs execution metadata
11. Client receives response and extracts result
12. Agent processes tool result
```

## Error Handling

The MCP layer provides comprehensive error handling:

```python
# Safe execution with default on error
result = client.execute_tool_safe(
    tool_name="retrieve_context",
    parameters={"query": question},
    default_on_error={"documents": [], "num_retrieved": 0}
)

# Exception-based error handling
try:
    result = client.execute_tool(tool_name="...", parameters={...})
except RuntimeError as e:
    logger.error(f"Tool execution failed: {e}")
```

## Logging and Metrics

MCP Server tracks:
- Total executions
- Success/failure counts
- Execution times
- Tool usage patterns
- Error types

Access via:
```python
stats = server.get_server_stats()
log = server.get_execution_log(limit=100)
```

## Adding New Tools

To add a new tool:

```python
# 1. Implement the tool function
def my_new_tool(param1: str, param2: int) -> dict:
    # Tool logic here
    return {"result": ...}

# 2. Register with MCP server
server.register_tool(
    name="my_new_tool",
    description="What the tool does",
    category=ToolCategory.COMPUTATION,
    implementation=my_new_tool,
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
    returns="Description of return value"
)

# 3. Tool is now available to all agents
result = client.execute_tool(
    tool_name="my_new_tool",
    parameters={"param1": "value", "param2": 42}
)
```

## Future Enhancements

- Remote MCP server (HTTP/WebSocket)
- Tool versioning and migration
- Rate limiting and quotas
- Tool chaining and composition
- Distributed execution
- Result caching
- Tool authentication/authorization

## Conclusion

The MCP layer provides a clean, modular architecture for tool execution
while maintaining all existing SR-MARE functionality. It enables easy
extension, testing, and observability of the research pipeline.
