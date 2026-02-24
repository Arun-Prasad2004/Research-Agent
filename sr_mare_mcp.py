"""
SR-MARE with Model Context Protocol (MCP) Integration

This is a refactored version of the Self-Reflective Multi-Agent Research Engine
with a custom MCP layer for modular tool execution.

Architecture:
- MCP Layer: Tool registry, protocol, server, and client
- Agents: Planner, Analyst, Critic, Refiner (MCP-enabled)
- Tools: Retriever, Evaluator, Memory Store, Confidence Scorer
- Orchestrator: Coordinates agents and tool execution via MCP

Key Features:
- Decoupled tool execution through MCP protocol
- JSON-based request/response format
- Async execution support
- Tool discovery and registration
- Comprehensive logging and metrics
"""

from sr_mare.core.orchestrator import ResearchOrchestrator
from sr_mare.mcp import (
    MCPServer,
    MCPClient,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolCategory
)

__version__ = "2.0.0-mcp"

__all__ = [
    "ResearchOrchestrator",
    "MCPServer",
    "MCPClient",
    "ToolDefinition",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
    "ToolCategory"
]
