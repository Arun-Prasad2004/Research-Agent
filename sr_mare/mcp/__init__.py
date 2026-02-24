"""
Model Context Protocol (MCP) layer for SR-MARE.

This module provides a custom MCP implementation for tool registration,
discovery, and execution with agents.
"""

from sr_mare.mcp.server import MCPServer
from sr_mare.mcp.registry import ToolRegistry
from sr_mare.mcp.protocol import MCPProtocol
from sr_mare.mcp.schema import (
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolParameter,
    ToolCategory
)

__all__ = [
    "MCPServer",
    "ToolRegistry",
    "MCPProtocol",
    "ToolDefinition",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
    "ToolParameter",
    "ToolCategory"
]
