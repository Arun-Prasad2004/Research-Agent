"""
MCP Server for tool registration, discovery, and execution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from sr_mare.mcp.registry import ToolRegistry
from sr_mare.mcp.protocol import MCPProtocol
from sr_mare.mcp.schema import (
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolDiscoveryRequest,
    ToolDiscoveryResponse,
    ToolCategory,
    ToolParameter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server for managing tool lifecycle and execution.
    
    Provides synchronous and asynchronous tool execution,
    tool discovery, and centralized tool management.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.registry = ToolRegistry()
        self.protocol = MCPProtocol(self.registry)
        self._execution_log: list = []
        
        logger.info("🚀 MCP Server initialized")
    
    def register_tool(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        implementation: Callable,
        parameters: list[ToolParameter],
        returns: str,
        version: str = "1.0.0"
    ) -> None:
        """
        Register a new tool with the server.
        
        Args:
            name: Unique tool name
            description: Tool description
            category: Tool category
            implementation: Callable implementing the tool
            parameters: List of tool parameters
            returns: Description of return value
            version: Tool version
        """
        definition = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            returns=returns,
            version=version
        )
        
        self.registry.register_tool(definition, implementation)
        logger.info(f"✓ Tool registered: {name}")
    
    def discover_tools(
        self,
        category: Optional[ToolCategory] = None,
        name_pattern: Optional[str] = None
    ) -> ToolDiscoveryResponse:
        """
        Discover available tools.
        
        Args:
            category: Filter by category
            name_pattern: Filter by name pattern
            
        Returns:
            ToolDiscoveryResponse with matching tools
        """
        request = ToolDiscoveryRequest(
            category=category,
            name_pattern=name_pattern
        )
        
        return self.protocol.handle_discovery_request(request)
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResponse:
        """
        Execute a tool synchronously.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Execution parameters
            request_id: Optional request identifier
            metadata: Optional metadata
            
        Returns:
            ToolExecutionResponse with result or error
        """
        request = ToolExecutionRequest(
            tool_name=tool_name,
            parameters=parameters,
            request_id=request_id,
            metadata=metadata or {}
        )
        
        response = self.protocol.handle_execution_request(request)
        
        # Log execution
        self._log_execution(request, response)
        
        return response
    
    async def execute_tool_async(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResponse:
        """
        Execute a tool asynchronously.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Execution parameters
            request_id: Optional request identifier
            metadata: Optional metadata
            
        Returns:
            ToolExecutionResponse with result or error
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.execute_tool,
            tool_name,
            parameters,
            request_id,
            metadata
        )
        
        return response
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get definition for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolDefinition or None if not found
        """
        return self.registry.get_tool_definition(tool_name)
    
    def list_all_tools(self) -> list[ToolDefinition]:
        """
        Get list of all registered tools.
        
        Returns:
            List of all tool definitions
        """
        return list(self.registry.get_all_tools().values())
    
    def get_tools_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """
        Get tools in a specific category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tools in the category
        """
        return self.registry.get_tools_by_category(category)
    
    def _log_execution(
        self,
        request: ToolExecutionRequest,
        response: ToolExecutionResponse
    ) -> None:
        """
        Log tool execution for audit/debugging.
        
        Args:
            request: Execution request
            response: Execution response
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": request.tool_name,
            "request_id": request.request_id,
            "success": response.success,
            "execution_time": response.execution_time,
            "error": response.error
        }
        
        self._execution_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self._execution_log) > 1000:
            self._execution_log = self._execution_log[-1000:]
    
    def get_execution_log(self, limit: int = 100) -> list[Dict[str, Any]]:
        """
        Get recent execution log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries
        """
        return self._execution_log[-limit:]
    
    def get_server_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Dictionary with server stats
        """
        total_executions = len(self._execution_log)
        successful = sum(1 for entry in self._execution_log if entry["success"])
        failed = total_executions - successful
        
        categories = {}
        for tool in self.list_all_tools():
            cat = tool.category
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "protocol_version": self.protocol.VERSION,
            "registered_tools": self.registry.get_tool_count(),
            "tools_by_category": categories,
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0.0
        }
    
    def shutdown(self) -> None:
        """Shutdown the server and cleanup resources."""
        logger.info("🛑 MCP Server shutting down")
        self._execution_log.clear()
