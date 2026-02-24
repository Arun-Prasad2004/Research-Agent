"""
MCP Client for agents to interact with the MCP server.
"""

import logging
from typing import Dict, Any, Optional, List
from sr_mare.mcp.schema import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolDiscoveryRequest,
    ToolDiscoveryResponse,
    ToolDefinition,
    ToolCategory
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for agents to interact with MCP server.
    
    Provides a simple interface for tool discovery and execution.
    """
    
    def __init__(self, server):
        """
        Initialize MCP client.
        
        Args:
            server: MCPServer instance
        """
        self.server = server
        self._tool_cache: Optional[Dict[str, ToolDefinition]] = None
        
        logger.info("🔌 MCP Client initialized")
    
    def discover_tools(
        self,
        category: Optional[ToolCategory] = None,
        refresh_cache: bool = False
    ) -> List[ToolDefinition]:
        """
        Discover available tools.
        
        Args:
            category: Optional category filter
            refresh_cache: Force refresh of tool cache
            
        Returns:
            List of available tools
        """
        if refresh_cache or self._tool_cache is None:
            response = self.server.discover_tools(category=category)
            self._tool_cache = {tool.name: tool for tool in response.tools}
            logger.info(f"📋 Discovered {len(response.tools)} tools")
        
        if category:
            return [tool for tool in self._tool_cache.values() if tool.category == category]
        return list(self._tool_cache.values())
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool definition or None if not found
        """
        if self._tool_cache is None:
            self.discover_tools()
        
        if self._tool_cache is None:
            return None
        
        return self._tool_cache.get(tool_name)
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> Any:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            request_id: Optional request identifier
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If tool execution fails
        """
        logger.info(f"⚙️ Client executing: {tool_name}")
        
        response = self.server.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            request_id=request_id
        )
        
        if not response.success:
            logger.error(f"Tool execution failed: {response.error}")
            raise RuntimeError(f"Tool '{tool_name}' execution failed: {response.error}")
        
        logger.info(f"✓ Execution successful ({response.execution_time:.3f}s)")
        return response.result
    
    def execute_tool_safe(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        default_on_error: Any = None
    ) -> Any:
        """
        Execute a tool with error handling.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            default_on_error: Value to return on error
            
        Returns:
            Tool result or default value on error
        """
        try:
            return self.execute_tool(tool_name, parameters)
        except Exception as e:
            logger.warning(f"Tool execution failed, returning default: {e}")
            return default_on_error
    
    def list_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        tools = self.discover_tools()
        return [tool.name for tool in tools]
    
    def check_tool_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool is available
        """
        return tool_name in self.list_available_tools()
