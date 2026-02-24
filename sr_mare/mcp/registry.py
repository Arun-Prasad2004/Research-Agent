"""
Tool registry for managing MCP tool definitions and implementations.
"""

import logging
from typing import Dict, List, Callable, Optional, Any
import re
from sr_mare.mcp.schema import ToolDefinition, ToolCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool definitions and implementations."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._implementations: Dict[str, Callable] = {}
        logger.info("🔧 Tool Registry initialized")
    
    def register_tool(
        self,
        definition: ToolDefinition,
        implementation: Callable
    ) -> None:
        """
        Register a new tool with its definition and implementation.
        
        Args:
            definition: Tool definition schema
            implementation: Callable that implements the tool
            
        Raises:
            ValueError: If tool name already exists
        """
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        
        self._tools[definition.name] = definition
        self._implementations[definition.name] = implementation
        
        logger.info(f"✓ Registered tool: {definition.name} ({definition.category})")
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the registry.
        
        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            del self._implementations[tool_name]
            logger.info(f"✓ Unregistered tool: {tool_name}")
        else:
            logger.warning(f"Tool '{tool_name}' not found in registry")
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get tool definition by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_tool_implementation(self, tool_name: str) -> Optional[Callable]:
        """
        Get tool implementation by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool implementation callable or None if not found
        """
        return self._implementations.get(tool_name)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        name_pattern: Optional[str] = None
    ) -> List[ToolDefinition]:
        """
        List all registered tools with optional filtering.
        
        Args:
            category: Filter by tool category
            name_pattern: Filter by name pattern (regex)
            
        Returns:
            List of matching tool definitions
        """
        tools = list(self._tools.values())
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.category == category]
        
        # Filter by name pattern
        if name_pattern:
            try:
                pattern = re.compile(name_pattern, re.IGNORECASE)
                tools = [t for t in tools if pattern.search(t.name)]
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{name_pattern}': {e}")
        
        return tools
    
    def get_all_tools(self) -> Dict[str, ToolDefinition]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tool name to definition
        """
        return self._tools.copy()
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Tool category to filter by
            
        Returns:
            List of tool definitions in the category
        """
        return [tool for tool in self._tools.values() if tool.category == category]
    
    def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool exists, False otherwise
        """
        return tool_name in self._tools
    
    def get_tool_count(self) -> int:
        """
        Get total number of registered tools.
        
        Returns:
            Number of registered tools
        """
        return len(self._tools)
    
    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parameters for a tool execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        tool_def = self.get_tool_definition(tool_name)
        if not tool_def:
            return False, f"Tool '{tool_name}' not found"
        
        # Check required parameters
        for param in tool_def.parameters:
            if param.required and param.name not in parameters:
                return False, f"Required parameter '{param.name}' is missing"
        
        # Check for unknown parameters
        valid_param_names = {param.name for param in tool_def.parameters}
        for param_name in parameters:
            if param_name not in valid_param_names:
                return False, f"Unknown parameter '{param_name}'"
        
        return True, None
