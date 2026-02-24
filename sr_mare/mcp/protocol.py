"""
Core MCP protocol implementation for request/response handling.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from sr_mare.mcp.schema import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolDiscoveryRequest,
    ToolDiscoveryResponse,
    MCPError
)
from sr_mare.mcp.registry import ToolRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPProtocol:
    """Protocol handler for MCP communication."""
    
    VERSION = "1.0.0"
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the MCP protocol handler.
        
        Args:
            registry: Tool registry instance
        """
        self.registry = registry
        logger.info(f"🔗 MCP Protocol v{self.VERSION} initialized")
    
    def handle_discovery_request(
        self,
        request: ToolDiscoveryRequest
    ) -> ToolDiscoveryResponse:
        """
        Handle tool discovery request.
        
        Args:
            request: Discovery request
            
        Returns:
            Discovery response with available tools
        """
        logger.info("📋 Handling tool discovery request")
        
        tools = self.registry.list_tools(
            category=request.category,
            name_pattern=request.name_pattern
        )
        
        response = ToolDiscoveryResponse(
            tools=tools,
            total_count=len(tools)
        )
        
        logger.info(f"✓ Found {len(tools)} matching tools")
        return response
    
    def handle_execution_request(
        self,
        request: ToolExecutionRequest
    ) -> ToolExecutionResponse:
        """
        Handle tool execution request.
        
        Args:
            request: Execution request
            
        Returns:
            Execution response with result or error
        """
        start_time = datetime.now()
        logger.info(f"⚙️ Executing tool: {request.tool_name}")
        
        try:
            # Validate tool exists
            if not self.registry.tool_exists(request.tool_name):
                raise ValueError(f"Tool '{request.tool_name}' not found")
            
            # Validate parameters
            is_valid, error_msg = self.registry.validate_parameters(
                request.tool_name,
                request.parameters
            )
            if not is_valid:
                raise ValueError(error_msg)
            
            # Get implementation
            implementation = self.registry.get_tool_implementation(request.tool_name)
            if not implementation:
                raise RuntimeError(f"No implementation found for '{request.tool_name}'")
            
            # Execute tool
            result = implementation(**request.parameters)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Build response
            response = ToolExecutionResponse(
                success=True,
                result=result,
                execution_time=execution_time,
                request_id=request.request_id,
                metadata={
                    "tool_name": request.tool_name,
                    "parameters": request.parameters,
                    **request.metadata
                }
            )
            
            logger.info(f"✓ Tool execution successful ({execution_time:.3f}s)")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"✗ Tool execution failed: {str(e)}")
            
            return ToolExecutionResponse(
                success=False,
                error=str(e),
                execution_time=execution_time,
                request_id=request.request_id,
                metadata={
                    "tool_name": request.tool_name,
                    "error_type": type(e).__name__
                }
            )
    
    def serialize_request(self, request: Any) -> str:
        """
        Serialize request to JSON string.
        
        Args:
            request: Request object (Pydantic model)
            
        Returns:
            JSON string
        """
        return request.model_dump_json(indent=2)
    
    def serialize_response(self, response: Any) -> str:
        """
        Serialize response to JSON string.
        
        Args:
            response: Response object (Pydantic model)
            
        Returns:
            JSON string
        """
        return response.model_dump_json(indent=2)
    
    def deserialize_execution_request(self, json_str: str) -> ToolExecutionRequest:
        """
        Deserialize JSON string to execution request.
        
        Args:
            json_str: JSON string
            
        Returns:
            ToolExecutionRequest object
        """
        try:
            return ToolExecutionRequest.model_validate_json(json_str)
        except Exception as e:
            logger.error(f"Failed to deserialize execution request: {e}")
            raise ValueError(f"Invalid execution request format: {e}")
    
    def deserialize_discovery_request(self, json_str: str) -> ToolDiscoveryRequest:
        """
        Deserialize JSON string to discovery request.
        
        Args:
            json_str: JSON string
            
        Returns:
            ToolDiscoveryRequest object
        """
        try:
            return ToolDiscoveryRequest.model_validate_json(json_str)
        except Exception as e:
            logger.error(f"Failed to deserialize discovery request: {e}")
            raise ValueError(f"Invalid discovery request format: {e}")
    
    def create_error_response(
        self,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> MCPError:
        """
        Create a standardized error response.
        
        Args:
            error_code: Error code
            error_message: Human-readable error message
            details: Additional error details
            
        Returns:
            MCPError object
        """
        return MCPError(
            error_code=error_code,
            error_message=error_message,
            details=details or {}
        )
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """
        Get protocol information.
        
        Returns:
            Dictionary with protocol metadata
        """
        return {
            "protocol": "MCP",
            "version": self.VERSION,
            "capabilities": [
                "tool_discovery",
                "tool_execution",
                "async_execution",
                "parameter_validation"
            ],
            "registered_tools": self.registry.get_tool_count()
        }
