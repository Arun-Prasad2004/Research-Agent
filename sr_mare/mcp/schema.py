"""
Pydantic schemas for MCP protocol messages and tool definitions.
"""

from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from datetime import datetime


class ToolCategory(str, Enum):
    """Categories for organizing tools."""
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    MEMORY = "memory"
    COMPUTATION = "computation"
    UTILITY = "utility"


class ToolParameter(BaseModel):
    """Schema for tool parameter definition."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (str, int, float, bool, list, dict)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")
    
    class Config:
        use_enum_values = True


class ToolDefinition(BaseModel):
    """Schema for tool definition in the registry."""
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="What the tool does")
    category: ToolCategory = Field(..., description="Tool category")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    returns: str = Field(..., description="Description of return value")
    version: str = Field(default="1.0.0", description="Tool version")
    
    class Config:
        use_enum_values = True


class ToolExecutionRequest(BaseModel):
    """Schema for tool execution request."""
    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True


class ToolExecutionResponse(BaseModel):
    """Schema for tool execution response."""
    success: bool = Field(..., description="Whether execution succeeded")
    result: Optional[Any] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        use_enum_values = True


class ToolDiscoveryRequest(BaseModel):
    """Schema for tool discovery request."""
    category: Optional[ToolCategory] = Field(default=None, description="Filter by category")
    name_pattern: Optional[str] = Field(default=None, description="Filter by name pattern")
    
    class Config:
        use_enum_values = True


class ToolDiscoveryResponse(BaseModel):
    """Schema for tool discovery response."""
    tools: List[ToolDefinition] = Field(..., description="Available tools")
    total_count: int = Field(..., description="Total number of matching tools")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        use_enum_values = True


class MCPError(BaseModel):
    """Schema for MCP error messages."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        use_enum_values = True
