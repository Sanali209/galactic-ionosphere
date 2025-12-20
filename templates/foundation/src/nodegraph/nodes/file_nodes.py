# -*- coding: utf-8 -*-
"""
File Nodes - File and directory manipulation.

Provides nodes for:
- Reading/writing files
- Directory operations
- Path manipulation
- File copying/moving/deleting
"""
import os
import shutil
import fnmatch
from pathlib import Path
from typing import List, Optional

from ..core.base_node import BaseNode, NodeMetadata
from ..core.pins import ExecutionPin, DataPin, PinType, PinDirection


class ReadFileNode(BaseNode):
    """Read text content from a file."""
    node_type = "ReadFile"
    metadata = NodeMetadata(
        category="File",
        display_name="Read File",
        description="Read text content from a file",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("encoding", PinType.STRING, default_value="utf-8"))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("content", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class WriteFileNode(BaseNode):
    """Write text content to a file."""
    node_type = "WriteFile"
    metadata = NodeMetadata(
        category="File",
        display_name="Write File",
        description="Write text content to a file",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("content", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("encoding", PinType.STRING, default_value="utf-8"))
        self.add_input_pin(DataPin("append", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class ListDirectoryNode(BaseNode):
    """List files in a directory with optional wildcards."""
    node_type = "ListDirectory"
    metadata = NodeMetadata(
        category="File",
        display_name="List Directory",
        description="List files in directory with wildcard filter",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("pattern", PinType.STRING, default_value="*"))
        self.add_input_pin(DataPin("recursive", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("files", PinType.ARRAY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("count", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class CreateDirectoryNode(BaseNode):
    """Create a directory (including parents)."""
    node_type = "CreateDirectory"
    metadata = NodeMetadata(
        category="File",
        display_name="Create Directory",
        description="Create directory and parents",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class DeleteFileNode(BaseNode):
    """Delete a file or directory."""
    node_type = "DeleteFile"
    metadata = NodeMetadata(
        category="File",
        display_name="Delete File",
        description="Delete file or directory",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class CopyFileNode(BaseNode):
    """Copy a file to new location."""
    node_type = "CopyFile"
    metadata = NodeMetadata(
        category="File",
        display_name="Copy File",
        description="Copy file to new location",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("source", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("destination", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("overwrite", PinType.BOOLEAN, default_value=False))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class MoveFileNode(BaseNode):
    """Move/rename a file."""
    node_type = "MoveFile"
    metadata = NodeMetadata(
        category="File",
        display_name="Move File",
        description="Move or rename file",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(ExecutionPin("exec"))
        self.add_input_pin(DataPin("source", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("destination", PinType.PATH, default_value=""))
        self.add_output_pin(ExecutionPin("success", PinDirection.OUTPUT))
        self.add_output_pin(ExecutionPin("failed", PinDirection.OUTPUT))
        self.add_output_pin(DataPin("error", PinType.STRING, PinDirection.OUTPUT))


class FileExistsNode(BaseNode):
    """Check if a file or directory exists."""
    node_type = "FileExists"
    metadata = NodeMetadata(
        category="File",
        display_name="File Exists",
        description="Check if path exists",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_output_pin(DataPin("exists", PinType.BOOLEAN, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("is_file", PinType.BOOLEAN, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("is_directory", PinType.BOOLEAN, PinDirection.OUTPUT))


class PathJoinNode(BaseNode):
    """Join path components."""
    node_type = "PathJoin"
    metadata = NodeMetadata(
        category="File",
        display_name="Path Join",
        description="Join path components",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("path1", PinType.PATH, default_value=""))
        self.add_input_pin(DataPin("path2", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("path3", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("result", PinType.PATH, PinDirection.OUTPUT))


class GetFileInfoNode(BaseNode):
    """Get file size, modification time, etc."""
    node_type = "GetFileInfo"
    metadata = NodeMetadata(
        category="File",
        display_name="Get File Info",
        description="Get file size and modification time",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_output_pin(DataPin("size", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("modified", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("extension", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("filename", PinType.STRING, PinDirection.OUTPUT))


class GetDirectoryNode(BaseNode):
    """Get directory part of path."""
    node_type = "GetDirectory"
    metadata = NodeMetadata(
        category="File",
        display_name="Get Directory",
        description="Get directory from path",
        color="#8B4513"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("path", PinType.PATH, default_value=""))
        self.add_output_pin(DataPin("directory", PinType.PATH, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("filename", PinType.STRING, PinDirection.OUTPUT))


# Export all nodes
ALL_NODES = [
    ReadFileNode,
    WriteFileNode,
    ListDirectoryNode,
    CreateDirectoryNode,
    DeleteFileNode,
    CopyFileNode,
    MoveFileNode,
    FileExistsNode,
    PathJoinNode,
    GetFileInfoNode,
    GetDirectoryNode,
]
