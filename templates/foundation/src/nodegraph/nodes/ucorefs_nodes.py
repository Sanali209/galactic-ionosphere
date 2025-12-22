# -*- coding: utf-8 -*-
"""
UCoreFS Integration Nodes for NodeGraph.

Bridges NodeGraph visual programming with UCoreFS filesystem database.
"""
from typing import List, Optional, Any
from loguru import logger

from src.nodegraph.core.base_node import BaseNode, NodeMetadata
from src.nodegraph.core.pins import DataPin, PinType, PinDirection


class FileQueryNode(BaseNode):
    """
    Query files from UCoreFS database.
    
    Inputs:
        - pattern: Search pattern (regex supported)
        - file_type: Optional file type filter (e.g., "image")
        - limit: Maximum results
        
    Outputs:
        - files: List of FileRecord IDs
        - count: Number of results
    """
    node_type = "ucorefs.FileQuery"
    metadata = NodeMetadata(
        category="UCoreFS",
        display_name="Query Files",
        description="Search files in UCoreFS database",
        color="#2E7D32"  # Green
    )
    
    def _setup_pins(self):
        # Inputs
        self.add_input_pin(DataPin("pattern", PinType.STRING, default_value=".*"))
        self.add_input_pin(DataPin("file_type", PinType.STRING, default_value=""))
        self.add_input_pin(DataPin("limit", PinType.INTEGER, default_value=100))
        
        # Outputs
        self.add_output_pin(DataPin("files", PinType.ANY, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("count", PinType.INTEGER, PinDirection.OUTPUT))
    
    async def execute(self, context: dict) -> dict:
        """Execute file query."""
        from src.ucorefs.core.fs_service import FSService
        
        pattern = self.get_input("pattern")
        file_type = self.get_input("file_type") or None
        limit = self.get_input("limit")
        
        try:
            fs_service = context.get("locator").get_system(FSService)
            results = await fs_service.search_by_name(pattern, file_type, limit)
            
            file_ids = [str(r.id) for r in results]
            
            self.set_output("files", file_ids)
            self.set_output("count", len(file_ids))
            
            return {"files": file_ids, "count": len(file_ids)}
        except Exception as e:
            self.set_error(str(e))
            return {"files": [], "count": 0}


class GetFileByPathNode(BaseNode):
    """
    Get a specific file by its path.
    
    Inputs:
        - path: Absolute file path
        
    Outputs:
        - file_id: FileRecord ID (or None)
        - found: Boolean indicating if file exists
    """
    node_type = "ucorefs.GetFileByPath"
    metadata = NodeMetadata(
        category="UCoreFS",
        display_name="Get File by Path",
        description="Find a file by its absolute path",
        color="#2E7D32"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("path", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("file_id", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("found", PinType.BOOLEAN, PinDirection.OUTPUT))
    
    async def execute(self, context: dict) -> dict:
        from src.ucorefs.core.fs_service import FSService
        
        path = self.get_input("path")
        
        try:
            fs_service = context.get("locator").get_system(FSService)
            record = await fs_service.get_by_path(path)
            
            if record:
                self.set_output("file_id", str(record.id))
                self.set_output("found", True)
                return {"file_id": str(record.id), "found": True}
            else:
                self.set_output("file_id", None)
                self.set_output("found", False)
                return {"file_id": None, "found": False}
        except Exception as e:
            self.set_error(str(e))
            return {"file_id": None, "found": False}


class TagFilesNode(BaseNode):
    """
    Apply tags to files.
    
    Inputs:
        - file_ids: List of file IDs to tag
        - tag_name: Tag name to apply (creates if doesn't exist)
        
    Outputs:
        - success_count: Number of files tagged
    """
    node_type = "ucorefs.TagFiles"
    metadata = NodeMetadata(
        category="UCoreFS",
        display_name="Tag Files",
        description="Apply a tag to multiple files",
        color="#1565C0"  # Blue
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("file_ids", PinType.ANY, default_value=[]))
        self.add_input_pin(DataPin("tag_name", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("success_count", PinType.INTEGER, PinDirection.OUTPUT))
    
    async def execute(self, context: dict) -> dict:
        from bson import ObjectId
        from src.ucorefs.tags.manager import TagManager
        from src.ucorefs.models.file_record import FileRecord
        
        file_ids = self.get_input("file_ids") or []
        tag_name = self.get_input("tag_name")
        
        if not tag_name:
            self.set_error("Tag name is required")
            return {"success_count": 0}
        
        try:
            tag_manager = context.get("locator").get_system(TagManager)
            
            # Get or create tag
            tag = await tag_manager.get_or_create(tag_name)
            
            success = 0
            for fid in file_ids:
                try:
                    record = await FileRecord.get(ObjectId(fid))
                    if record:
                        if tag.id not in (record.tag_ids or []):
                            record.tag_ids = (record.tag_ids or []) + [tag.id]
                            await record.save()
                            success += 1
                except Exception:
                    pass
            
            self.set_output("success_count", success)
            return {"success_count": success}
        except Exception as e:
            self.set_error(str(e))
            return {"success_count": 0}


class GetFileMetadataNode(BaseNode):
    """
    Get metadata for a file.
    
    Inputs:
        - file_id: FileRecord ID
        
    Outputs:
        - name: Filename
        - path: Full path
        - size: File size in bytes
        - extension: File extension
        - file_type: File type category
    """
    node_type = "ucorefs.GetFileMetadata"
    metadata = NodeMetadata(
        category="UCoreFS",
        display_name="Get File Metadata",
        description="Retrieve metadata for a file",
        color="#2E7D32"
    )
    
    def _setup_pins(self):
        self.add_input_pin(DataPin("file_id", PinType.STRING, default_value=""))
        self.add_output_pin(DataPin("name", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("path", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("size", PinType.INTEGER, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("extension", PinType.STRING, PinDirection.OUTPUT))
        self.add_output_pin(DataPin("file_type", PinType.STRING, PinDirection.OUTPUT))
    
    async def execute(self, context: dict) -> dict:
        from bson import ObjectId
        from src.ucorefs.models.file_record import FileRecord
        
        file_id = self.get_input("file_id")
        
        if not file_id:
            self.set_error("File ID is required")
            return {}
        
        try:
            record = await FileRecord.get(ObjectId(file_id))
            
            if record:
                self.set_output("name", record.name)
                self.set_output("path", record.path)
                self.set_output("size", record.size or 0)
                self.set_output("extension", record.extension or "")
                self.set_output("file_type", record.file_type or "")
                
                return {
                    "name": record.name,
                    "path": record.path,
                    "size": record.size or 0,
                    "extension": record.extension or "",
                    "file_type": record.file_type or ""
                }
            else:
                self.set_error("File not found")
                return {}
        except Exception as e:
            self.set_error(str(e))
            return {}


# Register nodes with NodeGraph
def register_ucorefs_nodes():
    """Register UCoreFS nodes with the NodeGraph registry."""
    from src.nodegraph.core.node_registry import NodeRegistry
    
    registry = NodeRegistry()
    registry.register(FileQueryNode)
    registry.register(GetFileByPathNode)
    registry.register(TagFilesNode)
    registry.register(GetFileMetadataNode)
    
    logger.info("Registered 4 UCoreFS bridge nodes")
