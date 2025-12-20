# -*- coding: utf-8 -*-
"""
File Node Executors - Execution logic for file operations.

Implements actual file I/O for all file nodes.
"""
import os
import shutil
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import List

from .node_executor import BaseNodeExecutor, register_executor


class ReadFileExecutor(BaseNodeExecutor):
    """Read file contents."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        encoding = executor.evaluate_input(node, "encoding") or "utf-8"
        
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            
            node.set_output("content", content)
            node.set_output("error", "")
            context.log(node, f"Read {len(content)} chars from {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("content", "")
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class WriteFileExecutor(BaseNodeExecutor):
    """Write content to file."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        content = executor.evaluate_input(node, "content") or ""
        encoding = executor.evaluate_input(node, "encoding") or "utf-8"
        append = executor.evaluate_input(node, "append") or False
        
        try:
            mode = "a" if append else "w"
            
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            
            node.set_output("error", "")
            context.log(node, f"Wrote {len(content)} chars to {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class ListDirectoryExecutor(BaseNodeExecutor):
    """List files in directory."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        pattern = executor.evaluate_input(node, "pattern") or "*"
        recursive = executor.evaluate_input(node, "recursive") or False
        
        try:
            files = []
            base_path = Path(path)
            
            # Support multiple patterns separated by ;
            patterns = [p.strip() for p in pattern.split(";")]
            
            if recursive:
                for item in base_path.rglob("*"):
                    if item.is_file():
                        if any(fnmatch.fnmatch(item.name, p) for p in patterns):
                            files.append(str(item))
            else:
                for item in base_path.iterdir():
                    if item.is_file():
                        if any(fnmatch.fnmatch(item.name, p) for p in patterns):
                            files.append(str(item))
            
            node.set_output("files", files)
            node.set_output("count", len(files))
            node.set_output("error", "")
            context.log(node, f"Found {len(files)} files matching {pattern}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("files", [])
            node.set_output("count", 0)
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class CreateDirectoryExecutor(BaseNodeExecutor):
    """Create directory."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            node.set_output("error", "")
            context.log(node, f"Created directory: {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class DeleteFileExecutor(BaseNodeExecutor):
    """Delete file or directory."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        
        try:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(path)
            else:
                p.unlink()
            
            node.set_output("error", "")
            context.log(node, f"Deleted: {path}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class CopyFileExecutor(BaseNodeExecutor):
    """Copy file."""
    
    async def execute(self, node, context, executor):
        source = executor.evaluate_input(node, "source") or ""
        dest = executor.evaluate_input(node, "destination") or ""
        overwrite = executor.evaluate_input(node, "overwrite") or False
        
        try:
            if not overwrite and Path(dest).exists():
                raise FileExistsError(f"Destination exists: {dest}")
            
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            
            node.set_output("error", "")
            context.log(node, f"Copied: {source} -> {dest}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class MoveFileExecutor(BaseNodeExecutor):
    """Move/rename file."""
    
    async def execute(self, node, context, executor):
        source = executor.evaluate_input(node, "source") or ""
        dest = executor.evaluate_input(node, "destination") or ""
        
        try:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, dest)
            
            node.set_output("error", "")
            context.log(node, f"Moved: {source} -> {dest}")
            
            await executor.execute_output_pin(node, "success")
            
        except Exception as e:
            node.set_output("error", str(e))
            context.log(node, f"Failed: {e}", "ERROR")
            
            await executor.execute_output_pin(node, "failed")


class FileExistsExecutor(BaseNodeExecutor):
    """Check if file exists."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        
        p = Path(path)
        node.set_output("exists", p.exists())
        node.set_output("is_file", p.is_file())
        node.set_output("is_directory", p.is_dir())


class PathJoinExecutor(BaseNodeExecutor):
    """Join path components."""
    
    async def execute(self, node, context, executor):
        path1 = executor.evaluate_input(node, "path1") or ""
        path2 = executor.evaluate_input(node, "path2") or ""
        path3 = executor.evaluate_input(node, "path3") or ""
        
        parts = [p for p in [path1, path2, path3] if p]
        result = str(Path(*parts)) if parts else ""
        
        node.set_output("result", result)


class GetFileInfoExecutor(BaseNodeExecutor):
    """Get file info."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        
        try:
            p = Path(path)
            stat = p.stat()
            
            node.set_output("size", stat.st_size)
            node.set_output("modified", datetime.fromtimestamp(stat.st_mtime).isoformat())
            node.set_output("extension", p.suffix)
            node.set_output("filename", p.name)
            
        except Exception:
            node.set_output("size", 0)
            node.set_output("modified", "")
            node.set_output("extension", "")
            node.set_output("filename", "")


class GetDirectoryExecutor(BaseNodeExecutor):
    """Get directory from path."""
    
    async def execute(self, node, context, executor):
        path = executor.evaluate_input(node, "path") or ""
        
        p = Path(path)
        node.set_output("directory", str(p.parent))
        node.set_output("filename", p.name)


def register_file_executors():
    """Register all file executors."""
    register_executor("ReadFile", ReadFileExecutor())
    register_executor("WriteFile", WriteFileExecutor())
    register_executor("ListDirectory", ListDirectoryExecutor())
    register_executor("CreateDirectory", CreateDirectoryExecutor())
    register_executor("DeleteFile", DeleteFileExecutor())
    register_executor("CopyFile", CopyFileExecutor())
    register_executor("MoveFile", MoveFileExecutor())
    register_executor("FileExists", FileExistsExecutor())
    register_executor("PathJoin", PathJoinExecutor())
    register_executor("GetFileInfo", GetFileInfoExecutor())
    register_executor("GetDirectory", GetDirectoryExecutor())
