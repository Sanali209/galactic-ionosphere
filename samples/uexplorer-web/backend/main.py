"""
UExplorer Web - FastAPI Backend

A web-based file explorer API that provides endpoints for browsing
the local file system, similar to the desktop UExplorer application.
"""
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# Models
class FileInfo(BaseModel):
    """File or directory information"""
    name: str
    path: str
    is_directory: bool
    size: Optional[int] = None
    modified: Optional[str] = None
    extension: Optional[str] = None


class DirectoryContent(BaseModel):
    """Directory contents response"""
    current_path: str
    parent_path: Optional[str]
    items: List[FileInfo]
    total_items: int


class DirectoryChangeRequest(BaseModel):
    """Request to change directory"""
    path: str


class SearchResult(BaseModel):
    """Search results"""
    query: str
    results: List[FileInfo]
    total_results: int


# Initialize FastAPI app
app = FastAPI(
    title="UExplorer Web API",
    description="File system browsing API inspired by UExplorer",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for current directory
current_directory = str(Path.home())


def get_file_info(path: Path) -> FileInfo:
    """Get information about a file or directory"""
    try:
        stat = path.stat()
        return FileInfo(
            name=path.name,
            path=str(path),
            is_directory=path.is_dir(),
            size=stat.st_size if path.is_file() else None,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            extension=path.suffix if path.is_file() else None
        )
    except Exception as e:
        return FileInfo(
            name=path.name,
            path=str(path),
            is_directory=path.is_dir(),
            size=None,
            modified=None,
            extension=None
        )


def is_safe_path(requested_path: str) -> bool:
    """
    Check if the requested path is safe to access.
    In production, implement proper security checks.
    """
    if not requested_path:
        return False
    try:
        path = Path(requested_path).resolve()
        return path.exists()
    except Exception:
        return False


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "UExplorer Web API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/api/directory/current", response_model=dict)
async def get_current_directory():
    """Get the current directory"""
    return {
        "path": current_directory,
        "exists": Path(current_directory).exists()
    }


@app.post("/api/directory/change")
async def change_directory(request: DirectoryChangeRequest):
    """Change the current directory"""
    global current_directory
    
    if not is_safe_path(request.path):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    path = Path(request.path).resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    current_directory = str(path)
    return {
        "success": True,
        "current_path": current_directory
    }


@app.get("/api/browse", response_model=DirectoryContent)
async def browse_directory(
    path: Optional[str] = Query(None, description="Directory path to browse"),
    show_hidden: bool = Query(False, description="Show hidden files")
):
    """
    Browse files and directories.
    If no path is provided, uses the current directory.
    """
    target_path = path if path else current_directory
    
    if not is_safe_path(target_path):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    dir_path = Path(target_path).resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    try:
        items = []
        for item in dir_path.iterdir():
            # Skip hidden files if not requested
            if not show_hidden and item.name.startswith('.'):
                continue
            items.append(get_file_info(item))
        
        # Sort: directories first, then by name
        items.sort(key=lambda x: (not x.is_directory, x.name.lower()))
        
        parent = str(dir_path.parent) if dir_path.parent != dir_path else None
        
        return DirectoryContent(
            current_path=str(dir_path),
            parent_path=parent,
            items=items,
            total_items=len(items)
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading directory: {str(e)}")


@app.get("/api/file/{path:path}", response_model=FileInfo)
async def get_file_details(path: str):
    """Get detailed information about a specific file"""
    if not is_safe_path(path):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return get_file_info(file_path)


@app.get("/api/search", response_model=SearchResult)
async def search_files(
    query: str = Query(..., description="Search query"),
    path: Optional[str] = Query(None, description="Directory to search in"),
    recursive: bool = Query(True, description="Search recursively")
):
    """
    Search for files and directories by name.
    """
    search_path = Path(path if path else current_directory).resolve()
    
    if not is_safe_path(str(search_path)):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    if not search_path.is_dir():
        raise HTTPException(status_code=400, detail="Search path is not a directory")
    
    results = []
    query_lower = query.lower()
    
    try:
        if recursive:
            # Recursive search
            for item in search_path.rglob("*"):
                if query_lower in item.name.lower():
                    results.append(get_file_info(item))
                    if len(results) >= 100:  # Limit results
                        break
        else:
            # Non-recursive search
            for item in search_path.iterdir():
                if query_lower in item.name.lower():
                    results.append(get_file_info(item))
        
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results)
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/home")
async def get_home_directory():
    """Get the user's home directory"""
    return {
        "home": str(Path.home()),
        "current": current_directory
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 UExplorer Web API Starting")
    print("=" * 60)
    print(f"📁 Initial directory: {current_directory}")
    print(f"🌐 API will be available at: http://localhost:8000")
    print(f"📚 API docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
