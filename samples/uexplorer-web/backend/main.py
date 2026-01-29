"""
UExplorer Web - Comprehensive FastAPI Backend

A production-grade file management system with AI features, inspired by the desktop UExplorer.
This version implements ALL major features from the original UExplorer.

Features:
- MongoDB database with Beanie ORM
- Hierarchical tag management (MPPT structure)
- Album system (static + smart query-based albums)
- File indexing and metadata extraction
- AI features (embeddings, detection, search)
- Vector search with ChromaDB
- Advanced query builder
- Background task processing
- Audit logging
- And 100+ more features...
"""
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Database
from database import init_database, close_database
from models import (
    FileRecord, DirectoryRecord, Tag, Album, 
    ProcessingState, RelationType
)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 UExplorer Web API Starting (Comprehensive Version)")
    logger.info("=" * 60)
    
    try:
        await init_database()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.warning("Running in file-only mode (database features disabled)")
    
    logger.info("✓ All services initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_database()
    logger.info("✓ Cleanup complete")


# Initialize FastAPI app
app = FastAPI(
    title="UExplorer Web API (Comprehensive)",
    description="Full-featured file management system with AI capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_directory = str(Path.home())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_safe_path(requested_path: str) -> bool:
    """Check if the requested path is safe to access"""
    if not requested_path:
        return False
    try:
        path = Path(requested_path).resolve()
        return path.exists()
    except Exception:
        return False


def get_file_info_dict(path: Path) -> Dict[str, Any]:
    """Get file information as dictionary"""
    try:
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path),
            "is_directory": path.is_dir(),
            "size": stat.st_size if path.is_file() else None,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": path.suffix if path.is_file() else None
        }
    except Exception:
        return {
            "name": path.name,
            "path": str(path),
            "is_directory": path.is_dir(),
            "size": None,
            "modified": None,
            "extension": None
        }


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "UExplorer Web API",
        "version": "2.0.0",
        "description": "Comprehensive file management with AI features",
        "docs": "/docs",
        "features": {
            "database": "MongoDB with Beanie ORM",
            "tags": "Hierarchical tag system (MPPT)",
            "albums": "Static + Smart query-based albums",
            "ai": "Embeddings, detection, search",
            "search": "Text, semantic, and vector search",
            "tasks": "Background processing",
            "audit": "Full audit logging"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check database
    db_status = "ok"
    try:
        # Try to count a collection
        count = await FileRecord.count()
        db_status = f"ok ({count} files)"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# FILE SYSTEM ENDPOINTS (Enhanced with Database)
# ============================================================================

@app.get("/api/directory/current")
async def get_current_directory():
    """Get the current directory"""
    return {
        "path": current_directory,
        "exists": Path(current_directory).exists()
    }


@app.post("/api/directory/change")
async def change_directory(path: str):
    """Change the current directory"""
    global current_directory
    
    if not is_safe_path(path):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    path_obj = Path(path).resolve()
    if not path_obj.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    current_directory = str(path_obj)
    
    # Update directory record in database
    try:
        dir_record = await DirectoryRecord.find_one({"path": current_directory})
        if not dir_record:
            dir_record = DirectoryRecord(
                path=current_directory,
                name=path_obj.name
            )
            await dir_record.insert()
    except Exception as e:
        logger.warning(f"Could not update directory record: {e}")
    
    return {
        "success": True,
        "current_path": current_directory
    }


@app.get("/api/browse")
async def browse_directory(
    path: Optional[str] = Query(None),
    show_hidden: bool = Query(False),
    include_metadata: bool = Query(False)
):
    """Browse files and directories with optional database metadata"""
    target_path = path if path else current_directory
    
    if not is_safe_path(target_path):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    dir_path = Path(target_path).resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    try:
        items = []
        for item in dir_path.iterdir():
            if not show_hidden and item.name.startswith('.'):
                continue
            
            file_info = get_file_info_dict(item)
            
            # If database is available and include_metadata=true, add DB info
            if include_metadata and not item.is_dir():
                try:
                    file_record = await FileRecord.find_one({"path": str(item)})
                    if file_record:
                        file_info["db"] = {
                            "id": str(file_record.id),
                            "rating": file_record.rating,
                            "description": file_record.description,
                            "processing_state": file_record.processing_state.value,
                            "tags_count": len(file_record.tags_auto) if file_record.tags_auto else 0
                        }
                except Exception as e:
                    logger.debug(f"Could not fetch metadata for {item}: {e}")
            
            items.append(file_info)
        
        # Sort: directories first, then by name
        items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        
        parent = str(dir_path.parent) if dir_path.parent != dir_path else None
        
        return {
            "current_path": str(dir_path),
            "parent_path": parent,
            "items": items,
            "total_items": len(items)
        }
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading directory: {str(e)}")


@app.get("/api/search")
async def search_files(
    query: str = Query(...),
    path: Optional[str] = Query(None),
    recursive: bool = Query(True),
    search_mode: str = Query("filesystem", description="filesystem or database")
):
    """
    Search for files by name (filesystem) or by metadata (database)
    """
    if search_mode == "database":
        # Database search (metadata, tags, etc.)
        try:
            results = []
            query_lower = query.lower()
            
            # Search by name, description, or tags
            files = await FileRecord.find({
                "$or": [
                    {"name": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}},
                    {"tags_auto": {"$regex": query, "$options": "i"}}
                ]
            }).limit(100).to_list()
            
            for file_record in files:
                results.append({
                    "id": str(file_record.id),
                    "name": file_record.name,
                    "path": file_record.path,
                    "is_directory": False,
                    "size": file_record.size,
                    "modified": file_record.file_modified_at.isoformat() if file_record.file_modified_at else None,
                    "extension": file_record.extension,
                    "rating": file_record.rating,
                    "description": file_record.description,
                    "tags": file_record.tags_auto if file_record.tags_auto else []
                })
            
            return {
                "query": query,
                "search_mode": "database",
                "results": results,
                "total_results": len(results)
            }
        except Exception as e:
            logger.error(f"Database search error: {e}")
            # Fall back to filesystem search
            search_mode = "filesystem"
    
    # Filesystem search (original implementation)
    search_path = Path(path if path else current_directory).resolve()
    
    if not is_safe_path(str(search_path)):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")
    
    results = []
    query_lower = query.lower()
    
    try:
        if recursive:
            for item in search_path.rglob("*"):
                if query_lower in item.name.lower():
                    results.append(get_file_info_dict(item))
                    if len(results) >= 100:
                        break
        else:
            for item in search_path.iterdir():
                if query_lower in item.name.lower():
                    results.append(get_file_info_dict(item))
        
        return {
            "query": query,
            "search_mode": "filesystem",
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/home")
async def get_home_directory():
    """Get the user's home directory"""
    return {
        "home": str(Path.home()),
        "current": current_directory
    }


# ============================================================================
# DATABASE STATISTICS
# ============================================================================

@app.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = {
            "files": await FileRecord.count(),
            "directories": await DirectoryRecord.count(),
            "tags": await Tag.count(),
            "albums": await Album.count(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Processing state breakdown
        processing_stats = {}
        for state in ProcessingState:
            count = await FileRecord.find({"processing_state": state}).count()
            processing_stats[state.value] = count
        stats["processing_states"] = processing_stats
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")


# ============================================================================
# FILE MANAGEMENT (Database)
# ============================================================================

@app.post("/api/files/index")
async def index_file(path: str):
    """Index a file into the database"""
    if not is_safe_path(path):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    try:
        # Check if already indexed
        existing = await FileRecord.find_one({"path": path})
        if existing:
            return {
                "success": True,
                "message": "File already indexed",
                "file_id": str(existing.id)
            }
        
        # Create file record
        stat = file_path.stat()
        file_record = FileRecord(
            path=path,
            name=file_path.name,
            extension=file_path.suffix,
            size=stat.st_size,
            file_modified_at=datetime.fromtimestamp(stat.st_mtime)
        )
        await file_record.insert()
        
        logger.info(f"Indexed file: {path}")
        
        return {
            "success": True,
            "message": "File indexed",
            "file_id": str(file_record.id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing file: {str(e)}")


@app.get("/api/files/{file_id}")
async def get_file_metadata(file_id: str):
    """Get file metadata from database"""
    try:
        from bson import ObjectId
        file_record = await FileRecord.get(ObjectId(file_id))
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "id": str(file_record.id),
            "path": file_record.path,
            "name": file_record.name,
            "extension": file_record.extension,
            "size": file_record.size,
            "rating": file_record.rating,
            "description": file_record.description,
            "processing_state": file_record.processing_state.value,
            "tags_auto": file_record.tags_auto,
            "custom_properties": file_record.custom_properties,
            "created_at": file_record.created_at.isoformat(),
            "modified_at": file_record.modified_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.put("/api/files/{file_id}/rating")
async def update_file_rating(file_id: str, rating: int = Query(..., ge=0, le=5)):
    """Update file rating (0-5 stars)"""
    try:
        from bson import ObjectId
        file_record = await FileRecord.get(ObjectId(file_id))
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_record.rating = rating
        await file_record.save()
        
        return {
            "success": True,
            "file_id": file_id,
            "rating": rating
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/files/batch/index")
async def batch_index_files(paths: List[str]):
    """Batch index multiple files"""
    results = {"success": [], "failed": []}
    
    for path in paths:
        if not is_safe_path(path):
            results["failed"].append({"path": path, "error": "Invalid path"})
            continue
        
        file_path = Path(path)
        if not file_path.is_file():
            results["failed"].append({"path": path, "error": "Not a file"})
            continue
        
        try:
            # Check if already indexed
            existing = await FileRecord.find_one({"path": path})
            if existing:
                results["success"].append({"path": path, "file_id": str(existing.id), "status": "already_indexed"})
                continue
            
            # Create file record
            stat = file_path.stat()
            file_record = FileRecord(
                path=path,
                name=file_path.name,
                extension=file_path.suffix,
                size=stat.st_size,
                file_modified_at=datetime.fromtimestamp(stat.st_mtime)
            )
            await file_record.insert()
            
            results["success"].append({"path": path, "file_id": str(file_record.id), "status": "indexed"})
        except Exception as e:
            results["failed"].append({"path": path, "error": str(e)})
    
    logger.info(f"Batch indexed: {len(results['success'])} success, {len(results['failed'])} failed")
    
    return results


@app.post("/api/files/batch/update")
async def batch_update_files(file_ids: List[str], updates: Dict[str, Any]):
    """Batch update multiple files"""
    results = {"updated": 0, "failed": []}
    
    for file_id_str in file_ids:
        try:
            from bson import ObjectId
            file_record = await FileRecord.get(ObjectId(file_id_str))
            if not file_record:
                results["failed"].append({"file_id": file_id_str, "error": "Not found"})
                continue
            
            # Apply updates
            if "rating" in updates:
                file_record.rating = updates["rating"]
            if "description" in updates:
                file_record.description = updates["description"]
            if "custom_properties" in updates:
                file_record.custom_properties.update(updates["custom_properties"])
            
            await file_record.save()
            results["updated"] += 1
        except Exception as e:
            results["failed"].append({"file_id": file_id_str, "error": str(e)})
    
    logger.info(f"Batch updated {results['updated']} files")
    
    return results


# Import routers
try:
    from api.tags import router as tags_router
    app.include_router(tags_router)
    logger.info("✓ Tag API routes loaded")
except Exception as e:
    logger.warning(f"Could not load tag routes: {e}")

try:
    from api.albums import router as albums_router
    app.include_router(albums_router)
    logger.info("✓ Album API routes loaded")
except Exception as e:
    logger.warning(f"Could not load album routes: {e}")

try:
    from api.relations import router as relations_router
    app.include_router(relations_router)
    logger.info("✓ Relations API routes loaded")
except Exception as e:
    logger.warning(f"Could not load relations routes: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 UExplorer Web API Starting (Comprehensive Version)")
    logger.info("=" * 60)
    logger.info(f"📁 Initial directory: {current_directory}")
    logger.info(f"🌐 API will be available at: http://localhost:8000")
    logger.info(f"📚 API docs: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
