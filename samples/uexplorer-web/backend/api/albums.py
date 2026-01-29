"""
Album Management API Routes

Static and smart (query-based) album system.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime

from models import Album, FileAlbum, FileRecord
from loguru import logger


router = APIRouter(prefix="/api/albums", tags=["Albums"])


# Request/Response Models
class AlbumCreate(BaseModel):
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    is_smart: bool = False
    query: Optional[dict] = None


class AlbumUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    query: Optional[dict] = None


class AlbumResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    icon: Optional[str]
    is_smart: bool
    file_count: int
    created_at: str
    updated_at: str


class AlbumFileAssignment(BaseModel):
    file_ids: List[str]
    album_id: str


@router.get("/", response_model=List[AlbumResponse])
async def list_albums():
    """List all albums"""
    albums = await Album.find().sort("name").to_list()
    
    return [
        AlbumResponse(
            id=str(album.id),
            name=album.name,
            description=album.description,
            icon=album.icon,
            is_smart=album.is_smart,
            file_count=album.file_count,
            created_at=album.created_at.isoformat(),
            updated_at=album.updated_at.isoformat()
        )
        for album in albums
    ]


@router.get("/{album_id}", response_model=AlbumResponse)
async def get_album(album_id: str):
    """Get a specific album"""
    album = await Album.get(ObjectId(album_id))
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    return AlbumResponse(
        id=str(album.id),
        name=album.name,
        description=album.description,
        icon=album.icon,
        is_smart=album.is_smart,
        file_count=album.file_count,
        created_at=album.created_at.isoformat(),
        updated_at=album.updated_at.isoformat()
    )


@router.post("/", response_model=AlbumResponse)
async def create_album(album_data: AlbumCreate):
    """Create a new album"""
    # Check if album name already exists
    existing = await Album.find_one({"name": album_data.name})
    if existing:
        raise HTTPException(status_code=400, detail="Album name already exists")
    
    # Validate smart album query
    if album_data.is_smart and not album_data.query:
        raise HTTPException(status_code=400, detail="Smart albums require a query")
    
    album = Album(
        name=album_data.name,
        description=album_data.description,
        icon=album_data.icon,
        is_smart=album_data.is_smart,
        query=album_data.query
    )
    await album.insert()
    
    logger.info(f"Created album: {album.name} (smart={album.is_smart})")
    
    return AlbumResponse(
        id=str(album.id),
        name=album.name,
        description=album.description,
        icon=album.icon,
        is_smart=album.is_smart,
        file_count=0,
        created_at=album.created_at.isoformat(),
        updated_at=album.updated_at.isoformat()
    )


@router.put("/{album_id}", response_model=AlbumResponse)
async def update_album(album_id: str, album_data: AlbumUpdate):
    """Update an album"""
    album = await Album.get(ObjectId(album_id))
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    if album_data.name:
        # Check for duplicate name
        existing = await Album.find_one({"name": album_data.name, "_id": {"$ne": ObjectId(album_id)}})
        if existing:
            raise HTTPException(status_code=400, detail="Album name already exists")
        album.name = album_data.name
    
    if album_data.description is not None:
        album.description = album_data.description
    
    if album_data.icon is not None:
        album.icon = album_data.icon
    
    if album_data.query is not None:
        album.query = album_data.query
    
    album.updated_at = datetime.utcnow()
    await album.save()
    
    return AlbumResponse(
        id=str(album.id),
        name=album.name,
        description=album.description,
        icon=album.icon,
        is_smart=album.is_smart,
        file_count=album.file_count,
        created_at=album.created_at.isoformat(),
        updated_at=album.updated_at.isoformat()
    )


@router.delete("/{album_id}")
async def delete_album(album_id: str):
    """Delete an album"""
    album = await Album.get(ObjectId(album_id))
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    # Remove file associations
    await FileAlbum.find({"album_id": album.id}).delete()
    
    # Delete album
    await album.delete()
    
    logger.info(f"Deleted album: {album.name}")
    
    return {"success": True, "deleted_id": album_id}


@router.post("/assign")
async def assign_files_to_album(assignment: AlbumFileAssignment):
    """Assign files to an album (static albums only)"""
    album = await Album.get(ObjectId(assignment.album_id))
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    if album.is_smart:
        raise HTTPException(status_code=400, detail="Cannot manually assign files to smart albums")
    
    assigned_count = 0
    for file_id_str in assignment.file_ids:
        file_id = ObjectId(file_id_str)
        
        # Check if file exists
        file_record = await FileRecord.get(file_id)
        if not file_record:
            continue
        
        # Check if already assigned
        existing = await FileAlbum.find_one({"file_id": file_id, "album_id": album.id})
        if existing:
            continue
        
        # Create assignment
        file_album = FileAlbum(file_id=file_id, album_id=album.id)
        await file_album.insert()
        assigned_count += 1
    
    # Update album count
    album.file_count = await FileAlbum.find({"album_id": album.id}).count()
    await album.save()
    
    logger.info(f"Assigned {assigned_count} files to album '{album.name}'")
    
    return {
        "success": True,
        "assigned_count": assigned_count,
        "album_file_count": album.file_count
    }


@router.post("/unassign")
async def unassign_files_from_album(assignment: AlbumFileAssignment):
    """Remove files from an album (static albums only)"""
    album = await Album.get(ObjectId(assignment.album_id))
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    if album.is_smart:
        raise HTTPException(status_code=400, detail="Cannot manually remove files from smart albums")
    
    removed_count = 0
    for file_id_str in assignment.file_ids:
        file_id = ObjectId(file_id_str)
        result = await FileAlbum.find({"file_id": file_id, "album_id": album.id}).delete()
        if result.deleted_count > 0:
            removed_count += 1
    
    # Update album count
    album.file_count = await FileAlbum.find({"album_id": album.id}).count()
    await album.save()
    
    logger.info(f"Removed {removed_count} files from album '{album.name}'")
    
    return {
        "success": True,
        "removed_count": removed_count,
        "album_file_count": album.file_count
    }


@router.get("/{album_id}/files")
async def get_album_files(
    album_id: str,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get files in an album"""
    album = await Album.get(ObjectId(album_id))
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    if album.is_smart:
        # Execute smart query
        # For now, simple implementation - can be enhanced with full query builder
        query = album.query or {}
        files = await FileRecord.find(query).skip(offset).limit(limit).to_list()
    else:
        # Get files from FileAlbum relationships
        file_albums = await FileAlbum.find({"album_id": album.id}).skip(offset).limit(limit).to_list()
        file_ids = [fa.file_id for fa in file_albums]
        files = await FileRecord.find({"_id": {"$in": file_ids}}).to_list()
    
    result = []
    for file_record in files:
        result.append({
            "id": str(file_record.id),
            "name": file_record.name,
            "path": file_record.path,
            "size": file_record.size,
            "extension": file_record.extension,
            "rating": file_record.rating,
            "description": file_record.description,
            "created_at": file_record.created_at.isoformat()
        })
    
    return {
        "album_id": album_id,
        "album_name": album.name,
        "is_smart": album.is_smart,
        "files": result,
        "count": len(result),
        "offset": offset,
        "limit": limit
    }


@router.get("/file/{file_id}")
async def get_file_albums(file_id: str):
    """Get all albums containing a file"""
    file_albums = await FileAlbum.find({"file_id": ObjectId(file_id)}).to_list()
    
    albums = []
    for fa in file_albums:
        album = await Album.get(fa.album_id)
        if album:
            albums.append({
                "id": str(album.id),
                "name": album.name,
                "icon": album.icon,
                "is_smart": album.is_smart
            })
    
    return {"file_id": file_id, "albums": albums, "count": len(albums)}
