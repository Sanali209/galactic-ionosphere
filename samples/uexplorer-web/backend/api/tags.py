"""
Tag Management API Routes

Hierarchical tag system with MPPT (Modified Preorder Tree Traversal) structure.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from bson import ObjectId

from models import Tag, FileTag, FileRecord
from loguru import logger


router = APIRouter(prefix="/api/tags", tags=["Tags"])


# Request/Response Models
class TagCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    parent_id: Optional[str] = None


class TagUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None


class TagResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    color: Optional[str]
    parent_id: Optional[str]
    file_count: int
    level: int
    has_children: bool


class TagFileAssignment(BaseModel):
    file_ids: List[str]
    tag_id: str


@router.get("/", response_model=List[TagResponse])
async def list_tags(parent_id: Optional[str] = None):
    """List all tags or tags under a parent"""
    query = {}
    if parent_id:
        query["parent_id"] = ObjectId(parent_id)
    
    tags = await Tag.find(query).sort("name").to_list()
    
    result = []
    for tag in tags:
        # Check if tag has children
        has_children = await Tag.find_one({"parent_id": tag.id}) is not None
        
        result.append(TagResponse(
            id=str(tag.id),
            name=tag.name,
            description=tag.description,
            color=tag.color,
            parent_id=str(tag.parent_id) if tag.parent_id else None,
            file_count=tag.file_count,
            level=tag.level,
            has_children=has_children
        ))
    
    return result


@router.get("/tree")
async def get_tag_tree():
    """Get full tag hierarchy as tree structure"""
    all_tags = await Tag.find().sort("lft").to_list()
    
    # Build tree structure
    tag_map = {}
    root_tags = []
    
    for tag in all_tags:
        tag_dict = {
            "id": str(tag.id),
            "name": tag.name,
            "description": tag.description,
            "color": tag.color,
            "file_count": tag.file_count,
            "level": tag.level,
            "children": []
        }
        tag_map[str(tag.id)] = tag_dict
        
        if tag.parent_id:
            parent = tag_map.get(str(tag.parent_id))
            if parent:
                parent["children"].append(tag_dict)
        else:
            root_tags.append(tag_dict)
    
    return {"tags": root_tags, "total": len(all_tags)}


@router.post("/", response_model=TagResponse)
async def create_tag(tag_data: TagCreate):
    """Create a new tag"""
    # Check if tag name already exists
    existing = await Tag.find_one({"name": tag_data.name})
    if existing:
        raise HTTPException(status_code=400, detail="Tag name already exists")
    
    # Calculate MPPT values
    parent = None
    if tag_data.parent_id:
        parent = await Tag.get(ObjectId(tag_data.parent_id))
        if not parent:
            raise HTTPException(status_code=404, detail="Parent tag not found")
    
    # Simple MPPT implementation
    if parent:
        lft = parent.rgt
        rgt = parent.rgt + 1
        level = parent.level + 1
        
        # Update existing tags
        await Tag.find({"rgt": {"$gte": parent.rgt}}).update({"$inc": {"rgt": 2}})
        await Tag.find({"lft": {"$gt": parent.rgt}}).update({"$inc": {"lft": 2}})
    else:
        # Root level tag
        max_tag = await Tag.find().sort([("rgt", -1)]).first_or_none()
        if max_tag:
            lft = max_tag.rgt + 1
            rgt = max_tag.rgt + 2
        else:
            lft = 1
            rgt = 2
        level = 0
    
    # Create tag
    tag = Tag(
        name=tag_data.name,
        description=tag_data.description,
        color=tag_data.color,
        parent_id=ObjectId(tag_data.parent_id) if tag_data.parent_id else None,
        lft=lft,
        rgt=rgt,
        level=level
    )
    await tag.insert()
    
    logger.info(f"Created tag: {tag.name} (level={level})")
    
    return TagResponse(
        id=str(tag.id),
        name=tag.name,
        description=tag.description,
        color=tag.color,
        parent_id=str(tag.parent_id) if tag.parent_id else None,
        file_count=0,
        level=level,
        has_children=False
    )


@router.put("/{tag_id}", response_model=TagResponse)
async def update_tag(tag_id: str, tag_data: TagUpdate):
    """Update a tag"""
    tag = await Tag.get(ObjectId(tag_id))
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    
    if tag_data.name:
        # Check for duplicate name
        existing = await Tag.find_one({"name": tag_data.name, "_id": {"$ne": ObjectId(tag_id)}})
        if existing:
            raise HTTPException(status_code=400, detail="Tag name already exists")
        tag.name = tag_data.name
    
    if tag_data.description is not None:
        tag.description = tag_data.description
    
    if tag_data.color is not None:
        tag.color = tag_data.color
    
    await tag.save()
    
    has_children = await Tag.find_one({"parent_id": tag.id}) is not None
    
    return TagResponse(
        id=str(tag.id),
        name=tag.name,
        description=tag.description,
        color=tag.color,
        parent_id=str(tag.parent_id) if tag.parent_id else None,
        file_count=tag.file_count,
        level=tag.level,
        has_children=has_children
    )


@router.delete("/{tag_id}")
async def delete_tag(tag_id: str, cascade: bool = False):
    """Delete a tag"""
    tag = await Tag.get(ObjectId(tag_id))
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    
    # Check for children
    has_children = await Tag.find_one({"parent_id": tag.id}) is not None
    if has_children and not cascade:
        raise HTTPException(
            status_code=400,
            detail="Tag has children. Use cascade=true to delete recursively"
        )
    
    if cascade:
        # Delete all descendants
        descendants = await Tag.find({"lft": {"$gt": tag.lft}, "rgt": {"$lt": tag.rgt}}).to_list()
        for desc in descendants:
            await FileTag.find({"tag_id": desc.id}).delete()
            await desc.delete()
    
    # Remove file associations
    await FileTag.find({"tag_id": tag.id}).delete()
    
    # Delete tag
    await tag.delete()
    
    logger.info(f"Deleted tag: {tag.name}")
    
    return {"success": True, "deleted_id": tag_id}


@router.post("/assign")
async def assign_tags_to_files(assignment: TagFileAssignment):
    """Assign a tag to multiple files"""
    tag = await Tag.get(ObjectId(assignment.tag_id))
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    
    assigned_count = 0
    for file_id_str in assignment.file_ids:
        file_id = ObjectId(file_id_str)
        
        # Check if file exists
        file_record = await FileRecord.get(file_id)
        if not file_record:
            continue
        
        # Check if already assigned
        existing = await FileTag.find_one({"file_id": file_id, "tag_id": tag.id})
        if existing:
            continue
        
        # Create assignment
        file_tag = FileTag(file_id=file_id, tag_id=tag.id)
        await file_tag.insert()
        assigned_count += 1
    
    # Update tag count
    tag.file_count = await FileTag.find({"tag_id": tag.id}).count()
    await tag.save()
    
    logger.info(f"Assigned tag '{tag.name}' to {assigned_count} files")
    
    return {
        "success": True,
        "assigned_count": assigned_count,
        "tag_file_count": tag.file_count
    }


@router.post("/unassign")
async def unassign_tags_from_files(assignment: TagFileAssignment):
    """Remove a tag from multiple files"""
    tag = await Tag.get(ObjectId(assignment.tag_id))
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    
    removed_count = 0
    for file_id_str in assignment.file_ids:
        file_id = ObjectId(file_id_str)
        result = await FileTag.find({"file_id": file_id, "tag_id": tag.id}).delete()
        if result.deleted_count > 0:
            removed_count += 1
    
    # Update tag count
    tag.file_count = await FileTag.find({"tag_id": tag.id}).count()
    await tag.save()
    
    logger.info(f"Removed tag '{tag.name}' from {removed_count} files")
    
    return {
        "success": True,
        "removed_count": removed_count,
        "tag_file_count": tag.file_count
    }


@router.get("/file/{file_id}")
async def get_file_tags(file_id: str):
    """Get all tags for a file"""
    file_tags = await FileTag.find({"file_id": ObjectId(file_id)}).to_list()
    
    tags = []
    for ft in file_tags:
        tag = await Tag.get(ft.tag_id)
        if tag:
            tags.append({
                "id": str(tag.id),
                "name": tag.name,
                "color": tag.color,
                "description": tag.description
            })
    
    return {"file_id": file_id, "tags": tags, "count": len(tags)}
