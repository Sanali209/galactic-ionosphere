"""
Direct MongoDB check for CLIP vectors in unified storage
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def deep_check():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["ucore"]
    
    # Count files
    total = await db.file_records.count_documents({})
    print(f"Total files: {total}")
    
    # Count with embeddings.clip (metadata)
    with_clip_meta = await db.file_records.count_documents({"embeddings.clip": {"$exists": True}})
    print(f"Files with embeddings.clip: {with_clip_meta}")
    
    # Count with embeddings.clip.vector (actual vectors)
    with_vectors = await db.file_records.count_documents({"embeddings.clip.vector": {"$exists": True}})
   print(f"Files with embeddings.clip.vector: {with_vectors}")
    
    # Sample one file
    if with_clip_meta > 0:
        sample = await db.file_records.find_one({"embeddings.clip": {"$exists": True}})
        print(f"\n=== Sample File ===")
        print(f"Name: {sample.get('name')}")
        print(f"has_vector: {sample.get('has_vector')}")
        
        clip_data = sample.get('embeddings', {}).get('clip', {})
        print(f"\nembeddings.clip keys: {list(clip_data.keys())}")
        
        if 'vector' in clip_data:
            vector = clip_data['vector']
            print(f"✅ Vector EXISTS! Length: {len(vector)}")
            print(f"   First 5 values: {vector[:5]}")
        else:
            print(f"❌ Vector MISSING! Only has: {clip_data}")
    
    client.close()

asyncio.run(deep_check())
