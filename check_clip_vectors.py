"""
Check if files have CLIP embeddings in the new unified storage
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check_embeddings():
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["ucore"]
    
    # Count total files
    total_files = await db.file_records.count_documents({})
    print(f"Total files in database: {total_files}")
    
    # Count files with CLIP embeddings  
    files_with_clip = await db.file_records.count_documents({
        "embeddings.clip": {"$exists": True}
    })
    print(f"Files with clip metadata: {files_with_clip}")
    
    # Count files with CLIP VECTORS (unified storage)
    files_with_vectors = await db.file_records.count_documents({
        "embeddings.clip.vector": {"$exists": True}
    })
    print(f"Files with clip VECTORS: {files_with_vectors}")
    
    if files_with_vectors > 0:
        # Show sample
        sample = await db.file_records.find_one({"embeddings.clip.vector": {"$exists": True}})
        if sample:
            print(f"\nSample file: {sample.get('name')}")
            clip_emb = sample.get('embeddings', {}).get('clip', {})
            print(f"  Model: {clip_emb.get('model')}")
            print(f"  Dimension: {clip_emb.get('dimension')}")
            vector = clip_emb.get('vector', [])
            print(f"  Vector length: {len(vector)}")
            print(f"  Vector sample: {vector[:5] if vector else 'EMPTY'}")
    else:
        print("\n⚠️ NO FILES WITH CLIP VECTORS FOUND!")
        print("You need to:")
        print("1. Add a library root in UExplorer")
        print("2. Wait for Phase 2 processing to complete")
        print("3. Then search again")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_embeddings())
