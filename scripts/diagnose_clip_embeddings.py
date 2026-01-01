"""
Diagnostic script to check CLIP embeddings and file processing states.

Run this to understand:
- How many files exist
- How many are images
- File states distribution
- How many have/don't have CLIP embeddings
"""
import asyncio
from src.core.database.manager import DatabaseManager
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.base import ProcessingState
from src.ucorefs.vectors.models import EmbeddingRecord


async def diagnose_clip_state():
    """Diagnose CLIP embedding coverage."""
    
    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    print("\n" + "="*60)
    print("CLIP EMBEDDINGS DIAGNOSTIC")
    print("="*60 + "\n")
    
    # 1. Total files
    total_files = await FileRecord.count_documents({})
    print(f"📁 Total files in database: {total_files}")
    
    # 2. Files by type
    image_files = await FileRecord.count_documents({"file_type": "image"})
    text_files = await FileRecord.count_documents({"file_type": "text"})
    unknown_files = await FileRecord.count_documents({"file_type": {"$exists": False}})
    
    print(f"\n📊 Files by type:")
    print(f"   Images: {image_files}")
    print(f"   Text: {text_files}")
    print(f"   Unknown/None: {unknown_files}")
    
    # 3. Files by processing state
    print(f"\n🔄 Files by processing state:")
    for state in ProcessingState:
        count = await FileRecord.count_documents({"processing_state": state.value})
        if count > 0:
            print(f"   {state.name}: {count}")
    
    # 4. CLIP embedding coverage
    clip_embeddings = await EmbeddingRecord.count_documents({"provider": "clip"})
    files_with_clip_metadata = await FileRecord.count_documents({
        "embeddings.clip": {"$exists": True}
    })
    files_with_has_vector = await FileRecord.count_documents({"has_vector": True})
    
    print(f"\n🎯 CLIP Embedding coverage:")
    print(f"   EmbeddingRecord (clip): {clip_embeddings}")
    print(f"   FileRecords with 'embeddings.clip': {files_with_clip_metadata}")
    print(f"   FileRecords with 'has_vector=True': {files_with_has_vector}")
    
    # 5. Files matching reprocess criteria (INDEXED/COMPLETE without CLIP)
    reprocess_candidates = await FileRecord.find({
        "processing_state": {"$gte": ProcessingState.INDEXED},
        "$or": [
            {"embeddings.clip": {"$exists": False}},
            {"embeddings": {}}
        ],
        "file_type": "image"
    })
    
    print(f"\n⚠️  Files needing reprocessing:")
    print(f"   Images at INDEXED/COMPLETE without CLIP: {len(reprocess_candidates)}")
    
    # 6. Sample files without CLIP at lower states
    pending_images = await FileRecord.find({
        "processing_state": {"$lt": ProcessingState.INDEXED},
        "file_type": "image"
    })
    
    print(f"\n⏳ Images still being processed:")
    print(f"   Images below INDEXED state: {len(pending_images)}")
    if pending_images:
        print(f"   (These will get CLIP embeddings when Phase 2 runs)")
    
    # 7. Images at INDEXED+ WITH clip embeddings (success!)
    successful_images = await FileRecord.find({
        "processing_state": {"$gte": ProcessingState.INDEXED},
        "embeddings.clip": {"$exists": True},
        "file_type": "image"
    })
    
    print(f"\n✅ Successfully processed images:")
    print(f"   Images at INDEXED+ WITH CLIP: {len(successful_images)}")
    
    # 8. Sample a few files
    print(f"\n📋 Sample files (first 5):")
    sample_files = await FileRecord.find({}, limit=5)
    for i, file in enumerate(sample_files, 1):
        has_clip = "clip" in file.embeddings if file.embeddings else False
        print(f"   {i}. {file.name}")
        print(f"      State: {file.processing_state.name}")
        print(f"      Type: {file.file_type}")
        print(f"      Has CLIP: {'✓' if has_clip else '✗'}")
        print(f"      has_vector: {file.has_vector}")
    
    # Cleanup
    await db_manager.shutdown()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(diagnose_clip_state())
