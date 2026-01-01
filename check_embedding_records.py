"""
Quick script to check EmbeddingRecord table for FAISS debugging
"""
import asyncio
from src.core.database.manager import DatabaseManager
from src.ucorefs.vectors.models import EmbeddingRecord

async def check_embedding_records():
    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    try:
        # Count total embedding records
        all_embeddings = await EmbeddingRecord.find({})
        print(f"\nTotal EmbeddingRecords: {len(all_embeddings)}")
        
        # Count by provider
        clip_embeddings = await EmbeddingRecord.find({"provider": "clip"})
        blip_embeddings = await EmbeddingRecord.find({"provider": "blip"})
        dino_embeddings = await EmbeddingRecord.find({"provider": "dino"})
        
        print(f"  CLIP embeddings: {len(clip_embeddings)}")
        print(f"  BLIP embeddings: {len(blip_embeddings)}")
        print(f"  DINO embeddings: {len(dino_embeddings)}")
        
        # Show sample
        if clip_embeddings:
            sample = clip_embeddings[0]
            print(f"\nSample CLIP embedding:")
            print(f"  file_id: {sample.file_id}")
            print(f"  provider: {sample.provider}")
            print(f"  dimension: {sample.dimension}")
            print(f"  vector length: {len(sample.vector) if sample.vector else 0}")
        else:
            print("\n⚠️ NO CLIP EMBEDDINGS FOUND IN EmbeddingRecord TABLE!")
            print("This is why FAISS search returns 0 results.")
            print("\nSolution: CLIP extractor is storing to FileRecord.embeddings")
            print("but NOT to EmbeddingRecord table (needed for FAISS).")
        
    finally:
        await db_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(check_embedding_records())
