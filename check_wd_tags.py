"""
Quick check: Do rating tags exist in MongoDB?
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check_ratings():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["foundation_app"]  # From config.json
    
    # Check for any wd_rating tags
    rating_tags = await db.tags.find({
        "full_path": {"$regex": "^auto/wd_rating/"}
    }).to_list(length=100)
    
    print(f"\n=== WD Rating Tags ===")
    print(f"Total rating tags found: {len(rating_tags)}")
    
    if rating_tags:
        for tag in rating_tags:
            print(f"  - {tag['full_path']} ({tag.get('file_count', 0)} files)")
    else:
        print("  ⚠️ NO RATING TAGS FOUND!")
        print("\nReason: Images haven't been processed with WD tagger yet.")
        print("Next step: Restart app + run maintenance 'Reprocess Phase 2'")
    
    # Check for character tags
    char_tags = await db.tags.find({
        "full_path": {"$regex": "^auto/wd_character/"}
    }).to_list(length=100)
    
    print(f"\n=== WD Character Tags ===")
    print(f"Total character tags found: {len(char_tags)}")
    if char_tags:
        for tag in char_tags[:10]:  # Show first 10
            print(f"  - {tag['full_path']} ({tag.get('file_count', 0)} files)")
    else:
        print("  ⚠️ NO CHARACTER TAGS FOUND!")
        print("Reason: Either no characters detected OR threshold too high (was 0.75, now 0.35)")
    
    client.close()

asyncio.run(check_ratings())
