"""
Check if WD tags have correct parent hierarchy
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

async def check_hierarchy():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["foundation_app"]
    
    # Get the "auto" root tag
    auto_tag = await db.tags.find_one({"name": "auto", "parent_id": None})
    
    if not auto_tag:
        print("❌ 'auto' root tag not found!")
        print("This means tags weren't created with hierarchy.")
        client.close()
        return
    
    print(f"✅ Found 'auto' root tag (ID: {auto_tag['_id']})")
    
    # Get children of "auto"
    children = await db.tags.find({"parent_id": auto_tag['_id']}).to_list(length=100)
    
    print(f"\nChildren of 'auto': {len(children)}")
    for child in children:
        print(f"  - {child['name']} (ID: {child['_id']})")
        
        # Get grandchildren
        grandchildren = await db.tags.find({"parent_id": child['_id']}).to_list(length=10)
        if grandchildren:
            print(f"    → {len(grandchildren)} tags inside (showing first 5):")
            for gc in grandchildren[:5]:
                print(f"       • {gc['name']}")
    
    client.close()

asyncio.run(check_hierarchy())
