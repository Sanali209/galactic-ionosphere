import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def wipe():
    print("Connecting to DB (localhost:27017)...")
    uri = "mongodb://localhost:27017"
    db_name = "gallery_db"
    
    client = AsyncIOMotorClient(uri)
    db = client[db_name]
    
    print(f"Dropping collections in {db_name}...")
    
    cols = await db.list_collection_names()
    for c in cols:
        await db[c].drop()
        print(f"Dropped {c}")
        
    print("Database wiped.")

if __name__ == "__main__":
    asyncio.run(wipe())
