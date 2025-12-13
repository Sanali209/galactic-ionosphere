import asyncio
from pymongo import AsyncMongoClient
import sys

async def main():
    print(f"Testing AsyncMongoClient from pymongo...")
    # Assume default local mongo, or use what's in config (but simple 27017 is fine for test)
    uri = "mongodb://localhost:27017"
    try:
        client = AsyncMongoClient(uri)
        print(f"Client created: {client}")
        # Ping
        await client.admin.command('ping')
        print("Ping successful!")
        
        # Insert/Find
        db = client.test_async_db
        coll = db.test_collection
        res = await coll.insert_one({"hello": "world"})
        print(f"Inserted: {res.inserted_id}")
        
        doc = await coll.find_one({"_id": res.inserted_id})
        print(f"Found: {doc}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
