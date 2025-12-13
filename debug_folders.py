import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import sys

async def check_folders():
    uri = "mongodb://localhost:27017"
    client = AsyncIOMotorClient(uri)
    db = client["gallery_db"]
    collection = db["folders"]
    
    print("--- Folder Records ---")
    async for doc in collection.find({}):
        print(f"Path: '{doc.get('path')}' | Parent: '{doc.get('parent_path')}'")
        
    print("----------------------")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_folders())
