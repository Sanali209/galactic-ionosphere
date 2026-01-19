import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    try:
        # Use default port
        # Note: serverSelectionTimeoutMS ensures we fail fast if Mongo is down
        client = AsyncIOMotorClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        db = client["foundation_app"]
        
        # Check connection
        try:
            await client.server_info()
            print("Connected to MongoDB")
        except Exception:
            print("Failed to connect to MongoDB")
            return

        # Count tasks
        pending = await db.tasks.count_documents({"status": "pending"})
        print(f"PENDING_TASKS: {pending}")
        
        running = await db.tasks.count_documents({"status": "running"})
        print(f"RUNNING_TASKS: {running}")
        
        completed = await db.tasks.count_documents({"status": "completed"})
        print(f"COMPLETED_TASKS: {completed}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check())
