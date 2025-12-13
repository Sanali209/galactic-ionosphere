import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.locator import sl
from src.core.database.manager import db_manager
from src.core.database.models.folder import FolderRecord

async def dedupe():
    sl.init("config.json")
    db_manager.init()
    
    print("Fetching all folders...")
    folders = await FolderRecord.find({})
    print(f"Total folders: {len(folders)}")
    
    path_map = {}
    duplicates = []
    
    for f in folders:
        if f.path not in path_map:
            path_map[f.path] = f
        else:
            duplicates.append(f)
            
    print(f"Found {len(duplicates)} duplicates.")
    
    if duplicates:
        print("Removing duplicates...")
        for d in duplicates:
            print(f"Removing duplicate: {d.path} (ID: {d.id})")
            # Assuming CollectionRecord has delete or we use safe delete
            # FolderRecord doesn't show delete method in snippet, but assuming standard ODM
            # We can use db_manager directly if needed, but let's try delete() if available or DB direct
            try:
                # Direct delete via DB to be safe
                res = await db_manager.db[FolderRecord.Meta.table_name].delete_one({"_id": d.id})
                if res.deleted_count == 0:
                     # Maybe string ID?
                     from bson import ObjectId
                     await db_manager.db[FolderRecord.Meta.table_name].delete_one({"_id": ObjectId(d.id)})
            except Exception as e:
                print(f"Error deleting {d.id}: {e}")
                
    print("Done.")

if __name__ == "__main__":
    from PySide6.QtCore import QCoreApplication
    import qasync
    
    # QSettings needs an app instance
    q_app = QCoreApplication(sys.argv)
    
    loop = qasync.QEventLoop(q_app)
    asyncio.set_event_loop(loop)
    
    with loop:
        loop.run_until_complete(dedupe())
