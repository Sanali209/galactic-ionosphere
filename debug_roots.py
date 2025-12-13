import asyncio
import os
from src.core.database.models.folder import FolderRecord
from src.core.database.manager import db_manager
from src.core.locator import sl

async def check_roots():
    await db_manager.reset_db() # Don't reset, just init
    # Re-init manually since we skipped main
    from src.core.config import Config
    sl.config = Config()
    sl.config.load()
    await db_manager.init()
    
    print("Checking Root Folders...")
    roots_none = await FolderRecord.find(FolderRecord.parent_path == None).to_list()
    print(f"Roots with parent=None: {len(roots_none)}")
    for r in roots_none:
        print(f" - {r.path} (Parent: {r.parent_path})")

    all_folders = await FolderRecord.find({}).to_list()
    print(f"Total Folders: {len(all_folders)}")
    
    # Check for self-referential
    self_ref = []
    for f in all_folders:
        if f.path == f.parent_path:
            self_ref.append(f)
            
    print(f"Self-referential Roots: {len(self_ref)}")
    for r in self_ref:
        print(f" - {r.path} (Parent: {r.parent_path})")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(check_roots())
