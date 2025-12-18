"""
Force cleanup of orphaned records - NO PROMPTS.
"""
import asyncio
import sys
from pathlib import Path

foundation_path = Path("D:/github/USCore/templates/foundation")
sys.path.insert(0, str(foundation_path))

from src.core.locator import ServiceLocator
from src.core.database.manager import DatabaseManager
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.directory import DirectoryRecord

async def force_cleanup():
    locator = ServiceLocator()
    locator.init()
    
    db_manager = locator.register_system(DatabaseManager)
    await locator.start_all()
    
    roots = await DirectoryRecord.find({"is_root": True})
    root_ids = {str(r._id) for r in roots}
    
    print(f"Found {len(roots)} roots")
    
    all_files = await FileRecord.find({})
    all_dirs = await DirectoryRecord.find({"is_root": {"$ne": True}})
    
    orphaned_files = [f for f in all_files if str(f.root_id) not in root_ids]
    orphaned_dirs = [d for d in all_dirs if str(d.root_id) not in root_ids]
    
    print(f"Deleting {len(orphaned_files)} orphaned files...")
    for i, f in enumerate(orphaned_files):
        await f.delete()
        if i % 1000 == 0:
            print(f"  Deleted {i}/{len(orphaned_files)} files")
    
    print(f"Deleting {len(orphaned_dirs)} orphaned directories...")
    for d in orphaned_dirs:
        await d.delete()
    
    print(f"✓ Cleanup complete!")
    
    await locator.stop_all()

if __name__ == "__main__":
    asyncio.run(force_cleanup())
