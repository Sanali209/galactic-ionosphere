"""
Database cleanup script - removes orphaned files/directories.
"""
import asyncio
import sys
from pathlib import Path

# Setup path
foundation_path = Path("D:/github/USCore/templates/foundation")
sys.path.insert(0, str(foundation_path))

from src.core.locator import ServiceLocator
from src.core.database.manager import DatabaseManager
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.models.directory import DirectoryRecord

async def cleanup_orphans():
    locator = ServiceLocator()
    locator.init()
    
    db_manager = locator.register_system(DatabaseManager)
    await locator.start_all()
    
    print("Finding all library roots...")
    roots = await DirectoryRecord.find({"is_root": True})
    root_ids = {str(r._id) for r in roots}
    
    print(f"Found {len(roots)} roots: {[r.path for r in roots]}")
    
    # Find all files
    all_files = await FileRecord.find({})
    all_dirs = await DirectoryRecord.find({"is_root": {"$ne": True}})
    
    print(f"\nTotal in database: {len(all_files)} files, {len(all_dirs)} directories")
    
    orphaned_files = [f for f in all_files if str(f.root_id) not in root_ids]
    orphaned_dirs = [d for d in all_dirs if str(d.root_id) not in root_ids]
    
    print(f"Orphaned: {len(orphaned_files)} files, {len(orphaned_dirs)} directories")
    
    if orphaned_files or orphaned_dirs:
        response = input("\nDelete orphaned records? (yes/no): ")
        if response.lower() == 'yes':
            for f in orphaned_files:
                await f.delete()
            for d in orphaned_dirs:
                await d.delete()
            print(f"✓ Deleted {len(orphaned_files)} files and {len(orphaned_dirs)} directories")
        else:
            print("Cancelled")
    else:
        print("\n✓ No orphaned records found")
    
    await locator.stop_all()

if __name__ == "__main__":
    asyncio.run(cleanup_orphans())
