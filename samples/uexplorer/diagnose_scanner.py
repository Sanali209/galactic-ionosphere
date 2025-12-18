"""
Diagnostic script for scanner issues.
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
from src.ucorefs.discovery.library_manager import LibraryManager
from src.ucorefs.discovery.scanner import DirectoryScanner
from src.ucorefs.core.fs_service import FSService

async def diagnose():
    locator = ServiceLocator()
    locator.init()
    
    db_manager = locator.register_system(DatabaseManager)
    await locator.start_all()
    
    print("=" * 60)
    print("UCOREFS DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Check roots
    roots = await DirectoryRecord.find({"is_root": True})
    print(f"\n1. LIBRARY ROOTS: {len(roots)}")
    for r in roots:
        print(f"   - {r.path} (ID: {r._id})")
    
    # 2. Check total DB content
    all_files = await FileRecord.find({})
    all_dirs = await DirectoryRecord.find({})
    print(f"\n2. DATABASE CONTENT:")
    print(f"   Files: {len(all_files)}")
    print(f"   Directories: {len(all_dirs)}")
    
    # 3. Check root_id distribution
    root_id_set = {str(r._id) for r in roots}
    files_with_valid_root = [f for f in all_files if str(f.root_id) in root_id_set]
    files_without_root = [f for f in all_files if f.root_id is None]
    files_orphaned = [f for f in all_files if f.root_id and str(f.root_id) not in root_id_set]
    
    print(f"\n3. FILE ROOT_ID DISTRIBUTION:")
    print(f"   Valid root_id: {len(files_with_valid_root)}")
    print(f"   No root_id (None): {len(files_without_root)}")
    print(f"   Orphaned (invalid root_id): {len(files_orphaned)}")
    
    # 4. Test scan on first root
    if roots:
        test_root = roots[0]
        print(f"\n4. TEST SCAN: {test_root.path}")
        
        fs_service = FSService()
        lib_manager = LibraryManager(fs_service)
        scanner = DirectoryScanner(lib_manager)
        
        scan_count = 0
        for batch in scanner.scan_directory(
            test_root.path,
            test_root.watch_extensions,
            test_root.blacklist_paths,
            recursive=True
        ):
            scan_count += len(batch)
            if scan_count <= 10:
                for item in batch[:3]:
                    print(f"   Scanned: {item.path}")
        
        print(f"   Total scanned items: {scan_count}")
        
        # 5. Sample path check
        if scan_count > 0:
            sample_path = None
            for batch in scanner.scan_directory(test_root.path, recursive=False):
                if batch:
                    sample_path = batch[0].path
                    break
            
            if sample_path:
                print(f"\n5. SAMPLE PATH CHECK: {sample_path}")
                
                # Check if exists in DB
                file_match = await FileRecord.find_one({"path": sample_path})
                dir_match = await DirectoryRecord.find_one({"path": sample_path})
                
                print(f"   File match: {file_match}")
                print(f"   Dir match: {dir_match}")
                
                # Check path format
                normalized = str(Path(sample_path).as_posix())
                print(f"   Normalized: {normalized}")
                
                # Check DB paths format
                if all_files:
                    sample_db_path = all_files[0].path
                    print(f"   Sample DB path format: {sample_db_path}")
    
    await locator.stop_all()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(diagnose())
