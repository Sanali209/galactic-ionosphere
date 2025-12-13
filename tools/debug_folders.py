import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.locator import sl
from src.core.database.manager import db_manager
from src.core.database.models.folder import FolderRecord

async def inspect():
    sl.init("config.json")
    db_manager.init()
    
    print("Searching for 'safe repo'...")
    # Regex case insensitive search
    records = await FolderRecord.find({"path": {"$regex": "safe repo", "$options": "i"}})
    
    path_counts = {}
    for r in records:
        print(f"ID: {r.id} | Path: {r.path} | Parent: {r.parent_path}")
        if r.path not in path_counts: path_counts[r.path] = 0
        path_counts[r.path] += 1
        
    print("-" * 20)
    print("Duplicates check:")
    for p, c in path_counts.items():
        if c > 1:
            print(f"FAILED: {c} records for path '{p}'")
    
    if all(c == 1 for c in path_counts.values()):
        print("No exact path duplicates found in search results.")

if __name__ == "__main__":
    from PySide6.QtCore import QCoreApplication
    import qasync
    q_app = QCoreApplication(sys.argv)
    loop = qasync.QEventLoop(q_app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.run_until_complete(inspect())
