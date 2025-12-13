import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.ui.models.flat_tree import TagFlatModel
    print("Import Successful: TagFlatModel")
except Exception as e:
    print(f"Import Failed: {e}")
    import traceback
    traceback.print_exc()
