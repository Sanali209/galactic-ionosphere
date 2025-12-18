"""
Test path resolution
"""
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
uexplorer_dir = current_file.parent
foundation_dir = uexplorer_dir.parent.parent / "templates" / "foundation"

print(f"Current file: {current_file}")
print(f"UExplorer dir: {uexplorer_dir}")
print(f"Foundation dir: {foundation_dir}")
print(f"Foundation exists: {foundation_dir.exists()}")
print(f"Foundation/src exists: {(foundation_dir / 'src').exists()}")
print(f"Foundation/src/core exists: {(foundation_dir / 'src' / 'core').exists()}")

# Add to path
sys.path.insert(0, str(foundation_dir))
sys.path.insert(0, str(uexplorer_dir))

print(f"\nSys.path[0]: {sys.path[0]}")
print(f"Sys.path[1]: {sys.path[1]}")

# Try import
try:
    from src.core.bootstrap import ApplicationBuilder
    print("\n✅ SUCCESS: ApplicationBuilder imported!")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
