"""
Migrate flat tags to hierarchical structure.

Finds tags with delimiter characters in name and converts to proper hierarchy.
"""
import asyncio
import sys
from pathlib import Path

foundation_path = Path("D:/github/USCore/templates/foundation")
sys.path.insert(0, str(foundation_path))

from src.core.locator import ServiceLocator
from src.core.database.manager import DatabaseManager
from src.ucorefs.tags.manager import TagManager
from src.ucorefs.tags.models import Tag

async def migrate_tags():
    locator = ServiceLocator()
    locator.init()
    
    db_manager = locator.register_system(DatabaseManager)
    tag_manager = locator.register_system(TagManager)
    await locator.start_all()
    
    print("Finding flat tags with delimiters...")
    
    all_tags = await Tag.find({})
    flat_tags = [t for t in all_tags if any(d in t.name for d in ['/', '|', '\\'])]
    
    print(f"Found {len(flat_tags)} flat tags to migrate:")
    for t in flat_tags:
        print(f"  - {t.name}")
    
    if not flat_tags:
        print("No flat tags with delimiters found!")
        await locator.stop_all()
        return
    
    response = input("\nMigrate these to hierarchy? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        await locator.stop_all()
        return
    
    for old_tag in flat_tags:
        print(f"\nMigrating: {old_tag.name}")
        
        # Create new hierarchical tag
        new_tag = await tag_manager.create_tag_from_path(old_tag.name)
        print(f"  -> Created: {new_tag.full_path}")
        
        # TODO: Update files that reference old tag to use new tag
        # For now, just delete the old flat tag
        await old_tag.delete()
        print(f"  -> Deleted old flat tag")
    
    print(f"\n✓ Migrated {len(flat_tags)} tags to hierarchy!")
    
    await locator.stop_all()

if __name__ == "__main__":
    asyncio.run(migrate_tags())
