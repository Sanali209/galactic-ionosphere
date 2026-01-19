#!/usr/bin/env python3
"""
UCoreFS File Scanner - Console Application Example

Demonstrates using UCoreFS framework without GUI dependencies.
This tool can scan directories, search files, and query metadata
entirely from the command line.

Usage:
    python file_scanner.py scan /path/to/directory
    python file_scanner.py search "query text"
    python file_scanner.py list --limit 10
    python file_scanner.py stats
"""
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.bundles import UCoreFSDataBundle


async def scan_directory(path: str, locator) -> int:
    """
    Scan a directory and import files into the database.
    
    Args:
        path: Directory path to scan
        locator: Service locator
        
    Returns:
        Number of files found
    """
    from src.ucorefs.services.fs_service import FSService
    
    fs = locator.get_system(FSService)
    
    print(f"Scanning directory: {path}")
    files = await fs.scan_directory(path)
    
    print(f"✓ Found {len(files)} files")
    
    # Show first 5 files as preview
    if files:
        print("\nPreview (first 5):")
        for i, file in enumerate(files[:5], 1):
            print(f"  {i}. {file.name} ({file.size} bytes)")
    
    return len(files)


async def search_files(query: str, locator) -> list:
    """
    Search for files by text query.
    
    Args:
        query: Search query
        locator: Service locator
        
    Returns:
        List of matching files
    """
    from src.ucorefs.search.service import SearchService
    
    search = locator.get_system(SearchService)
    
    print(f"Searching for: '{query}'")
    results = await search.text_search(query, fields=["name", "path"])
    
    print(f"✓ Found {len(results)} matches")
    
    if results:
        print("\nResults:")
        for i, file in enumerate(results[:10], 1):
            print(f"  {i}. {file.name}")
            print(f"     {file.path}")
    
    return results


async def list_files(limit: int, locator) -> list:
    """
    List files in database.
    
    Args:
        limit: Maximum number of files to show
        locator: Service locator
        
    Returns:
        List of files
    """
    from src.ucorefs.services.fs_service import FSService
    
    fs = locator.get_system(FSService)
    
    print(f"Listing files (limit: {limit})")
    
    # Get recent files
    files = await fs.get_recent_files(limit=limit)
    
    print(f"✓ Retrieved {len(files)} files")
    
    if files:
        print("\nFiles:")
        for i, file in enumerate(files, 1):
            print(f"  {i}. {file.name}")
            print(f"     Size: {file.size} bytes | Path: {file.path}")
    
    return files


async def show_stats(locator):
    """
    Show database statistics.
    
    Args:
        locator: Service locator
    """
    from src.ucorefs.services.fs_service import FSService
    from src.ucorefs.tags.manager import TagManager
    from src.ucorefs.albums.manager import AlbumManager
    
    fs = locator.get_system(FSService)
    tags = locator.get_system(TagManager)
    albums = locator.get_system(AlbumManager)
    
    print("Database Statistics:")
    print("=" * 50)
    
    # File count
    file_count = await fs.count_files()
    print(f"  Files: {file_count}")
    
    # Tag count
    all_tags = await tags.get_all_tags()
    print(f"  Tags: {len(all_tags)}")
    
    # Album count
    all_albums = await albums.get_all_albums()
    print(f"  Albums: {len(all_albums)}")
    
    print("=" * 50)


async def main():
    """Main entry point for console application."""
    parser = argparse.ArgumentParser(
        description="UCoreFS File Scanner - Console Tool"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan directory")
    scan_parser.add_argument("path", help="Directory path to scan")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search files")
    search_parser.add_argument("query", help="Search query")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List files")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of files to show")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Build console application (NO PySide6 needed!)
    print("Initializing UCoreFS (console mode)...")
    
    builder = (ApplicationBuilder.for_console("FileScanner", "config.json")
        .add_bundle(UCoreFSDataBundle()))
    
    locator = await builder.build()
    
    print("✓ Initialized\n")
    
    try:
        # Execute command
        if args.command == "scan":
            await scan_directory(args.path, locator)
        
        elif args.command == "search":
            await search_files(args.query, locator)
        
        elif args.command == "list":
            await list_files(args.limit, locator)
        
        elif args.command == "stats":
            await show_stats(locator)
        
        return 0
    
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
    
    finally:
        # Cleanup
        await locator.stop_all()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
