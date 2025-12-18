"""
DuckDuckGo image search service.
"""
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger
import sys
from pathlib import Path

# Temporary path setup (until pip install -e foundation)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "templates/foundation"))

from foundation import BaseSystem

@dataclass
class ImageSearchResult:
    """Result from image search."""
    url: str
    thumbnail_url: str
    title: str
    width: int
    height: int
    source: str

class SearchService(BaseSystem):
    """
    Service for searching images using DuckDuckGo.
    """
    
    async def initialize(self):
        """Initialize the search service."""
        logger.info("SearchService initialized")
    
    async def search_images(self, query: str, max_results: int = 20) -> List[ImageSearchResult]:
        """
        Search for images.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of image search results
        """
        logger.info(f"🔍 Starting image search: '{query}' (max: {max_results})")
        
        try:
            logger.debug("Importing ddgs module...")
            from ddgs import DDGS
            logger.debug("✅ ddgs imported successfully")
            
            results = []
            
            logger.info(f"📡 Calling DuckDuckGo API...")
            # Use DuckDuckGo search
            with DDGS() as ddgs:
                search_results = ddgs.images(query, max_results=max_results)
                
                logger.debug(f"Processing search results...")
                for idx, item in enumerate(search_results, 1):
                    result = ImageSearchResult(
                        url=item.get('image', ''),
                        thumbnail_url=item.get('thumbnail', ''),
                        title=item.get('title', 'Untitled'),
                        width=item.get('width', 0),
                        height=item.get('height', 0),
                        source=item.get('source', '')
                    )
                    results.append(result)
                    logger.debug(f"  [{idx}] {result.title} ({result.width}x{result.height})")
            
            logger.info(f"✅ Search complete: Found {len(results)} images for '{query}'")
            return results
            
        except ImportError as e:
            logger.error(f"❌ Import Error: {e}")
            logger.error("💡 Fix: Run 'pip install ddgs' in the correct environment")
            logger.error(f"   Current Python: {sys.executable}")
            return []
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            logger.exception("Full traceback:")
            return []
    
    async def shutdown(self):
        """Shutdown the search service."""
        logger.info("SearchService shutdown")
