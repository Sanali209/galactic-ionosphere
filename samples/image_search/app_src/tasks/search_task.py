"""
Search handler for async image search.
"""
from datetime import datetime
from loguru import logger

# Use proper app_src module imports
from app_src.models.search_history import SearchHistory
from app_src.models.image_record import ImageRecord
from app_src.core.search_service import SearchService
from src.core.locator import ServiceLocator


async def search_images_handler(query: str, count: str):
    """
    Handler function for image search task.
    Args are passed as strings from TaskSystem.
    """
    logger.info(f"🚀 Search handler started: '{query}' (count: {count})")
    
    try:
        # Convert count to int
        count_int = int(count)
        logger.debug(f"Converted count to integer: {count_int}")
        
        # Get search service from ServiceLocator
        logger.debug("Getting SearchService from ServiceLocator...")
        sl = ServiceLocator()
        search_service = sl.get_system(SearchService)
        logger.debug("✅ SearchService retrieved")
        
        # Perform search
        logger.info(f"🔎 Executing search via SearchService...")
        results = await search_service.search_images(query, count_int)
        
        logger.info(f"📊 Search returned {len(results)} results")
        
        # Create search history record
        logger.debug("Creating SearchHistory record...")
        search_history = SearchHistory()
        search_history.query = query
        search_history.timestamp = datetime.now()
        search_history.result_count = len(results)
        search_history.max_results = count_int
        await search_history.save()
        
        logger.info(f"💾 Search history saved: {search_history._id}")
        
        # Save image records to database
        logger.debug(f"Saving {len(results)} image records to MongoDB...")
        for idx, result in enumerate(results, 1):
            image_record = ImageRecord()
            image_record.url = result.url
            image_record.thumbnail_url = result.thumbnail_url
            image_record.title = result.title
            image_record.width = result.width
            image_record.height = result.height
            image_record.source = result.source
            image_record.search_id = search_history._id
            await image_record.save()
            if idx % 5 == 0:  # Log every 5 images
                logger.debug(f"  Saved {idx}/{len(results)} images...")
        
        logger.info(f"✅ Successfully saved {len(results)} image records to database")
        logger.info(f"🎉 Search complete! Query: '{query}', Results: {len(results)}")
        
        # Return success message
        return f"Found {len(results)} images"
        
    except Exception as e:
        logger.error(f"❌ Search handler failed: {e}")
        logger.exception("Full traceback:")
        raise
