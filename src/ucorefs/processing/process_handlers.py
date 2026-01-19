"""
Process-safe handlers for Phase 2 batch processing.

These functions run in separate processes - they MUST be:
1. Top-level functions (not methods or closures)
2. Use only picklable arguments
3. Not access shared state directly
4. Load resources locally within the function

Phase 2 handles: Thumbnails, EXIF, XMP, CLIP embeddings
"""
from typing import Dict, List, Any
from bson import ObjectId


def process_phase2_batch_sync(file_ids_str: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process Phase 2 batch in isolated process.
    
    Args:
        file_ids_str: Comma-separated file ObjectIds
        config: Extractor configuration dict (must include 'db_uri' and 'db_name')
        
    Returns:
        Dict with processing results
    """
    # Import inside function to ensure clean process state
    import asyncio
    from loguru import logger
    
    # Parse file IDs
    file_ids = [ObjectId(fid) for fid in file_ids_str.split(",") if fid]
    
    logger.info(f"[PROCESS] Phase 2 batch processing {len(file_ids)} files")
    
    # Initialize database connection in this process
    db_uri = config.get("db_uri", "mongodb://localhost:27017")
    db_name = config.get("db_name", "foundation_app")
    
    # Run async logic with DB initialization
    return asyncio.run(_process_phase2_with_db(file_ids, config, db_uri, db_name))


async def _process_phase2_with_db(file_ids: List[ObjectId], config: Dict[str, Any], db_uri: str, db_name: str) -> Dict[str, Any]:
    """Initialize DB and run Phase 2 processing."""
    from motor.motor_asyncio import AsyncIOMotorClient
    from src.core.database.manager import DatabaseManager
    from loguru import logger
    
    # Create standalone DB connection for this process
    try:
        client = AsyncIOMotorClient(db_uri)
        db = client[db_name]
        
        # Initialize DatabaseManager with this client
        # Use singleton pattern workaround for process isolation
        DatabaseManager._instance = None  # Reset singleton
        db_manager = DatabaseManager.__new__(DatabaseManager)
        db_manager._client = client
        db_manager._db = db
        db_manager._collections = {}
        DatabaseManager._instance = db_manager
        
        logger.debug(f"[PROCESS] Initialized DB connection to {db_name}")
        
        result = await _process_phase2_async(file_ids, config)
        
        # Cleanup
        client.close()
        DatabaseManager._instance = None
        
        return result
        
    except Exception as e:
        logger.error(f"[PROCESS] DB initialization failed: {e}")
        return {"processed": 0, "errors": len(file_ids), "by_extractor": {}}



async def _process_phase2_async(file_ids: List[ObjectId], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async implementation of Phase 2 processing.
    
    Extracted to allow proper async/await within the process.
    """
    from src.ucorefs.models.file_record import FileRecord
    from src.ucorefs.extractors import ExtractorRegistry
    from src.ucorefs.models.base import ProcessingState
    from loguru import logger
    
    results = {
        "processed": 0,
        "errors": 0,
        "by_extractor": {}
    }
    
    # Load files
    files = []
    for file_id in file_ids:
        try:
            file = await FileRecord.get(file_id)
            if file:
                files.append(file)
        except Exception as e:
            logger.error(f"[PROCESS] Failed to load {file_id}: {e}")
            results["errors"] += 1
    
    if not files:
        logger.warning("[PROCESS] No files loaded")
        return results
    
    # Get Phase 2 extractors
    # NOTE: locator is None in process context, config passed as dict
    extractors = ExtractorRegistry.get_for_phase(2, locator=None, config=config)
    
    logger.info(f"[PROCESS] Running {len(extractors)} extractors")
    
    # Process with each extractor
    for extractor in extractors:
        try:
            processable = [f for f in files if extractor.can_process(f)]
            
            if processable:
                extractor_results = await extractor.process(processable)
                success_count = sum(1 for v in extractor_results.values() if v)
                results["by_extractor"][extractor.name] = success_count
                results["processed"] += success_count
                
                logger.info(f"[PROCESS] {extractor.name}: {success_count}/{len(processable)} successful")
        except Exception as e:
            logger.error(f"[PROCESS] Extractor {extractor.name} failed: {e}")
            results["errors"] += 1
    
    # Update processing state
    for file in files:
        file = await FileRecord.get(file._id)  # Re-fetch
        if file and results["processed"] > 0:
            if file.processing_state < ProcessingState.INDEXED:
                file.processing_state = ProcessingState.INDEXED
                await file.save()
    
    logger.info(f"[PROCESS] Phase 2 complete: {results}")
    return results
