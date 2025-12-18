"""
UCoreFS - Thumbnail Service

Manages thumbnail generation and caching.
"""
import os
from typing import Optional, List
from pathlib import Path
from hashlib import md5
from loguru import logger
from bson import ObjectId

from src.core.base_system import BaseSystem
from src.ucorefs.models.file_record import FileRecord
from src.ucorefs.types.registry import registry


class ThumbnailService(BaseSystem):
    """
    Thumbnail generation and caching service.
    
    Features:
    - Configurable thumbnail sizes
    - Hash-based directory structure
    - WebP format for efficiency
    - Lazy generation (create on first request)
    """
    
    async def initialize(self) -> None:
        """Initialize thumbnail service."""
        logger.info("ThumbnailService initializing")
        
        # Get configuration
        thumbnail_config = self.config.data.thumbnail if hasattr(self.config.data, 'thumbnail') else None
        
        # Set defaults
        self.sizes = [128, 256, 512] if not thumbnail_config else thumbnail_config.sizes
        self.format = "webp"
        self.quality = 85
        self.cache_path = Path("./thumbnails")
        
        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        await super().initialize()
        logger.info(f"ThumbnailService ready (sizes: {self.sizes}, cache: {self.cache_path})")
    
    async def shutdown(self) -> None:
        """Shutdown thumbnail service."""
        logger.info("ThumbnailService shutting down")
        await super().shutdown()
    
    async def get_or_create(
        self,
        file_id: ObjectId,
        size: int = 256
    ) -> Optional[bytes]:
        """
        Get thumbnail, creating if needed.
        
        Args:
            file_id: File ObjectId
            size: Thumbnail size (must be in configured sizes)
            
        Returns:
            Thumbnail bytes or None
        """
        if size not in self.sizes:
            logger.warning(f"Invalid thumbnail size: {size}, using 256")
            size = 256
        
        # Check cache first
        cached_path = self.get_path(file_id, size)
        if cached_path and cached_path.exists():
            try:
                return cached_path.read_bytes()
            except Exception as e:
                logger.error(f"Failed to read cached thumbnail: {e}")
        
        # Generate new thumbnail
        thumbnail = await self._generate(file_id, size)
        
        if thumbnail:
            # Cache it
            await self._cache_thumbnail(file_id, size, thumbnail)
        
        return thumbnail
    
    async def regenerate(self, file_id: ObjectId, size: int = None) -> bool:
        """
        Force regenerate thumbnail(s).
        
        Args:
            file_id: File ObjectId
            size: Specific size to regenerate (None = all sizes)
            
        Returns:
            True if successful
        """
        sizes_to_generate = [size] if size else self.sizes
        
        success = True
        for s in sizes_to_generate:
            thumbnail = await self._generate(file_id, s)
            if thumbnail:
                await self._cache_thumbnail(file_id, s, thumbnail)
            else:
                success = False
        
        return success
    
    def get_path(self, file_id: ObjectId, size: int = 256) -> Optional[Path]:
        """
        Get cached thumbnail path.
        
        Args:
            file_id: File ObjectId
            size: Thumbnail size
            
        Returns:
            Path to cached thumbnail or None
        """
        file_id_str = str(file_id)
        hash_prefix = file_id_str[:4]  # First 4 chars
        
        # Structure: thumbnails/ab/cd/abcd1234_256.webp
        dir_path = self.cache_path / hash_prefix[:2] / hash_prefix[2:4]
        thumb_path = dir_path / f"{file_id_str}_{size}.{self.format}"
        
        return thumb_path
    
    async def _generate(self, file_id: ObjectId, size: int) -> Optional[bytes]:
        """
        Generate thumbnail using file driver.
        
        Args:
            file_id: File ObjectId
            size: Thumbnail size
            
        Returns:
            Thumbnail bytes or None
        """
        try:
            # Get file record
            file = await FileRecord.get(file_id)
            if not file:
                logger.error(f"File not found: {file_id}")
                return None
            
            # Get driver for file type
            driver = registry.get_driver(file.path, file.extension)
            
            # Generate thumbnail
            thumbnail = await driver.get_thumbnail(file)
            
            if thumbnail:
                # Resize to requested size if needed
                thumbnail = await self._resize_thumbnail(thumbnail, size)
            
            return thumbnail
        
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {file_id}: {e}")
            return None
    
    async def _resize_thumbnail(self, thumbnail_bytes: bytes, target_size: int) -> bytes:
        """
        Resize thumbnail to target size (runs in thread pool).
        
        Args:
            thumbnail_bytes: Original thumbnail bytes
            target_size: Target size
            
        Returns:
            Resized thumbnail bytes
        """
        import asyncio
        return await asyncio.to_thread(self._resize_impl, thumbnail_bytes, target_size)

    def _resize_impl(self, thumbnail_bytes: bytes, target_size: int) -> bytes:
        """Synchronous implementation of resize."""
        try:
            from PIL import Image
            import io
            
            # Load image
            img = Image.open(io.BytesIO(thumbnail_bytes))
            
            # Resize
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Save as WebP
            buffer = io.BytesIO()
            img.save(buffer, format="WEBP", quality=self.quality)
            
            return buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Failed to resize thumbnail: {e}")
            return thumbnail_bytes  # Return original on error
    
    async def _cache_thumbnail(
        self,
        file_id: ObjectId,
        size: int,
        thumbnail: bytes
    ) -> None:
        """
        Save thumbnail to cache (runs in thread pool).
        
        Args:
            file_id: File ObjectId
            size: Thumbnail size
            thumbnail: Thumbnail bytes
        """
        import asyncio
        await asyncio.to_thread(self._cache_impl, file_id, size, thumbnail)

    def _cache_impl(self, file_id: ObjectId, size: int, thumbnail: bytes) -> None:
        """Synchronous implementation of cache write."""
        try:
            thumb_path = self.get_path(file_id, size)
            
            # Create directory
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write thumbnail
            thumb_path.write_bytes(thumbnail)
            
            logger.debug(f"Cached thumbnail: {thumb_path}")
        
        except Exception as e:
            logger.error(f"Failed to cache thumbnail: {e}")
