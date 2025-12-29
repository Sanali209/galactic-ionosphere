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
    
    depends_on = ["DatabaseManager"]
    
    async def initialize(self) -> None:
        """Initialize thumbnail service."""
        logger.info("ThumbnailService initializing")
        
        # Get configuration
        thumbnail_config = self.config.data.thumbnail if hasattr(self.config.data, 'thumbnail') else None
        
        # Set defaults
        self.sizes = [128, 256, 512] if not thumbnail_config else thumbnail_config.sizes
        self.format = "jpeg"  # Use JPEG for better compatibility
        self.extension = "jpg"
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

    # ... (skipping methods that don't need changes: regenerate) ...

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
        
        # Structure: thumbnails/ab/cd/abcd1234_256.jpg
        dir_path = self.cache_path / hash_prefix[:2] / hash_prefix[2:4]
        thumb_path = dir_path / f"{file_id_str}_{size}.{self.extension}"
        
        return thumb_path

    async def _generate(self, file_id: ObjectId, size: int) -> Optional[bytes]:
        """
        Generate thumbnail from source file.
        
        Args:
            file_id: File ObjectId
            size: Target thumbnail size
            
        Returns:
            Thumbnail bytes or None
        """
        import asyncio
        
        # Get file record to find source path
        try:
            # FileRecord.find_one is async, so await it directly
            file_record = await FileRecord.find_one({"_id": file_id})
            if not file_record or not file_record.path:
                logger.debug(f"File not found for thumbnail: {file_id}")
                return None
            
            source_path = Path(file_record.path)
            if not source_path.exists():
                logger.debug(f"Source file not found: {source_path}")
                return None
            
            # Check if it's an image (simple extension check)
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
            if source_path.suffix.lower() not in image_extensions:
                return None
            
            # Generate thumbnail in thread
            return await asyncio.to_thread(
                self._generate_impl, source_path, size
            )
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None
    
    def _generate_impl(self, source_path: Path, size: int) -> Optional[bytes]:
        """Synchronous thumbnail generation."""
        try:
            from PIL import Image
            import io
            
            # Open and resize
            with Image.open(source_path) as img:
                # Convert mode for JPEG compatibility
                if img.mode in ("RGBA", "LA", "P"):
                    if img.mode == "P" and "transparency" in img.info:
                        img = img.convert("RGBA")
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode in ("RGBA", "LA"):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize maintaining aspect ratio
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format=self.format, quality=self.quality)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {source_path}: {e}")
            return None

    def _resize_impl(self, thumbnail_bytes: bytes, target_size: int) -> bytes:
        """Synchronous implementation of resize."""
        try:
            from PIL import Image
            import io
            
            # Load image
            img = Image.open(io.BytesIO(thumbnail_bytes))
            
            # Resize
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

            # Convert to RGB if saving as JPEG (handling RGBA/P)
            if self.format.lower() in ("jpeg", "jpg"):
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode == "P":
                     img = img.convert("RGB")
            
            # Save using configured format
            buffer = io.BytesIO()
            img.save(buffer, format=self.format, quality=self.quality)
            
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
