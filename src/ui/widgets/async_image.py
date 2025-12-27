"""
Async image widget for loading images from URLs.
"""
import asyncio
from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, QByteArray
from loguru import logger
import aiohttp

class AsyncImageWidget(QLabel):
    """
    Widget that loads and displays images asynchronously.
    Useful for loading images from URLs without blocking the UI.
    """
    image_loaded = Signal()
    load_failed = Signal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self._loading = False
        self._placeholder_text = "Loading..."
        
    def set_placeholder(self, text: str = "Loading..."):
        """Show placeholder text while loading."""
        self._placeholder_text = text
        self.setText(text)
        self.setStyleSheet("color: gray; font-style: italic;")
    
    def set_error(self, message: str = "Failed to load"):
        """Show error message."""
        self.setText(message)
        self.setStyleSheet("color: red; font-style: italic;")
    
    async def load_from_url(self, url: str, timeout: int = 10) -> bool:
        """
        Load image from URL asynchronously.
        
        Args:
            url: Image URL
            timeout: Timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if self._loading:
            logger.warning("Already loading an image")
            return False
        
        self._loading = True
        self.set_placeholder()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    data = await response.read()
                    
                    # Load image from bytes
                    image = QImage()
                    if not image.loadFromData(QByteArray(data)):
                        raise Exception("Invalid image data")
                    
                    pixmap = QPixmap.fromImage(image)
                    self.setPixmap(pixmap)
                    self.setStyleSheet("")  # Clear placeholder style
                    
                    self.image_loaded.emit()
                    logger.debug(f"Loaded image from {url}")
                    return True
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout loading image: {url}")
            self.set_error("Timeout")
            self.load_failed.emit("Timeout")
            return False
        except Exception as e:
            logger.error(f"Failed to load image from {url}: {e}")
            self.set_error()
            self.load_failed.emit(str(e))
            return False
        finally:
            self._loading = False
    
    def load_from_file(self, path: Path) -> bool:
        """
        Load image from local file.
        
        Args:
            path: Path to image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                raise Exception("Failed to load image")
            
            self.setPixmap(pixmap)
            self.setStyleSheet("")
            self.image_loaded.emit()
            return True
        except Exception as e:
            logger.error(f"Failed to load image from {path}: {e}")
            self.set_error()
            self.load_failed.emit(str(e))
            return False
    
    def clear_image(self):
        """Clear the current image."""
        self.clear()
        self.setStyleSheet("")
