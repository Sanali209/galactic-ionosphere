import sys
from loguru import logger
import os

def setup_logging(debug_mode: bool = True, log_dir: str = "logs"):
    """
    Configures Loguru logger.
    """
    # Remove default handler
    logger.remove()

    # Console Handler
    level = "DEBUG" if debug_mode else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    # File Handler
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger.add(os.path.join(log_dir, "app_{time}.log"), rotation="10 MB", retention="1 week", level="DEBUG")

    logger.info("Logging initialized.")
