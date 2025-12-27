"""
Core Services.

Generic services for application infrastructure.

Note: FSService has been moved to src.ucorefs.services.
This module provides backward compatibility imports.
"""
# Re-export for backward compatibility
from src.ucorefs.services import FSService

__all__ = ['FSService']

