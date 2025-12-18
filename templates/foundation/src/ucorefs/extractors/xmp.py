"""
UCoreFS - XMP Metadata Extractor

Extracts XMP metadata from image files using pyexiv2.
"""
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger


class XMPExtractor:
    """
    Extracts XMP metadata from images.
    
    Handles hierarchical tags with separators: / | \
    Imports label and description fields.
    """
    
    HIERARCHY_SEPARATORS = ['|', '/', '\\']
    
    def __init__(self):
        """Initialize XMP extractor."""
        self._pyexiv2_available = False
        try:
            import pyexiv2
            self._pyexiv2_available = True
        except ImportError:
            logger.warning("pyexiv2 not available, XMP extraction disabled")
    
    def is_available(self) -> bool:
        """Check if pyexiv2 is available."""
        return self._pyexiv2_available
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract XMP metadata from file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with extracted metadata
        """
        if not self._pyexiv2_available:
            return {}
        
        try:
            import pyexiv2
            
            img = pyexiv2.Image(file_path)
            xmp_data = img.read_xmp()
            img.close()
            
            result = {
                "tags": [],
                "label": "",
                "description": "",
                "raw_xmp": xmp_data
            }
            
            # Extract tags
            if "Xmp.dc.subject" in xmp_data:
                raw_tags = xmp_data["Xmp.dc.subject"]
                if isinstance(raw_tags, list):
                    result["tags"] = self._parse_hierarchical_tags(raw_tags)
                elif isinstance(raw_tags, str):
                    result["tags"] = self._parse_hierarchical_tags([raw_tags])
            
            # Extract label
            if "Xmp.xmp.Label" in xmp_data:
                result["label"] = xmp_data["Xmp.xmp.Label"]
            
            # Extract description
            if "Xmp.dc.description" in xmp_data:
                desc = xmp_data["Xmp.dc.description"]
                if isinstance(desc, dict) and "x-default" in desc:
                    result["description"] = desc["x-default"]
                elif isinstance(desc, str):
                    result["description"] = desc
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to extract XMP from {file_path}: {e}")
            return {}
    
    def _parse_hierarchical_tags(self, raw_tags: List[str]) -> List[str]:
        """
        Parse hierarchical tags and expand them.
        
        Args:
            raw_tags: List of raw tag strings
            
        Returns:
            List of expanded tag paths
        """
        expanded_tags = set()
        
        for tag in raw_tags:
            # Try each separator
            for sep in self.HIERARCHY_SEPARATORS:
                if sep in tag:
                    # Found hierarchy
                    parts = tag.split(sep)
                    
                    # Build path progressively
                    path_parts = []
                    for part in parts:
                        part = part.strip()
                        if part:
                            path_parts.append(part)
                            # Add full path up to this point
                            expanded_tags.add(sep.join(path_parts))
                    
                    break  # Stop after first separator found
            else:
                # No separator found, add as-is
                expanded_tags.add(tag.strip())
        
        return sorted(list(expanded_tags))


# Global instance
xmp_extractor = XMPExtractor()
