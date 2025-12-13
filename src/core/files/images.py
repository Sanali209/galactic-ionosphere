import os
import asyncio
import re
from typing import Dict, Any, List, Union
from loguru import logger

# Try importing pyexiv2, handle failure gracefully (though it should be installed)
try:
    import pyexiv2
except ImportError:
    pyexiv2 = None
    logger.warning("pyexiv2 not found. Metadata operations will be limited.")

from src.core.files.base import FileHandler

class JpgHandler(FileHandler):
    @property
    def supported_extensions(self) -> List[str]:
        return ['.jpg', '.jpeg']

    async def extract_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extracts XMP and key Exif data using pyexiv2.
        Returns a dictionary with raw XMP and normalized fields.
        """
        if not pyexiv2:
            return {}

        return await asyncio.to_thread(self._read_metadata_sync, path)

    def _read_metadata_sync(self, path: str) -> Dict[str, Any]:
        try:
            with pyexiv2.Image(path) as img:
                xmp = img.read_xmp()
                exif = img.read_exif()
                iptc = img.read_iptc()

                # Helper to get description safely
                description = ""
                raw_desc = xmp.get("Xmp.dc.description", {})
                if isinstance(raw_desc, dict):
                    # Check for 'lang="x-default"' or "lang='x-default'"
                    # Pyexiv2 2.15+ behavior check: keys might be 'lang="x-default"'
                    for key, val in raw_desc.items():
                        if "x-default" in key:
                            description = val
                            break
                    # Fallback: take first value if no default found?
                    if not description and raw_desc:
                        description = list(raw_desc.values())[0]
                elif isinstance(raw_desc, str):
                    description = raw_desc

                # Tag Parsing logic
                # 1. Prefer Hierarchical Subject (Lightroom)
                # 2. Fallback to dc:subject, splitting by delimiters

                tags_flat = xmp.get("Xmp.dc.subject", [])
                if not isinstance(tags_flat, list):
                    tags_flat = [tags_flat] if tags_flat else []

                tags_hierarchical_raw = xmp.get("Xmp.lr.hierarchicalSubject", [])
                if not isinstance(tags_hierarchical_raw, list):
                    tags_hierarchical_raw = [tags_hierarchical_raw] if tags_hierarchical_raw else []

                # Normalize to our internal path format "A|B|C"
                # If we have explicit hierarchical tags, use them.
                # If we rely on flat tags, check for separators.

                final_tag_paths = []

                if tags_hierarchical_raw:
                     # Already paths, just ensure separator is pipe '|'
                     for t in tags_hierarchical_raw:
                         # normalize separators: replace / or \ with |
                         norm = re.sub(r'[\\/]', '|', t)
                         final_tag_paths.append(norm)
                else:
                    # Parse flat tags for hierarchy
                    for t in tags_flat:
                         # check for / \ | :
                         norm = re.sub(r'[\\/:]', '|', t)
                         final_tag_paths.append(norm)

                data = {
                    "raw_xmp": xmp,
                    "raw_exif": exif,
                    "raw_iptc": iptc,
                    "rating": int(xmp.get("Xmp.xmp.Rating", 0)),
                    "label": xmp.get("Xmp.xmp.Label", ""),
                    "description": description,
                    "tags": final_tag_paths, # List of "Path|To|Tag"
                    "leaf_tags": [p.split('|')[-1] for p in final_tag_paths]
                }

                return data
        except Exception as e:
            logger.error(f"Failed to read metadata from {path}: {e}")
            return {}

    async def write_metadata(self, path: str, metadata: Dict[str, Any]) -> None:
        """
        Writes XMP data to the file.
        'metadata' dict can contain:
        - 'rating': int
        - 'label': str
        - 'description': str
        - 'tags': List[str] (Expected to be full paths 'A|B|C')
        - 'xmp': Dict[str, str] (Raw XMP updates)
        """
        if not pyexiv2:
            logger.warning("pyexiv2 missing, cannot write metadata.")
            return

        await asyncio.to_thread(self._write_metadata_sync, path, metadata)

    def _write_metadata_sync(self, path: str, metadata: Dict[str, Any]):
        try:
            with pyexiv2.Image(path) as img:
                # Prepare updates
                xmp_updates = metadata.get("xmp", {}).copy()

                # Map standard fields to XMP
                if "rating" in metadata:
                    xmp_updates["Xmp.xmp.Rating"] = str(metadata["rating"])

                if "label" in metadata:
                    xmp_updates["Xmp.xmp.Label"] = metadata["label"]

                if "description" in metadata:
                    # Write as dict for AltLang. Pyexiv2 requires explicit lang qualifier key.
                    # Use double quotes for lang attribute as seen in read output.
                    xmp_updates["Xmp.dc.description"] = {'lang="x-default"': metadata["description"]}

                if "tags" in metadata:
                    # metadata['tags'] is list of paths "A|B|C"
                    paths = metadata["tags"]

                    # 1. Update Hierarchical
                    # Ensure we use | as separator? Or standard LR format?
                    # LR usually accepts | but displays as hierarchy.
                    xmp_updates["Xmp.lr.hierarchicalSubject"] = paths

                    # 2. Update Flat Subject (Leafs only)
                    leafs = [p.split('|')[-1] for p in paths]
                    # Deduplicate
                    leafs = list(set(leafs))
                    xmp_updates["Xmp.dc.subject"] = leafs

                img.modify_xmp(xmp_updates)
                logger.info(f"Updated metadata for {path}: {xmp_updates.keys()}")

        except Exception as e:
            logger.error(f"Failed to write metadata to {path}: {e}")
            raise

    async def generate_thumbnail(self, source_path: str, target_path: str, size: tuple = (256, 256)):
        # Stub: In real world use PIL
        try:
            from PIL import Image
            # Ensure folder exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with Image.open(source_path) as im:
                im.thumbnail(size)
                im.save(target_path, "JPEG")
        except ImportError:
            logger.warning("PIL not installed, cannot generate thumbnail.")
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")

    async def get_dimensions(self, path: str) -> Dict[str, int]:
        try:
            from PIL import Image
            with Image.open(path) as im:
                return {"width": im.width, "height": im.height}
        except ImportError:
            return {"width": 0, "height": 0}
        except Exception as e:
            logger.error(f"Failed to get dimensions: {e}")
            return {"width": 0, "height": 0}

class PngHandler(FileHandler):
    # PNG also supports XMP, so we can reuse similar logic or inherit a BaseXmpHandler
    @property
    def supported_extensions(self) -> List[str]:
        return ['.png']

    async def extract_metadata(self, path: str) -> Dict[str, Any]:
        if not pyexiv2: return {}
        return await JpgHandler().extract_metadata(path)

    async def write_metadata(self, path: str, metadata: Dict[str, Any]) -> None:
         await JpgHandler().write_metadata(path, metadata)

    async def generate_thumbnail(self, source_path: str, target_path: str, size: tuple = (256, 256)):
        await JpgHandler().generate_thumbnail(source_path, target_path, size)

    async def get_dimensions(self, path: str) -> Dict[str, int]:
        return await JpgHandler().get_dimensions(path)

class FileHandlerFactory:
    _handlers: Dict[str, FileHandler] = {}
    
    @classmethod
    def register(cls, handler: FileHandler):
        for ext in handler.supported_extensions:
            cls._handlers[ext.lower()] = handler
            
    @classmethod
    def get_handler(cls, ext: str) -> FileHandler:
        return cls._handlers.get(ext.lower())

# Register
FileHandlerFactory.register(JpgHandler())
FileHandlerFactory.register(PngHandler())
