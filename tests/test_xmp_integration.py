import os
import shutil
import pytest
import asyncio
from src.core.files.images import JpgHandler, FileHandlerFactory

# We need a sample image. We can create a dummy one or use PIL to make one.
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
@pytest.mark.asyncio
async def test_xmp_read_write(tmp_path):
    # 1. Create a dummy JPEG
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(img_path)

    # 2. Initialize Handler
    handler = JpgHandler()

    # 3. Write Metadata
    # Note: Description handling depends on pyexiv2 version/behavior.
    # We verified it returns {'lang="x-default"': val}
    meta_to_write = {
        "rating": 4,
        "label": "Blue",
        "description": "A test image",
        "tags": ["test", "unit", "demo"]
    }

    await handler.write_metadata(str(img_path), meta_to_write)

    # 4. Read Metadata
    read_meta = await handler.extract_metadata(str(img_path))

    # 5. Verify
    assert read_meta["rating"] == 4
    assert read_meta["label"] == "Blue"
    assert read_meta["description"] == "A test image"
    assert set(read_meta["tags"]) == set(["test", "unit", "demo"])

    # Verify Raw XMP presence
    assert "Xmp.xmp.Rating" in read_meta["raw_xmp"]
    assert read_meta["raw_xmp"]["Xmp.xmp.Rating"] == "4"

@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
@pytest.mark.asyncio
async def test_update_existing_metadata(tmp_path):
    if not HAS_PIL: return

    img_path = tmp_path / "update_test.jpg"
    img = Image.new('RGB', (100, 100), color = 'blue')
    img.save(img_path)

    handler = JpgHandler()

    # Initial Write
    await handler.write_metadata(str(img_path), {"rating": 1, "description": "Original"})

    # Update
    await handler.write_metadata(str(img_path), {"rating": 5})

    # Check
    read_meta = await handler.extract_metadata(str(img_path))
    assert read_meta["rating"] == 5
    assert read_meta["description"] == "Original" # Should persist

@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
@pytest.mark.asyncio
async def test_hierarchical_tags_read(tmp_path):
    # Test that we can parse "A/B/C" from Xmp.dc.subject if Xmp.lr.hierarchicalSubject is missing
    # Or explicitly write hierarchical and read it back.

    img_path = tmp_path / "tree.jpg"
    Image.new('RGB', (50, 50)).save(img_path)

    handler = JpgHandler()

    # Case 1: Write explicit paths
    paths = ["Animals|Mammals|Cat", "Places|Home"]
    await handler.write_metadata(str(img_path), {"tags": paths})

    read_meta = await handler.extract_metadata(str(img_path))

    # Should have normalized paths
    assert "Animals|Mammals|Cat" in read_meta["tags"]
    assert "Places|Home" in read_meta["tags"]

    # Should have leaf tags in 'leaf_tags'
    assert "Cat" in read_meta["leaf_tags"]
    assert "Home" in read_meta["leaf_tags"]

    # Verify raw XMP has Hierarchical Subject (LR style)
    # Note: pyexiv2 might format it slightly differently, but we expect keys.
    raw_keys = read_meta["raw_xmp"].keys()
    # Check if LR key exists (it should since we wrote it)
    assert "Xmp.lr.hierarchicalSubject" in raw_keys

@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
@pytest.mark.asyncio
async def test_hierarchical_tags_parsing_separators(tmp_path):
    # Manually inject XMP with / separators to test normalization
    img_path = tmp_path / "legacy.jpg"
    Image.new('RGB', (50, 50)).save(img_path)

    import pyexiv2
    with pyexiv2.Image(str(img_path)) as img:
        # Simulate old software writing paths to dc:subject with slash
        img.modify_xmp({"Xmp.dc.subject": ["Old/Path/To/Tag", "SimpleTag"]})

    handler = JpgHandler()
    read_meta = await handler.extract_metadata(str(img_path))

    # Should normalize / to |
    assert "Old|Path|To|Tag" in read_meta["tags"]
    assert "SimpleTag" in read_meta["tags"]
