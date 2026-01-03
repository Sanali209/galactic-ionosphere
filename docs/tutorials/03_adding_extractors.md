# Tutorial: Adding a Metadata Extractor

Extractors are plugins that generate metadata for files during the Indexing Phase.

## 1. Create Extractor Class

Inherit from `BaseExtractor` and implement `extract()`.

```python
from typing import List, Dict, Any
from src.ucorefs.extractors.base import BaseExtractor
from src.ucorefs.models.file_record import FileRecord

class WordCountExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "word_count"

    @property
    def phase(self) -> int:
        return 2  # Run during batch processing

    def can_process(self, file: FileRecord) -> bool:
        return file.extension.lower() == ".txt"

    async def process(self, files: List[FileRecord]) -> Dict[str, bool]:
        results = {}
        for file in files:
            try:
                # Logic: Count words in text file
                path = file.get_path()
                content = path.read_text()
                count = len(content.split())
                
                # Store metadata
                file.metadata["word_count"] = count
                await file.save()
                
                results[file._id] = True
            except Exception:
                results[file._id] = False
        return results
```

## 2. Register Extractor

In your bootstrap code (or usually `UCoreFSBundle`):

```python
from src.ucorefs.extractors import ExtractorRegistry

# Register globally
ExtractorRegistry.register(WordCountExtractor)
```

Now, every time a `.txt` file is indexed, `WordCountExtractor` will run and populate `metadata.word_count`.
