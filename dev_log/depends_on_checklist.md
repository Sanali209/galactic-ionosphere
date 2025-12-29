# Remaining depends_on Additions

This file lists all remaining services that need `depends_on` declarations.  
Apply these manually or use a batch script.

## Services Needing depends_on

### Detection Services
```python
# src/ucorefs/detection/service.py
class DetectionService(BaseSystem):
    depends_on = []  # Independent service
```

### Vector Services
```python
# src/ucorefs/vectors/f aiss_service.py
class FAISSIndexService(BaseSystem):
    depends_on = ["DatabaseManager"]

# src/ucorefs/vectors/service.py
class VectorService(BaseSystem):
    depends_on = ["FAISSIndexService", "DatabaseManager"]
```

### Search & Discovery
```python
# src/ucorefs/search/service.py
class SearchService(BaseSystem):
    depends_on = ["DatabaseManager", "VectorService"]

# src/ucorefs/discovery/service.py
class DiscoveryService(BaseSystem):
    depends_on = ["ProcessingPipeline", "DatabaseManager"]
```

### Data Management
```python
# src/ucorefs/tags/manager.py
class TagManager(BaseSystem):
    depends_on = ["DatabaseManager"]

# src/ucorefs/albums/manager.py
class AlbumManager(BaseSystem):
    depends_on = ["DatabaseManager"]

# src/ucorefs/relations/service.py
class RelationService(BaseSystem):
    depends_on = ["DatabaseManager", "ThumbnailService"]

# src/ucorefs/rules/engine.py
class RulesEngine(BaseSystem):
    depends_on = ["DatabaseManager", "TagManager"]

# src/ucorefs/annotation/service.py
class AnnotationService(BaseSystem):
    depends_on = ["DatabaseManager"]
```

### AI Services
```python
# src/ucorefs/ai/similarity_service.py
class SimilarityService(BaseSystem):
    depends_on = ["VectorService"]

# src/ucorefs/ai/llm_service.py
class LLMService(BaseSystem):
    depends_on = []  # Independent
```

### UI Services
```python
# src/ui/state/session_state.py
class SessionState(BaseSystem):
    depends_on = []  # Independent

# src/ui/navigation/service.py
class NavigationService(BaseSystem):
    depends_on = []  # Independent
```

### DockingService (NOT a BaseSystem)
```python
# src/ui/docking/service.py
class DockingService:
    async def shutdown(self):
        """Cleanup docking manager."""
        logger.info("DockingService shutting down")
```

## Application Instructions

For each file above:
1. Open the file
2. Find the class definition
3. Add `depends_on = [...]` right after the class docstring
4. For DockingService, add the shutdown() method at the end of the class
