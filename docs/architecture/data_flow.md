# Data Flow Diagrams

## 1. Discovery & Indexing Flow

How files move from the filesystem to the database and search index.

```mermaid
sequenceDiagram
    participant OS as Filesystem
    participant Scanner as DirectoryScanner
    participant Diff as DiffDetector
    participant Sync as SyncManager
    participant Pipeline as ProcessingPipeline
    participant DB as MongoDB
    
    note over OS, Scanner: Phase 1: Discovery
    Scanner->>OS: Scan Directory (Async Thread Pool)
    OS-->>Scanner: File Entries
    Scanner->>Diff: Batch Results
    Diff->>DB: Compare with existing
    Diff-->>Sync: Changes (Added/Modified/Deleted)
    Sync->>DB: Update Records (State: DISCOVERED)
    Sync-->>Pipeline: Enqueue Added IDs
    
    note over Pipeline, DB: Phase 2: Indexing
    Pipeline->>DB: Fetch Metadata
    Pipeline->>Pipeline: Extract Metadata (EXIF/XMP)
    Pipeline->>Pipeline: Generate Thumbnails
    Pipeline->>Pipeline: Generate CLIP Embeddings
    Pipeline->>DB: Update Record (State: INDEXED)
```

## 2. Search & Retrieval Flow

How user queries are processed and results are ranked.

```mermaid
sequenceDiagram
    participant UI as SearchPanel
    participant Service as SearchService
    participant MongoDB
    participant FAISS
    
    UI->>Service: Search("cat", tags=[nature])
    
    par MongoDB Filter
        Service->>MongoDB: Find IDs where tags=nature
        MongoDB-->>Service: {id1, id2, id3...}
    and Vector Search
        Service->>Service: Embed "cat" -> Vector
        Service->>FAISS: Search(Vector, k=100)
        FAISS-->>Service: {id2: 0.9, id4: 0.8...}
    end
    
    Service->>Service: Intersect & Rank Results
    Service-->>UI: Ordered SearchResults
```
