# UCoreFS Overview

UCoreFS ("Universal Core File System") is the intelligent data layer of USCore. It treats the filesystem as a database, enriching static files with dynamic metadata and AI understanding.

## Core Modules

| Module | Description | Doc Link |
| :--- | :--- | :--- |
| **Discovery** | Watches filesystem for changes. | [Discovery](discovery.md) |
| **Pipeline** | The AI indexing engine (Phases 1-3). | [Pipeline](pipeline.md) |
| **Search** | Hybrid Vector + Metadata search. | [Search](search.md) |
| **Organization** | Tags, Albums, and Rules. | [Organization](organization.md) |
| **Models** | Data schema (`FileRecord`, etc.). | [Models](models.md) |

## Key Concepts

-   **Mirroring**: UCoreFS mirrors the physical filesystem state. If a file is deleted on disk, it is marked as `MISSING` or deleted from DB.
-   **Extensibility**: New file types and metadata extractors can be added via plugins.
-   **Performance**: Uses MongoDB for metadata and FAISS for vectors, ensuring sub-second queries over 100k+ files.
