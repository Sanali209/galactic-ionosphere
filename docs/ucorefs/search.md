# Search System

The Search System (`src.ucorefs.search`) provides a unified interface for querying files using metadata filters, text search, and AI vector similarity.

## Search Service

The `SearchService` is the single entry point for all queries. It orchestrates:
1.  **MongoDB Query**: Filters by tags, rating, file type, or regex text matching.
2.  **Vector Search**: Queries FAISS for semantic similarity.
3.  **Hybrid Reranking**: Combines text scores and vector scores.

### Query capabilities

-   **Text**: "cat on a boat" (Matches filename, description, or semantic content).
-   **Filters**: `tag_ids=[...]`, `rating >= 4`, `file_type="image"`.
-   **Similar**: Find images similar to `file_id`.

## Vector Search (FAISS)

The system uses **FAISS** (Facebook AI Similarity Search) for high-performance vector retrieval.

-   **Service**: `FAISSIndexService` (`src.ucorefs.vectors.faiss_service`).
-   **Storage**: In-memory IVF (Inverted File) index, persisted to disk (`faiss_index.bin`).
-   **Providers**:
    -   **CLIP**: 512-dim vectors for text/image similarity.
    -   **MobileNet** (Legacy): Visual similarity.

### Hybrid Search Logic

1.  **Fetch Candidates**: MongoDB finds files matching hard filters (e.g., only "Nature" tag).
2.  **Vector Query**: If text query exists (e.g., "sunset"), generate CLIP embedding for text.
3.  **Filter FAISS**: Restrict FAISS search to the IDs returned by MongoDB (ID mapping).
4.  **Score Fusion**:
    -   `Final Score = (TextScore * 0.4) + (VectorScore * 0.6)`
    -   Boosts exact name matches while allowing semantic fuzzy matches.
