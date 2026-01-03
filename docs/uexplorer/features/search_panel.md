# Feature: Search Panel

The `UnifiedSearchPanel` allows users to construct complex queries.

## Components

1.  **Text Input**: Free text search.
    -   Matches filename, description, or AI caption.
    -   Auto-detects semantic queries vs filters.
2.  **Filter Builder**:
    -   Tag selector (Include/Exclude).
    -   Rating slider.
    -   File type checkboxes.
3.  **Sort Controls**: Relevancy vs Date/Name.

## Unified Query Builder

The `UnifiedQueryBuilder` class aggregates state from multiple panels:
-   **Text** from Search Panel.
-   **Tags** from Tags Panel.
-   **Album** from Album Panel.
-   **Dir** from Directory Panel.

It produces a `SearchQuery` object sent to `SearchService`.

## Integration

-   **Live Updates**: Changing a filter immediately updates the active Browser Document.
-   **Badge Display**: Active filters are shown as distinct chips in the UI.
