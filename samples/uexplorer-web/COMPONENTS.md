# Component Documentation

## Svelte Component Architecture

### App.svelte (Main Container)

The root component that orchestrates the application:

```svelte
<script>
  - Imports DirectoryBrowser and SearchBar
  - Manages global state (currentPath, searchQuery)
  - Loads initial directory on mount
</script>

<template>
  <header>
    - Application title
    - Subtitle
  </header>
  
  <container>
    <SearchBar /> - Search interface
    <DirectoryBrowser /> - Main file browser
  </container>
</template>

<style>
  - Dark theme styling
  - Gradient header
  - Responsive layout
</style>
```

### DirectoryBrowser.svelte

Main file browsing interface with full directory navigation:

**Props:**
- `currentPath` (string, bindable) - Current directory path
- `searchQuery` (string) - Active search query

**Features:**
- Navigation toolbar with Up, Home, Refresh buttons
- Show/hide hidden files toggle
- Custom path input with Go button
- File list in grid layout
- Column headers (Icon, Name, Size, Modified)
- Directory click navigation
- Empty state handling
- Loading and error states

**API Integration:**
- `GET /api/browse` - Load directory contents
- `GET /api/home` - Navigate to home directory

**Styling:**
- Dark card with rounded corners
- Toolbar with controls
- Monospace path display
- Grid layout for file list
- Hover effects on rows
- Responsive column sizing

### SearchBar.svelte

Real-time search interface with debounced queries:

**Props:**
- `query` (string, bindable) - Search query text
- `currentPath` (string) - Directory to search in

**Features:**
- Search input with icon
- Debounced search (300ms delay)
- Clear button when query is active
- Loading indicator during search
- Results list with file details
- Result count display
- Empty state for no results

**API Integration:**
- `GET /api/search?query=<text>&path=<dir>` - Perform search

**Styling:**
- Styled input with focus states
- Results card with scrollable list
- Result items with file info
- File size formatting
- Path truncation

## Data Flow

```
User Action
    ↓
Component Event Handler
    ↓
API Request (fetch)
    ↓
FastAPI Backend
    ↓
File System Operation
    ↓
JSON Response
    ↓
Component State Update
    ↓
UI Re-render
```

## Component Communication

### Parent to Child (Props)
- App → DirectoryBrowser: `currentPath`, `searchQuery`
- App → SearchBar: `query`, `currentPath`

### Child to Parent (Bindings)
- DirectoryBrowser → App: `currentPath` (bind:currentPath)
- SearchBar → App: `query` (bind:query)

### Sibling Communication
- Via shared parent state (App.svelte)
- SearchBar sets `query` → App updates → DirectoryBrowser receives

## Styling Approach

All components use scoped CSS with:
- Dark color scheme (#1e1e1e, #2d2d2d, #3d3d3d)
- Accent colors (#667eea, #764ba2)
- System font stack
- Responsive units (rem, em)
- Transitions for smooth interactions
- Hover states for interactive elements

## API Response Formats

### Directory Content
```json
{
  "current_path": "/home/user",
  "parent_path": "/home",
  "items": [
    {
      "name": "Documents",
      "path": "/home/user/Documents",
      "is_directory": true,
      "size": null,
      "modified": "2024-01-29T10:30:00",
      "extension": null
    }
  ],
  "total_items": 15
}
```

### Search Results
```json
{
  "query": "test",
  "results": [...],
  "total_results": 5
}
```

## Component Lifecycle

### DirectoryBrowser
1. Component created
2. `loadDirectory()` called automatically
3. API request to `/api/browse`
4. Update `files` array
5. Render file list
6. User interactions trigger new loads

### SearchBar
1. Component created
2. User types in search input
3. Debounce timer starts (300ms)
4. On timer completion: API request
5. Update `searchResults` array
6. Render results

## Error Handling

Both components handle:
- Network errors (fetch failures)
- API errors (4xx, 5xx responses)
- Permission denied (403)
- Path not found (404)
- Display user-friendly error messages

## Performance Considerations

- Debounced search to reduce API calls
- Lazy loading (only current directory)
- Result limiting (max 100 search results)
- Efficient array updates (Svelte reactivity)
- No unnecessary re-renders
