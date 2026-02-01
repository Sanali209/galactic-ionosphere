<script>
  export let query = '';
  export let currentPath = '';
  
  let searchResults = [];
  let isSearching = false;
  let searchTimeout;
  
  async function performSearch() {
    if (!query || query.length < 2) {
      searchResults = [];
      return;
    }
    
    isSearching = true;
    try {
      const response = await fetch(`/api/search?query=${encodeURIComponent(query)}&path=${encodeURIComponent(currentPath)}`);
      const data = await response.json();
      searchResults = data.results;
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      isSearching = false;
    }
  }
  
  function handleInput() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(performSearch, 300);
  }
  
  function clearSearch() {
    query = '';
    searchResults = [];
  }
  
  function formatFileSize(bytes) {
    if (!bytes) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }
</script>

<div class="search-container">
  <div class="search-bar">
    <span class="search-icon">🔍</span>
    <input
      type="text"
      bind:value={query}
      on:input={handleInput}
      placeholder="Search files and folders..."
      class="search-input"
    />
    {#if query}
      <button class="clear-btn" on:click={clearSearch}>✕</button>
    {/if}
  </div>
  
  {#if isSearching}
    <div class="search-status">Searching...</div>
  {/if}
  
  {#if searchResults.length > 0}
    <div class="search-results">
      <div class="results-header">
        <strong>{searchResults.length}</strong> results for "{query}"
      </div>
      <div class="results-list">
        {#each searchResults as result}
          <div class="result-item">
            <span class="result-icon">
              {result.is_directory ? '📁' : '📄'}
            </span>
            <div class="result-info">
              <div class="result-name">{result.name}</div>
              <div class="result-path">{result.path}</div>
            </div>
            {#if result.size}
              <div class="result-size">{formatFileSize(result.size)}</div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}
  
  {#if query && !isSearching && searchResults.length === 0}
    <div class="no-results">No results found for "{query}"</div>
  {/if}
</div>

<style>
  .search-container {
    margin-bottom: 2rem;
  }
  
  .search-bar {
    display: flex;
    align-items: center;
    background: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    transition: border-color 0.2s;
  }
  
  .search-bar:focus-within {
    border-color: #667eea;
  }
  
  .search-icon {
    font-size: 1.2rem;
    margin-right: 0.75rem;
  }
  
  .search-input {
    flex: 1;
    background: none;
    border: none;
    color: #e0e0e0;
    font-size: 1rem;
    outline: none;
  }
  
  .search-input::placeholder {
    color: #808080;
  }
  
  .clear-btn {
    background: none;
    border: none;
    color: #808080;
    cursor: pointer;
    font-size: 1.2rem;
    padding: 0.25rem;
    transition: color 0.2s;
  }
  
  .clear-btn:hover {
    color: #e0e0e0;
  }
  
  .search-status {
    margin-top: 1rem;
    color: #667eea;
    font-style: italic;
  }
  
  .search-results {
    margin-top: 1rem;
    background: #2d2d2d;
    border-radius: 8px;
    overflow: hidden;
  }
  
  .results-header {
    padding: 1rem;
    background: #3d3d3d;
    border-bottom: 1px solid #4d4d4d;
  }
  
  .results-list {
    max-height: 400px;
    overflow-y: auto;
  }
  
  .result-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #3d3d3d;
    transition: background 0.2s;
    cursor: pointer;
  }
  
  .result-item:hover {
    background: #3d3d3d;
  }
  
  .result-icon {
    font-size: 1.5rem;
    margin-right: 1rem;
  }
  
  .result-info {
    flex: 1;
  }
  
  .result-name {
    font-weight: 500;
    margin-bottom: 0.25rem;
  }
  
  .result-path {
    font-size: 0.85rem;
    color: #808080;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .result-size {
    color: #808080;
    font-size: 0.9rem;
  }
  
  .no-results {
    margin-top: 1rem;
    padding: 2rem;
    text-align: center;
    color: #808080;
    background: #2d2d2d;
    border-radius: 8px;
  }
</style>
