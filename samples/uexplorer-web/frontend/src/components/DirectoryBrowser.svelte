<script>
  export let currentPath = '';
  export let searchQuery = '';
  
  let files = [];
  let parentPath = null;
  let loading = false;
  let error = null;
  let showHidden = false;
  let customPath = '';
  
  async function loadDirectory(path = null) {
    loading = true;
    error = null;
    
    try {
      const targetPath = path || currentPath;
      const url = `/api/browse?path=${encodeURIComponent(targetPath)}&show_hidden=${showHidden}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to load directory: ${response.statusText}`);
      }
      
      const data = await response.json();
      currentPath = data.current_path;
      parentPath = data.parent_path;
      files = data.items;
    } catch (err) {
      error = err.message;
      console.error('Error loading directory:', err);
    } finally {
      loading = false;
    }
  }
  
  async function navigateToPath(path) {
    await loadDirectory(path);
  }
  
  async function goToParent() {
    if (parentPath) {
      await navigateToPath(parentPath);
    }
  }
  
  async function goToHome() {
    try {
      const response = await fetch('/api/home');
      const data = await response.json();
      await navigateToPath(data.home);
    } catch (err) {
      error = 'Failed to navigate to home directory';
    }
  }
  
  function handleFileClick(file) {
    if (file.is_directory) {
      navigateToPath(file.path);
    }
  }
  
  async function changeDirectory() {
    if (customPath) {
      await navigateToPath(customPath);
      customPath = '';
    }
  }
  
  function formatFileSize(bytes) {
    if (!bytes) return '-';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }
  
  function formatDate(isoString) {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleString();
  }
  
  function toggleHidden() {
    showHidden = !showHidden;
    loadDirectory();
  }
  
  // Load initial directory
  loadDirectory();
</script>

<div class="browser">
  <div class="toolbar">
    <div class="navigation">
      <button on:click={goToParent} disabled={!parentPath} class="nav-btn" title="Go to parent directory">
        ⬆️ Up
      </button>
      <button on:click={goToHome} class="nav-btn" title="Go to home directory">
        🏠 Home
      </button>
      <button on:click={() => loadDirectory()} class="nav-btn" title="Refresh">
        🔄 Refresh
      </button>
      <label class="checkbox-label">
        <input type="checkbox" bind:checked={showHidden} on:change={toggleHidden} />
        Show Hidden
      </label>
    </div>
    
    <div class="path-selector">
      <input
        type="text"
        bind:value={customPath}
        placeholder="Enter path to navigate..."
        class="path-input"
        on:keydown={(e) => e.key === 'Enter' && changeDirectory()}
      />
      <button on:click={changeDirectory} class="go-btn">Go</button>
    </div>
  </div>
  
  <div class="current-path">
    <strong>📂 Current:</strong> {currentPath}
  </div>
  
  {#if loading}
    <div class="status">Loading directory...</div>
  {:else if error}
    <div class="error">❌ {error}</div>
  {:else}
    <div class="file-list">
      <div class="file-header">
        <div class="col-icon"></div>
        <div class="col-name">Name</div>
        <div class="col-size">Size</div>
        <div class="col-modified">Modified</div>
      </div>
      
      {#each files as file}
        <div 
          class="file-row" 
          class:directory={file.is_directory}
          on:click={() => handleFileClick(file)}
        >
          <div class="col-icon">
            {file.is_directory ? '📁' : '📄'}
          </div>
          <div class="col-name" title={file.name}>
            {file.name}
            {#if file.extension && !file.is_directory}
              <span class="extension">{file.extension}</span>
            {/if}
          </div>
          <div class="col-size">
            {formatFileSize(file.size)}
          </div>
          <div class="col-modified">
            {formatDate(file.modified)}
          </div>
        </div>
      {/each}
      
      {#if files.length === 0}
        <div class="empty-message">
          This directory is empty
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .browser {
    background: #2d2d2d;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  
  .toolbar {
    background: #3d3d3d;
    padding: 1rem;
    border-bottom: 2px solid #4d4d4d;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
  }
  
  .navigation {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
  }
  
  .nav-btn {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background 0.2s;
  }
  
  .nav-btn:hover:not(:disabled) {
    background: #5568d3;
  }
  
  .nav-btn:disabled {
    background: #4d4d4d;
    color: #808080;
    cursor: not-allowed;
  }
  
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #e0e0e0;
    cursor: pointer;
  }
  
  .path-selector {
    display: flex;
    gap: 0.5rem;
    flex: 1;
    min-width: 300px;
  }
  
  .path-input {
    flex: 1;
    background: #2d2d2d;
    border: 1px solid #4d4d4d;
    color: #e0e0e0;
    padding: 0.5rem;
    border-radius: 6px;
    font-size: 0.9rem;
  }
  
  .path-input:focus {
    outline: none;
    border-color: #667eea;
  }
  
  .go-btn {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.5rem 1.5rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background 0.2s;
  }
  
  .go-btn:hover {
    background: #5568d3;
  }
  
  .current-path {
    padding: 1rem;
    background: #3d3d3d;
    border-bottom: 1px solid #4d4d4d;
    font-family: monospace;
    font-size: 0.9rem;
    overflow-x: auto;
    white-space: nowrap;
  }
  
  .status, .error {
    padding: 2rem;
    text-align: center;
  }
  
  .error {
    color: #ff6b6b;
  }
  
  .file-list {
    min-height: 400px;
  }
  
  .file-header {
    display: grid;
    grid-template-columns: 40px 1fr 120px 200px;
    gap: 1rem;
    padding: 1rem;
    background: #3d3d3d;
    border-bottom: 2px solid #4d4d4d;
    font-weight: 600;
    font-size: 0.9rem;
  }
  
  .file-row {
    display: grid;
    grid-template-columns: 40px 1fr 120px 200px;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #3d3d3d;
    transition: background 0.2s;
    cursor: default;
    align-items: center;
  }
  
  .file-row.directory {
    cursor: pointer;
  }
  
  .file-row:hover {
    background: #3d3d3d;
  }
  
  .file-row.directory:hover {
    background: #4d4d4d;
  }
  
  .col-icon {
    font-size: 1.5rem;
    text-align: center;
  }
  
  .col-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .extension {
    color: #667eea;
    font-size: 0.85rem;
    margin-left: 0.25rem;
  }
  
  .col-size, .col-modified {
    color: #b0b0b0;
    font-size: 0.9rem;
  }
  
  .empty-message {
    padding: 3rem;
    text-align: center;
    color: #808080;
    font-style: italic;
  }
</style>
