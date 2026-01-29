<script>
  import { files, tags, albums, formatFileSize, formatDate, formatRating } from '../lib/api.js';
  
  export let fileId = null;
  
  let fileData = null;
  let fileTags = null;
  let fileAlbums = null;
  let loading = false;
  let error = null;
  let newRating = 0;
  
  async function loadFileMetadata() {
    if (!fileId) return;
    
    loading = true;
    error = null;
    
    try {
      [fileData, fileTags, fileAlbums] = await Promise.all([
        files.get(fileId),
        tags.getForFile(fileId),
        albums.getForFile(fileId)
      ]);
      newRating = fileData.rating || 0;
    } catch (err) {
      error = err.message;
      console.error('Failed to load file metadata:', err);
    } finally {
      loading = false;
    }
  }
  
  async function updateRating() {
    if (!fileId) return;
    
    try {
      await files.updateRating(fileId, newRating);
      fileData.rating = newRating;
    } catch (err) {
      alert(`Failed to update rating: ${err.message}`);
    }
  }
  
  function setRating(rating) {
    newRating = rating;
    updateRating();
  }
  
  $: if (fileId) loadFileMetadata();
</script>

<div class="metadata-panel">
  <div class="panel-header">
    <h3>📄 File Metadata</h3>
  </div>
  
  {#if !fileId}
    <div class="no-selection">
      <p>No file selected</p>
      <p class="hint">Select a file to view metadata</p>
    </div>
  {:else if loading}
    <div class="status">Loading metadata...</div>
  {:else if error}
    <div class="error">❌ {error}</div>
  {:else if fileData}
    <div class="metadata-content">
      <!-- File Info -->
      <div class="section">
        <h4>File Information</h4>
        <div class="info-grid">
          <div class="info-item">
            <span class="label">Name:</span>
            <span class="value">{fileData.name}</span>
          </div>
          <div class="info-item">
            <span class="label">Path:</span>
            <span class="value path">{fileData.path}</span>
          </div>
          <div class="info-item">
            <span class="label">Extension:</span>
            <span class="value">{fileData.extension || 'None'}</span>
          </div>
          <div class="info-item">
            <span class="label">Size:</span>
            <span class="value">{formatFileSize(fileData.size)}</span>
          </div>
          <div class="info-item">
            <span class="label">Created:</span>
            <span class="value">{formatDate(fileData.created_at)}</span>
          </div>
          <div class="info-item">
            <span class="label">Modified:</span>
            <span class="value">{formatDate(fileData.modified_at)}</span>
          </div>
          <div class="info-item">
            <span class="label">State:</span>
            <span class="value state-{fileData.processing_state}">
              {fileData.processing_state}
            </span>
          </div>
        </div>
      </div>
      
      <!-- Rating -->
      <div class="section">
        <h4>Rating</h4>
        <div class="rating-widget">
          {#each [1, 2, 3, 4, 5] as star}
            <button
              class="star"
              class:filled={star <= newRating}
              on:click={() => setRating(star)}
            >
              {star <= newRating ? '★' : '☆'}
            </button>
          {/each}
          {#if newRating > 0}
            <button class="clear-rating" on:click={() => setRating(0)}>Clear</button>
          {/if}
        </div>
      </div>
      
      <!-- Description -->
      {#if fileData.description}
        <div class="section">
          <h4>Description</h4>
          <p class="description">{fileData.description}</p>
        </div>
      {/if}
      
      <!-- Tags -->
      <div class="section">
        <h4>Tags ({fileTags?.count || 0})</h4>
        {#if fileTags && fileTags.tags.length > 0}
          <div class="tag-list">
            {#each fileTags.tags as tag}
              <span class="tag-badge" style="border-color: {tag.color || '#667eea'}">
                {tag.name}
              </span>
            {/each}
          </div>
        {:else}
          <p class="empty">No tags</p>
        {/if}
      </div>
      
      <!-- Albums -->
      <div class="section">
        <h4>Albums ({fileAlbums?.count || 0})</h4>
        {#if fileAlbums && fileAlbums.albums.length > 0}
          <div class="album-list">
            {#each fileAlbums.albums as album}
              <div class="album-item">
                <span class="album-icon">{album.icon || '📚'}</span>
                <span class="album-name">{album.name}</span>
                {#if album.is_smart}
                  <span class="smart-badge">🔮</span>
                {/if}
              </div>
            {/each}
          </div>
        {:else}
          <p class="empty">Not in any albums</p>
        {/if}
      </div>
      
      <!-- Auto Tags -->
      {#if fileData.tags_auto && fileData.tags_auto.length > 0}
        <div class="section">
          <h4>Auto-Generated Tags</h4>
          <div class="tag-list">
            {#each fileData.tags_auto as tag}
              <span class="tag-badge auto">{tag}</span>
            {/each}
          </div>
        </div>
      {/if}
      
      <!-- Custom Properties -->
      {#if fileData.custom_properties && Object.keys(fileData.custom_properties).length > 0}
        <div class="section">
          <h4>Custom Properties</h4>
          <div class="properties">
            {#each Object.entries(fileData.custom_properties) as [key, value]}
              <div class="property-item">
                <span class="prop-key">{key}:</span>
                <span class="prop-value">{JSON.stringify(value)}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .metadata-panel {
    background: #2d2d2d;
    border-radius: 8px;
    padding: 1rem;
    max-height: 600px;
    overflow-y: auto;
  }
  
  .panel-header {
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #4d4d4d;
  }
  
  .panel-header h3 {
    margin: 0;
    font-size: 1.2rem;
  }
  
  .no-selection {
    padding: 3rem 1rem;
    text-align: center;
    color: #808080;
  }
  
  .no-selection .hint {
    font-size: 0.85rem;
    margin-top: 0.5rem;
  }
  
  .metadata-content {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .section h4 {
    margin: 0 0 0.75rem 0;
    font-size: 1rem;
    color: #b0b0b0;
  }
  
  .info-grid {
    display: grid;
    gap: 0.5rem;
  }
  
  .info-item {
    display: grid;
    grid-template-columns: 100px 1fr;
    gap: 0.5rem;
    font-size: 0.9rem;
  }
  
  .label {
    color: #808080;
  }
  
  .value {
    color: #e0e0e0;
  }
  
  .value.path {
    word-break: break-all;
    font-size: 0.85rem;
  }
  
  .state-raw {
    color: #ffa500;
  }
  
  .state-processing {
    color: #667eea;
  }
  
  .state-complete {
    color: #4caf50;
  }
  
  .state-error {
    color: #ff6b6b;
  }
  
  .rating-widget {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }
  
  .star {
    background: none;
    border: none;
    color: #ffd700;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0.25rem;
    transition: transform 0.1s;
  }
  
  .star:hover {
    transform: scale(1.2);
  }
  
  .star:not(.filled) {
    color: #4d4d4d;
  }
  
  .clear-rating {
    background: #4d4d4d;
    border: none;
    color: #e0e0e0;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    margin-left: 0.5rem;
  }
  
  .description {
    color: #e0e0e0;
    font-size: 0.9rem;
    line-height: 1.4;
    margin: 0;
  }
  
  .tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  .tag-badge {
    background: #3d3d3d;
    border-left: 3px solid #667eea;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.85rem;
  }
  
  .tag-badge.auto {
    border-left-color: #ffa500;
    opacity: 0.8;
  }
  
  .album-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .album-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    background: #3d3d3d;
    border-radius: 4px;
    font-size: 0.9rem;
  }
  
  .album-icon {
    font-size: 1.2rem;
  }
  
  .album-name {
    flex: 1;
  }
  
  .smart-badge {
    font-size: 0.8rem;
  }
  
  .properties {
    display: grid;
    gap: 0.5rem;
  }
  
  .property-item {
    display: flex;
    gap: 0.5rem;
    font-size: 0.85rem;
    padding: 0.5rem;
    background: #3d3d3d;
    border-radius: 4px;
  }
  
  .prop-key {
    color: #b0b0b0;
    font-weight: 500;
  }
  
  .prop-value {
    color: #e0e0e0;
  }
  
  .empty {
    color: #808080;
    font-style: italic;
    font-size: 0.9rem;
    margin: 0;
  }
  
  .status,
  .error {
    padding: 2rem;
    text-align: center;
  }
  
  .error {
    color: #ff6b6b;
  }
</style>
