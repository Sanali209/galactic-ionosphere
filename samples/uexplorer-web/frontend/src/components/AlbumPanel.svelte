<script>
  import { onMount } from 'svelte';
  import { albums } from '../lib/api.js';
  
  let albumList = [];
  let loading = false;
  let error = null;
  let showCreateForm = false;
  let newAlbum = { name: '', description: '', icon: '📚', is_smart: false };
  let selectedAlbum = null;
  let albumFiles = null;
  
  async function loadAlbums() {
    loading = true;
    error = null;
    try {
      albumList = await albums.list();
    } catch (err) {
      error = err.message;
      console.error('Failed to load albums:', err);
    } finally {
      loading = false;
    }
  }
  
  async function createAlbum() {
    if (!newAlbum.name.trim()) {
      alert('Album name is required');
      return;
    }
    
    try {
      await albums.create(newAlbum);
      newAlbum = { name: '', description: '', icon: '📚', is_smart: false };
      showCreateForm = false;
      await loadAlbums();
    } catch (err) {
      alert(`Failed to create album: ${err.message}`);
    }
  }
  
  async function deleteAlbum(albumId, albumName) {
    if (!confirm(`Delete album "${albumName}"?`)) return;
    
    try {
      await albums.delete(albumId);
      if (selectedAlbum?.id === albumId) {
        selectedAlbum = null;
        albumFiles = null;
      }
      await loadAlbums();
    } catch (err) {
      alert(`Failed to delete album: ${err.message}`);
    }
  }
  
  async function selectAlbum(album) {
    selectedAlbum = album;
    try {
      albumFiles = await albums.getFiles(album.id, 50, 0);
    } catch (err) {
      console.error('Failed to load album files:', err);
      albumFiles = null;
    }
  }
  
  onMount(loadAlbums);
</script>

<div class="album-panel">
  <div class="panel-header">
    <h3>📚 Albums ({albumList.length})</h3>
    <div class="header-actions">
      <button on:click={loadAlbums} class="icon-btn" title="Refresh">🔄</button>
      <button on:click={() => showCreateForm = !showCreateForm} class="icon-btn" title="Create Album">➕</button>
    </div>
  </div>
  
  {#if showCreateForm}
    <div class="create-form">
      <h4>Create New Album</h4>
      <input
        type="text"
        bind:value={newAlbum.name}
        placeholder="Album name"
        class="form-input"
      />
      <input
        type="text"
        bind:value={newAlbum.description}
        placeholder="Description (optional)"
        class="form-input"
      />
      <input
        type="text"
        bind:value={newAlbum.icon}
        placeholder="Icon (emoji)"
        class="form-input"
        maxlength="2"
      />
      <label class="checkbox-label">
        <input type="checkbox" bind:checked={newAlbum.is_smart} />
        Smart Album (query-based)
      </label>
      <div class="form-actions">
        <button on:click={createAlbum} class="btn-primary">Create</button>
        <button on:click={() => showCreateForm = false} class="btn-secondary">Cancel</button>
      </div>
    </div>
  {/if}
  
  {#if loading}
    <div class="status">Loading albums...</div>
  {:else if error}
    <div class="error">❌ {error}</div>
  {:else if albumList.length === 0}
    <div class="empty-state">
      <p>No albums yet</p>
      <button on:click={() => showCreateForm = true} class="btn-primary">Create First Album</button>
    </div>
  {:else}
    <div class="album-list">
      {#each albumList as album}
        <div
          class="album-item"
          class:selected={selectedAlbum?.id === album.id}
          on:click={() => selectAlbum(album)}
        >
          <span class="album-icon">{album.icon || '📚'}</span>
          <div class="album-info">
            <div class="album-name">
              {album.name}
              {#if album.is_smart}
                <span class="smart-badge">🔮</span>
              {/if}
            </div>
            {#if album.description}
              <div class="album-description">{album.description}</div>
            {/if}
          </div>
          <span class="album-count">{album.file_count}</span>
          <button
            on:click|stopPropagation={() => deleteAlbum(album.id, album.name)}
            class="delete-btn"
            title="Delete album"
          >
            🗑️
          </button>
        </div>
      {/each}
    </div>
  {/if}
  
  {#if selectedAlbum}
    <div class="album-details">
      <h4>{selectedAlbum.icon} {selectedAlbum.name}</h4>
      {#if selectedAlbum.description}
        <p class="description">{selectedAlbum.description}</p>
      {/if}
      <div class="detail-stats">
        <span>Type: {selectedAlbum.is_smart ? 'Smart' : 'Static'}</span>
        <span>Files: {selectedAlbum.file_count}</span>
      </div>
      
      {#if albumFiles}
        <div class="album-files">
          <h5>Files ({albumFiles.count})</h5>
          {#if albumFiles.files.length === 0}
            <p class="no-files">No files in this album</p>
          {:else}
            <div class="file-list">
              {#each albumFiles.files as file}
                <div class="file-item">
                  <span class="file-icon">📄</span>
                  <span class="file-name">{file.name}</span>
                  {#if file.rating > 0}
                    <span class="file-rating">{'★'.repeat(file.rating)}</span>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .album-panel {
    background: #2d2d2d;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    max-height: 600px;
    display: flex;
    flex-direction: column;
  }
  
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #4d4d4d;
  }
  
  .panel-header h3 {
    margin: 0;
    font-size: 1.2rem;
  }
  
  .header-actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .icon-btn {
    background: #3d3d3d;
    border: none;
    padding: 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
  }
  
  .icon-btn:hover {
    background: #4d4d4d;
  }
  
  .create-form {
    background: #3d3d3d;
    padding: 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
  }
  
  .create-form h4 {
    margin: 0 0 1rem 0;
  }
  
  .form-input {
    width: 100%;
    background: #2d2d2d;
    border: 1px solid #4d4d4d;
    color: #e0e0e0;
    padding: 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }
  
  .form-input:focus {
    outline: none;
    border-color: #667eea;
  }
  
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    cursor: pointer;
  }
  
  .form-actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .btn-primary {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
  }
  
  .btn-primary:hover {
    background: #5568d3;
  }
  
  .btn-secondary {
    background: #4d4d4d;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
  }
  
  .btn-secondary:hover {
    background: #5d5d5d;
  }
  
  .album-list {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 1rem;
  }
  
  .album-item {
    display: flex;
    align-items: center;
    padding: 0.75rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
    gap: 0.75rem;
  }
  
  .album-item:hover {
    background: #3d3d3d;
  }
  
  .album-item.selected {
    background: #4d4d4d;
  }
  
  .album-icon {
    font-size: 1.5rem;
  }
  
  .album-info {
    flex: 1;
  }
  
  .album-name {
    color: #e0e0e0;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .smart-badge {
    font-size: 0.8rem;
  }
  
  .album-description {
    color: #b0b0b0;
    font-size: 0.85rem;
    margin-top: 0.25rem;
  }
  
  .album-count {
    background: #3d3d3d;
    color: #667eea;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: bold;
  }
  
  .delete-btn {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1rem;
    opacity: 0.6;
    padding: 0.25rem;
    transition: opacity 0.2s;
  }
  
  .delete-btn:hover {
    opacity: 1;
  }
  
  .album-details {
    background: #3d3d3d;
    padding: 1rem;
    border-radius: 6px;
    max-height: 300px;
    overflow-y: auto;
  }
  
  .album-details h4 {
    margin: 0 0 0.5rem 0;
  }
  
  .description {
    color: #b0b0b0;
    font-size: 0.9rem;
    margin: 0 0 1rem 0;
  }
  
  .detail-stats {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #b0b0b0;
  }
  
  .album-files h5 {
    margin: 1rem 0 0.5rem 0;
    font-size: 1rem;
  }
  
  .file-list {
    max-height: 150px;
    overflow-y: auto;
  }
  
  .file-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
  }
  
  .file-item:hover {
    background: #2d2d2d;
  }
  
  .file-icon {
    font-size: 1rem;
  }
  
  .file-name {
    flex: 1;
    color: #e0e0e0;
  }
  
  .file-rating {
    color: #ffd700;
    font-size: 0.8rem;
  }
  
  .no-files {
    color: #808080;
    font-style: italic;
    text-align: center;
    padding: 1rem;
  }
  
  .status,
  .error,
  .empty-state {
    padding: 2rem;
    text-align: center;
  }
  
  .error {
    color: #ff6b6b;
  }
  
  .empty-state p {
    margin-bottom: 1rem;
    color: #b0b0b0;
  }
</style>
