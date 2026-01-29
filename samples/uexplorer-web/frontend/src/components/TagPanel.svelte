<script>
  import { onMount } from 'svelte';
  import { tags } from '../lib/api.js';
  
  let tagTree = { tags: [], total: 0 };
  let loading = false;
  let error = null;
  let showCreateForm = false;
  let newTag = { name: '', description: '', color: '#667eea', parent_id: null };
  let selectedTag = null;
  
  async function loadTags() {
    loading = true;
    error = null;
    try {
      tagTree = await tags.getTree();
    } catch (err) {
      error = err.message;
      console.error('Failed to load tags:', err);
    } finally {
      loading = false;
    }
  }
  
  async function createTag() {
    if (!newTag.name.trim()) {
      alert('Tag name is required');
      return;
    }
    
    try {
      await tags.create(newTag);
      newTag = { name: '', description: '', color: '#667eea', parent_id: null };
      showCreateForm = false;
      await loadTags();
    } catch (err) {
      alert(`Failed to create tag: ${err.message}`);
    }
  }
  
  async function deleteTag(tagId, tagName) {
    if (!confirm(`Delete tag "${tagName}"?`)) return;
    
    try {
      await tags.delete(tagId, false);
      await loadTags();
    } catch (err) {
      if (err.message.includes('children')) {
        if (confirm(`Tag "${tagName}" has children. Delete all children too?`)) {
          await tags.delete(tagId, true);
          await loadTags();
        }
      } else {
        alert(`Failed to delete tag: ${err.message}`);
      }
    }
  }
  
  function selectTag(tag) {
    selectedTag = tag;
  }
  
  function renderTag(tag, level = 0) {
    return { tag, level };
  }
  
  function flattenTags(tags, level = 0) {
    let result = [];
    for (const tag of tags) {
      result.push({ ...tag, level });
      if (tag.children && tag.children.length > 0) {
        result.push(...flattenTags(tag.children, level + 1));
      }
    }
    return result;
  }
  
  $: flatTags = flattenTags(tagTree.tags || []);
  
  onMount(loadTags);
</script>

<div class="tag-panel">
  <div class="panel-header">
    <h3>🏷️ Tags ({tagTree.total})</h3>
    <div class="header-actions">
      <button on:click={loadTags} class="icon-btn" title="Refresh">🔄</button>
      <button on:click={() => showCreateForm = !showCreateForm} class="icon-btn" title="Create Tag">➕</button>
    </div>
  </div>
  
  {#if showCreateForm}
    <div class="create-form">
      <h4>Create New Tag</h4>
      <input
        type="text"
        bind:value={newTag.name}
        placeholder="Tag name"
        class="form-input"
      />
      <input
        type="text"
        bind:value={newTag.description}
        placeholder="Description (optional)"
        class="form-input"
      />
      <div class="color-picker">
        <label>Color:</label>
        <input type="color" bind:value={newTag.color} />
      </div>
      <div class="form-actions">
        <button on:click={createTag} class="btn-primary">Create</button>
        <button on:click={() => showCreateForm = false} class="btn-secondary">Cancel</button>
      </div>
    </div>
  {/if}
  
  {#if loading}
    <div class="status">Loading tags...</div>
  {:else if error}
    <div class="error">❌ {error}</div>
  {:else if flatTags.length === 0}
    <div class="empty-state">
      <p>No tags yet</p>
      <button on:click={() => showCreateForm = true} class="btn-primary">Create First Tag</button>
    </div>
  {:else}
    <div class="tag-list">
      {#each flatTags as tag}
        <div
          class="tag-item"
          class:selected={selectedTag?.id === tag.id}
          style="padding-left: {tag.level * 20 + 10}px"
          on:click={() => selectTag(tag)}
        >
          <span class="tag-icon" style="color: {tag.color || '#667eea'}">
            {tag.children && tag.children.length > 0 ? '📁' : '🏷️'}
          </span>
          <span class="tag-name">{tag.name}</span>
          <span class="tag-count">({tag.file_count})</span>
          <button
            on:click|stopPropagation={() => deleteTag(tag.id, tag.name)}
            class="delete-btn"
            title="Delete tag"
          >
            🗑️
          </button>
        </div>
      {/each}
    </div>
  {/if}
  
  {#if selectedTag}
    <div class="tag-details">
      <h4>Tag Details</h4>
      <div class="detail-row">
        <strong>Name:</strong> {selectedTag.name}
      </div>
      {#if selectedTag.description}
        <div class="detail-row">
          <strong>Description:</strong> {selectedTag.description}
        </div>
      {/if}
      <div class="detail-row">
        <strong>Files:</strong> {selectedTag.file_count}
      </div>
      <div class="detail-row">
        <strong>Level:</strong> {selectedTag.level}
      </div>
    </div>
  {/if}
</div>

<style>
  .tag-panel {
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
  
  .color-picker {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  
  .color-picker label {
    color: #b0b0b0;
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
  
  .tag-list {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 1rem;
  }
  
  .tag-item {
    display: flex;
    align-items: center;
    padding: 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
    gap: 0.5rem;
  }
  
  .tag-item:hover {
    background: #3d3d3d;
  }
  
  .tag-item.selected {
    background: #4d4d4d;
  }
  
  .tag-icon {
    font-size: 1.2rem;
  }
  
  .tag-name {
    flex: 1;
    color: #e0e0e0;
  }
  
  .tag-count {
    color: #b0b0b0;
    font-size: 0.85rem;
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
  
  .tag-details {
    background: #3d3d3d;
    padding: 1rem;
    border-radius: 6px;
  }
  
  .tag-details h4 {
    margin: 0 0 0.75rem 0;
  }
  
  .detail-row {
    padding: 0.5rem 0;
    border-bottom: 1px solid #4d4d4d;
    font-size: 0.9rem;
  }
  
  .detail-row:last-child {
    border-bottom: none;
  }
  
  .detail-row strong {
    color: #b0b0b0;
    margin-right: 0.5rem;
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
