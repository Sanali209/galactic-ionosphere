<script>
  import { onMount } from 'svelte';
  import { database } from '../lib/api.js';
  
  let stats = null;
  let health = null;
  let loading = true;
  let error = null;
  
  async function loadStats() {
    loading = true;
    error = null;
    try {
      [stats, health] = await Promise.all([
        database.getStats(),
        database.health()
      ]);
    } catch (err) {
      error = err.message;
      console.error('Failed to load stats:', err);
    } finally {
      loading = false;
    }
  }
  
  onMount(loadStats);
</script>

<div class="dashboard">
  <div class="header">
    <h2>📊 System Dashboard</h2>
    <button on:click={loadStats} class="refresh-btn" title="Refresh">🔄</button>
  </div>
  
  {#if loading}
    <div class="status">Loading statistics...</div>
  {:else if error}
    <div class="error">❌ {error}</div>
  {:else}
    <div class="stats-grid">
      <!-- Database Stats -->
      <div class="stat-card">
        <div class="stat-icon">📄</div>
        <div class="stat-value">{stats?.files || 0}</div>
        <div class="stat-label">Files</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">📁</div>
        <div class="stat-value">{stats?.directories || 0}</div>
        <div class="stat-label">Directories</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">🏷️</div>
        <div class="stat-value">{stats?.tags || 0}</div>
        <div class="stat-label">Tags</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">📚</div>
        <div class="stat-value">{stats?.albums || 0}</div>
        <div class="stat-label">Albums</div>
      </div>
    </div>
    
    <!-- Processing State Breakdown -->
    {#if stats?.processing_states}
      <div class="processing-states">
        <h3>Processing States</h3>
        <div class="states-list">
          {#each Object.entries(stats.processing_states) as [state, count]}
            <div class="state-item">
              <span class="state-name">{state}</span>
              <span class="state-count">{count}</span>
            </div>
          {/each}
        </div>
      </div>
    {/if}
    
    <!-- Health Status -->
    {#if health}
      <div class="health-status">
        <h3>System Health</h3>
        <div class="health-item">
          <span class="health-label">Status:</span>
          <span class="health-value status-{health.status}">{health.status}</span>
        </div>
        <div class="health-item">
          <span class="health-label">Database:</span>
          <span class="health-value">{health.database}</span>
        </div>
        <div class="health-item">
          <span class="health-label">Updated:</span>
          <span class="health-value">{new Date(health.timestamp).toLocaleTimeString()}</span>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .dashboard {
    background: #2d2d2d;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }
  
  .header h2 {
    margin: 0;
    font-size: 1.5rem;
  }
  
  .refresh-btn {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
  }
  
  .refresh-btn:hover {
    background: #5568d3;
  }
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  
  .stat-card {
    background: #3d3d3d;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s;
  }
  
  .stat-card:hover {
    transform: translateY(-2px);
  }
  
  .stat-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
  }
  
  .stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: #667eea;
    margin-bottom: 0.25rem;
  }
  
  .stat-label {
    color: #b0b0b0;
    font-size: 0.9rem;
  }
  
  .processing-states,
  .health-status {
    background: #3d3d3d;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  
  .processing-states h3,
  .health-status h3 {
    margin: 0 0 1rem 0;
    font-size: 1.1rem;
  }
  
  .states-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.5rem;
  }
  
  .state-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem;
    background: #2d2d2d;
    border-radius: 4px;
  }
  
  .state-name {
    color: #e0e0e0;
  }
  
  .state-count {
    color: #667eea;
    font-weight: bold;
  }
  
  .health-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #4d4d4d;
  }
  
  .health-item:last-child {
    border-bottom: none;
  }
  
  .health-label {
    color: #b0b0b0;
  }
  
  .health-value {
    color: #e0e0e0;
  }
  
  .health-value.status-healthy {
    color: #4caf50;
    font-weight: bold;
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
