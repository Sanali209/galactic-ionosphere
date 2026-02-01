<script>
  import DirectoryBrowser from './components/DirectoryBrowser.svelte';
  import SearchBar from './components/SearchBar.svelte';
  import StatsDashboard from './components/StatsDashboard.svelte';
  import TagPanel from './components/TagPanel.svelte';
  import AlbumPanel from './components/AlbumPanel.svelte';
  
  let currentPath = '';
  let searchQuery = '';
  let activeView = 'browser'; // browser, tags, albums, stats
  
  async function loadCurrentDirectory() {
    try {
      const response = await fetch('/api/directory/current');
      const data = await response.json();
      currentPath = data.path;
    } catch (error) {
      console.error('Failed to load current directory:', error);
    }
  }
  
  loadCurrentDirectory();
</script>

<main>
  <header>
    <h1>🗂️ UExplorer Web</h1>
    <p class="subtitle">Comprehensive File Manager - FastAPI + Svelte</p>
  </header>
  
  <nav class="view-tabs">
    <button
      class="tab"
      class:active={activeView === 'browser'}
      on:click={() => activeView = 'browser'}
    >
      📁 Browser
    </button>
    <button
      class="tab"
      class:active={activeView === 'tags'}
      on:click={() => activeView = 'tags'}
    >
      🏷️ Tags
    </button>
    <button
      class="tab"
      class:active={activeView === 'albums'}
      on:click={() => activeView = 'albums'}
    >
      📚 Albums
    </button>
    <button
      class="tab"
      class:active={activeView === 'stats'}
      on:click={() => activeView = 'stats'}
    >
      📊 Statistics
    </button>
  </nav>
  
  <div class="container">
    {#if activeView === 'browser'}
      <SearchBar bind:query={searchQuery} currentPath={currentPath} />
      <DirectoryBrowser bind:currentPath={currentPath} searchQuery={searchQuery} />
    {:else if activeView === 'tags'}
      <TagPanel />
    {:else if activeView === 'albums'}
      <AlbumPanel />
    {:else if activeView === 'stats'}
      <StatsDashboard />
    {/if}
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: #1e1e1e;
    color: #e0e0e0;
  }
  
  main {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  
  header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }
  
  header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 600;
  }
  
  .subtitle {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
    font-size: 1rem;
  }
  
  .view-tabs {
    display: flex;
    background: #2d2d2d;
    padding: 0.5rem;
    gap: 0.5rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  }
  
  .tab {
    background: #3d3d3d;
    color: #e0e0e0;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
  }
  
  .tab:hover {
    background: #4d4d4d;
  }
  
  .tab.active {
    background: #667eea;
    color: white;
    font-weight: 600;
  }
  
  .container {
    flex: 1;
    max-width: 1400px;
    width: 100%;
    margin: 0 auto;
    padding: 2rem;
  }
</style>
