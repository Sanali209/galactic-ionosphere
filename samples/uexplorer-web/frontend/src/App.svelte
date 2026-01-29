<script>
  import DirectoryBrowser from './components/DirectoryBrowser.svelte';
  import SearchBar from './components/SearchBar.svelte';
  
  let currentPath = '';
  let searchQuery = '';
  
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
    <p class="subtitle">Modern File Manager - FastAPI + Svelte</p>
  </header>
  
  <div class="container">
    <SearchBar bind:query={searchQuery} currentPath={currentPath} />
    <DirectoryBrowser bind:currentPath={currentPath} searchQuery={searchQuery} />
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
  
  .container {
    flex: 1;
    max-width: 1400px;
    width: 100%;
    margin: 0 auto;
    padding: 2rem;
  }
</style>
