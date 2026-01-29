<script>
  export let query = {};
  export let onQueryChange = null;
  
  let filters = [
    { field: 'name', operator: 'contains', value: '' }
  ];
  
  const fields = [
    { id: 'name', label: 'Name', type: 'text' },
    { id: 'extension', label: 'Extension', type: 'text' },
    { id: 'size', label: 'Size', type: 'number' },
    { id: 'rating', label: 'Rating', type: 'number' },
    { id: 'description', label: 'Description', type: 'text' }
  ];
  
  const operators = {
    text: [
      { id: 'contains', label: 'Contains' },
      { id: 'equals', label: 'Equals' },
      { id: 'starts_with', label: 'Starts with' },
      { id: 'ends_with', label: 'Ends with' }
    ],
    number: [
      { id: 'equals', label: 'Equals' },
      { id: 'gt', label: 'Greater than' },
      { id: 'lt', label: 'Less than' },
      { id: 'gte', label: 'Greater or equal' },
      { id: 'lte', label: 'Less or equal' }
    ]
  };
  
  function addFilter() {
    filters = [...filters, { field: 'name', operator: 'contains', value: '' }];
  }
  
  function removeFilter(index) {
    filters = filters.filter((_, i) => i !== index);
  }
  
  function getOperatorsForField(fieldId) {
    const field = fields.find(f => f.id === fieldId);
    return field ? operators[field.type] : operators.text;
  }
  
  function buildQuery() {
    const conditions = {};
    
    filters.forEach(filter => {
      if (!filter.value) return;
      
      const field = fields.find(f => f.id === filter.field);
      if (!field) return;
      
      switch (filter.operator) {
        case 'contains':
          conditions[filter.field] = { $regex: filter.value, $options: 'i' };
          break;
        case 'equals':
          conditions[filter.field] = filter.value;
          break;
        case 'starts_with':
          conditions[filter.field] = { $regex: `^${filter.value}`, $options: 'i' };
          break;
        case 'ends_with':
          conditions[filter.field] = { $regex: `${filter.value}$`, $options: 'i' };
          break;
        case 'gt':
          conditions[filter.field] = { $gt: Number(filter.value) };
          break;
        case 'lt':
          conditions[filter.field] = { $lt: Number(filter.value) };
          break;
        case 'gte':
          conditions[filter.field] = { $gte: Number(filter.value) };
          break;
        case 'lte':
          conditions[filter.field] = { $lte: Number(filter.value) };
          break;
      }
    });
    
    query = conditions;
    if (onQueryChange) {
      onQueryChange(conditions);
    }
  }
  
  function clearFilters() {
    filters = [{ field: 'name', operator: 'contains', value: '' }];
    query = {};
    if (onQueryChange) {
      onQueryChange({});
    }
  }
  
  $: {
    // Rebuild query when filters change
    buildQuery();
  }
</script>

<div class="query-builder">
  <div class="builder-header">
    <h4>🔍 Query Builder</h4>
    <div class="header-actions">
      <button on:click={clearFilters} class="btn-secondary btn-sm">Clear All</button>
      <button on:click={addFilter} class="btn-primary btn-sm">+ Add Filter</button>
    </div>
  </div>
  
  <div class="filters-list">
    {#each filters as filter, index}
      <div class="filter-row">
        <select bind:value={filter.field} class="filter-select">
          {#each fields as field}
            <option value={field.id}>{field.label}</option>
          {/each}
        </select>
        
        <select bind:value={filter.operator} class="filter-select">
          {#each getOperatorsForField(filter.field) as op}
            <option value={op.id}>{op.label}</option>
          {/each}
        </select>
        
        <input
          type="text"
          bind:value={filter.value}
          placeholder="Value"
          class="filter-input"
        />
        
        <button
          on:click={() => removeFilter(index)}
          class="remove-btn"
          title="Remove filter"
        >
          ✕
        </button>
      </div>
    {/each}
  </div>
  
  {#if Object.keys(query).length > 0}
    <div class="query-preview">
      <strong>Query:</strong>
      <pre>{JSON.stringify(query, null, 2)}</pre>
    </div>
  {/if}
</div>

<style>
  .query-builder {
    background: #2d2d2d;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  
  .builder-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #4d4d4d;
  }
  
  .builder-header h4 {
    margin: 0;
    font-size: 1rem;
  }
  
  .header-actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .btn-primary,
  .btn-secondary {
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.2s;
  }
  
  .btn-primary {
    background: #667eea;
    color: white;
  }
  
  .btn-primary:hover {
    background: #5568d3;
  }
  
  .btn-secondary {
    background: #4d4d4d;
    color: #e0e0e0;
  }
  
  .btn-secondary:hover {
    background: #5d5d5d;
  }
  
  .btn-sm {
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
  }
  
  .filters-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .filter-row {
    display: grid;
    grid-template-columns: 1fr 1fr 2fr auto;
    gap: 0.5rem;
    align-items: center;
  }
  
  .filter-select,
  .filter-input {
    background: #3d3d3d;
    border: 1px solid #4d4d4d;
    color: #e0e0e0;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.9rem;
  }
  
  .filter-select:focus,
  .filter-input:focus {
    outline: none;
    border-color: #667eea;
  }
  
  .remove-btn {
    background: #4d4d4d;
    border: none;
    color: #ff6b6b;
    padding: 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
  }
  
  .remove-btn:hover {
    background: #5d5d5d;
  }
  
  .query-preview {
    margin-top: 1rem;
    padding: 1rem;
    background: #3d3d3d;
    border-radius: 6px;
    font-size: 0.85rem;
  }
  
  .query-preview strong {
    color: #667eea;
  }
  
  .query-preview pre {
    margin: 0.5rem 0 0 0;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    overflow-x: auto;
  }
</style>
