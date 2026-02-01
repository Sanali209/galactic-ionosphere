/**
 * API Client for UExplorer Web
 * 
 * Centralized API calls to the FastAPI backend
 */

const API_BASE = '';  // Vite proxy handles /api routing

/**
 * Generic API call wrapper
 */
async function apiCall(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`API call failed: ${endpoint}`, error);
    throw error;
  }
}

// ============================================================================
// File System API
// ============================================================================

export const fileSystem = {
  getCurrentDirectory: () => apiCall('/api/directory/current'),
  changeDirectory: (path) => apiCall('/api/directory/change', {
    method: 'POST',
    body: JSON.stringify({ path })
  }),
  browse: (path = null, showHidden = false, includeMetadata = false) => {
    const params = new URLSearchParams();
    if (path) params.append('path', path);
    params.append('show_hidden', showHidden);
    params.append('include_metadata', includeMetadata);
    return apiCall(`/api/browse?${params}`);
  },
  search: (query, path = null, recursive = true, searchMode = 'filesystem') => {
    const params = new URLSearchParams({ query, recursive, search_mode: searchMode });
    if (path) params.append('path', path);
    return apiCall(`/api/search?${params}`);
  },
  getHome: () => apiCall('/api/home')
};

// ============================================================================
// Database API
// ============================================================================

export const database = {
  getStats: () => apiCall('/api/stats'),
  health: () => apiCall('/health')
};

// ============================================================================
// File Management API
// ============================================================================

export const files = {
  index: (path) => apiCall('/api/files/index', {
    method: 'POST',
    body: JSON.stringify({ path })
  }),
  get: (fileId) => apiCall(`/api/files/${fileId}`),
  updateRating: (fileId, rating) => apiCall(`/api/files/${fileId}/rating?rating=${rating}`, {
    method: 'PUT'
  })
};

// ============================================================================
// Tag API
// ============================================================================

export const tags = {
  list: (parentId = null) => {
    const params = parentId ? `?parent_id=${parentId}` : '';
    return apiCall(`/api/tags/${params}`);
  },
  getTree: () => apiCall('/api/tags/tree'),
  create: (tagData) => apiCall('/api/tags/', {
    method: 'POST',
    body: JSON.stringify(tagData)
  }),
  update: (tagId, tagData) => apiCall(`/api/tags/${tagId}`, {
    method: 'PUT',
    body: JSON.stringify(tagData)
  }),
  delete: (tagId, cascade = false) => apiCall(`/api/tags/${tagId}?cascade=${cascade}`, {
    method: 'DELETE'
  }),
  assign: (fileIds, tagId) => apiCall('/api/tags/assign', {
    method: 'POST',
    body: JSON.stringify({ file_ids: fileIds, tag_id: tagId })
  }),
  unassign: (fileIds, tagId) => apiCall('/api/tags/unassign', {
    method: 'POST',
    body: JSON.stringify({ file_ids: fileIds, tag_id: tagId })
  }),
  getForFile: (fileId) => apiCall(`/api/tags/file/${fileId}`)
};

// ============================================================================
// Album API
// ============================================================================

export const albums = {
  list: () => apiCall('/api/albums/'),
  get: (albumId) => apiCall(`/api/albums/${albumId}`),
  create: (albumData) => apiCall('/api/albums/', {
    method: 'POST',
    body: JSON.stringify(albumData)
  }),
  update: (albumId, albumData) => apiCall(`/api/albums/${albumId}`, {
    method: 'PUT',
    body: JSON.stringify(albumData)
  }),
  delete: (albumId) => apiCall(`/api/albums/${albumId}`, {
    method: 'DELETE'
  }),
  assign: (fileIds, albumId) => apiCall('/api/albums/assign', {
    method: 'POST',
    body: JSON.stringify({ file_ids: fileIds, album_id: albumId })
  }),
  unassign: (fileIds, albumId) => apiCall('/api/albums/unassign', {
    method: 'POST',
    body: JSON.stringify({ file_ids: fileIds, album_id: albumId })
  }),
  getFiles: (albumId, limit = 100, offset = 0) => apiCall(`/api/albums/${albumId}/files?limit=${limit}&offset=${offset}`),
  getForFile: (fileId) => apiCall(`/api/albums/file/${fileId}`)
};

// ============================================================================
// Relations API
// ============================================================================

export const relations = {
  list: (relationType = null, limit = 100) => {
    const params = new URLSearchParams({ limit });
    if (relationType) params.append('relation_type', relationType);
    return apiCall(`/api/relations/?${params}`);
  },
  create: (relationData) => apiCall('/api/relations/', {
    method: 'POST',
    body: JSON.stringify(relationData)
  }),
  delete: (relationId) => apiCall(`/api/relations/${relationId}`, {
    method: 'DELETE'
  }),
  getForFile: (fileId) => apiCall(`/api/relations/file/${fileId}`),
  markWrong: (relationId) => apiCall(`/api/relations/${relationId}/mark-wrong`, {
    method: 'PUT'
  })
};

// ============================================================================
// Utility Functions
// ============================================================================

export function formatFileSize(bytes) {
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

export function formatDate(isoString) {
  if (!isoString) return '-';
  const date = new Date(isoString);
  return date.toLocaleString();
}

export function formatRating(rating) {
  const filled = '★'.repeat(rating);
  const empty = '☆'.repeat(5 - rating);
  return filled + empty;
}

export default {
  fileSystem,
  database,
  files,
  tags,
  albums,
  relations,
  formatFileSize,
  formatDate,
  formatRating
};
