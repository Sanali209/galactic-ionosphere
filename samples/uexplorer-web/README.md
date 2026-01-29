# UExplorer Web - FastAPI + Svelte File Manager

A modern web-based file manager inspired by the UExplorer desktop application, built with FastAPI backend and Svelte frontend. This sample demonstrates porting the key features of the PySide6-based UExplorer to a web-based architecture.

## Features

- **Directory Browsing**: Navigate through local file system with a clean, modern interface
- **Directory Selection**: Choose any directory on the local machine to browse
- **File Operations**: List files with details (size, modified date, type)
- **Search Functionality**: Real-time search across files and directories
- **Custom Svelte Components**: Reusable UI components for directory browsing and search
- **Modern UI**: Dark-themed responsive interface with smooth interactions
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation

## Architecture

```
uexplorer-web/
├── backend/           # FastAPI backend
│   ├── main.py       # API server entry point
│   ├── api/          # API endpoints
│   └── requirements.txt
├── frontend/         # Svelte frontend
│   ├── src/
│   │   ├── components/  # Custom Svelte components
│   │   │   ├── DirectoryBrowser.svelte
│   │   │   └── SearchBar.svelte
│   │   ├── App.svelte
│   │   └── main.js
│   └── package.json
└── start.sh         # Convenience script to start both servers
```

## Comparison with Desktop UExplorer

This web version implements the core features of the desktop UExplorer:

| Feature | Desktop (PySide6) | Web (FastAPI + Svelte) |
|---------|-------------------|------------------------|
| File Browsing | ✅ Dual-pane | ✅ Single pane with navigation |
| Directory Selection | ✅ Native dialog | ✅ Path input + navigation |
| Search | ✅ Full-text + filters | ✅ Name-based recursive search |
| File Details | ✅ Properties panel | ✅ Size, date, type display |
| Tags/Albums | ✅ Full support | ⏳ Future enhancement |
| Database | ✅ MongoDB | ⏳ Can be added |

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm or pnpm

## Quick Start

Use the provided start script to run both servers:

```bash
cd uexplorer-web
./start.sh
```

This will:
1. Install backend dependencies
2. Install frontend dependencies
3. Start the FastAPI backend on `http://localhost:8000`
4. Start the Svelte frontend on `http://localhost:5173`

## Manual Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`  
API documentation: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

- `GET /` - API info
- `GET /api/directory/current` - Get current directory
- `POST /api/directory/change` - Change current directory
- `GET /api/browse` - List files in a directory
  - Query params: `path` (optional), `show_hidden` (boolean)
- `GET /api/file/{path}` - Get file details
- `GET /api/search` - Search files
  - Query params: `query` (required), `path` (optional), `recursive` (boolean)
- `GET /api/home` - Get user's home directory

## Custom Svelte Components

### DirectoryBrowser Component

The main file browsing interface with features:
- Grid display with file icons, names, sizes, and dates
- Navigate up/down the directory tree
- Go to home directory
- Custom path navigation
- Show/hide hidden files
- Responsive design

### SearchBar Component

Search interface with:
- Real-time search as you type
- Displays results with file paths
- File size display for files
- Clear search functionality
- Empty state handling

## Security Note

⚠️ This application allows browsing the local file system. In production, you should:
- Implement proper authentication and authorization
- Restrict accessible directories to specific paths
- Add permission checks for sensitive operations
- Use HTTPS for all connections
- Add rate limiting to prevent abuse
- Implement input validation and sanitization
- Add audit logging

## Development

The application demonstrates:
- FastAPI async endpoints with proper error handling
- Pydantic models for request/response validation
- Custom Svelte components with reactive state
- Clean separation of backend/frontend concerns
- CORS configuration for local development
- Directory browsing with local machine behavior

### Backend Structure

- `main.py` - FastAPI application with all endpoints
- Pydantic models for type safety
- Async/await for I/O operations
- Automatic OpenAPI documentation

### Frontend Structure

- Vite for fast development and building
- Svelte 4 for reactive UI
- Component-based architecture
- CSS-in-JS styling
- Proxy configuration for API calls

## Testing

Test the backend API:
```bash
cd backend
# Start the server
python main.py

# In another terminal, test endpoints
curl http://localhost:8000/
curl http://localhost:8000/api/directory/current
curl http://localhost:8000/api/browse
curl "http://localhost:8000/api/search?query=test"
```

Test the frontend:
```bash
cd frontend
npm run dev
# Open http://localhost:5173 in your browser
```

## Building for Production

Build the frontend:
```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/` and can be served by any static file server or integrated with the FastAPI backend.

## Future Enhancements

Potential additions to match more desktop UExplorer features:
- [ ] File tagging system with backend database
- [ ] Album/collection management
- [ ] File preview (images, text, etc.)
- [ ] Dual-pane browsing mode
- [ ] File operations (copy, move, delete)
- [ ] Drag and drop support
- [ ] Thumbnail generation
- [ ] Advanced filtering options
- [ ] Keyboard shortcuts
- [ ] Breadcrumb navigation
- [ ] File upload capability

## License

MIT
