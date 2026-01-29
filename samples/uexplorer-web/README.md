# UExplorer Web - FastAPI + Svelte File Manager

A modern web-based file manager inspired by the UExplorer desktop application, built with FastAPI backend and Svelte frontend.

## Features

- **Directory Browsing**: Navigate through local file system
- **Directory Selection**: Choose any directory on the local machine
- **File Operations**: List files, view details, search
- **Modern UI**: Responsive Svelte components
- **RESTful API**: FastAPI backend with automatic documentation

## Architecture

```
uexplorer-web/
├── backend/           # FastAPI backend
│   ├── main.py       # API server entry point
│   ├── api/          # API endpoints
│   └── requirements.txt
└── frontend/         # Svelte frontend
    ├── src/
    │   ├── components/  # Custom Svelte components
    │   └── App.svelte
    └── package.json
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm or pnpm

## Setup

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

- `GET /api/browse` - List files in a directory
- `GET /api/directory/select` - Get current directory
- `POST /api/directory/change` - Change current directory
- `GET /api/file/{path}` - Get file details
- `GET /api/search` - Search files

## Security Note

This application allows browsing the local file system. In production, you should:
- Implement proper authentication
- Restrict accessible directories
- Add permission checks
- Use HTTPS

## Development

The application is designed for local development and demonstrates:
- FastAPI async endpoints
- Custom Svelte components
- Directory browsing with local machine behavior
- Clean separation of backend/frontend

## License

MIT
