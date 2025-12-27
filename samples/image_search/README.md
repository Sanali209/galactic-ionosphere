# Image Search Sample Application

DuckDuckGo image search with download capabilities - demonstrating the Foundation Template.

## Features

- Search images via DuckDuckGo
- **CardView gallery** with virtualization (handles 1000+ images)
- **Sort/Group/Filter** via toolbar
- Multi-selection with Ctrl/Shift+click
- Batch download
- Search history tracking

## Quick Start

```bash
# From foundation/samples/image_search directory
cd templates/foundation/samples/image_search

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Foundation Template Features Used

✅ **MainWindow** - Base window with menus, actions, status bar  
✅ **DockingService** - Panel layout management  
✅ **CardView** - Virtualized image gallery with grouping/sorting  
✅ **CardViewModel** - MVVM data binding  
✅ **ThumbnailService** - Async thumbnail loading + caching  
✅ **ServiceLocator** - Dependency injection  
✅ **ORM** - MongoDB for search history  
✅ **TaskSystem** - Async search & downloads  
✅ **ConfigManager** - App settings  
✅ **CommandBus** - Command execution  
✅ **Journal** - Activity logging  

## Requirements

- Python 3.11+
- MongoDB (local or remote)
- Internet connection for searches

## Project Structure

```
image_search/
├── main.py                  # Entry point (ApplicationBuilder)
├── config.json              # App configuration
├── src/
│   ├── core/
│   │   └── search_service.py   # DuckDuckGo search
│   ├── models/
│   │   ├── image_record.py     # Image ORM model
│   │   └── search_history.py   # Search history ORM
│   ├── tasks/
│   │   └── search_task.py      # Async search task
│   └── ui/
│       ├── main_window.py      # Foundation MainWindow
│       ├── panels/
│       │   ├── gallery_panel.py  # CardView gallery
│       │   └── search_panel.py   # Search input
│       └── viewmodels/
│           └── main_viewmodel.py
└── thumbnails/              # Cached thumbnails
```
