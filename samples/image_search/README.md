# Image Search Sample Application

DuckDuckGo image search with download capabilities - demonstrating the Foundation Template.

## Features

- Search images via DuckDuckGo
- Thumbnail gallery view
- Batch download
- Save to folder
- Search history tracking

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Foundation Template Features Used

✅ ServiceLocator - Dependency injection
✅ ORM - MongoDB for search history
✅ TaskSystem - Async search & downloads
✅ ConfigManager - App settings
✅ CommandBus - Command execution
✅ Journal - Activity logging
✅ GUI Framework - Panels, menus, settings

## Requirements

- Python 3.11+
- MongoDB (local or remote)
- Internet connection for searches

See [Implementation Plan](../../.gemini/antigravity/brain/.../implementation_plan.md) for details.
