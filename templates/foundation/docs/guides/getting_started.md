# Getting Started

## Prerequisites
- Python 3.10+
- MongoDB (running locally or accessible)
- pip (Python package manager)

## Installation

### Method 1: Using Foundation as a Package (Recommended)

1. **Install Foundation in Development Mode**
   ```bash
   cd templates/foundation
   pip install -e .
   ```

2. **Create Your Application**
   ```python
   # my_app/main.py
   from foundation import ApplicationBuilder, run_app
   from src.ui.main_window import MainWindow
   from src.ui.viewmodels.main_viewmodel import MainViewModel
   
   if __name__ == "__main__":
       builder = (ApplicationBuilder("My App", "config.json")
                  .with_default_systems()
                  .with_logging(True))
       
       run_app(MainWindow, MainViewModel, builder=builder)
   ```

### Method 2: Copy Template (Legacy)

1. **Clone the Template**
   Copy the `templates/foundation` folder to your new project location.

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### MongoDB Connection
Edit `config.json` (auto-created on first run):

```json
{
  "mongo": {
    "host": "localhost",
    "port": 27017,
    "database_name": "my_app_db"
  },
  "general": {
    "debug_mode": true,
    "theme": "dark"
  }
}
```

## Running the App

```bash
python main.py
```

The application will:
1. Setup logging
2. Initialize ServiceLocator
3. Register and start all systems (Database, CommandBus, TaskSystem, etc.)
4. Create and show the main window
5. Run the Qt event loop

## Project Layout

```
your_app/
├── main.py              # Bootstrap with ApplicationBuilder
├── config.json          # Auto-generated configuration
├── src/
│   ├── models/          # ORM models (CollectionRecord)
│   ├── core/            # Custom systems (BaseSystem)
│   └── ui/              # PySide6 widgets & panels
│       ├── main_window.py
│       ├── panels/      # Custom dock panels
│       └── viewmodels/  # MVVM view models
├── data/                # Default location for local files
└── logs/                # Application logs (auto-created)
```

## Quick Example: Creating a Model

```python
from foundation import CollectionRecord, Field

class User(CollectionRecord):
    # Auto collection name: "users"
    username: str = Field(default="", index=True, unique=True)
    email: str = Field(default="")
    role: str = Field(default="user")

# Use it
user = User(username="alice", email="alice@example.com")
await user.save()

# Query
users = await User.find({"role": "user"})
```

## Next Steps

- Read the [MVVM Pattern Guide](./mvvm_pattern.md)
- Learn about [ORM Features](../components/orm.md)
- See [Migration Guide](../../MIGRATION.md) if updating from old pattern
- Check the [GUI User Guide](./gui_user_guide.md) for keyboard shortcuts

