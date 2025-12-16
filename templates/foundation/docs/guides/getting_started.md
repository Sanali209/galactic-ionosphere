# Getting Started

## Prerequisites
- Python 3.9+
- MongoDB (running locally or accessible)

## Installation

1. **Clone the Template**
   Copy the `templates/foundation` folder to your new project location.

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure MongoDB**
   By default, the app looks for `mongodb://localhost:27017`.
   Edit `config.json` (auto-created on first run) to change settings.

## Running the App

```bash
python main.py
```

## Project Layout

- `src/main.py`: Bootstraps the application.
- `src/core/`: Contains all business logic.
- `src/ui/`: Contains PySide6 widgets.
- `data/`: Default location for local files.
- `logs/`: Application logs.
