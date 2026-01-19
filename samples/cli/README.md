# UCoreFS Console Tools

This directory contains command-line tools that demonstrate using the UCoreFS framework **without any GUI dependencies**.

## Key Features

✅ **No PySide6 Required** - Works without Qt  
✅ **Framework-agnostic** - Uses `ApplicationBuilder.for_console()`  
✅ **Full Data Access** - Query files, tags, albums, search  
✅ **Lightweight** - Fast startup, minimal dependencies  

## Installation

```bash
# Install core dependencies only (no PySide6!)
pip install motor loguru pydantic

# From project root
cd samples/cli
```

## Tools

### File Scanner

Command-line tool for scanning directories and querying the database.

**Usage**:
```bash
# Scan a directory
python file_scanner.py scan /path/to/directory

# Search for files
python file_scanner.py search "vacation photos"

# List recent files
python file_scanner.py list --limit 20

# Show database statistics
python file_scanner.py stats
```

**Examples**:
```bash
# Scan your Pictures folder
python file_scanner.py scan ~/Pictures

# Find all PDFs
python file_scanner.py search "*.pdf"

# List 50 most recent files
python file_scanner.py list --limit 50
```

## Architecture

Console tools use the same core framework as the GUI application:

```python
from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.bundles import UCoreFSDataBundle

# Build console app (NO PySide6!)
locator = await (ApplicationBuilder.for_console("MyCLI", "config.json")
    .add_bundle(UCoreFSDataBundle())
    .build())

# Use services
from src.ucorefs.services.fs_service import FSService
fs = locator.get_system(FSService)
files = await fs.scan_directory("/path")
```

## Creating Your Own Console Tool

1. **Import framework** (no GUI imports!)
```python
from src.core.bootstrap import ApplicationBuilder
from src.ucorefs.bundles import UCoreFSDataBundle
```

2. **Build application**
```python
locator = await (ApplicationBuilder.for_console("MyTool")
    .add_bundle(UCoreFSDataBundle())
    .build())
```

3. **Use services**
```python
fs = locator.get_system(FSService)
tags = locator.get_system(TagManager)
search = locator.get_system(SearchService)
```

4. **Cleanup on exit**
```python
await locator.stop_all()
```

## Comparison: Console vs GUI

| Feature | Console | GUI |
|---------|---------|-----|
| **Dependencies** | motor, loguru | + PySide6, qasync |
| **Startup** | Fast (~1s) | Slower (~3s) |
| **Use Case** | Automation, scripts | Interactive UI |
| **Builder** | `.for_console()` | `.for_gui()` |
| **Bundles** | `UCoreFSDataBundle` | + `PySideBundle` |

## Development

### Adding a New Command

Edit `file_scanner.py`:

```python
# Add subparser
cmd_parser = subparsers.add_parser("mycommand", help="My command")
cmd_parser.add_argument("arg", help="Argument")

# Add handler
async def handle_mycommand(arg, locator):
    # Your code here
    pass

# Add to main()
elif args.command == "mycommand":
    await handle_mycommand(args.arg, locator)
```

### Error Handling

All commands have try/except to handle errors gracefully:

```python
try:
    await scan_directory(path, locator)
    return 0
except Exception as e:
    logger.error(f"Failed: {e}", exc_info=True)
    print(f"❌ Error: {e}")
    return 1
```

## Deployment

Console tools can be deployed to servers without GUI:

```bash
# On headless server
pip install motor loguru pydantic  # No PySide6!

# Run in background
nohup python file_scanner.py scan /data &

# Or in Docker
docker run -v /data:/data myapp python file_scanner.py scan /data
```

## Troubleshooting

**"No module named 'PySide6'"**  
✅ Correct - console tools don't need PySide6!

**"Database connection failed"**  
Check `config.json` has correct MongoDB URL

**"Module not found"**  
Run from project root or adjust `sys.path`

## License

Same as main UCoreFS project.
