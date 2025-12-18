# GUI Framework - User Guide

## Overview

The Foundation Template includes a professional GUI framework with document management, docking panels, settings, and command palette.

## Quick Start

Launch the application:
```bash
python main.py
```

---

## Core Features

### 1. Menu System

**File Menu**
- **New** (Ctrl+N) - Create new document
- **Open** (Ctrl+O) - Open file
- **Save** (Ctrl+S) - Save current file
- **Close** (Ctrl+W) - Close current file
- **Exit** (Alt+F4) - Exit application

**View Menu**
- **Panels** → Toggle panel visibility
- **Split Horizontal** (Ctrl+\\) - Split view side-by-side
- **Split Vertical** (Ctrl+Shift+\\) - Split view top-down
- **Reset Layout** - Restore default layout

**Tools Menu**
- **Settings** (Ctrl+,) - Open settings dialog
- **Command Palette** (Ctrl+Shift+P) - Search all commands

### 2. Docking Panels

**Output Panel**
- Displays application logs and messages
- Toggle: View → Panels → Output
- Resizable and dockable (left/right/bottom/floating)
- Position saved on close

**Panel Features:**
- Drag panel title to move
- Resize using splitter handles
- State persists across sessions

### 3. Settings Dialog

**Open:** Ctrl+, or Tools → Settings

**Features:**
- **Category Tree** - Organized settings (General/Editor/Appearance)
- **Search** - Filter settings by keyword
- **Live Updates** - Changes apply immediately
- **Reset** - Restore category defaults

**Available Settings:**
- Application Name
- Database Name
- (More categories coming soon)

### 4. Command Palette

**Open:** Ctrl+Shift+P or Tools → Command Palette

**Usage:**
1. Press Ctrl+Shift+P
2. Type to search (fuzzy matching)
3. Use ↑↓ to navigate
4. Press Enter to execute
5. Esc to cancel

**Shows:** All commands with their keyboard shortcuts

### 5. Document Splits (Future)

**Planned Features:**
- Side-by-side document comparison
- Multiple document layouts
- Nested splits (unlimited depth)
- Per-split tab management

---

## Keyboard Shortcuts

### File Operations
- `Ctrl+N` - New
- `Ctrl+O` - Open
- `Ctrl+S` - Save
- `Ctrl+Shift+S` - Save As
- `Ctrl+W` - Close

### Edit Operations
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo
- `Ctrl+X` - Cut
- `Ctrl+C` - Copy
- `Ctrl+V` - Paste
- `Ctrl+F` - Find
- `Ctrl+H` - Replace

### View Operations
- `Ctrl+\\` - Split Horizontal
- `Ctrl+Shift+\\` - Split Vertical

### Tools
- `Ctrl+,` - Settings
- `Ctrl+Shift+P` - Command Palette

### Window Navigation
- `Ctrl+Tab` - Next Document
- `Ctrl+Shift+Tab` - Previous Document

### Help
- `F1` - Documentation

---

## Tips & Tricks

### Layout Management
1. Arrange panels to your preference
2. Close application - layout saved automatically
3. Reopen - panels restore to saved positions
4. Use View → Reset Layout to restore defaults

### Efficient Navigation
- Use Command Palette (Ctrl+Shift+P) to find any action
- Most actions have keyboard shortcuts
- Right-click for context menus

### Customization
- Settings dialog provides live configuration
- Changes save automatically
- Search settings to find options quickly

---

## Troubleshooting

**Panel disappeared?**
- Check View → Panels menu
- Click panel name to show/hide

**Keyboard shortcut not working?**
- Check for conflicts with OS shortcuts
- View shortcuts in Command Palette

**Settings not saving?**
- Check file permissions
- Verify config.json is writable

---

## Next Steps

- Explore all menus
- Try keyboard shortcuts
- Customize settings
- Arrange panels to your workflow
