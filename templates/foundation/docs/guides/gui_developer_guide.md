# GUI Framework - Developer Guide

## Architecture Overview

The GUI framework is built on 4 main subsystems:

1. **Documents** - Split management and document views
2. **Docking** - Panel lifecycle and state
3. **Menus** - Action registry and menu building
4. **Settings** - Configuration UI
5. **Commands** - Command palette

---

## Extending the Framework

### Creating Custom Panels

**1. Create Panel Class**

```python
from src.ui.docking.panel_base import BasePanelWidget
from PySide6.QtWidgets import QVBoxLayout, QLabel

class MyPanel(BasePanelWidget):
    def __init__(self, title: str, locator, parent=None):
        super().__init__(title, locator, parent)
    
    def initialize_ui(self):
        """Build your panel UI here."""
        layout = QVBoxLayout(self._content)
        layout.addWidget(QLabel("My Custom Panel"))
    
    def on_show(self):
        """Called when panel becomes visible."""
        print("Panel shown")
    
    def on_hide(self):
        """Called when panel is hidden."""
        print("Panel hidden")
    
    def get_state(self) -> dict:
        """Return custom state for persistence."""
        return {"my_setting": "value"}
    
    def set_state(self, state: dict):
        """Restore custom state."""
        my_setting = state.get("my_setting")
```

**2. Register Panel**

In `MainWindow.__init__`:

```python
self.dock_manager.register_panel("my_panel", MyPanel)

# Add toggle action
self.action_registry.register_action(
    "view_panel_my_panel",
    "&My Panel",
    lambda: self.dock_manager.toggle_panel("my_panel"),
    checkable=True
)
```

**3. Update Menu**

In `MenuBuilder.build_view_menu`:

```python
panels_menu.addAction(self.actions.get_action("view_panel_my_panel"))
```

### Creating Custom Document Types

**1. Create Document View**

```python
from src.ui.documents.document_view import DocumentView, DocumentViewModel

class MyDocumentView(DocumentView):
    def __init__(self, viewmodel: DocumentViewModel, parent=None):
        super().__init__(viewmodel, parent)
        self._build_ui()
    
    def _build_ui(self):
        """Build document UI."""
        # Add your widgets here
        pass
    
    def get_content(self) -> str:
        """Return document content."""
        return self._content
    
    def set_content(self, content: str):
        """Set document content."""
        self._content = content
        self.content_changed.emit()
    
    def save(self):
        """Save document."""
        # Save logic here
        self.viewmodel.mark_saved()
```

**2. Use Document**

```python
# Create ViewModel
vm = DocumentViewModel(locator, file_path="path/to/file.txt")

# Create View
doc = MyDocumentView(vm)

# Add to split container
container.add_document(doc, vm.title)
```

### Adding Menu Actions

**1. Register Action**

```python
self.action_registry.register_action(
    name="my_action",
    text="&My Action",
    callback=self._on_my_action,
    shortcut="Ctrl+M",
    tooltip="Execute my action",
    checkable=False
)
```

**2. Add to Menu**

```python
# In MenuBuilder
my_menu = self.menubar.addMenu("&MyMenu")
my_menu.addAction(self.actions.get_action("my_action"))
```

### Adding Settings

**1. Update Config Model**

In `src/core/config.py`:

```python
class AppConfig(BaseModel):
    app_name: str = "Foundation App"
    db_name: str = "foundation_demo"
    my_setting: str = "default_value"  # New setting
```

**2. Add to Settings Dialog**

In `SettingsDialog._create_general_settings`:

```python
layout.addWidget(QLabel("My Setting:"))
self.my_setting_input = QLineEdit()
self.my_setting_input.setText(self.config.data.my_setting)
self.my_setting_input.textChanged.connect(
    lambda text: self._on_setting_changed("my_setting", text))
layout.addWidget(self.my_setting_input)
```

---

## Architecture Details

### Document Split System

**SplitManager**
- Tree structure for managing splits
- Methods: `split_node()`, `remove_split()`, `get_all_containers()`

**SplitNode**
- Represents container (leaf) or splitter (branch)
- Serializes to/from JSON

**Layout State**
- Saves split tree to config
- Restores on app startup

### Docking System

**DockManager**
- Central panel registry
- Methods: `register_panel()`, `create_panel()`, `save_state()`, `restore_state()`

**BasePanelWidget**
- Base class for all panels
- Lifecycle hooks: `initialize_ui()`, `on_show()`, `on_hide()`, `on_update()`

**PanelState**
- Tracks visibility, position, size
- Serializes to config

### Menu System

**ActionRegistry**
- Manages all QActions
- Provides: `register_action()`, `get_action()`, `set_enabled()`, `set_checked()`

**MenuBuilder**
- Builds menus from actions
- Methods: `build_file_menu()`, `build_edit_menu()`, etc.

### Settings System

**SettingsDialog**
- Category tree on left
- Settings panels on right
- Search filtering
- Direct ConfigManager integration

### Command System

**CommandPalette**
- Fuzzy search through actions
- Keyboard navigation
- Shows shortcuts

---

## MVVM Integration

All GUI components follow MVVM:

**ViewModel**
- Extends `BaseViewModel`
- Properties with signals
- Example: `status_message` emits `statusMessageChanged`

**View**
- Connects to ViewModel signals
- Updates UI on changes
- Example: `viewmodel.statusMessageChanged.connect(self.update_status)`

---

## Best Practices

### Panel Development
1. Always call `super().__init__()` first
2. Build UI in `initialize_ui()`, not `__init__`
3. Use lifecycle hooks for setup/cleanup
4. Implement `get_state()`/`set_state()` for persistence

### Document Development
1. Inherit from `DocumentView`
2. Create corresponding `DocumentViewModel`
3. Emit `content_changed` when modified
4. Implement `save()` and `can_close()`

### Action Development
1. Register all actions centrally
2. Use meaningful keyboard shortcuts
3. Update action state based on context
4. Add tooltips for discoverability

### Settings Development
1. Add to Pydantic model first
2. Create widget in settings dialog
3. Connect to `_on_setting_changed`
4. Test persistence (save/restart)

---

## Testing

### Panel Tests
```python
def test_my_panel(qapp):
    config = MagicMock()
    panel = MyPanel("Test", config)
    panel.initialize_ui()
    
    assert panel.isVisible()
```

### Action Tests
```python
def test_my_action(qapp):
    registry = ActionRegistry(QWidget())
    callback = MagicMock()
    
    action = registry.register_action("test", "Test", callback)
    action.trigger()
    
    callback.assert_called_once()
```

---

## Common Patterns

### Context-Aware Actions
```python
# Enable save only when document modified
doc.content_changed.connect(
    lambda: self.action_registry.set_enabled("file_save", True))
```

### Panel Communication
```python
# Use signals
class MyPanel(BasePanelWidget):
    data_changed = Signal(object)
    
    def update_data(self, data):
        self.data_changed.emit(data)

# Connect in MainWindow
my_panel.data_changed.connect(other_panel.on_data_update)
```

### Settings Reactivity
```python
# Listen for config changes
self.config.on_changed.connect(self._on_config_changed)

def _on_config_changed(self, key, value):
    if key == "my_setting":
        self.apply_setting(value)
```

---

## API Reference

See individual module documentation:
- `src/ui/documents/` - Document split system
- `src/ui/docking/` - Panel management
- `src/ui/menus/` - Menu and action system
- `src/ui/settings/` - Settings dialog
- `src/ui/commands/` - Command palette
