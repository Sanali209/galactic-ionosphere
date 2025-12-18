# Foundation Template - GUI Framework API

Quick reference for the GUI framework APIs.

## Documents Module

### SplitManager
```python
from src.ui.documents.split_manager import SplitManager, SplitOrientation

manager = SplitManager()

# Split a node
new_id = manager.split_node(node_id, SplitOrientation.HORIZONTAL)

# Remove split
manager.remove_split(node_id)

# Get all containers
containers = manager.get_all_containers()

# Serialize
data = manager.to_dict()

# Deserialize
manager = SplitManager.from_dict(data)
```

### DocumentView
```python
from src.ui.documents.document_view import DocumentView, DocumentViewModel

# Create ViewModel
vm = DocumentViewModel(locator, file_path="file.txt")

# Create View
class MyDoc(DocumentView):
    def initialize_ui(self):
        # Build UI
        pass

doc = MyDoc(vm)

# Properties
doc.viewmodel.title           # Document title
doc.viewmodel.is_modified     # Modified flag
doc.viewmodel.file_path       # File path

# Methods
doc.viewmodel.mark_modified()
doc.viewmodel.mark_saved()
```

## Docking Module

### DockManager
```python
from src.ui.docking.dock_manager import DockManager

manager = DockManager(main_window, config)

# Register panel type
manager.register_panel("my_panel", MyPanelClass)

# Create/show panel
manager.create_panel("my_panel")
manager.show_panel("my_panel")
manager.hide_panel("my_panel")
manager.toggle_panel("my_panel")

# State management
state = manager.save_state()
manager.restore_state(state)
```

### BasePanelWidget
```python
from src.ui.docking.panel_base import BasePanelWidget

class MyPanel(BasePanelWidget):
    def initialize_ui(self):
        """Required: Build UI here."""
        pass
    
    def on_show(self):
        """Optional: Called when shown."""
        pass
    
    def on_hide(self):
        """Optional: Called when hidden."""
        pass
    
    def on_update(self, context=None):
        """Optional: Refresh content."""
        pass
    
    def get_state(self) -> dict:
        """Optional: Return custom state."""
        return {}
    
    def set_state(self, state: dict):
        """Optional: Restore custom state."""
        pass

# Signals
panel.panel_shown.connect(callback)
panel.panel_hidden.connect(callback)
```

## Menus Module

### ActionRegistry
```python
from src.ui.menus.action_registry import ActionRegistry

registry = ActionRegistry(parent_widget)

# Register action
action = registry.register_action(
    name="my_action",
    text="&My Action",
    callback=my_function,
    shortcut="Ctrl+M",
    tooltip="My tooltip",
    checkable=False
)

# Get action
action = registry.get_action("my_action")

# Update action
registry.set_enabled("my_action", True)
registry.set_checked("my_action", True)
registry.update_text("my_action", "New Text")
```

### MenuBuilder
```python
from src.ui.menus.menu_builder import MenuBuilder

builder = MenuBuilder(main_window, action_registry)

# Build individual menus
builder.build_file_menu()
builder.build_edit_menu()
builder.build_view_menu(dock_manager)
builder.build_tools_menu()
builder.build_window_menu()
builder.build_help_menu()

# Build all at once
builder.build_all_menus(dock_manager)
```

## Settings Module

### SettingsDialog
```python
from src.ui.settings.settings_dialog import SettingsDialog

dialog = SettingsDialog(config_manager, parent)

# Show dialog
dialog.exec()

# Signals
dialog.settings_changed.connect(on_setting_changed)

# Methods (internal)
dialog._build_categories()
dialog._add_category(name, tree_item, widget)
```

## Commands Module

### CommandPalette
```python
from src.ui.commands.command_palette import CommandPalette

palette = CommandPalette(action_registry, parent)

# Show palette
palette.exec()

# Signals
palette.command_selected.connect(on_command)
```

## Common Patterns

### Creating a Panel
```python
# 1. Define panel class
class MyPanel(BasePanelWidget):
    def initialize_ui(self):
        layout = QVBoxLayout(self._content)
        layout.addWidget(QLabel("My Panel"))

# 2. Register in MainWindow
self.dock_manager.register_panel("my_panel", MyPanel)

# 3. Add toggle action
self.action_registry.register_action(
    "view_panel_my_panel",
    "&My Panel",
    lambda: self.dock_manager.toggle_panel("my_panel"),
    checkable=True
)

# 4. Add to menu
view_menu.addAction(self.actions.get_action("view_panel_my_panel"))
```

### Adding a Menu Action
```python
# 1. Register action
self.action_registry.register_action(
    "my_action",
    "&My Action",
    self._on_my_action,
    "Ctrl+M"
)

# 2. Add to menu
menu.addAction(self.actions.get_action("my_action"))

# 3. Implement handler
def _on_my_action(self):
    print("Action executed")
```

### Adding a Setting
```python
# 1. Update config model
class AppConfig(BaseModel):
    my_setting: str = "default"

# 2. Add to settings dialog
layout.addWidget(QLabel("My Setting:"))
input = QLineEdit()
input.setText(self.config.data.my_setting)
input.textChanged.connect(
    lambda text: self._on_setting_changed("my_setting", text))
layout.addWidget(input)
```

---

## Type Hints

```python
from typing import Optional, List, Dict, Any, Callable
from PySide6.QtWidgets import QWidget, QMainWindow
from src.ui.docking.panel_base import BasePanelWidget
from src.ui.menus.action_registry import ActionRegistry

def register_panel(
    self,
    name: str,
    panel_class: type[BasePanelWidget]
) -> None: ...

def register_action(
    self,
    name: str,
    text: str,
    callback: Callable,
    shortcut: Optional[str] = None,
    tooltip: Optional[str] = None,
    checkable: bool = False
) -> QAction: ...
```
