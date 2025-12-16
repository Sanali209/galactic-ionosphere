# MVVM Pattern Guide

The template supports the Model-View-ViewModel (MVVM) pattern to separate UI (PySide6) from business logic (Core Systems).

## Components

### 1. View (`src.ui.main_window.py`)
- **Responsibility**: Display data, capture user input.
- **Dependency**: Knows only the `ViewModel`.
- **Binding**: Connects Signals from VM to Slots in View.

```python
class MainWindow(QMainWindow):
    def __init__(self, vm: MainViewModel):
        self.vm = vm
        # One-way Binding (VM -> View)
        self.vm.statusMessageChanged.connect(self.update_label)

    @Slot(str)
    def update_label(self, text):
        self.label.setText(text)
```

### 2. ViewModel (`src.ui.viewmodels.*`)
- **Responsibility**: Prepare data for View, handle View logic.
- **Dependency**: Knows `ServiceLocator` (to access Models/Systems).
- **Inheritance**: Inherits `BaseViewModel` (`QObject`).

```python
class MyViewModel(BaseViewModel):
    textChanged = Signal(str)

    @property
    def text(self): ...
    
    @text.setter
    def text(self, val):
        self._text = val
        self.textChanged.emit(val)
        
    def save_data(self):
        # Access Model via Locator
        self.locator.get_system(Database).save(...)
```

### 3. Model (`src.core.*`)
- **Responsibility**: Business logic, data persistence.
- **Dependency**: Independent of UI or VM.

## Usage

1. **Create VM**: In `src/ui/viewmodels/`, create `YourViewModel`.
2. **Register**: (Optional) or just instantiate via `ViewModelProvider`.
3. **Inject**: Passing `vm` to the View constructor.

```python
provider = ViewModelProvider(sl)
vm = provider.get(YourViewModel)
view = YourView(vm)
```
