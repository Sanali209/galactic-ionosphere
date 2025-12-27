# Binding Demo Sample

Demonstrates WPF-style data binding features from Foundation.

## Features Shown

- **BindableProperty**: Automatic signal emission on property change
- **Two-Way Binding**: VM ↔ Widget synchronization
- **One-Way Binding**: VM → Widget with converters
- **Command Binding**: Button → ViewModel method

## Run

```bash
cd templates/foundation/samples/binding_demo
python main.py
```

## Code Highlights

### ViewModel with BindableProperty

```python
class DemoViewModel(BindableBase):
    nameChanged = Signal(str)
    name = BindableProperty(default="World")
```

### Two-Way Binding

```python
bind(vm, "name", self.name_edit, "text", mode=BindingMode.TWO_WAY)
```

### Command Binding

```python
bind_command(vm, "reset", self.reset_button, "clicked")
```
