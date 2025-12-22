"""
WPF-Style Data Binding Utilities.

Provides declarative binding between ViewModel properties and View widgets.

Usage:
    from src.ui.mvvm.binding import bind, BindingMode
    
    # One-way binding (VM -> View)
    bind(vm, "username", line_edit, "text")
    
    # Two-way binding (VM <-> View)
    bind(vm, "username", line_edit, "text", mode=BindingMode.TWO_WAY)
"""
from enum import Enum
from typing import Any, Optional, Callable
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLineEdit, QLabel, QCheckBox, QSpinBox, QComboBox, QSlider


class BindingMode(Enum):
    """Binding direction modes, inspired by WPF."""
    ONE_WAY = "OneWay"           # Source -> Target (ViewModel -> View)
    TWO_WAY = "TwoWay"           # Source <-> Target (bidirectional)
    ONE_WAY_TO_SOURCE = "OneWayToSource"  # Target -> Source (View -> ViewModel)
    ONE_TIME = "OneTime"         # Initial sync only


# Widget property mappings for common Qt widgets
_WIDGET_PROPERTY_MAP = {
    QLineEdit: {"text": ("text", "textChanged")},
    QLabel: {"text": ("text", None)},
    QCheckBox: {"checked": ("isChecked", "stateChanged")},
    QSpinBox: {"value": ("value", "valueChanged")},
    QSlider: {"value": ("value", "valueChanged")},
    QComboBox: {"currentIndex": ("currentIndex", "currentIndexChanged")},
}


def _get_widget_accessors(widget: QObject, prop_name: str) -> tuple:
    """
    Get getter/setter/signal for a widget property.
    
    Returns:
        (getter_callable, setter_callable, change_signal_or_None)
    """
    widget_type = type(widget)
    
    # Check if we have a mapping for this widget type
    for wtype, props in _WIDGET_PROPERTY_MAP.items():
        if isinstance(widget, wtype) and prop_name in props:
            getter_name, signal_name = props[prop_name]
            getter = getattr(widget, getter_name, None)
            setter = getattr(widget, f"set{prop_name[0].upper()}{prop_name[1:]}", None)
            if setter is None:
                # Try direct property setter for some widgets
                setter = lambda v, w=widget, p=prop_name: setattr(w, p, v)
            signal = getattr(widget, signal_name, None) if signal_name else None
            return (getter, setter, signal)
    
    # Fallback: try generic property access
    getter = lambda w=widget, p=prop_name: getattr(w, p, None)
    setter = lambda v, w=widget, p=prop_name: setattr(w, p, v)
    signal_name = f"{prop_name}Changed"
    signal = getattr(widget, signal_name, None)
    
    return (getter, setter, signal)


def bind(
    source: QObject,
    source_property: str,
    target: QObject,
    target_property: str,
    mode: BindingMode = BindingMode.ONE_WAY,
    converter: Optional[Callable[[Any], Any]] = None,
    converter_back: Optional[Callable[[Any], Any]] = None
) -> None:
    """
    Bind a ViewModel property to a View widget property.
    
    Args:
        source: ViewModel instance (must have {property}Changed signal).
        source_property: Property name on ViewModel (e.g., "username").
        target: QWidget instance.
        target_property: Property name on Widget (e.g., "text").
        mode: Binding direction mode.
        converter: Optional function to convert source value to target value.
        converter_back: Optional function to convert target value back to source.
    
    Example:
        bind(vm, "username", self.line_edit, "text", mode=BindingMode.TWO_WAY)
    """
    # Get source signal
    source_signal_name = f"{source_property}Changed"
    source_signal = getattr(source, source_signal_name, None)
    
    # Get target accessors
    target_getter, target_setter, target_signal = _get_widget_accessors(target, target_property)
    
    # Flag to prevent infinite loops in two-way binding
    _updating = [False]
    
    # Source -> Target
    if mode in (BindingMode.ONE_WAY, BindingMode.TWO_WAY, BindingMode.ONE_TIME):
        def update_target(value):
            if _updating[0]:
                return
            _updating[0] = True
            try:
                if converter:
                    value = converter(value)
                if callable(target_setter):
                    target_setter(value)
            finally:
                _updating[0] = False
        
        # Initial sync
        initial_value = getattr(source, source_property, None)
        update_target(initial_value)
        
        # Connect signal for ongoing updates (unless OneTime)
        if mode != BindingMode.ONE_TIME and source_signal is not None:
            source_signal.connect(update_target)
    
    # Target -> Source (Two-way or OneWayToSource)
    if mode in (BindingMode.TWO_WAY, BindingMode.ONE_WAY_TO_SOURCE):
        if target_signal is not None:
            def update_source(*args):
                if _updating[0]:
                    return
                _updating[0] = True
                try:
                    value = target_getter() if callable(target_getter) else args[0] if args else None
                    if converter_back:
                        value = converter_back(value)
                    setattr(source, source_property, value)
                finally:
                    _updating[0] = False
            
            target_signal.connect(update_source)


def bind_command(
    source: QObject,
    command_name: str,
    trigger: QObject,
    trigger_signal: str = "clicked"
) -> None:
    """
    Bind a ViewModel command/method to a widget signal.
    
    Args:
        source: ViewModel instance.
        command_name: Method name on ViewModel to call.
        trigger: Widget that triggers the command (e.g., QPushButton).
        trigger_signal: Signal name on widget (default: "clicked").
    
    Example:
        bind_command(vm, "save_document", self.save_button)
    """
    command = getattr(source, command_name, None)
    if command is None:
        raise ValueError(f"Command '{command_name}' not found on {source}")
    
    signal = getattr(trigger, trigger_signal, None)
    if signal is None:
        raise ValueError(f"Signal '{trigger_signal}' not found on {trigger}")
    
    signal.connect(command)
