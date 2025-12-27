# -*- coding: utf-8 -*-
"""
PropertiesPanel - Panel for editing selected node properties.

Auto-generates widgets based on pin types:
- BOOLEAN: Checkbox
- INTEGER: SpinBox
- FLOAT: DoubleSpinBox
- STRING: LineEdit
- PATH: FileDialog button + LineEdit
"""
from typing import Optional, TYPE_CHECKING
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QFileDialog, QScrollArea,
    QFrame, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

if TYPE_CHECKING:
    from ..core.base_node import BaseNode
    from ..core.pins import BasePin, PinType


class PropertiesPanel(QWidget):
    """
    Panel for editing node properties.
    
    Displays the selected node's input pins as editable
    fields with appropriate widgets for each type.
    
    Signals:
        value_changed: Emitted when any value is changed
    """
    
    value_changed = Signal(str, object)  # pin_name, new_value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_node: Optional['BaseNode'] = None
        self._widgets = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Properties")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # No selection label
        self._no_selection = QLabel("No node selected")
        self._no_selection.setStyleSheet("color: #888;")
        layout.addWidget(self._no_selection)
        
        # Scroll area for properties
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(4)
        
        scroll.setWidget(self._content)
        layout.addWidget(scroll)
        
        self._content.hide()
    
    def set_node(self, node: Optional['BaseNode']):
        """
        Set the node to edit.
        
        Args:
            node: Node to display, or None to clear
        """
        self._current_node = node
        self._clear_widgets()
        
        if node is None:
            self._no_selection.show()
            self._content.hide()
            return
        
        self._no_selection.hide()
        self._content.show()
        
        # Node info header
        header = QLabel(f"<b>{node.metadata.display_name or node.node_type}</b>")
        header.setStyleSheet(f"color: {node.metadata.color};")
        self._content_layout.addWidget(header)
        
        # Description
        if node.metadata.description:
            desc = QLabel(node.metadata.description)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #aaa; font-size: 10px;")
            self._content_layout.addWidget(desc)
        
        # Create widget for each input pin (if not execution)
        form = QFormLayout()
        form.setContentsMargins(0, 8, 0, 0)
        form.setSpacing(6)
        
        for pin_name, pin in node.input_pins.items():
            # Skip execution pins
            if pin.pin_type.name == "EXECUTION":
                continue
            
            widget = self._create_widget_for_pin(pin)
            if widget:
                self._widgets[pin_name] = widget
                form.addRow(pin_name + ":", widget)
        
        if form.rowCount() > 0:
            form_widget = QWidget()
            form_widget.setLayout(form)
            self._content_layout.addWidget(form_widget)
        
        self._content_layout.addStretch()
    
    def _clear_widgets(self):
        """Remove all property widgets."""
        self._widgets.clear()
        
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _create_widget_for_pin(self, pin: 'BasePin') -> Optional[QWidget]:
        """
        Create appropriate widget for pin type.
        
        Args:
            pin: Pin to create widget for
            
        Returns:
            Widget or None
        """
        from ..core.pins import PinType
        
        pin_type = pin.pin_type
        default = pin.default_value
        
        if pin_type == PinType.BOOLEAN:
            widget = QCheckBox()
            widget.setChecked(bool(default) if default is not None else False)
            widget.stateChanged.connect(
                lambda state, name=pin.name: self._on_value_changed(name, bool(state))
            )
            return widget
        
        elif pin_type == PinType.INTEGER:
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(int(default) if default is not None else 0)
            widget.valueChanged.connect(
                lambda val, name=pin.name: self._on_value_changed(name, val)
            )
            return widget
        
        elif pin_type == PinType.FLOAT:
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(3)
            widget.setValue(float(default) if default is not None else 0.0)
            widget.valueChanged.connect(
                lambda val, name=pin.name: self._on_value_changed(name, val)
            )
            return widget
        
        elif pin_type == PinType.STRING:
            widget = QLineEdit()
            widget.setText(str(default) if default is not None else "")
            widget.textChanged.connect(
                lambda text, name=pin.name: self._on_value_changed(name, text)
            )
            return widget
        
        elif pin_type == PinType.PATH:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            
            line_edit = QLineEdit()
            line_edit.setText(str(default) if default is not None else "")
            line_edit.textChanged.connect(
                lambda text, name=pin.name: self._on_value_changed(name, text)
            )
            layout.addWidget(line_edit)
            
            browse_btn = QPushButton("...")
            browse_btn.setFixedWidth(30)
            browse_btn.clicked.connect(
                lambda _, le=line_edit: self._browse_file(le)
            )
            layout.addWidget(browse_btn)
            
            return container
        
        else:
            # Generic string editor for other types
            widget = QLineEdit()
            widget.setText(str(default) if default is not None else "")
            widget.textChanged.connect(
                lambda text, name=pin.name: self._on_value_changed(name, text)
            )
            return widget
    
    def _browse_file(self, line_edit: QLineEdit):
        """Open file dialog and set path."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", line_edit.text()
        )
        if path:
            line_edit.setText(path)
    
    def _on_value_changed(self, pin_name: str, value):
        """Handle value change."""
        if self._current_node:
            # Update pin default value
            pin = self._current_node.get_input_pin(pin_name)
            if pin:
                pin.default_value = value
            
            self.value_changed.emit(pin_name, value)
