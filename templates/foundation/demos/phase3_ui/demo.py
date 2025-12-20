# -*- coding: utf-8 -*-
"""
Phase 3 Demo - Node Editor UI

This demo shows the visual node editor:
- NodeGraphWidget with pan/zoom
- Node creation via context menu
- Connection drag-and-drop
- Properties panel for editing values
- Node palette for browsing available nodes

Run with: py demos/phase3_ui/demo.py
"""
import sys
from pathlib import Path

# Add foundation to path
FOUNDATION_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FOUNDATION_DIR))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout
)
from PySide6.QtCore import Qt

from src.nodegraph.core.graph import NodeGraph
from src.nodegraph.core.registry import NodeRegistry
from src.nodegraph.ui import NodeGraphWidget, PropertiesPanel, NodePalettePanel


class MockLocator:
    """Mock ServiceLocator for standalone demo."""
    pass


class MockConfig:
    """Mock ConfigManager for standalone demo."""
    pass


class NodeEditorDemo(QMainWindow):
    """
    Demonstration of the Node Editor UI.
    
    Features:
    - Main canvas with pan/zoom
    - Left dock: Node Palette
    - Right dock: Properties Panel
    - Context menu to add nodes
    - Connection drag-and-drop
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("NodeGraph Editor - Phase 3 Demo")
        self.setMinimumSize(1200, 800)
        
        # Create components
        self._setup_registry()
        self._setup_graph()
        self._setup_ui()
        self._add_sample_nodes()
    
    def _setup_registry(self):
        """Initialize node registry with built-in nodes."""
        self._registry = NodeRegistry(MockLocator(), MockConfig())
        # Manually register nodes since we're not using async init
        from src.nodegraph.nodes import ALL_NODES
        for node_cls in ALL_NODES:
            self._registry.register(node_cls)
    
    def _setup_graph(self):
        """Create a new graph."""
        self._graph = NodeGraph("Demo Graph")
    
    def _setup_ui(self):
        """Setup the UI layout."""
        # Central widget - Node Graph
        self._canvas = NodeGraphWidget()
        self._canvas.set_graph(self._graph, self._registry)
        self.setCentralWidget(self._canvas)
        
        # Left dock - Node Palette
        palette_dock = QDockWidget("Node Palette", self)
        palette_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        self._palette = NodePalettePanel()
        self._palette.set_registry(self._registry)
        self._palette.node_requested.connect(self._on_node_requested)
        palette_dock.setWidget(self._palette)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, palette_dock)
        
        # Right dock - Properties Panel
        props_dock = QDockWidget("Properties", self)
        props_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        self._properties = PropertiesPanel()
        props_dock.setWidget(self._properties)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, props_dock)
        
        # Connect signals
        self._canvas.node_selected.connect(self._on_node_selected)
    
    def _add_sample_nodes(self):
        """Add some sample nodes to demonstrate the editor."""
        from src.nodegraph.nodes.events import StartNode
        from src.nodegraph.nodes.flow_control import BranchNode, SequenceNode
        from src.nodegraph.nodes.utilities import PrintNode
        
        # Create nodes
        start = StartNode()
        start.position = (100, 200)
        self._graph.add_node(start)
        
        branch = BranchNode()
        branch.position = (300, 200)
        self._graph.add_node(branch)
        
        print_true = PrintNode()
        print_true.position = (550, 100)
        print_true._input_pins["message"].default_value = "Condition is TRUE!"
        self._graph.add_node(print_true)
        
        print_false = PrintNode()
        print_false.position = (550, 300)
        print_false._input_pins["message"].default_value = "Condition is FALSE!"
        self._graph.add_node(print_false)
        
        # Create connections
        self._graph.connect(start.node_id, "exec", branch.node_id, "exec")
        self._graph.connect(branch.node_id, "true", print_true.node_id, "exec")
        self._graph.connect(branch.node_id, "false", print_false.node_id, "exec")
        
        # Rebuild display
        self._canvas.set_graph(self._graph, self._registry)
        
        # Fit in view
        self._canvas.fit_in_view()
    
    def _on_node_requested(self, node_type: str):
        """Handle node requested from palette."""
        # Add at center of view
        center = self._canvas.mapToScene(
            self._canvas.viewport().rect().center()
        )
        self._canvas.add_node(node_type, center)
    
    def _on_node_selected(self, node):
        """Handle node selection change."""
        self._properties.set_node(node)


def main():
    # Set dark theme
    import os
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
    
    app = QApplication(sys.argv)
    
    # Basic dark stylesheet
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2d;
            color: #e0e0e0;
        }
        QDockWidget {
            color: #e0e0e0;
            titlebar-close-icon: url(none);
        }
        QDockWidget::title {
            background-color: #3c3c3e;
            padding: 6px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #3c3c3e;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px;
            color: #e0e0e0;
        }
        QCheckBox {
            color: #e0e0e0;
        }
        QPushButton {
            background-color: #4a4a4c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px 12px;
            color: #e0e0e0;
        }
        QPushButton:hover {
            background-color: #5a5a5c;
        }
        QTreeWidget {
            background-color: #2b2b2d;
            border: none;
            color: #e0e0e0;
        }
        QTreeWidget::item:hover {
            background-color: #3c3c3e;
        }
        QTreeWidget::item:selected {
            background-color: #4a90d9;
        }
        QScrollBar:vertical {
            background-color: #2b2b2d;
            width: 12px;
        }
        QScrollBar::handle:vertical {
            background-color: #555;
            border-radius: 4px;
            min-height: 20px;
        }
    """)
    
    window = NodeEditorDemo()
    window.show()
    
    print("=" * 60)
    print("NodeGraph Phase 3 Demo - Node Editor UI")
    print("=" * 60)
    print("\nControls:")
    print("  - Pan: Middle-mouse drag or Space+drag")
    print("  - Zoom: Mouse wheel")
    print("  - Add node: Right-click canvas -> select node type")
    print("  - Connect: Drag from output pin to input pin")
    print("  - Delete: Select + Delete key")
    print("  - Select: Click node")
    print("  - Fit view: Press 'F'")
    print("\nDouble-click nodes in palette to add them.")
    print("Select a node to edit its properties.")
    print("=" * 60)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
