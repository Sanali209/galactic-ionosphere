# -*- coding: utf-8 -*-
"""
Node Editor Sample Application

A complete visual node programming environment demonstrating:
- Full node editor with all UI panels
- Graph execution with logging
- File save/load
- Example workflows

This is the reference implementation for the nodegraph subsystem.
"""
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QToolBar,
    QFileDialog, QMessageBox, QStatusBar, QWidget, QVBoxLayout,
    QSplitter, QLabel, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QIcon

# Add src to path
foundation_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(foundation_dir))

from src.nodegraph.core import NodeGraph, NodeRegistry
from src.nodegraph.ui import (
    NodeGraphWidget, PropertiesPanel, NodePalettePanel,
    VariablesPanel, ExecutionLogPanel
)
from src.nodegraph.execution import GraphExecutor, ExecutionState


class NodeEditorWindow(QMainWindow):
    """
    Main window for the Node Editor application.
    
    Features:
    - Central canvas with node graph
    - Dockable panels: Palette, Properties, Variables, Log
    - Toolbar with common actions
    - File menu for save/load
    - Graph execution with progress
    """
    
    def __init__(self):
        super().__init__()
        
        self._graph = NodeGraph("Untitled")
        self._registry = NodeRegistry()
        self._registry.register_all_builtin()
        self._current_file: Optional[Path] = None
        self._executor: Optional[GraphExecutor] = None
        
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_docks()
        self._connect_signals()
        
        self.setWindowTitle("Node Editor - Untitled")
        self.resize(1400, 900)
        
        # Apply dark theme
        self._apply_theme()
    
    def _setup_ui(self):
        """Setup main UI components."""
        # Central widget - the node graph canvas
        self._canvas = NodeGraphWidget()
        self._canvas.set_graph(self._graph, self._registry)
        self.setCentralWidget(self._canvas)
        
        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")
        
        # Progress bar for execution
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setVisible(False)
        self._status.addPermanentWidget(self._progress)
    
    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = file_menu.addAction("&New")
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_graph)
        
        open_action = file_menu.addAction("&Open...")
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_graph)
        
        save_action = file_menu.addAction("&Save")
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_graph)
        
        save_as_action = file_menu.addAction("Save &As...")
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_graph_as)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = edit_menu.addAction("&Undo")
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.setEnabled(False)  # TODO: implement undo
        
        redo_action = edit_menu.addAction("&Redo")
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.setEnabled(False)  # TODO: implement redo
        
        edit_menu.addSeparator()
        
        cut_action = edit_menu.addAction("Cu&t")
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        cut_action.triggered.connect(self._canvas.cut_selected)
        
        copy_action = edit_menu.addAction("&Copy")
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self._canvas.copy_selected)
        
        paste_action = edit_menu.addAction("&Paste")
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self._canvas.paste)
        
        delete_action = edit_menu.addAction("&Delete")
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self._canvas.delete_selected)
        
        edit_menu.addSeparator()
        
        select_all = edit_menu.addAction("Select &All")
        select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all.triggered.connect(self._canvas.select_all)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_action = view_menu.addAction("&Fit All")
        fit_action.setShortcut(Qt.Key.Key_F)
        fit_action.triggered.connect(self._canvas.fit_in_view)
        
        view_menu.addSeparator()
        
        # Panel toggles will be added in _setup_docks
        self._view_menu = view_menu
        
        # Run menu
        run_menu = menubar.addMenu("&Run")
        
        self._run_action = run_menu.addAction("&Execute Graph")
        self._run_action.setShortcut(Qt.Key.Key_F5)
        self._run_action.triggered.connect(self._execute_graph)
        
        self._stop_action = run_menu.addAction("&Stop")
        self._stop_action.setShortcut(Qt.Key.Key_Escape)
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._stop_execution)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self._show_about)
    
    def _setup_toolbar(self):
        """Setup main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # File actions
        toolbar.addAction("New").triggered.connect(self._new_graph)
        toolbar.addAction("Open").triggered.connect(self._open_graph)
        toolbar.addAction("Save").triggered.connect(self._save_graph)
        
        toolbar.addSeparator()
        
        # Edit actions
        toolbar.addAction("Copy").triggered.connect(self._canvas.copy_selected)
        toolbar.addAction("Paste").triggered.connect(self._canvas.paste)
        toolbar.addAction("Delete").triggered.connect(self._canvas.delete_selected)
        
        toolbar.addSeparator()
        
        # View actions
        toolbar.addAction("Fit All").triggered.connect(self._canvas.fit_in_view)
        
        toolbar.addSeparator()
        
        # Run actions
        self._run_btn = toolbar.addAction("▶ Run")
        self._run_btn.triggered.connect(self._execute_graph)
        
        self._stop_btn = toolbar.addAction("■ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.triggered.connect(self._stop_execution)
    
    def _setup_docks(self):
        """Setup dockable panels."""
        # Node Palette (Left)
        self._palette_dock = QDockWidget("Node Palette", self)
        self._palette = NodePalettePanel()
        self._palette.set_registry(self._registry)
        self._palette.node_requested.connect(self._add_node_from_palette)
        self._palette_dock.setWidget(self._palette)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._palette_dock)
        
        # Variables Panel (Left, tabbed with Palette)
        self._variables_dock = QDockWidget("Variables", self)
        self._variables = VariablesPanel()
        self._variables.set_graph(self._graph)
        self._variables_dock.setWidget(self._variables)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._variables_dock)
        self.tabifyDockWidget(self._palette_dock, self._variables_dock)
        self._palette_dock.raise_()
        
        # Properties Panel (Right)
        self._properties_dock = QDockWidget("Properties", self)
        self._properties = PropertiesPanel()
        self._properties_dock.setWidget(self._properties)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)
        
        # Execution Log (Bottom)
        self._log_dock = QDockWidget("Execution Log", self)
        self._log = ExecutionLogPanel()
        self._log_dock.setWidget(self._log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        
        # Add toggle actions to View menu
        self._view_menu.addAction(self._palette_dock.toggleViewAction())
        self._view_menu.addAction(self._variables_dock.toggleViewAction())
        self._view_menu.addAction(self._properties_dock.toggleViewAction())
        self._view_menu.addAction(self._log_dock.toggleViewAction())
    
    def _connect_signals(self):
        """Connect signals between components."""
        # Canvas -> Properties panel
        self._canvas.node_selected.connect(self._properties.set_node)
        
        # Canvas -> Status bar
        self._canvas.graph_changed.connect(self._on_graph_changed)
        
        # Log panel -> Canvas (for error navigation)
        self._log.node_clicked.connect(self._canvas.highlight_node)
    
    def _apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2d;
            }
            QMenuBar {
                background-color: #3c3c3e;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #505052;
            }
            QMenu {
                background-color: #3c3c3e;
                color: #e0e0e0;
                border: 1px solid #505052;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
            QToolBar {
                background-color: #3c3c3e;
                border: none;
                spacing: 5px;
                padding: 5px;
            }
            QToolButton {
                background-color: transparent;
                color: #e0e0e0;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QToolButton:hover {
                background-color: #505052;
                border: 1px solid #606062;
            }
            QToolButton:pressed {
                background-color: #0078d4;
            }
            QDockWidget {
                color: #e0e0e0;
                titlebar-close-icon: url(close.png);
            }
            QDockWidget::title {
                background-color: #3c3c3e;
                padding: 6px;
            }
            QStatusBar {
                background-color: #3c3c3e;
                color: #e0e0e0;
            }
            QProgressBar {
                background-color: #2b2b2d;
                border: 1px solid #505052;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    @Slot()
    def _new_graph(self):
        """Create new empty graph."""
        if not self._confirm_discard():
            return
        
        self._graph = NodeGraph("Untitled")
        self._canvas.set_graph(self._graph, self._registry)
        self._variables.set_graph(self._graph)
        self._properties.set_node(None)
        self._log.clear()
        self._current_file = None
        self._update_title()
        self._status.showMessage("Created new graph")
    
    @Slot()
    def _open_graph(self):
        """Open graph from file."""
        if not self._confirm_discard():
            return
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Graph", "",
            "Graph Files (*.graph *.json);;All Files (*)"
        )
        
        if filepath:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                self._graph = NodeGraph.from_dict(data, self._registry)
                self._canvas.set_graph(self._graph, self._registry)
                self._variables.set_graph(self._graph)
                self._properties.set_node(None)
                self._log.clear()
                self._current_file = Path(filepath)
                self._update_title()
                self._status.showMessage(f"Opened: {filepath}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file:\n{e}")
    
    @Slot()
    def _save_graph(self):
        """Save graph to current file or prompt."""
        if self._current_file:
            self._do_save(self._current_file)
        else:
            self._save_graph_as()
    
    @Slot()
    def _save_graph_as(self):
        """Save graph to new file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "",
            "Graph Files (*.graph);;JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            if not filepath.endswith(('.graph', '.json')):
                filepath += '.graph'
            self._do_save(Path(filepath))
    
    def _do_save(self, filepath: Path):
        """Perform save operation."""
        try:
            data = self._graph.to_dict()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            self._current_file = filepath
            self._graph._modified = False
            self._update_title()
            self._status.showMessage(f"Saved: {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")
    
    @Slot(str)
    def _add_node_from_palette(self, node_type: str):
        """Add node from palette double-click."""
        # Add at center of view
        center = self._canvas.mapToScene(
            self._canvas.viewport().rect().center()
        )
        self._canvas.add_node(node_type, center)
    
    @Slot()
    def _execute_graph(self):
        """Execute the current graph."""
        self._log.clear()
        self._log.add_message("Starting execution...", "INFO")
        
        self._run_action.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._stop_action.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # Indeterminate
        
        # Run async
        self._executor = GraphExecutor(self._graph)
        
        # Use QTimer to run asyncio
        QTimer.singleShot(100, self._run_executor)
    
    def _run_executor(self):
        """Run executor in asyncio."""
        import asyncio
        
        async def run():
            try:
                context = await self._executor.run_async()
                return context
            except Exception as e:
                return e
        
        # Run synchronously for simplicity
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run())
            loop.close()
            
            if isinstance(result, Exception):
                self._log.add_message(f"Execution error: {result}", "ERROR")
                self._status.showMessage("Execution failed")
            else:
                # Add logs to panel
                self._log.add_logs(result.logs)
                
                if result.error_node_id:
                    self._log.add_message(
                        f"Error in node {result.error_node_id}: {result.error_message}",
                        "ERROR"
                    )
                    self._canvas.highlight_node(result.error_node_id)
                    self._status.showMessage("Execution completed with errors")
                else:
                    self._log.add_message("Execution completed successfully", "INFO")
                    self._status.showMessage("Execution completed")
                    
        except Exception as e:
            self._log.add_message(f"Runtime error: {e}", "ERROR")
            self._status.showMessage("Execution failed")
        
        self._execution_finished()
    
    @Slot()
    def _stop_execution(self):
        """Stop current execution."""
        if self._executor:
            self._executor.stop()
        self._log.add_message("Execution stopped by user", "WARNING")
        self._execution_finished()
    
    def _execution_finished(self):
        """Reset UI after execution."""
        self._run_action.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._stop_action.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._executor = None
    
    @Slot()
    def _on_graph_changed(self):
        """Handle graph modification."""
        self._graph._modified = True
        self._update_title()
    
    @Slot()
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Node Editor",
            "<h2>Node Editor</h2>"
            "<p>Visual Node Programming Environment</p>"
            "<p>Part of the Foundation Template</p>"
            "<hr>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>80 built-in nodes</li>"
            "<li>File, String, Array, Image operations</li>"
            "<li>Matplotlib visualization</li>"
            "<li>Flow control: Loops, Branches</li>"
            "<li>Variables and execution logging</li>"
            "</ul>"
        )
    
    def _update_title(self):
        """Update window title."""
        name = self._current_file.name if self._current_file else "Untitled"
        modified = "*" if getattr(self._graph, '_modified', False) else ""
        self.setWindowTitle(f"Node Editor - {name}{modified}")
    
    def _confirm_discard(self) -> bool:
        """Confirm discarding unsaved changes."""
        if getattr(self._graph, '_modified', False):
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes
        return True
    
    def closeEvent(self, event):
        """Handle window close."""
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()


def main():
    """Run the Node Editor application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Node Editor")
    app.setStyle("Fusion")
    
    window = NodeEditorWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
