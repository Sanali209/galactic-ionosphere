"""
Test script to verify ActionRegistry implementation.

This tests that all actions are properly registered and accessible.
"""
import sys
from pathlib import Path

# Add foundation to path
foundation_path = Path(__file__).parent.parent.parent.parent / "templates" / "foundation"
sys.path.insert(0, str(foundation_path))

def test_action_registry():
    """Test that ActionRegistry has all expected actions."""
    from PySide6.QtWidgets import QApplication, QMainWindow
    from src.ui.menus.action_registry import ActionRegistry
    
    app = QApplication([])
    window = QMainWindow()
    
    # Create registry
    registry = ActionRegistry(window)
    
    # Register some test actions
    registry.register_action("test.action1", "Test Action 1", lambda: None, "Ctrl+T")
    registry.register_action("test.action2", "Test Action 2", lambda: None)
    
    # Verify actions exist
    action1 = registry.get_action("test.action1")
    assert action1 is not None, "Action test.action1 should exist"
    assert action1.text() == "Test Action 1", f"Expected 'Test Action 1', got '{action1.text()}'"
    assert action1.shortcut().toString() == "Ctrl+T", f"Expected 'Ctrl+T', got '{action1.shortcut().toString()}'"
    
    action2 = registry.get_action("test.action2")
    assert action2 is not None, "Action test.action2 should exist"
    assert action2.shortcut().toString() == "", "Action2 should have no shortcut"
    
    print("✓ ActionRegistry test passed")
    return True

def test_uexplorer_actions():
    """Test that UExplorer actions are all registered."""
    from PySide6.QtWidgets import QApplication, QMainWindow
    from src.ui.menus.action_registry import ActionRegistry
    from src.ui.actions.action_definitions import register_all_actions
    
    app = QApplication([])
    
    # Create mock window with required methods
    class MockWindow(QMainWindow):
        def new_window(self): pass
        def show_settings_dialog(self): pass
        def _toggle_panel(self, name): pass
        def _split_horizontal(self): pass
        def _split_vertical(self): pass
        def _close_split(self): pass
        def reset_layout(self): pass
        def show_library_dialog(self): pass
        def show_command_palette(self): pass
        def show_shortcuts_dialog(self): pass
        def show_about_dialog(self): pass
    
    window = MockWindow()
    registry = ActionRegistry(window)
    
    # Register UExplorer actions
    register_all_actions(registry, window)
    
    # Verify expected actions
    expected_actions = [
        "file.new_window",
        "file.exit",
        "edit.settings",
        "view.panel.tags",
        "view.panel.albums",
        "view.panel.relations",
        "view.panel.properties",
        "view.split_horizontal",
        "view.split_vertical",
        "view.close_split",
        "view.reset_layout",
        "view.thumbnails",
        "tools.scan",
        "tools.library",
        "tools.rules",
        "tools.command_palette",
        "help.shortcuts",
        "help.about",
    ]
    
    for action_name in expected_actions:
        action = registry.get_action(action_name)
        assert action is not None, f"Action {action_name} should exist"
        print(f"  ✓ {action_name}: {action.text().replace('&', '')} ({action.shortcut().toString() or 'no shortcut'})")
    
    print(f"\n✓ All {len(expected_actions)} UExplorer actions registered successfully")
    return True

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Testing ActionRegistry Implementation")
        print("=" * 60)
        
        print("\n1. Testing basic ActionRegistry...")
        test_action_registry()
        
        print("\n2. Testing UExplorer action definitions...")
        test_uexplorer_actions()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
