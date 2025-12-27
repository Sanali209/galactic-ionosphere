import loguru
import pathlib
import dataclasses # Import dataclasses for type checking in _start_drag

from PySide6.QtCore import Qt, QPoint, QMimeData
from PySide6.QtGui import QDrag, QDropEvent, QDragEnterEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

# Import the new DataTransferHelper and base classes
# Use relative import '.' assuming it's in the same 'shared' directory
from .data_transfer_helper import DataTransferHelper, MongoRecordWrapper, DataTClass


# Forward declaration for type hinting and defining delegate protocol
from typing import  List, Any,  Protocol, Optional

class DropTargetDelegate(Protocol):
    """Defines the methods a target widget must implement to handle drops and initiate drags."""
    def get_selected_items_for_drag(self) -> List[Any]:
        """Return a list of selected objects eligible for dragging."""
        ...

    def handle_dropped_files(self, paths: List[pathlib.Path], event: QDropEvent):
        """Handle dropped file/folder paths."""
        ...

    def handle_dropped_mongo_wrappers(self, objects: List[MongoRecordWrapper], event: QDropEvent):
        """Handle dropped MongoRecordWrapper instances."""
        ...

    def handle_dropped_custom_data(self, objects: List[Any], event: QDropEvent):
        """Handle dropped custom data instances (DataTClass, dataclasses)."""
        ...

    # Optional: Allow delegate to specify accepted drop types
    def accepts_drop_type(self, drop_type: str) -> bool:
        """Check if the delegate accepts a specific drop type ('files', 'mongo_wrapper', 'custom_data', 'generic_json', 'text')."""
        return True # Default to accepting all known types

    # Optional: Handle generic JSON or text if needed
    # def handle_dropped_generic_json(self, data: Any, event: QDropEvent): ...
    # def handle_dropped_text(self, text: str, event: QDropEvent): ...

    # Optional: Handle completion of a move action
    # def handle_drag_move_completed(self, moved_items: List[Any]): ...

    # The target widget must also inherit from QWidget or a relevant subclass
    def setAcceptDrops(self, accept: bool): ...
    # Example method if target is a list/tree view, adjust as needed
    # def itemAt(self, pos: QPoint) -> Optional[QWidget]: ...



class DragDropHandler:
    """
    Handles drag initiation and drop events for a target widget,
    delegating application-specific logic. Uses DataTransferHelper.

    Integration:
    1. Instantiate DragDropHandler in your QWidget: `self.drag_handler = DragDropHandler(self)`
    2. Make your QWidget implement the `DropTargetDelegate` protocol methods.
    3. Forward Qt events from your QWidget to the handler:
       - `dragEnterEvent(self, event)` -> `self.drag_handler.dragEnterEvent(event)`
       - `dropEvent(self, event)` -> `self.drag_handler.dropEvent(event)`
       - `mousePressEvent(self, event)` -> `self.drag_handler.mousePressEvent(event)` (call super if needed)
       - `mouseMoveEvent(self, event)` -> `self.drag_handler.mouseMoveEvent(event)` (call super if needed)
    """

    def __init__(self, target_widget: QWidget):

        self.target = target_widget
        self.start_drag_pos: Optional[QPoint] = None
        self.helper = DataTransferHelper() # Instantiate the helper
        # Ensure the target widget accepts drops (redundant if called in widget's __init__)
        self.target.setAcceptDrops(True)
        loguru.logger.info(f"DragDropHandler initialized for widget: {target_widget.__class__.__name__}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accepts drag events if they contain supported mime types accepted by the delegate."""
        mime_data = event.mimeData()
        accepted = False

        # Check standard formats first
        if mime_data.hasUrls() and self._delegate_accepts('files'):
            loguru.logger.debug("Drag enter: Accepting URLs (files).")
            accepted = True
        # Check custom formats
        elif mime_data.hasFormat(self.helper.MONGO_WRAPPER_MIME_TYPE) and self._delegate_accepts('mongo_wrapper'):
            loguru.logger.debug(f"Drag enter: Accepting '{self.helper.MONGO_WRAPPER_MIME_TYPE}'.")
            accepted = True
        elif mime_data.hasFormat(self.helper.CUSTOM_DATA_MIME_TYPE) and self._delegate_accepts('custom_data'):
            loguru.logger.debug(f"Drag enter: Accepting '{self.helper.CUSTOM_DATA_MIME_TYPE}'.")
            accepted = True
        # Fallback: Check if generic JSON might be acceptable
        elif mime_data.hasFormat('application/json') and self._delegate_accepts('generic_json'):
             loguru.logger.debug("Drag enter: Accepting generic 'application/json'.")
             accepted = True
        # Fallback: Check simple text if needed
        elif mime_data.hasText() and self._delegate_accepts('text'):
             loguru.logger.debug("Drag enter: Accepting generic 'text/plain'.")
             accepted = True

        if accepted:
            event.acceptProposedAction()
        else:
            loguru.logger.debug(f"Drag enter rejected. Mime types: {mime_data.formats()}")
            event.ignore()

    def _delegate_accepts(self, drop_type: str) -> bool:
        """Checks if the delegate accepts the drop type."""
        if hasattr(self.target, 'accepts_drop_type') and callable(getattr(self.target, 'accepts_drop_type')):
            try:
                return self.target.accepts_drop_type(drop_type)
            except Exception as e:
                loguru.logger.error(f"Error calling delegate 'accepts_drop_type' for type '{drop_type}': {e}")
                return False # Reject on error
        return True # Default if method not implemented

    def dropEvent(self, event: QDropEvent):
        """Handles dropping data by decoding and delegating to the target widget."""
        loguru.logger.debug(f"Drop event at pos: {event.position()}")
        decoded_data = self.helper.decode(event.mimeData())

        if decoded_data:
            data_type = decoded_data.get('type')
            data = decoded_data.get('data')

            # Ensure data is not None before proceeding
            if data is None:
                loguru.warning(f"Decoded data for type '{data_type}' is None. Ignoring drop.")
                event.ignore()
                return

            loguru.logger.info(f"Processing dropped data: type='{data_type}', data_count={len(data) if isinstance(data, list) else 1}")

            handled = False
            try:
                if data_type == 'files':
                    self.target.handle_dropped_files(data, event)
                    handled = True
                elif data_type == 'mongo_wrapper':
                    self.target.handle_dropped_mongo_wrappers(data, event)
                    handled = True
                elif data_type == 'custom_data':
                    self.target.handle_dropped_custom_data(data, event)
                    handled = True
                elif data_type == 'generic_json':
                    # Delegate generic JSON if a handler exists
                    if hasattr(self.target, 'handle_dropped_generic_json') and callable(getattr(self.target, 'handle_dropped_generic_json')):
                        self.target.handle_dropped_generic_json(data, event)
                        handled = True
                    else:
                        loguru.warning(f"No handler method 'handle_dropped_generic_json' in {self.target.__class__.__name__}")
                elif data_type == 'text':
                     # Delegate plain text if a handler exists
                     if hasattr(self.target, 'handle_dropped_text') and callable(getattr(self.target, 'handle_dropped_text')):
                         # DataTransferHelper doesn't explicitly return 'text', but QMimeData might have it.
                         # Re-check mime data if needed, or assume helper handles it via generic_json/other means.
                         # For now, assume 'text' type won't be returned by helper.decode() unless added explicitly.
                         # text_content = event.mimeData().text()
                         # self.target.handle_dropped_text(text_content, event)
                         # handled = True
                         loguru.warning(f"Received 'text' type from decode, but handler logic might need adjustment.")
                     else:
                         loguru.warning(f"No handler method 'handle_dropped_text' in {self.target.__class__.__name__}")
                else:
                    loguru.warning(f"Unhandled decoded data type from helper: {data_type}")

                if handled:
                    loguru.logger.info(f"Drop handled by delegate for type '{data_type}'.")
                    # Accept proposed action only if delegate didn't ignore it
                    if event.isAccepted():
                        event.acceptProposedAction() # Confirm acceptance
                    else:
                        loguru.logger.info("Delegate ignored the event during handling.")
                        # Event remains ignored
                else:
                    loguru.logger.warning(f"Drop not handled by delegate for type '{data_type}'.")
                    event.ignore()

            except Exception as e:
                loguru.logger.error(f"Error during drop handling by delegate '{self.target.__class__.__name__}': {e}", exc_info=True)
                event.ignore() # Ignore if delegate raises an error
        else:
            loguru.logger.debug("Drop event ignored (no supported data decoded by helper).")
            event.ignore()

    def mousePressEvent(self, event: QMouseEvent):
        """Stores the starting position for a potential drag operation."""
        # This method should be called *from* the target widget's mousePressEvent
        if event.button() == Qt.MouseButton.LeftButton:
            # Store position. Drag start eligibility is checked in _start_drag.
            self.start_drag_pos = event.position().toPoint() # Use position() for QMouseEvent
            loguru.logger.debug(f"Mouse press stored at {self.start_drag_pos}, potential drag start.")
        else:
             self.start_drag_pos = None # Reset on non-left button press

    def mouseMoveEvent(self, event: QMouseEvent):
        """Initiates a drag operation if the mouse moves sufficiently after a press."""
        # This method should be called *from* the target widget's mouseMoveEvent
        if event.buttons() & Qt.MouseButton.LeftButton and self.start_drag_pos:
            # Use globalPos for distance calculation if widget might move during drag setup
            # distance = (event.globalPosition().toPoint() - self.start_global_drag_pos).manhattanLength()
            # Or use local position if widget is static during this phase
            distance = (event.position().toPoint() - self.start_drag_pos).manhattanLength()

            app_instance = QApplication.instance()
            if app_instance and distance >= app_instance.startDragDistance():
                loguru.logger.debug("Drag distance threshold reached, attempting to start drag...")
                # Store items *before* resetting start_drag_pos
                items_to_drag = self._get_items_from_delegate()
                # Reset position immediately to prevent multiple starts even if _start_drag fails
                current_start_pos = self.start_drag_pos
                self.start_drag_pos = None
                if items_to_drag:
                    self._start_drag(items_to_drag, current_start_pos)
                else:
                    loguru.logger.debug("Drag aborted: No items returned by delegate.")


    def _get_items_from_delegate(self) -> Optional[List[Any]]:
        """Safely gets items to drag from the delegate."""
        try:
            selected_items = self.target.get_selected_items_for_drag()
            if not isinstance(selected_items, list):
                loguru.error(f"Delegate 'get_selected_items_for_drag' must return a list, got {type(selected_items)}")
                return None
            return selected_items
        except Exception as e:
            loguru.logger.error(f"Error calling get_selected_items_for_drag on delegate '{self.target.__class__.__name__}': {e}", exc_info=True)
            return None

    def _start_drag(self, selected_items: List[Any], press_pos: QPoint):
        """Creates QMimeData based on selected items and executes the drag."""
        if not selected_items: # Should be caught by caller, but double-check
            loguru.logger.warning("_start_drag called with no selected items.")
            return

        loguru.logger.info(f"Preparing drag for {len(selected_items)} items.")

        # Determine data type and prepare mime data
        mime_data: Optional[QMimeData] = None
        first_item = selected_items[0]

        # Check types and prepare data using the helper
        try:
            if isinstance(first_item, MongoRecordWrapper):
                # Filter for consistency, though delegate should ideally provide uniform list
                mongo_items = [item for item in selected_items if isinstance(item, MongoRecordWrapper)]
                if mongo_items:
                    loguru.logger.debug(f"Preparing MongoWrapper data for {len(mongo_items)} items.")
                    mime_data = self.helper.prepare_mongo_wrapper_data(mongo_items)
            elif isinstance(first_item, DataTClass) or dataclasses.is_dataclass(first_item.__class__):
                 custom_items = [item for item in selected_items if isinstance(item, DataTClass) or dataclasses.is_dataclass(item.__class__)]
                 if custom_items:
                     loguru.logger.debug(f"Preparing CustomClass data for {len(custom_items)} items.")
                     mime_data = self.helper.prepare_custom_class_data(custom_items)
            elif isinstance(first_item, (str, pathlib.Path)):
                 path_items = [item for item in selected_items if isinstance(item, (str, pathlib.Path))]
                 if path_items:
                     loguru.logger.debug(f"Preparing File/Folder data for {len(path_items)} items.")
                     mime_data = self.helper.prepare_file_folder_data(path_items)
            # Add more type checks as needed
            else:
                 loguru.warning(f"No specific data preparation method for item type: {type(first_item).__name__}")

        except Exception as e:
             loguru.logger.error(f"Error preparing mime data: {e}", exc_info=True)
             return # Abort drag if data preparation fails

        if not mime_data or not mime_data.formats():
            loguru.logger.error(f"Failed to create valid mime data for selected items (first type: {type(first_item).__name__}).")
            return

        # Create and execute the drag object
        drag = QDrag(self.target)
        drag.setMimeData(mime_data)
        # Set pixmap (optional, shows representation during drag)
        # if hasattr(self.target, 'create_drag_pixmap'):
        #    pixmap = self.target.create_drag_pixmap(selected_items)
        #    drag.setPixmap(pixmap)
        #    drag.setHotSpot(pixmap.rect().center()) # Adjust hotspot if needed

        loguru.logger.debug(f"Executing drag with mime types: {mime_data.formats()}")

        # Execute the drag and capture the result action
        # Use Qt.DropAction constants
        result_action = drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

        # Log result and potentially notify delegate about move completion
        if result_action == Qt.DropAction.MoveAction:
            loguru.logger.info("Drag resulted in MoveAction (accepted by target).")
            # Check delegate for completion handler
            if hasattr(self.delegate, 'handle_drag_move_completed') and callable(getattr(self.delegate, 'handle_drag_move_completed')):
                try:
                    self.delegate.handle_drag_move_completed(selected_items)
                except Exception as e:
                    loguru.logger.error(f"Error calling delegate '{self.delegate.__class__.__name__}.handle_drag_move_completed': {e}", exc_info=True)
        elif result_action == Qt.DropAction.CopyAction:
            loguru.logger.info("Drag resulted in CopyAction (accepted by target).")
        else:
            loguru.logger.info("Drag operation cancelled or failed (ignored by target or user).")


# --- Example Integration (Commented Out) ---
#
# class FileSearchView(QWidget, DropTargetDelegate): # Inherit QWidget and implement protocol
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         # Assuming self.list_view_widget and self.query_text_edit exist
#         # self.list_view_widget = QListWidget() # Example
#         # self.query_text_edit = QLineEdit() # Example
#         self.drag_handler = DragDropHandler(self)
#         self.setAcceptDrops(True) # Ensure drops are enabled on the widget itself
#         # ... other initializations ...
#
#     # === Implement DragDropHandler event forwarding ===
#     def dragEnterEvent(self, event: QDragEnterEvent):
#         self.drag_handler.dragEnterEvent(event)
#
#     def dropEvent(self, event: QDropEvent):
#         self.drag_handler.dropEvent(event)
#
#     def mousePressEvent(self, event: QMouseEvent):
#         # Forward press event to handler *first* to store position
#         self.drag_handler.mousePressEvent(event)
#         # Then call base implementation or handle other press logic
#         super().mousePressEvent(event)
#
#     def mouseMoveEvent(self, event: QMouseEvent):
#         # Forward move event to handler *first* to potentially start drag
#         self.drag_handler.mouseMoveEvent(event)
#         # Then call base implementation or handle other move logic
#         super().mouseMoveEvent(event)
#
#     # === Implement DropTargetDelegate methods ===
#     def get_selected_items_for_drag(self) -> List[Any]:
#         loguru.logger.debug("Delegate: get_selected_items_for_drag called")
#         # --- Replace with actual logic to get selected data objects ---
#         # Example for QListWidget where items store data objects:
#         # selected_qt_items = self.list_view_widget.selectedItems()
#         # return [self.list_view_widget.itemWidget(item).data_object for item in selected_qt_items if hasattr(self.list_view_widget.itemWidget(item), 'data_object')]
#         # Example: Return dummy data for testing
#         # return [FileRecord()] # Or [TagRecord()] or [MyCustomData()]
#         return [] # Placeholder - MUST BE IMPLEMENTED
#
#     def handle_dropped_files(self, paths: List[pathlib.Path], event: QDropEvent):
#         loguru.logger.info(f"Delegate: Handling {len(paths)} dropped files/folders.")
#         if paths:
#             # Example: Set query text based on the first path dropped
#             first_path = paths[0]
#             # self.query_text_edit.setText(f'path REGEX "{first_path.name}"') # Example action
#             # self.find() # Example action
#             loguru.logger.info(f"Example action: Set query for path '{first_path.name}'")
#             event.accept() # Accept the event if handled
#         else:
#             event.ignore()
#
#     def handle_dropped_mongo_wrappers(self, objects: List[MongoRecordWrapper], event: QDropEvent):
#         loguru.logger.info(f"Delegate: Handling {len(objects)} dropped MongoWrappers.")
#         if not objects:
#             event.ignore()
#             return
#
#         first_obj = objects[0]
#         if isinstance(first_obj, TagRecord):
#             tag_names = [obj.fullName for obj in objects if isinstance(obj, TagRecord) and hasattr(obj, 'fullName')]
#             if tag_names:
#                 loguru.logger.info(f"Applying dropped Tags: {tag_names}")
#                 # Example: Set query text based on the first tag dropped
#                 # self.query_text_edit.setText(f'tags REGEX "{tag_names[0]}"')
#                 # self.find()
#                 loguru.logger.info(f"Example action: Set query for tag '{tag_names[0]}'")
#                 event.accept()
#             else:
#                 event.ignore()
#         elif isinstance(first_obj, FileRecord):
#             loguru.logger.info(f"Applying dropped FileRecords (IDs: {[obj._id for obj in objects]})")
#             # Example: Add file paths or IDs to somewhere, maybe trigger loading
#             event.accept()
#         else:
#             loguru.warning(f"Unhandled MongoWrapper type in delegate: {type(first_obj).__name__}")
#             event.ignore()
#
#     def handle_dropped_custom_data(self, objects: List[Any], event: QDropEvent):
#         loguru.logger.info(f"Delegate: Handling {len(objects)} dropped custom data objects.")
#         if objects:
#             # Example: Process objects based on their type
#             loguru.logger.info(f"Example action: Received custom data of type {type(objects[0]).__name__}")
#             event.accept()
#         else:
#             event.ignore()
#
#     # Optional: Customize accepted types
#     # def accepts_drop_type(self, drop_type: str) -> bool:
#     #     # Only accept files and mongo wrappers
#     #     return drop_type in ['files', 'mongo_wrapper']
