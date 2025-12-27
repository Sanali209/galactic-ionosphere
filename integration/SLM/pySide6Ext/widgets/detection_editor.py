from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QWidget, QGraphicsItem,
    QGraphicsEllipseItem, QLineEdit, QCompleter
)
from PySide6.QtGui import QPixmap, QPen, QColor, QCursor
from PySide6.QtCore import QRectF, Qt, Signal, QPointF, QSizeF
import sys


###############################################################################
# Dialog to edit text with autocompletion
###############################################################################
class LabelEditDialog(QDialog):
    """
    A simple dialog with a QLineEdit that supports autocompletion.
    """

    def __init__(self, current_text, completions, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Edit Label")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Create a line edit with a QCompleter
        self.line_edit = QLineEdit(self)
        completer = QCompleter(completions, self)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.line_edit.setCompleter(completer)
        self.line_edit.setText(current_text)

        layout.addWidget(self.line_edit)

        # Simple "OK" button
        self.button_ok = QPushButton("OK", self)
        self.button_ok.clicked.connect(self.accept)
        layout.addWidget(self.button_ok)

        self.setLayout(layout)

    def getText(self):
        return self.line_edit.text()


###############################################################################
# Editable label item with double-click to show LabelEditDialog
###############################################################################
class EditableLabelItem(QGraphicsTextItem):
    """
    A QGraphicsTextItem that, on double-click, opens a dialog for text editing
    with autocompletion provided.
    """

    def __init__(self, initial_text, completions, parent=None):
        super().__init__(initial_text, parent)
        self.completions = completions or []

        # Example style: make the text color blue.
        self.setDefaultTextColor(QColor('blue'))

        # Allow item to be selected, if you like
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def mouseDoubleClickEvent(self, event):
        """
        On double-click, open a dialog to edit the text with autocompletion.
        """
        # Current text
        current_text = self.toPlainText()

        # Create & run the dialog
        dialog = LabelEditDialog(current_text, self.completions)
        if dialog.exec() == QDialog.Accepted:
            new_text = dialog.getText()
            self.setPlainText(new_text)

        # Call the base implementation (optional)
        super().mouseDoubleClickEvent(event)


###############################################################################
# HandleItem and ResizableRectItem (unchanged except for using EditableLabelItem)
###############################################################################
class HandleItem(QGraphicsEllipseItem):
    def __init__(self, corner, parent_rect_item):
        super().__init__(QRectF(0, 0, 8, 8), parent_rect_item)
        self.corner = corner
        self.parent_rect_item = parent_rect_item

        self.setBrush(QColor('blue'))
        self.setPen(QPen(Qt.black, 1))
        self.setCursor(QCursor(Qt.SizeFDiagCursor))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(1000)

    def mouseMoveEvent(self, event):
        parent = self.parent_rect_item
        # Convert the mouse position into the parent's local coordinates
        new_scene_pos = self.mapToScene(event.pos())
        new_parent_pos = parent.mapFromScene(new_scene_pos)

        # Adjust the parent's rectangle
        rect = parent.rect()
        if self.corner == "bottom_right":
            rect.setBottomRight(new_parent_pos)
        elif self.corner == "top_left":
            rect.setTopLeft(new_parent_pos)

        # Enforce a valid rectangle
        rect = rect.normalized()

        # Call our overridden setRect on the parent
        parent.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # Final alignment
        self.parent_rect_item.updateHandles()


class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, rect, label_item, parent=None):
        super().__init__(rect, parent)
        self.xc = rect.x()
        self.yc = rect.y()
        self.wc = rect.width()
        self.hc = rect.height()
        self.label_item = label_item
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        self.setPen(QPen(QColor('red'), 2))
        self.setBrush(QColor(255, 0, 0, 50))
        self.handles = []

        # Force initial label position
        self.updateLabelPosition(self.pos())

    def setRect(self, new_rect):
        """
        Override setRect() to also update label & handles whenever the rect changes.
        """
        # Only update if there's an actual change
        if new_rect == self.rect():
            return

        self.prepareGeometryChange()
        super().setRect(new_rect)
        self.wc = new_rect.width()
        self.hc = new_rect.height()
        print(new_rect)
        self.updateHandles()
        # Also move the label above the rectangle
        self.updateLabelPosition(self.pos())

    def addHandlesToScene(self):
        """Create handles as child items of this rectangle."""
        if self.scene() and not self.handles:
            # top-left
            self.handles.append(HandleItem("top_left", self))
            # bottom-right
            self.handles.append(HandleItem("bottom_right", self))
            self.updateHandles()

    def updateHandles(self):
        """Position each handle at the correct corner in local coords."""

        rect = self.rect()
        handle_size = 8
        for handle in self.handles:
            if handle.corner == "top_left":
                x = rect.topLeft().x() - handle_size / 2
                y = rect.topLeft().y() - handle_size / 2
                handle.setPos(x, y)
            elif handle.corner == "bottom_right":
                x = rect.bottomRight().x() - handle_size / 2
                y = rect.bottomRight().y() - handle_size / 2
                handle.setPos(x, y)


    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.updateLabelPosition(value)
            self.updateHandles()
        elif change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            self.addHandlesToScene()

        return super().itemChange(change, value)

    def updateLabelPosition(self, new_item_pos):
        """
        Place the label slightly above the rectangle, in scene coordinates.
        """
        # rect's local top-left:
        top_left_local = self.rect().topLeft()
        # Convert to scene
        top_left_scene = self.mapToScene(top_left_local)
        # Then offset above:
        self.label_item.setPos(top_left_scene + QPointF(0, -20))
        self.xc = top_left_scene.x()
        self.yc = top_left_scene.y()
        print(self.xc, self.yc, self.wc, self.hc)


###############################################################################
# DetectionEditorWidget and DetectionEditorDialog
###############################################################################
class DetectionEditorWidget(QWidget):
    detectionUpdated = Signal(int, QRectF)

    def __init__(self, image_path, detections, completions=None, parent=None):
        """
        :param image_path: Path to the image file to display.
        :param detections: List[dict] with keys:
            'id', 'label', 'rect'=(x,y,x1,y1)
        :param completions: A list of strings for label autocompletion.
        """
        super().__init__(parent)
        self.image_path = image_path
        self.detections = detections
        self.selected_detection = None
        self.completions = completions or []  # store auto-complete strings

        self.initUI()

    def initUI(self):
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)

        # Load and display the image
        pixmap = QPixmap(self.image_path)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        # Add detection rectangles and labels
        self.rect_items = {}
        for detection in self.detections:
            rect = QRectF(*detection['rect'])

            # Instead of QGraphicsTextItem, use EditableLabelItem
            label_item = EditableLabelItem(detection['label'], self.completions)
            self.scene.addItem(label_item)

            rect_item = ResizableRectItem(rect, label_item)
            rect_item.setData(0, detection['id'])
            self.scene.addItem(rect_item)
            rect_item.addHandlesToScene()

            self.rect_items[detection['id']] = (rect_item, label_item)

        self.scene.selectionChanged.connect(self.onSelectionChanged)
        self.view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.view.setSceneRect(self.image_item.boundingRect())

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

    def onSelectionChanged(self):
        selected_items = self.scene.selectedItems()
        if selected_items:
            self.selected_detection = selected_items[0]
        else:
            self.selected_detection = None

    def updateDetection(self):
        if self.selected_detection:
            rect = self.selected_detection.rect()
            label_id = self.selected_detection.data(0)
            self.detectionUpdated.emit(label_id, rect)

    def setDetections(self, detections):
        self.detections = detections
        self.scene.clear()

        pixmap = QPixmap(self.image_path)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        self.rect_items = {}
        for detection in self.detections:
            rect = QRectF(*detection['rect'])
            label_item = EditableLabelItem(detection['label'], self.completions)
            self.scene.addItem(label_item)

            rect_item = ResizableRectItem(rect, label_item)
            rect_item.setData(0, detection['id'])
            self.scene.addItem(rect_item)
            rect_item.addHandlesToScene()

            self.rect_items[detection['id']] = (rect_item, label_item)

    def getDetections(self):
        """
        Return the current state of detections.
        Each detection dict: {'id', 'label', 'rect'=(x, y, x+w, y+h)}
        """
        results = []
        for label_id, (rect_item, label_item) in self.rect_items.items():
            rect = rect_item.rect()
            label_text = label_item.toPlainText()
            x, y, w, h = rect_item.xc, rect_item.yc, rect_item.wc, rect_item.hc

            results.append({
                'id': label_id,
                'label': label_text,
                'rect': (int(x), int(y), int(w), int(h))
            })
        return results


class DetectionEditorDialog(QDialog):
    def __init__(self, image_path, detections, completions=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Editor")
        self.resize(800, 600)

        # Pass completions to the editor:
        self.editor = DetectionEditorWidget(image_path, detections, completions, self)
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.saveDetections)

        layout = QVBoxLayout()
        layout.addWidget(self.editor)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def saveDetections(self):
        detections = self.editor.getDetections()
        print("Updated Detections:", detections)
        self.accept()

    def get_detections(self):
        return self.editor.getDetections()


###############################################################################
# Main Entry
###############################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)

    image_path = r"E:\rawimagedb\repository\nsfv repo\drawn\_site rip\Reiq.ws + Jigglygirls.com\mix\Akame Ga Kill! - Akame 01 Clean 02.jpg"
    detections = [
        {'id': 1, 'label': 'Label 1', 'rect': (50, 50, 150, 150)},
    ]
    # Example completions for the label editor
    completions = [
        "Person", "Car", "Dog", "Cat", "Tree", "Building",
        "Label 1", "Label 2", "SomethingElse"
    ]

    dialog = DetectionEditorDialog(image_path, detections, completions)
    dialog.exec()
    print("Dialog closed with OK.")
    updated_detections = dialog.get_detections()
    dialog = DetectionEditorDialog(image_path, updated_detections, completions)
    dialog.exec()
    sys.exit(app.exec())
