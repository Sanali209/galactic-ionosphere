import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout,
    QLabel)

from SLM import Allocator
from SLM.files_data_cache.thumbnail import ImageThumbCache
from SLM.files_db.components.File_record_wraper import FileRecord
from SLM.pySide6Ext.widgets.ImageWidget import ImageOverlayWidget


class ImageDisplayWidget(QWidget ):
    """Widget to display an image and its ratings."""

    clicked = Signal()

    def __init__(self, title="Image",w=600, h=600):
        super().__init__()
        self.setMinimumSize(w, h)  # Ensure a decent minimum size
        layout = QVBoxLayout(self)

        self.title_label = QLabel(title)
        self.title_label.setFixedHeight(25)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.image_label = ImageOverlayWidget(w - 25, h - 25)
        self.image_label.setMinimumSize(w-30, h-30)  # Min size for image area
        layout.addWidget(self.image_label)

        self.user_rating_label = QLabel("User Rating: N/A")
        self.user_rating_label.setFixedHeight(30)
        self.user_rating_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_rating_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 50);"
            "color: white;"
            "padding: 1px;"
            "font-size: 12px;"
        )
        self.image_label.overlay_layout.addWidget(self.user_rating_label)

        self.predicted_rating_label = QLabel("Pred. Rating: N/A")
        self.predicted_rating_label.setFixedHeight(30)
        self.predicted_rating_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.predicted_rating_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 50);"
            "color: white;"
            "padding: 1px;"
            "font-size: 12px;"
        )
        self.image_label.overlay_layout.addWidget(self.predicted_rating_label)

    def set_size(self, width: int, height: int):
        self.setMinimumSize(width, height)
        self.image_label.setFixedSize(width - 100, height - 200)

    def set_image(self, file_record: FileRecord | None):
        if file_record and file_record.full_path and os.path.exists(file_record.full_path):
            tumb_path = Allocator.res.get_by_type_one(ImageThumbCache).get_thumb(file_record.full_path)
            self.image_label.load_image(tumb_path)
        else:
            pass

    def set_ratings(self, user_rating: int | None, predicted_rating: str | None):  # Takes int for display logic
        self.user_rating_label.setText(f"User Rating: {user_rating if user_rating is not None else 'N/A'}")
        self.predicted_rating_label.setText(
            predicted_rating if predicted_rating is not None else "Pred. Rating: N/A")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)  # Call parent to ensure default behavior is preserved
