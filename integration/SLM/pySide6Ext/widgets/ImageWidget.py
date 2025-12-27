from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PySide6.QtGui import QPixmap, QImage, QPalette
from PySide6.QtCore import Qt, QThread, Signal, Slot
import sys
import io

from SLM.files_data_cache.pool import PILPool


class ImageLoaderThread(QThread):
    """
    Thread to load and optionally resize an image asynchronously.
    Emits the loaded QPixmap via the `image_loaded` signal.
    """
    image_loaded = Signal(QPixmap)

    def __init__(self, image_path, width=None, height=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.width = width
        self.height = height

    def run(self):
        try:
            # Load and convert image using PIL
            image = PILPool.get_pil_image(self.image_path)
            # Resize if width and height are provided
            #if self.width and self.height:
            #image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)

            # Convert PIL image to QImage
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            qimage = QImage()
            qimage.loadFromData(image_bytes.read())
            if qimage.isNull():
                raise ValueError("Invalid image file")
            pixmap = QPixmap.fromImage(qimage)
            # Resize pixmap while keeping aspect ratio

            if self.width and self.height:
                pixmap = pixmap.scaled(self.width, self.height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation
                                       )
            self.image_loaded.emit(pixmap)
        except Exception as e:
            print("Error loading image:", e)


class ImageOverlayWidget(QWidget):
    """
    A widget that displays a background image loaded asynchronously
    with an interactive transparent overlay on top.
    """

    def __init__(self, width=256, height=256, parent=None):
        super().__init__(parent)
        self.fixed_width = width
        self.fixed_height = height
        if width>0 and height > 0:
            self.setFixedSize(width, height)
        # Background image label (covers entire widget)
        self.image_label = QLabel(self)
        #self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Overlay widget for controls; set transparent background
        self.overlay_widget = QWidget(self)
        self.overlay_widget.setStyleSheet("background: transparent;")
        self.overlay_layout = QVBoxLayout(self.overlay_widget)
        self.overlay_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.overlay_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_widget.setAttribute(Qt.WA_AlwaysShowToolTips, True)

    def resizeEvent(self, event):
        """
        Ensure that both the image label and the overlay widget
        always fill the entire widget area.
        """
        super().resizeEvent(event)
        rect = self.rect()
        self.image_label.setGeometry(rect)
        self.overlay_widget.setGeometry(rect)

    def setToolTip(self, hint):
        super().setToolTip(hint)

    @Slot(QPixmap)
    def update_image(self, pixmap):
        """Update the background image with the loaded pixmap."""
        self.image_label.setPixmap(pixmap)

    def load_image(self, image_path):
        '''sinchronous load image'''
        image = PILPool.get_pil_image(image_path)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        qimage = QImage()
        qimage.loadFromData(image_bytes.read())
        if qimage.isNull():
            raise ValueError("Invalid image file")
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.fixed_width, self.fixed_height,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation
                               )
        self.update_image(pixmap)

    def load_from_pixmap(self, pixmap):
        pixmap = pixmap.scaled(self.fixed_width, self.fixed_height,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation
                               )
        self.update_image(pixmap)

    def load_image_async(self, image_path):
        """
        Load an image asynchronously with an optional size.
        Once loaded, the background image is updated.
        """
        self.thread = ImageLoaderThread(image_path, self.fixed_width, self.fixed_height)
        self.thread.image_loaded.connect(self.update_image)
        self.thread.start()

    def get_layout(self):
        return self.overlay_layout


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImageOverlayWidget()
    window.resize(500, 400)
    layout = window.get_layout()
    # Example interactive controls on the overlay:
    text_label = QLabel("Rating: ★★★★☆\nSize: ???\nTags: ???", window.overlay_widget)
    text_label.setStyleSheet(
        "background-color: rgba(0, 0, 0, 128);"
        "color: white;"
        "padding: 5px;"
        "font-size: 14px;"
    )
    window.overlay_layout.addWidget(text_label)

    button = QPushButton("Reload Image", window.overlay_widget)
    window.overlay_layout.addWidget(button)
    button.clicked.connect(lambda: window.load_image_async("example.jpg"))
    # Start asynchronous loading (image will be resized to 500x400)
    window.load_image_async(r"C:\Users\User\Pictures\Screenshot_1.png")
    window.show()

    sys.exit(app.exec())
