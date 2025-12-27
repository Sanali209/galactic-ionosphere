from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap, QPen
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QPushButton, QVBoxLayout, \
    QGraphicsRectItem, QGraphicsItem, QStackedLayout, QStackedWidget, QSizePolicy


class DetectedObjectItem(QGraphicsRectItem):
    def __init__(self, rect, label, parent=None):
        super().__init__(rect, parent)
        self.label = label
        self.setFlags(
            QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPen(QPen(Qt.yellow, 2))
        self.setBrush(Qt.transparent)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(Qt.yellow)
        painter.drawText(self.rect().topLeft(), self.label)


class TransparentOverlayWidget(QWidget):
    """
    Специальный виджет-оверлей, который пересылает события мыши нижележащему
    QGraphicsView, если событие не попало ни на один интерактивный дочерний элемент.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        # Не устанавливаем WA_TransparentForMouseEvents на весь виджет,
        # чтобы его дочерние кнопки могли получать события.
        # Вместо этого переопределяем обработчики событий мыши.

    def mousePressEvent(self, event):
        if self.childAt(event.pos()) is None:
            # Передаём событие нижележащему виджету (QGraphicsView)
            if hasattr(self.parent(), 'view'):
                self.parent().view.mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.childAt(event.pos()) is None:
            if hasattr(self.parent(), 'view'):
                self.parent().view.mouseMoveEvent(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.childAt(event.pos()) is None:
            if hasattr(self.parent(), 'view'):
                self.parent().view.mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)


class ImageView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- QGraphicsView для отображения изображения ---
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # --- Overlay widget для размещения кнопок ---
        self.overlay_widget = TransparentOverlayWidget(self)
        self.overlay_widget.setStyleSheet("background: transparent;")
        # Создаем один основной layout для overlay_widget
        self.overlay_layout = QVBoxLayout(self.overlay_widget)
        self.overlay_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_layout.setSpacing(0)

        # Контейнер для верхних элементов (например, зум-кнопок)
        self.top_container = TransparentOverlayWidget(self.overlay_widget)
        self.top_container.setStyleSheet("background: transparent;")
        self.top_layout = QVBoxLayout(self.top_container)
        self.top_layout.setContentsMargins(10, 10, 10, 10)
        self.top_layout.setSpacing(5)
        self.top_layout.setAlignment(Qt.AlignTop)
        # Создаем кнопки зума и Fit
        self.zoom_in_button = QPushButton("Zoom +", self.top_container)
        self.zoom_in_button.setFixedSize(100, 30)
        self.zoom_out_button = QPushButton("Zoom -", self.top_container)
        self.zoom_out_button.setFixedSize(100, 30)
        self.fit_button = QPushButton("Fit", self.top_container)
        self.fit_button.setFixedSize(100, 30)
        self.top_layout.addWidget(self.zoom_in_button)
        self.top_layout.addWidget(self.zoom_out_button)
        self.top_layout.addWidget(self.fit_button)

        # Контейнер для нижних элементов (например, тестовой кнопки)
        self.bottom_container = TransparentOverlayWidget(self.overlay_widget)
        self.bottom_container.setStyleSheet("background: transparent;")
        self.bottom_layout = QVBoxLayout(self.bottom_container)
        self.bottom_layout.setContentsMargins(10, 10, 10, 10)
        self.bottom_layout.setSpacing(5)
        self.bottom_layout.setAlignment(Qt.AlignBottom)
        self.testing_button = QPushButton("Testing", self.bottom_container)
        self.testing_button.setFixedSize(100, 30)
        self.bottom_layout.addWidget(self.testing_button)

        # Добавляем верхний контейнер, растягивающий spacer и нижний контейнер в основной overlay layout
        self.overlay_layout.addWidget(self.top_container)
        self.overlay_layout.addStretch()
        self.overlay_layout.addWidget(self.bottom_container)

        # Подключаем сигналы
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.fit_button.clicked.connect(self.fit_in_view)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Растягиваем QGraphicsView и overlay_widget на всю область родительского виджета
        rect = self.rect()
        self.view.setGeometry(rect)
        self.overlay_widget.setGeometry(rect)




    def load_image(self, path):
        pixmap = QPixmap(path)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(pixmap.rect())
        self.fit_in_view()

    def zoom_in(self):
        self.view.scale(1.25, 1.25)

    def zoom_out(self):
        self.view.scale(0.8, 0.8)

    def fit_in_view(self):
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def add_detected_object(self, rect, label):
        item = DetectedObjectItem(rect, label)
        self.scene.addItem(item)
        return item