import os
import sys
import pathlib
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

from applications.image_graph_qt.drag_drop_handler import DragDropHandler
from menu_actions import MenuActions
from model_config import model_sigma_config
from qtWidjets import ImageDisplayWidget
from six_image_ranking_view import SixImageRankingWidget

# Add the Python directory to the path so SLM module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(current_dir, "..", "..", "..")
sys.path.insert(0, python_dir)
from analitics_manager import analytics_manager
from anotation_tool import TrueSkillAnnotationRecordTools
from comparison_service import ComparisonService, BaseAnnotationMode
from constants import MODEL_SIGMA, DEFAULT_MU, AUTO_LOAD_ROUND, AUTO_WIN_PAIRS, DEFAULT_RATING
from data_manager import data_manager
from rating_helpers import RatingHelpers
from rating_operations import RatingOperations
from data_io_operations import DataIOOperations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QTabWidget, QCheckBox, QInputDialog,
    QDialog, QListWidget, QListWidgetItem, QProgressDialog
)

from tqdm import tqdm
from PySide6.QtWidgets import QComboBox, QListView, QMenu
from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, QRect, QEvent
from PySide6.QtGui import QPainter, QPixmap, QFontMetrics, QDropEvent, QDragEnterEvent
from PySide6.QtWidgets import QStyledItemDelegate
from PySide6.QtWidgets import QStyle

from SLM.appGlue.core import Allocator
from SLM.pySide6Ext.pySide6Q import PySide6GlueApp, PySide6GlueWidget
from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from SLM.files_db.components.File_record_wraper import FileRecord
from loguru import logger
import trueskill

# Import drag and drop handler from parent directory
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



class ImageRatingModel(QAbstractListModel):
    """Model for displaying rated images - READ ONLY"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records: List[AnnotationRecord] = []
        self._last_refresh_time = 0

    def rowCount(self, parent=QModelIndex()):
        return len(self.records)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.records)):
            return None
        if role == Qt.DisplayRole:
            return self.records[index.row()]
        return None

    def load_data(self, anchors_only=False):
        """Load data from the centralized data manager

        Args:
            anchors_only: If True, only load records marked as anchors
        """
        try:
            self.beginResetModel()
            # Force reload of manual_voted_list from database
            data_manager.load_manual_voted_list()

            valid_records = [r for r in data_manager.manual_voted_list if r.file]

            # Apply anchor filtering if requested
            if anchors_only:
                valid_records = [r for r in valid_records if r.get_field_val("ankor", False)]

            # Sort by TrueSkill rating (mu - 3*sigma) in descending order
            valid_records.sort(
                key=lambda r: data_manager.get_record_trueskill_key(r),
                reverse=True
            )
            self.records = valid_records
            self.endResetModel()

            filter_desc = " (anchors only)" if anchors_only else ""
            logger.debug(f"Loaded {len(self.records)} records into model{filter_desc}")
            return len(self.records)

        except Exception as e:
            logger.error(f"Error loading data into model: {e}")
            self.records = []
            self.endResetModel()
            return 0

    def sort_records(self):
        """Sort records by TrueSkill rating"""
        try:
            self.beginResetModel()
            self.records.sort(
                key=lambda r: data_manager.get_record_trueskill_key(r),
                reverse=True
            )
            self.endResetModel()
            logger.debug(f"Sorted {len(self.records)} records by rating")
        except Exception as e:
            logger.error(f"Error sorting records: {e}")

    def get_record_at(self, index: int) -> Optional[AnnotationRecord]:
        """Get record at specific index"""
        if 0 <= index < len(self.records):
            return self.records[index]
        return None


class ImageRatingDelegate(QStyledItemDelegate):
    """Delegate for rendering image items"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumb_size = 150
        self.padding = 5

    def paint(self, painter: QPainter, option, index: QModelIndex):
        record = index.data(Qt.DisplayRole)
        if not record:
            return

        painter.save()

        # Bounding rect
        rect = option.rect

        # Draw selection highlight
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, option.palette.highlight())

        # Draw anchor indicator
        if record.get_field_val("ankor", False):
            painter.save()
            pen = painter.pen()
            pen.setColor(Qt.red)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()

        # Draw thumbnail
        thumb_path = record.file.get_thumb("medium")
        if thumb_path and os.path.exists(thumb_path):
            pixmap = QPixmap(thumb_path)
            scaled_pixmap = pixmap.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_rect = QRect(rect.x() + self.padding, rect.y() + self.padding, self.thumb_size, self.thumb_size)
            painter.drawPixmap(img_rect, scaled_pixmap)
        else:
            painter.drawText(rect, Qt.AlignCenter, "No Thumb")

        # Prepare for text drawing
        font = painter.font()
        metrics = QFontMetrics(font)
        text_y = rect.y() + self.thumb_size + self.padding * 3

        # Draw file name
        name_rect = QRect(rect.x() + self.padding, text_y, rect.width() - self.padding * 2, metrics.height())
        elided_text = metrics.elidedText(record.file.name, Qt.ElideRight, name_rect.width())
        painter.drawText(name_rect, Qt.AlignLeft, elided_text)
        text_y += metrics.height()

        # Draw rating details
        mu = record.get_field_val("avg_rating", DEFAULT_MU)
        sigma = record.get_field_val("trueskill_sigma", MODEL_SIGMA)
        rating = mu - 3 * sigma

        painter.drawText(QRect(rect.x() + self.padding, text_y, rect.width(), metrics.height()),
                         f"Rating: {rating:.2f}")
        text_y += metrics.height()
        painter.drawText(QRect(rect.x() + self.padding, text_y, rect.width(), metrics.height()), f"μ: {mu:.2f}")
        text_y += metrics.height()
        painter.drawText(QRect(rect.x() + self.padding, text_y, rect.width(), metrics.height()), f"σ: {sigma:.2f}")

        painter.restore()

    def sizeHint(self, option, index: QModelIndex):
        return QSize(200, 220)


class CachedRelationsEditor(QDialog):
    """Dialog for editing cached win/loss relationships"""

    def __init__(self, record: AnnotationRecord, parent=None):
        super().__init__(parent)
        self.record = record
        self.setWindowTitle(f"Edit Cached Relations - {record.file.name if record.file else 'Unknown'}")
        self.setGeometry(100, 100, 800, 600)
        self.setModal(True)

        self.init_ui()
        self.refresh_data()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)

        # Header with image info
        header_layout = QHBoxLayout()
        header_label = QLabel(f"Editing relationships for: {self.record.file.name if self.record.file else 'Unknown'}")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(header_label)

        # Stats label
        self.stats_label = QLabel("Stats: Loading...")
        header_layout.addWidget(self.stats_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Tab widget for wins/losses
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Wins tab
        self.wins_widget = QWidget()
        wins_layout = QVBoxLayout(self.wins_widget)
        self.wins_list = QListWidget()
        wins_layout.addWidget(QLabel("Images this record has beaten:"))
        wins_layout.addWidget(self.wins_list)
        self.tabs.addTab(self.wins_widget, "Wins")

        # Losses tab
        self.losses_widget = QWidget()
        losses_layout = QVBoxLayout(self.losses_widget)
        self.losses_list = QListWidget()
        losses_layout.addWidget(QLabel("Images this record has lost to:"))
        losses_layout.addWidget(self.losses_list)
        self.tabs.addTab(self.losses_widget, "Losses")

        # Buttons
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_data)
        button_layout.addWidget(self.refresh_button)

        button_layout.addStretch()

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def refresh_data(self):
        """Refresh the wins and losses data"""
        try:
            record_id = str(self.record.id)

            # Get wins and losses from cache
            wins = data_manager.cache_manager.get_wins_for_record(record_id)
            losses = data_manager.cache_manager.get_losses_for_record(record_id)

            # Update stats
            self.stats_label.setText(f"Wins: {len(wins)} | Losses: {len(losses)}")

            # Populate wins list
            self.populate_relations_list(self.wins_list, wins, "win")

            # Populate losses list
            self.populate_relations_list(self.losses_list, losses, "loss")

        except Exception as e:
            logger.error(f"Error refreshing cached relations data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to refresh data: {e}")

    def populate_relations_list(self, list_widget: QListWidget, relations: List[Tuple[str, str]], relation_type: str):
        """Populate a list widget with relation data"""
        list_widget.clear()

        for opponent_id, pair_key in relations:
            try:
                # Find the opponent record
                opponent_records = [r for r in data_manager.manual_voted_list if str(r.id) == opponent_id]
                if not opponent_records:
                    continue

                opponent = opponent_records[0]
                if not opponent.file:
                    continue

                # Create list item
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 5, 5, 5)

                # Thumbnail (if available)
                thumb_label = QLabel()
                try:
                    from SLM.files_data_cache.thumbnail import ImageThumbCache
                    from SLM import Allocator
                    thumb_path = Allocator.res.get_by_type_one(ImageThumbCache).get_thumb(opponent.file.full_path)
                    if thumb_path and os.path.exists(thumb_path):
                        pixmap = QPixmap(thumb_path)
                        scaled_pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        thumb_label.setPixmap(scaled_pixmap)
                    else:
                        thumb_label.setText("No\nThumb")
                        thumb_label.setStyleSheet("border: 1px solid gray;")
                except Exception as e:
                    # Fallback to old method if new method fails
                    thumb_path = opponent.file.get_thumb("small")
                    if thumb_path and os.path.exists(thumb_path):
                        pixmap = QPixmap(thumb_path)
                        scaled_pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        thumb_label.setPixmap(scaled_pixmap)
                    else:
                        thumb_label.setText("No\nThumb")
                        thumb_label.setStyleSheet("border: 1px solid gray;")
                thumb_label.setFixedSize(64, 64)
                item_layout.addWidget(thumb_label)

                # Info section
                info_layout = QVBoxLayout()

                # Filename
                name_label = QLabel(opponent.file.name)
                name_label.setStyleSheet("font-weight: bold;")
                info_layout.addWidget(name_label)

                # Rating info
                mu = opponent.get_field_val("avg_rating", DEFAULT_MU)
                sigma = opponent.get_field_val("trueskill_sigma", MODEL_SIGMA)
                rating = mu - 3 * sigma
                rating_label = QLabel(f"Rating: {rating:.2f} (μ={mu:.2f}, σ={sigma:.2f})")
                info_layout.addWidget(rating_label)

                # Pair key info
                key_label = QLabel(f"Cache Key: {pair_key}")
                key_label.setStyleSheet("color: gray; font-size: 10px;")
                info_layout.addWidget(key_label)

                item_layout.addLayout(info_layout)
                item_layout.addStretch()

                # Delete button
                delete_button = QPushButton("Delete")
                delete_button.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; }")
                delete_button.clicked.connect(lambda checked, oid=opponent_id: self.delete_relationship(oid))
                item_layout.addWidget(delete_button)

                # Add to list
                list_item = QListWidgetItem()
                list_item.setSizeHint(item_widget.sizeHint())
                list_widget.addItem(list_item)
                list_widget.setItemWidget(list_item, item_widget)

            except Exception as e:
                logger.error(f"Error creating list item for opponent {opponent_id}: {e}")
                continue

    def delete_relationship(self, opponent_id: str):
        """Delete a cached relationship"""
        try:
            reply = QMessageBox.question(
                self,
                'Confirm Deletion',
                f'Delete cached comparison result with this image?\n\nThis will remove the win/loss relationship from cache.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Delete the cached pair relationship
                data_manager.cache_manager.delete_cached_pair_relationship(str(self.record.id), opponent_id)

                # Refresh the display
                self.refresh_data()

                QMessageBox.information(self, "Success", "Cached relationship deleted successfully.")

        except Exception as e:
            logger.error(f"Error deleting relationship with {opponent_id}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to delete relationship: {e}")


class ImageListRaterWidget(QWidget):
    """List view widget for displaying rated images"""

    def __init__(self):
        super().__init__()
        self.model = ImageRatingModel()
        self.anchors_only = False  # Filter state
        self.init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Initialize drag and drop handler
        self.drag_handler = DragDropHandler(self)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Add refresh and sort buttons
        refresh_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Data")
        self.refresh_button.clicked.connect(self.refresh_data)
        refresh_layout.addWidget(self.refresh_button)

        self.sort_button = QPushButton("Sort by Rating")
        self.sort_button.clicked.connect(self.sort_data)
        refresh_layout.addWidget(self.sort_button)

        # Add anchor filter checkbox
        self.anchors_checkbox = QCheckBox("Show Anchors Only")
        self.anchors_checkbox.stateChanged.connect(self.on_filter_changed)
        refresh_layout.addWidget(self.anchors_checkbox)

        self.status_label = QLabel("List View - Click Refresh to load data")
        refresh_layout.addWidget(self.status_label)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)

        self.list_view = QListView()
        self.delegate = ImageRatingDelegate()

        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setFlow(QListView.Flow.LeftToRight)
        self.list_view.setWrapping(True)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setGridSize(QSize(210, 230))

        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)

        self.list_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_view.customContextMenuRequested.connect(self.show_context_menu)
        self.list_view.installEventFilter(self)
        layout.addWidget(self.list_view)

    def on_filter_changed(self):
        """Handle anchor filter checkbox state change"""
        self.anchors_only = self.anchors_checkbox.isChecked()
        self.refresh_data()

    def refresh_data(self):
        """Refresh data from the centralized data manager"""
        try:
            count = self.model.load_data(self.anchors_only)
            filter_mode = " (anchors only)" if self.anchors_only else ""
            self.status_label.setText(f"Data refreshed - {count} records loaded{filter_mode}")
            logger.debug(f"Refreshed list view with {count} records")
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to refresh data: {e}")

    def sort_data(self):
        """Sort data by TrueSkill rating"""
        try:
            self.model.sort_records()
            count = len(self.model.records)
            self.status_label.setText(f"Data sorted by rating - {count} records")
            logger.debug(f"Sorted list view with {count} records")
        except Exception as e:
            logger.error(f"Error sorting data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to sort data: {e}")

    def eventFilter(self, source, event):
        if source is self.list_view and event.type() == QEvent.KeyPress:
            selected_indexes = self.list_view.selectedIndexes()
            if selected_indexes:
                index = selected_indexes[0]
                if event.key() == Qt.Key_Plus or event.text() == '+':
                    self.adjust_mu(index, 1)
                    return True
                elif event.key() == Qt.Key_Minus or event.text() == '-':
                    self.adjust_mu(index, -1)
                    return True
        return super().eventFilter(source, event)

    def show_context_menu(self, pos):
        index = self.list_view.indexAt(pos)
        selected_indexes = self.list_view.selectedIndexes()
        context_menu = QMenu(self)

        # Check if multiple items are selected
        if len(selected_indexes) >= 2:
            mark_equal_action = context_menu.addAction(f"Mark {len(selected_indexes)} Selected as Equal")
            context_menu.addSeparator()

        if index.isValid():
            record = self.model.data(index, Qt.DisplayRole)
            if record.get_field_val("ankor"):
                remove_anchor_action = context_menu.addAction("Remove Anchor")
            else:
                mark_as_anchor_action = context_menu.addAction("Mark as Anchor")

            context_menu.addSeparator()

            rate_up_action = context_menu.addAction("Rate Up")
            rate_down_action = context_menu.addAction("Rate Down")
            rate_swap_up_action = context_menu.addAction("Rate Swap (Up)")
            reset_sigma_action = context_menu.addAction("Reset Sigma")

            mu_submenu = context_menu.addMenu("Adjust Mu")
            add_10_mu_action = mu_submenu.addAction("Add 10")
            subtract_10_mu_action = mu_submenu.addAction("Subtract 10")
            add_1_mu_action = mu_submenu.addAction("Add 1")
            subtract_1_mu_action = mu_submenu.addAction("Subtract 1")

            context_menu.addSeparator()

            # Cached relations editing
            edit_wins_action = context_menu.addAction("Edit Wins")
            edit_losses_action = context_menu.addAction("Edit Losses")

            context_menu.addSeparator()

            # Edge solving for this item
            solve_edges_action = context_menu.addAction("Solve Edges for This Item")

            context_menu.addSeparator()
            normalize_midpoint_action = context_menu.addAction("Normalize with This as Midpoint (5.0)")
            context_menu.addSeparator()

            delete_item = context_menu.addAction("Delete from List")
            context_menu.addSeparator()

        refresh_action = context_menu.addAction("Refresh")
        reset_all_sigma_action = context_menu.addAction("Reset All Sigma")

        action = context_menu.exec(self.list_view.mapToGlobal(pos))

        # Handle multi-selection actions first
        if len(selected_indexes) >= 2 and 'mark_equal_action' in locals() and action == mark_equal_action:
            self.mark_selected_as_equal(selected_indexes)
            return

        if not index.isValid():
            if action == refresh_action:
                self.refresh_data()
            elif action == reset_all_sigma_action:
                self.reset_all_sigma()
            return

        if 'mark_as_anchor_action' in locals() and action == mark_as_anchor_action:
            self.mark_as_anchor(index)
        elif 'remove_anchor_action' in locals() and action == remove_anchor_action:
            self.remove_anchor(index)
        elif action == rate_up_action:
            self.rate_up(index)
        elif action == rate_down_action:
            self.rate_down(index)
        elif action == rate_swap_up_action:
            self.rate_swap_up(index)
        elif action == reset_sigma_action:
            self.reset_sigma(index)
        elif action == add_10_mu_action:
            self.adjust_mu(index, 10)
        elif action == subtract_10_mu_action:
            self.adjust_mu(index, -10)
        elif action == add_1_mu_action:
            self.adjust_mu(index, 1)
        elif action == subtract_1_mu_action:
            self.adjust_mu(index, -1)
        elif action == reset_all_sigma_action:
            self.reset_all_sigma()
        elif 'edit_wins_action' in locals() and action == edit_wins_action:
            self.edit_wins(index)
        elif 'edit_losses_action' in locals() and action == edit_losses_action:
            self.edit_losses(index)
        elif action == delete_item:
            self.delete_from_list(index)
        elif 'solve_edges_action' in locals() and action == solve_edges_action:
            self.solve_edges_for_item(index)
        elif 'normalize_midpoint_action' in locals() and action == normalize_midpoint_action:
            self.normalize_with_midpoint(index)
        elif action == refresh_action:
            self.refresh_data()

    def rate_up(self, index):
        """Rate up - improve rating of selected item"""
        row = index.row()
        if row == 0:
            return

        try:
            winner = self.model.get_record_at(row)
            loser = self.model.get_record_at(row - 1)

            if winner and loser:
                # Get current ratings
                win_mu, win_sigma = TrueSkillAnnotationRecordTools.get_ts_values(winner)
                lose_mu, lose_sigma = TrueSkillAnnotationRecordTools.get_ts_values(loser)

                # Calculate new ratings
                win_rating = trueskill.Rating(mu=win_mu, sigma=win_sigma)
                lose_rating = trueskill.Rating(mu=lose_mu, sigma=lose_sigma)
                new_win, new_lose = trueskill.rate_1vs1(win_rating, lose_rating)

                # Update through data manager
                data_manager.update_record_rating(winner, new_win.mu, new_win.sigma)
                data_manager.update_record_rating(loser, new_lose.mu, new_lose.sigma)

                logger.debug(f"Rated up: {winner.id} vs {loser.id}")

        except Exception as e:
            logger.error(f"Error rating up: {e}")
            QMessageBox.warning(self, "Error", f"Failed to rate up: {e}")

    def rate_down(self, index):
        """Rate down - worsen rating of selected item"""
        row = index.row()
        if row >= self.model.rowCount() - 1:
            return

        try:
            winner = self.model.get_record_at(row + 1)
            loser = self.model.get_record_at(row)

            if winner and loser:
                # Get current ratings
                win_mu, win_sigma = TrueSkillAnnotationRecordTools.get_ts_values(winner)
                lose_mu, lose_sigma = TrueSkillAnnotationRecordTools.get_ts_values(loser)

                # Calculate new ratings
                win_rating = trueskill.Rating(mu=win_mu, sigma=win_sigma)
                lose_rating = trueskill.Rating(mu=lose_mu, sigma=lose_sigma)
                new_win, new_lose = trueskill.rate_1vs1(win_rating, lose_rating)

                # Update through data manager
                data_manager.update_record_rating(winner, new_win.mu, new_win.sigma)
                data_manager.update_record_rating(loser, new_lose.mu, new_lose.sigma)

                logger.debug(f"Rated down: {loser.id} vs {winner.id}")

        except Exception as e:
            logger.error(f"Error rating down: {e}")
            QMessageBox.warning(self, "Error", f"Failed to rate down: {e}")

    def rate_swap_up(self, index):
        """Rate swap up - swap ratings with item above"""
        row = index.row()
        if row == 0:
            return

        try:
            selected_record = self.model.get_record_at(row)
            upper_record = self.model.get_record_at(row - 1)

            if selected_record and upper_record:
                # Get current ratings
                selected_mu = selected_record.get_field_val("avg_rating", DEFAULT_MU)
                selected_sigma = selected_record.get_field_val("trueskill_sigma", MODEL_SIGMA)
                upper_mu = upper_record.get_field_val("avg_rating", DEFAULT_MU)
                upper_sigma = upper_record.get_field_val("trueskill_sigma", MODEL_SIGMA)

                # Swap the values
                data_manager.update_record_rating(selected_record, upper_mu, upper_sigma)
                data_manager.update_record_rating(upper_record, selected_mu, selected_sigma)

                # Sort immediately after swap
                self.model.sort_records()

                logger.debug(f"Swapped ratings: {selected_record.id} <-> {upper_record.id}")

        except Exception as e:
            logger.error(f"Error swapping ratings: {e}")
            QMessageBox.warning(self, "Error", f"Failed to swap ratings: {e}")

    def reset_sigma(self, index):
        """Reset sigma for selected item"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                data_manager.update_record_rating(record, record.get_field_val("avg_rating", DEFAULT_MU), MODEL_SIGMA)
                logger.debug(f"Reset sigma for record {record.id}")
        except Exception as e:
            logger.error(f"Error resetting sigma: {e}")
            QMessageBox.warning(self, "Error", f"Failed to reset sigma: {e}")

    def adjust_mu(self, index, amount):
        """Adjust mu for selected item"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                current_mu = record.get_field_val("avg_rating", DEFAULT_MU)
                new_mu = current_mu + amount
                data_manager.update_record_rating(record, new_mu, record.get_field_val("trueskill_sigma", MODEL_SIGMA))
                logger.debug(f"Adjusted mu for record {record.id} by {amount}")
        except Exception as e:
            logger.error(f"Error adjusting mu: {e}")
            QMessageBox.warning(self, "Error", f"Failed to adjust mu: {e}")

    def reset_all_sigma(self):
        """Reset sigma for all items"""
        try:
            for record in self.model.records:
                if not record.get_field_val("ankor"):
                    data_manager.update_record_rating(record, record.get_field_val("avg_rating", DEFAULT_MU),
                                                      MODEL_SIGMA)
            logger.debug("Reset all sigma values")
        except Exception as e:
            logger.error(f"Error resetting all sigma: {e}")
            QMessageBox.warning(self, "Error", f"Failed to reset all sigma: {e}")

    def mark_as_anchor(self, index):
        """Mark item as anchor"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                record.set_field_val("ankor", True)
                record.set_field_val("trueskill_sigma", 1.0)
                logger.debug(f"Marked record {record.id} as anchor")
        except Exception as e:
            logger.error(f"Error marking as anchor: {e}")
            QMessageBox.warning(self, "Error", f"Failed to mark as anchor: {e}")

    def remove_anchor(self, index):
        """Remove anchor mark"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                record.set_field_val("ankor", False)
                logger.debug(f"Removed anchor mark from record {record.id}")
        except Exception as e:
            logger.error(f"Error removing anchor: {e}")
            QMessageBox.warning(self, "Error", f"Failed to remove anchor: {e}")

    def edit_wins(self, index):
        """Open wins editor for selected item"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                editor = CachedRelationsEditor(record, self)
                editor.tabs.setCurrentIndex(0)  # Show wins tab
                editor.exec()
        except Exception as e:
            logger.error(f"Error opening wins editor: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open wins editor: {e}")

    def edit_losses(self, index):
        """Open losses editor for selected item"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                editor = CachedRelationsEditor(record, self)
                editor.tabs.setCurrentIndex(1)  # Show losses tab
                editor.exec()
        except Exception as e:
            logger.error(f"Error opening losses editor: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open losses editor: {e}")

    def delete_from_list(self, index):
        """Delete item from list"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                record.delete_rec()
                # Force data manager to reload the manual_voted_list from database
                data_manager.load_manual_voted_list()
                # Refresh data after deletion
                self.refresh_data()
                logger.debug(f"Deleted record {record.id} from list")
        except Exception as e:
            logger.error(f"Error deleting from list: {e}")
            QMessageBox.warning(self, "Error", f"Failed to delete from list: {e}")

    def solve_edges_for_item(self, index):
        """Solve edges for the selected item"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                # Get the main widget from parent widgets
                parent = self.parent()
                while parent and not hasattr(parent, 'solve_edges_with_threshold'):
                    parent = parent.parent()

                if parent and hasattr(parent, 'solve_edges_with_threshold'):
                    # Call the main widget's solve edges method with this specific record
                    parent.solve_edges_with_threshold(record=record)
                else:
                    QMessageBox.warning(self, "Error", "Cannot find main widget to perform edge solving")

                logger.debug(f"Solved edges for record {record.id}")
        except Exception as e:
            logger.error(f"Error solving edges for item: {e}")
            QMessageBox.warning(self, "Error", f"Failed to solve edges for item: {e}")

    def normalize_with_midpoint(self, index):
        """Normalize ratings with the selected item as midpoint (5.0)"""
        try:
            record = self.model.data(index, Qt.DisplayRole)
            if record:
                # Get the main widget from parent widgets
                parent = self.parent()
                while parent and not hasattr(parent, 'normalize_ratings_with_midpoint'):
                    parent = parent.parent()

                if parent and hasattr(parent, 'normalize_ratings_with_midpoint'):
                    # Call the main widget's normalize method with this specific record as midpoint
                    parent.normalize_ratings_with_midpoint(record)
                else:
                    QMessageBox.warning(self, "Error", "Cannot find main widget to perform normalization")

                logger.debug(f"Normalized ratings with record {record.id} as midpoint")
        except Exception as e:
            logger.error(f"Error normalizing with midpoint: {e}")
            QMessageBox.warning(self, "Error", f"Failed to normalize with midpoint: {e}")

    def mark_selected_as_equal(self, selected_indexes):
        """Mark multiple selected items as equal"""
        try:
            if len(selected_indexes) < 2:
                QMessageBox.warning(self, "Error", "Please select at least 2 items to mark as equal.")
                return

            # Get all selected records
            selected_records = []
            for index in selected_indexes:
                record = self.model.data(index, Qt.DisplayRole)
                if record:
                    selected_records.append(record)

            if len(selected_records) < 2:
                QMessageBox.warning(self, "Error", "Could not retrieve selected records.")
                return

            # Determine primary record (anchor has priority)
            primary_record = None
            for record in selected_records:
                if record.get_field_val("ankor", False):
                    primary_record = record
                    break

            # If no anchor, use the first selected item
            if not primary_record:
                primary_record = selected_records[0]

            # Build list of items to display
            item_names = [f"- {r.file.name if r.file else str(r.id)}" for r in selected_records]
            primary_name = primary_record.file.name if primary_record.file else str(primary_record.id)

            # Confirm with user
            reply = QMessageBox.question(
                self,
                'Confirm Mark as Equal',
                f'Mark {len(selected_records)} items as equal?\n\n'
                f'Primary item (will keep manual=True):\n- {primary_name}\n\n'
                f'Items to merge (will set manual=False):\n' + '\n'.join(
                    [name for rec, name in zip(selected_records, item_names) if rec != primary_record]) + '\n\n'
                                                                                                          f'The merged items will disappear from the list.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            # Perform the merge
            primary_record.set_field_val("manual", True)

            merged_ids = []
            for record in selected_records:
                if record != primary_record:
                    TrueSkillAnnotationRecordTools.merge_items(primary_record, record)
                    merged_ids.append(str(record.id))
                    logger.debug(f"Merged {record.id} into {primary_record.id}, set manual=False")

            # Force data manager to reload the manual_voted_list from database
            data_manager.load_manual_voted_list()

            # Refresh the list to show changes
            self.refresh_data()

            QMessageBox.information(
                self,
                "Success",
                f"Marked {len(selected_records)} items as equal.\n\n"
                f"Primary: {primary_name}\n"
                f"Merged: {len(selected_records) - 1} items"
            )

            logger.info(f"Marked {len(selected_records)} items as equal with {primary_record.id} as primary")

        except Exception as e:
            logger.error(f"Error marking selected as equal: {e}")
            QMessageBox.warning(self, "Error", f"Failed to mark selected as equal: {e}")

    # ========== Drag and Drop Protocol Methods ==========
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Forward drag enter events to the drag handler"""
        self.drag_handler.dragEnterEvent(event)
    
    def dropEvent(self, event: QDropEvent):
        """Forward drop events to the drag handler"""
        self.drag_handler.dropEvent(event)
    
    def get_selected_items_for_drag(self) -> List[Any]:
        """Return empty list - we don't support dragging from this widget"""
        return []
    
    def handle_dropped_mongo_wrappers(self, objects, event: QDropEvent):
        """Ignore mongo wrapper drops"""
        event.ignore()
    
    def handle_dropped_custom_data(self, objects, event: QDropEvent):
        """Ignore custom data drops"""
        event.ignore()
    
    def handle_dropped_files(self, paths: List[pathlib.Path], event: QDropEvent):
        """Handle dropped image files from Windows Explorer"""
        try:
            if not paths:
                event.ignore()
                return
            
            logger.info(f"Handling {len(paths)} dropped files")
            added_count = self.add_files_to_ratings(paths)
            
            if added_count > 0:
                event.accept()
                self.status_label.setText(f"Added {added_count} images from drag & drop")
                logger.info(f"Successfully added {added_count} images via drag & drop")
            else:
                event.ignore()
                self.status_label.setText("No new images added (all may already exist)")
                
        except Exception as e:
            logger.error(f"Error handling dropped files: {e}")
            QMessageBox.warning(self, "Error", f"Failed to add dropped files: {e}")
            event.ignore()
    
    def add_files_to_ratings(self, file_paths: List[pathlib.Path]) -> int:
        """Add image files to the rating list
        
        Args:
            file_paths: List of file paths to add
            
        Returns:
            Number of files successfully added
        """
        try:
            # Filter for image files only
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
            image_paths = []
            
            for path in file_paths:
                path_obj = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
                if path_obj.suffix.lower() in image_extensions and path_obj.is_file():
                    image_paths.append(path_obj)
            
            if not image_paths:
                logger.warning("No valid image files found in dropped files")
                return 0
            
            # Show progress dialog for multiple files
            if len(image_paths) > 5:
                progress = QProgressDialog(
                    f"Adding {len(image_paths)} images to ratings...", 
                    None, 0, len(image_paths), self
                )
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
            else:
                progress = None
            
            # Calculate mean conservative rating for new items
            mean_conservative_rating = data_manager.calculate_mean_conservative_rating()
            
            # Get sigma value from model config or use default
            sigma_to_use = model_sigma_config.get_sigma_for_new_items()
            logger.info(f"Using sigma value: {sigma_to_use:.4f} (from {'model' if model_sigma_config.use_llm_sigma else 'default'})")
            
            # Get normalization params for denormalization
            transform_mean = data_manager.rating_job.get_field_val("transform_mean", DEFAULT_MU)
            transform_std_dev = data_manager.rating_job.get_field_val("transform_std_dev", 1.0)
            
            added_count = 0
            skipped_count = 0
            
            for idx, img_path in enumerate(image_paths):
                if progress:
                    progress.setValue(idx)
                    QApplication.processEvents()
                
                try:
                    # Get or create file record
                    file_record = FileRecord.get_record_by_path(str(img_path))
                    if not file_record:
                        logger.warning(f"Could not find/create file record for {img_path}")
                        continue
                    
                    # Check if annotation record exists
                    existing_ar = data_manager.rating_job.get_annotation_record(file_record)
                    
                    # Skip if already manually voted
                    if existing_ar and existing_ar.get_field_val("manual", False):
                        logger.debug(f"Skipping {img_path.name} - already in manual voted list")
                        skipped_count += 1
                        continue
                    
                    # Create new annotation record if doesn't exist
                    if not existing_ar:
                        ar_data = {
                            "parent_id": data_manager.rating_job.id,
                            "file_id": file_record.id,
                            "manual": True,
                            "avg_rating": DEFAULT_MU,
                            "trueskill_sigma": MODEL_SIGMA,
                            "value": DEFAULT_RATING
                        }
                        ar_id = AnnotationRecord.collection().insert_one(ar_data).inserted_id
                        existing_ar = AnnotationRecord(ar_id)
                        logger.debug(f"Created new annotation record for {img_path.name}")
                    
                    # Try to predict rating if enabled
                    mu = mean_conservative_rating
                    sigma = sigma_to_use
                    
                    if model_sigma_config.use_predictions and model_sigma_config.is_model_loaded():
                        try:
                            predicted_rating = model_sigma_config.predict_rating(str(img_path))
                            
                            if predicted_rating is not None:
                                # Denormalize predicted rating to mu
                                mu = RatingHelpers.denormalize_rating(
                                    predicted_rating, 
                                    transform_mean, 
                                    transform_std_dev
                                )
                                logger.debug(f"Using prediction for {img_path.name}: rating={predicted_rating:.2f}, mu={mu:.2f}")
                        except Exception as e:
                            logger.warning(f"Prediction error for {img_path.name}: {e}, using fallback")
                    
                    # Update the record
                    existing_ar.set_field_val("manual", True)
                    existing_ar.set_field_val("avg_rating", mu)
                    existing_ar.set_field_val("trueskill_sigma", sigma)
                    
                    # Clear caches for this record
                    data_manager.cache_manager.clear_record_caches(str(existing_ar.id))
                    
                    added_count += 1
                    logger.debug(f"Added {img_path.name} to ratings (mu={mu:.2f}, sigma={sigma:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
            
            if progress:
                progress.close()
            
            # Reload the manual voted list and refresh display
            if added_count > 0:
                data_manager.load_manual_voted_list()
                self.refresh_data()
            
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} files that were already in the ratings list")
            
            return added_count
            
        except Exception as e:
            logger.error(f"Error in add_files_to_ratings: {e}")
            raise


class ImageRatingWidget(PySide6GlueWidget):
    """Main image rating widget with dual interface"""

    def __init__(self):
        super().__init__()

        self.image_record1: Optional[AnnotationRecord] = None
        self.image_record2: Optional[AnnotationRecord] = None
        self.round_counter = 0

        self.init_ui()
        self.setup_application_logic()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Create tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Comparison View
        self.comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_widget)
        image_comparison_layout = QHBoxLayout()
        self.image_display1 = ImageDisplayWidget("Image 1")
        self.image_display2 = ImageDisplayWidget("Image 2")
        image_comparison_layout.addWidget(self.image_display1)
        image_comparison_layout.addWidget(self.image_display2)
        comparison_layout.addLayout(image_comparison_layout)

        controls_layout = QHBoxLayout()
        self.btn_img1_better = QPushButton("Image 1 is Better")
        self.btn_img1_better.clicked.connect(self.on_image1_better_click)
        self.image_display1.clicked.connect(self.on_image1_better_click)
        controls_layout.addWidget(self.btn_img1_better)

        self.btn_img2_better = QPushButton("Image 2 is Better")
        self.btn_img2_better.clicked.connect(self.on_image2_better_click)
        self.image_display2.clicked.connect(self.on_image2_better_click)
        controls_layout.addWidget(self.btn_img2_better)

        self.btn_equal = QPushButton("Images are Equal")
        self.btn_equal.clicked.connect(self.on_images_equal_click)
        controls_layout.addWidget(self.btn_equal)

        self.btn_skip = QPushButton("Skip Pair")
        self.btn_skip.clicked.connect(self.load_next_pair)
        controls_layout.addWidget(self.btn_skip)

        self.auto_load_checkbox = QCheckBox("Auto-load next pair")
        self.auto_load_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_load_checkbox)

        self.predict = QCheckBox("Show Predicted Rating")
        self.predict.setChecked(True)
        controls_layout.addWidget(self.predict)

        # Rating method selection
        self.rating_method_combo = QComboBox()
        items_names = ComparisonService.select_pair_mode_manager.modes.keys()
        self.rating_method_combo.addItems(items_names)
        self.rating_method_combo.setCurrentIndex(0)
        self.rating_method_combo.currentIndexChanged.connect(self.on_rating_method_changed)
        controls_layout.addWidget(QLabel("Rating Method:"))
        controls_layout.addWidget(self.rating_method_combo)

        comparison_layout.addLayout(controls_layout)
        self.status_label = QLabel("Status: Ready to begin.")
        self.status_label.setFixedHeight(40)
        comparison_layout.addWidget(self.status_label)
        self.tabs.addTab(self.comparison_widget, "Comparison View")

        # Tab 2: List View
        self.list_widget = ImageListRaterWidget()
        self.tabs.addTab(self.list_widget, "List View")

        self.sixImageRankingWidget = SixImageRankingWidget()
        self.tabs.addTab(self.sixImageRankingWidget, "Six Image Ranking")

        # Set focus policy to accept key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def on_rating_method_changed(self, idx):
        """Handle rating method change"""
        method_name = self.rating_method_combo.currentText()
        if method_name in ComparisonService.select_pair_mode_manager.modes:
            ComparisonService.select_pair_mode_manager.set_mode(method_name)
            self.status_label.setText(f"Switched to '{method_name}' mode.")
            self.load_next_pair()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if not self.btn_img1_better.isEnabled():
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key.Key_1:
            self.on_image1_better_click()
        elif event.key() == Qt.Key.Key_2:
            self.on_image2_better_click()
        elif event.key() == Qt.Key.Key_3:
            self.on_images_equal_click()
        elif event.key() == Qt.Key.Key_Space:
            self.load_next_pair()
        else:
            super().keyPressEvent(event)

    def setup_application_logic(self):
        """Setup application logic"""
        self.load_next_pair()

    def disable_controls(self, disabled=True):
        """Enable/disable controls"""
        self.btn_img1_better.setDisabled(disabled)
        self.btn_img2_better.setDisabled(disabled)
        self.btn_equal.setDisabled(disabled)
        self.btn_skip.setDisabled(disabled)

    def get_image_rating(self, record: Optional[AnnotationRecord]) -> float:
        """Get image rating"""
        if record and record.value is not None:
            try:
                return float(record.value)
            except ValueError:
                logger.warning(f"Could not convert rating '{record.value}' to float for record {record.id}")
                return DEFAULT_RATING
        return DEFAULT_RATING

    def get_predicted_rating_for_display(self, record: Optional[AnnotationRecord]) -> str:
        """Get predicted rating for display"""
        if not record or not record.file:
            return "N/A"

        # TrueSkill rating
        ts_mu = record.get_field_val("avg_rating", DEFAULT_MU)
        ts_sigma = record.get_field_val('trueskill_sigma', MODEL_SIGMA)
        ts_rating = ts_mu - 3 * ts_sigma

        # Try to predict if enabled
        if self.predict.isChecked() and model_sigma_config.is_model_loaded():
            try:
                # Get normalization params for denormalization
                transform_mean = data_manager.rating_job.get_field_val("transform_mean", DEFAULT_MU)
                transform_std_dev = data_manager.rating_job.get_field_val("transform_std_dev", 1.0)

                # Predict rating (1-10 scale)
                predicted_rating = model_sigma_config.predict_rating(record.file.full_path)
                if predicted_rating is not None:
                    # Denormalize to mu value
                    predicted_mu = RatingHelpers.denormalize_rating(
                        predicted_rating, transform_mean, transform_std_dev
                    )
                    return (f"TrueSkill: {ts_rating:.2f} (μ={ts_mu:.2f}, σ={ts_sigma:.2f})\n"
                            f"Predicted: {predicted_rating:.2f}/10 → μ={predicted_mu:.2f}")
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")

        # Fallback to TrueSkill only
        return f"TrueSkill: {ts_rating:.2f} (μ={ts_mu:.2f}, σ={ts_sigma:.2f})"

    def load_next_pair(self):
        """Load next pair for comparison"""
        self.image_display1.setStyleSheet("border: none;")
        self.image_display2.setStyleSheet("border: none;")

        while True:  # Loop to process cached pairs quickly
            mode: BaseAnnotationMode = ComparisonService.select_pair_mode_manager.current_mode
            pair = mode.get_next_pair()

            if not pair or pair[0] is None or pair[1] is None:
                self.image_record1, self.image_record2 = None, None
                self.image_display1.set_image(None)
                self.image_display1.set_ratings(None, None)
                self.image_display2.set_image(None)
                self.image_display2.set_ratings(None, None)
                msg = "Status: No images in job. Add folder." if not data_manager.all_annotations else "Status: Not enough images or all pairs shown recently."
                self.status_label.setText(msg)
                self.disable_controls()
                return

            self.image_record1, self.image_record2 = pair

            # Check for cached result using centralized cache manager
            cached_result = data_manager.cache_manager.get_pair_winner(str(self.image_record1.id),
                                                                       str(self.image_record2.id))
            if cached_result and AUTO_WIN_PAIRS:
                winner_id, loser_id = cached_result
                winner = self.image_record1 if str(self.image_record1.id) == winner_id else self.image_record2
                loser = self.image_record2 if winner is self.image_record1 else self.image_record1

                self.status_label.setText(f"Auto-voting cached pair... Round {self.round_counter + 1}")
                QApplication.processEvents()  # Update UI to show message
                self._update_ratings(winner, loser)

                # Auto-load new pairs every AUTO_LOAD_ROUND rounds
                if self.round_counter > 0 and self.round_counter % AUTO_LOAD_ROUND == 0:
                    self.round_counter += 1
                    self.normalize_ratings()
                    self.add_rand_from_all_to_human_voted()
                continue  # Immediately get the next pair

            # If we are here, it means we need a manual vote
            self.image_display1.set_image(self.image_record1.file)
            self.image_display1.set_ratings(self.get_image_rating(self.image_record1),
                                            self.get_predicted_rating_for_display(self.image_record1))
            self.image_display2.set_image(self.image_record2.file)
            self.image_display2.set_ratings(self.get_image_rating(self.image_record2),
                                            self.get_predicted_rating_for_display(self.image_record2))
            self.status_label.setText(f"Comparing images. Round {self.round_counter + 1}.")

            # Auto-load new pairs every AUTO_LOAD_ROUND rounds
            if self.round_counter > 0 and self.round_counter % AUTO_LOAD_ROUND == 0:
                self.round_counter += 1
                self.normalize_ratings()
                self.add_rand_from_all_to_human_voted()

            self.disable_controls(False)
            break  # Exit the loop and wait for user input

    def _update_ratings(self, win_rec: AnnotationRecord, lose_rec: AnnotationRecord):
        """Update ratings for winner and loser"""
        self.round_counter += 1

        # Cache the result
        data_manager.cache_manager.set_pair_winner(str(win_rec.id), str(lose_rec.id), str(win_rec.id))

        RatingHelpers.process_pair_with_trueskill(win_rec, lose_rec, data_manager)

        # Update analytics
        analytics_manager.increment_comparison_count()

    def _handle_comparison_result(self, winner_rec_id: str):
        """Handle comparison result"""
        win_rec = self.image_record1 if str(self.image_record1.id) == winner_rec_id else self.image_record2
        lose_rec = self.image_record2 if win_rec is self.image_record1 else self.image_record1

        self._update_ratings(win_rec, lose_rec)

        if self.auto_load_checkbox.isChecked():
            self.load_next_pair()
        else:
            self.image_display1.set_ratings(self.get_image_rating(self.image_record1),
                                            self.get_predicted_rating_for_display(self.image_record1))
            self.image_display2.set_ratings(self.get_image_rating(self.image_record2),
                                            self.get_predicted_rating_for_display(self.image_record2))

    def on_image1_better_click(self):
        """Handle image 1 better click"""
        if not self.image_record1 or not self.image_record2:
            return
        self._handle_comparison_result(str(self.image_record1.id))

    def on_image2_better_click(self):
        """Handle image 2 better click"""
        if not self.image_record1 or not self.image_record2:
            return
        self._handle_comparison_result(str(self.image_record2.id))

    def on_images_equal_click(self):
        """Handle images equal click"""
        if not self.image_record1 or not self.image_record2:
            return

        if self.image_record2.get_field_val("ankor", False):
            TrueSkillAnnotationRecordTools.merge_items(self.image_record2, self.image_record1)
        else:
            TrueSkillAnnotationRecordTools.merge_items(self.image_record1, self.image_record2)

        data_manager.load_manual_voted_list()
        self.load_next_pair()

    def normalize_ratings(self):
        """Normalize ratings using Z-score method"""
        result = RatingOperations.normalize_ratings(self, data_manager)
        if result:
            self.status_label.setText(result)

    def calculate_mean_conservative_rating(self) -> float:
        """Calculate mean conservative rating from existing manual voted items"""
        return RatingOperations.calculate_mean_conservative_rating(data_manager)

    def add_rand_from_all_to_human_voted(self):
        """Add random images from all to human voted list"""
        result = RatingOperations.add_rand_from_all_to_human_voted(self, data_manager)
        if result:
            self.status_label.setText(result)

    def clear_all_manual_flags(self):
        """Clear all manual flags"""
        result = RatingOperations.clear_all_manual_flags(self, data_manager)
        if result:
            self.status_label.setText(result)

    def shift_all_mu_plus_10(self):
        """Shift all manual voted items' mu by +10"""
        result = RatingOperations.shift_all_mu_plus_10(self, data_manager)
        if result:
            self.status_label.setText(result)

    def normalize_ratings_with_midpoint(self, midpoint_record: AnnotationRecord):
        """
        Normalize ratings using Z-score method with a selected record as the midpoint.

        This method uses Z-score normalization where the selected record's rating
        becomes the "mean" (midpoint), with other records distributed around it
        based on their deviation from this midpoint, scaled to 1-10 range.

        Args:
            midpoint_record: The record to use as the midpoint (will become ~5.0)
        """
        try:
            # Get manual voted list
            norm_list = data_manager.manual_voted_list.copy()

            values = [ar.get_field_val("avg_rating", 0) for ar in norm_list]

            if not values:
                QMessageBox.warning(self, "No Ratings", "No valid ratings found.")
                return

            # Use the midpoint record's rating as the custom mean
            custom_mean = midpoint_record.get_field_val("avg_rating", DEFAULT_MU)
            std_dev_rating = np.std(values)

            if std_dev_rating == 0:
                QMessageBox.warning(self, "No Variation", "All ratings are the same.")
                return

            # Store custom mean and std_dev for later use
            data_manager.rating_job.set_field_val("transform_mean", custom_mean)
            data_manager.rating_job.set_field_val("transform_std_dev", std_dev_rating)

            for ar in tqdm(norm_list, desc="Normalizing ratings with midpoint"):
                avg_rating = ar.get_field_val("avg_rating", custom_mean)
                z_score = (avg_rating - custom_mean) / std_dev_rating

                # Scale z-score to 1-10 range
                scaled_rating = (((z_score - (-3)) / (3 - (-3))) * (10.0 - 1.0)) + 1.0

                # Clamp values to be within 1.0 and 10.0
                scaled_rating = max(1.0, min(10.0, scaled_rating))

                ar.value = round(scaled_rating, 2)

            logger.info("Ratings normalized successfully with midpoint")
            self.status_label.setText(
                f"Ratings normalized with {midpoint_record.file.name if midpoint_record.file else str(midpoint_record.id)} as midpoint.")
            QMessageBox.information(
                self,
                "Normalization Complete",
                f"Successfully normalized {len(norm_list)} ratings using Z-score method.\n\n"
                f"Midpoint record: {midpoint_record.file.name if midpoint_record.file else str(midpoint_record.id)}\n"
                f"Custom mean: {custom_mean:.2f}, Std dev: {std_dev_rating:.2f}"
            )

        except Exception as e:
            logger.error(f"Error normalizing ratings with midpoint: {e}")
            QMessageBox.warning(self, "Error", f"Failed to normalize ratings: {e}")

    def add_1_to_all_sigma(self):
        """Add +1 to all manual voted items' sigma"""
        result = RatingOperations.add_1_to_all_sigma(self, data_manager)
        if result:
            self.status_label.setText(result)


class ImageRatingApp(PySide6GlueApp):
    """Main application"""

    def __init__(self):
        super().__init__()
        self._main_window.resize(1200, 800)
        # Initialize data manager
        data_manager.initialize()
        # Model will auto-load if predictions are enabled
        self.main_widget = ImageRatingWidget()
        self.set_main_widget(self.main_widget)
        self._main_window.setWindowTitle("Image Rating Competition v4 - Improved")

        # Analytics window (initially None)
        self.menu_actions = MenuActions(self._main_window, self.main_widget)

        self.setup_menu()

    def setup_menu(self):
        """Setup application menu with organized sub-menus"""
        menu_bar = self._main_window.menuBar()

        # File Menu with organized sub-menus
        file_menu = menu_bar.addMenu("File")

        add_folder_action = file_menu.addAction("Add Folder to Rating Job...")
        add_folder_action.triggered.connect(self.menu_actions.add_folder_to_rating_job)

        add_rand_action = file_menu.addAction("Add Random from All to Human Voted")
        add_rand_action.triggered.connect(self.main_widget.add_rand_from_all_to_human_voted)

        imp_a = file_menu.addAction("Import Scores (JSON)")
        imp_a.triggered.connect(lambda: DataIOOperations.import_data_json(self.main_widget, data_manager))
        exp_a = file_menu.addAction("Export Scores (JSON)")
        exp_a.triggered.connect(lambda: DataIOOperations.export_data_json(self.main_widget, data_manager))

        processing_submenu = file_menu.addMenu("Data Processing")

        normalize_res = processing_submenu.addAction("Normalize Ratings")
        normalize_res.triggered.connect(self.main_widget.normalize_ratings)

        cleanup_submenu = file_menu.addMenu("Data Cleanup")

        data_filters_menu = cleanup_submenu.addMenu("Data Filters")
        clear_manual = data_filters_menu.addAction("Clear All Manual Flags")
        clear_manual.triggered.connect(self.main_widget.clear_all_manual_flags)

        sigma_cleanup_menu = cleanup_submenu.addMenu("Sigma Operations")

        rating_adjust_submenu = file_menu.addMenu("Manual Adjustments")

        mu_operations_menu = rating_adjust_submenu.addMenu("Mu Operations")
        shift_mu_plus_10 = mu_operations_menu.addAction("Shift All Mu +10")
        shift_mu_plus_10.triggered.connect(self.main_widget.shift_all_mu_plus_10)
        set_al_mu_tu_def = mu_operations_menu.addAction("Set All Mu to Default")
        set_al_mu_tu_def.triggered.connect(
            lambda: TrueSkillAnnotationRecordTools.set_all_mu_to_default(DEFAULT_MU, MODEL_SIGMA))

        sigma_operations_menu = rating_adjust_submenu.addMenu("Sigma Operations")
        add_sigma_plus_1 = sigma_operations_menu.addAction("Add +1 to All Sigma")
        add_sigma_plus_1.triggered.connect(self.main_widget.add_1_to_all_sigma)
        set_al_sigma_to_default = sigma_operations_menu.addAction("Set All Sigma to Default")
        set_al_sigma_to_default.triggered.connect(
            lambda: TrueSkillAnnotationRecordTools.set_all_sigma_to_default(MODEL_SIGMA))

        sigma_operations_menu.addSeparator()
        configure_model_sigma_action = sigma_operations_menu.addAction("Configure Model Sigma...")
        configure_model_sigma_action.triggered.connect(self.menu_actions.configure_model_sigma)

        cache_management_submenu = file_menu.addMenu("Cache Management")

        clear_cache_action = cache_management_submenu.addAction("Clear Cache")
        clear_cache_action.triggered.connect(self.menu_actions.clear_all_cache)
        clean_cache_duplicates_action = cache_management_submenu.addAction("Clean Cache Duplicates")
        clean_cache_duplicates_action.triggered.connect(self.menu_actions.clean_cache_duplicates)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.exit_app)

        # View Menu with Analytics
        view_menu = menu_bar.addMenu("View")

        analytics_action = view_menu.addAction("Show Analytics")
        analytics_action.triggered.connect(self.menu_actions.show_analytics_window)

    def exit_app(self):
        """Exit application"""
        QApplication.instance().quit()


if __name__ == "__main__":
    # Configure application
    config = Allocator.config
    Allocator.config.def_annotation_jobs.dict = {
        "rating_1_10": {"type": "multiclass/image", "chooses": [str(i) for i in range(1, 11)]}
    }
    config.fileDataManager.path = r"D:\data\ImageDataManager"
    config.mongoConfig.database_name = "files_db"

    #logger.remove()  # Remove default handler
    #logger.add(sys.stdout, level="INFO")
    # Create and run application
    app = ImageRatingApp()
    app.run()
