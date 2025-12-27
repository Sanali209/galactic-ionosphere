"""
Six Image Ranking View - A view for ranking 6 images simultaneously
"""
import random
from typing import Optional, List
from loguru import logger

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
)

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from SLM.pySide6Ext.pySide6Q import FlowLayout
from qtWidjets import ImageDisplayWidget
from data_manager import data_manager
from rating_helpers import RatingHelpers
from anotation_tool import TrueSkillAnnotationRecordTools
from constants import DEFAULT_MU, MODEL_SIGMA
import trueskill


class SixImageRankingWidget(QWidget):
    """Widget for ranking 6 images simultaneously"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current state - 6 images in ranking order
        self.image_slots = None
        self.images: List[Optional[AnnotationRecord]] = [None] * 6
        self.image_displays: List[ImageDisplayWidget] = []
        self.rating_labels: List[QLabel] = []
        self.position_labels: List[QLabel] = []
        self.images_count = 10
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)

        # Images container
        images_container = FlowLayout()
        self.image_slots = []
        # Create 6 image slots
        for i in range(self.images_count):
            image_slot = QVBoxLayout()
            self.image_slots += [image_slot]

            # Up button (move higher/left)
            up_btn = QPushButton("↑ Higher")
            up_btn.clicked.connect(lambda checked, idx=i: self.move_image_up(idx))
            up_btn.setEnabled(False)  # Initially disabled
            image_slot.addWidget(up_btn)

            # Image display
            image_display = ImageDisplayWidget(f"Position {i + 1}", 360, 360)
            self.image_displays.append(image_display)
            image_slot.addWidget(image_display)


            # Down button (move lower/right)
            down_btn = QPushButton("↓ Lower")
            down_btn.clicked.connect(lambda checked, idx=i: self.move_image_down(idx))
            down_btn.setEnabled(False)  # Initially disabled
            image_slot.addWidget(down_btn)

            images_container.addItem(image_slot)

        main_layout.addLayout(images_container)

        # Control buttons
        control_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load 6 Images")
        self.load_btn.clicked.connect(self.load_random_six)
        control_layout.addWidget(self.load_btn)

        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.refresh_data)
        control_layout.addWidget(self.refresh_btn)

        self.auto_load_checkbox = QPushButton("Auto-load after confirm")
        self.auto_load_checkbox.setCheckable(True)
        self.auto_load_checkbox.setChecked(True)
        control_layout.addWidget(self.auto_load_checkbox)
        self.confirm_btn = QPushButton("CONFIRM RANKING")
        self.confirm_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 5px 3px; font-size: 14px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.confirm_btn.clicked.connect(self.on_confirm_click)
        self.confirm_btn.setEnabled(False)
        control_layout.addWidget(self.confirm_btn)
        # Status display
        self.status_label = QLabel("Status: Click 'Load 6 Images' to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 1px; background-color: #f0f0f0;")
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

    def refresh_data(self):
        """Refresh data from data manager"""
        try:
            data_manager.initialize()
            count = len([r for r in data_manager.manual_voted_list if r.file])
            self.status_label.setText(f"Data refreshed - {count} records available")
            logger.info(f"Six-image ranking view: {count} records available")
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to refresh data: {e}")

    def load_random_six(self):
        """Load 6 random images"""
        try:
            self.images.clear()
            # Get valid records
            valid_records = [r for r in data_manager.manual_voted_list if r.file]

            if len(valid_records) < self.images_count:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Need at least 6 images. Only {len(valid_records)} available."
                )
                return


            # sort by trueskill sigma
            valid_records.sort(key=lambda r: r.get_field_val("trueskill_sigma", MODEL_SIGMA), reverse=True)

            # Select 4 random images

            # get 4 images with bigest sigma
            self.images.extend(valid_records[:5])
            for item in self.images:
                valid_records.remove(item)

            while len(self.images) < self.images_count:
                rand_image = random.choice(valid_records)
                for record in self.images:
                    i_mu, i_sigma = TrueSkillAnnotationRecordTools.get_ts_values(record)
                    rand_mu, rand_sigma = TrueSkillAnnotationRecordTools.get_ts_values(rand_image)
                    quality = trueskill.quality_1vs1(trueskill.Rating(mu=i_mu, sigma=i_sigma), trueskill.Rating(mu=rand_mu, sigma=rand_sigma))
                    if quality > 0.5:
                        valid_records.remove(rand_image)
                        self.images.append(rand_image)
                        break

            self.images.sort(key=lambda key: key.get_field_val("avg_rating", DEFAULT_MU), reverse=True)

            # Update displays
            self.update_all_displays()

            # Enable buttons
            self.enable_movement_buttons()
            self.confirm_btn.setEnabled(True)

            self.status_label.setText("10 images loaded - Arrange from best (left) to worst (right)")
            logger.debug("Loaded 10 random images for ranking")

        except Exception as e:
            logger.error(f"Error loading 10 images: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load images: {e}")

    def update_all_displays(self):
        """Update all image displays"""
        for i in range(self.images_count):
            self.update_display(i)

    def update_display(self, index: int):
        """Update display for a specific position"""
        try:
            image = self.images[index]
            if image and image.file:
                self.image_displays[index].set_image(image.file)

                # Update rating label
                rating = RatingHelpers.calculate_conservative_rating(image)
                mu = image.get_field_val("avg_rating", DEFAULT_MU)
                sigma = image.get_field_val("trueskill_sigma", MODEL_SIGMA)
                self.image_displays[index].user_rating_label.setText(
                    f"Rating: {rating:.2f}\\n(μ={mu:.2f}, σ={sigma:.2f})"
                )
            else:
                self.image_displays[index].set_image(None)
                self.rating_labels[index].setText("No image")

        except Exception as e:
            logger.error(f"Error updating display {index}: {e}")

    def enable_movement_buttons(self):
        """Enable/disable movement buttons based on current state"""
        # Get all buttons
        for i in range(self.images_count):
            image_slot = self.image_slots[i]
            up_btn = image_slot.itemAt(0).widget()
            down_btn = image_slot.itemAt(2).widget()

            # Enable up button if not first position
            up_btn.setEnabled(i > 0 and self.images[i] is not None)

            # Enable down button if not last position
            down_btn.setEnabled(i < self.images_count-1 and self.images[i] is not None)

    def move_image_up(self, index: int):
        """Move image to higher position (swap with left neighbor)"""
        if index <= 0 or index >= self.images_count:
            return

        # Swap with previous position
        self.images[index], self.images[index - 1] = self.images[index - 1], self.images[index]

        # Update displays
        self.update_display(index)
        self.update_display(index - 1)
        self.enable_movement_buttons()

        logger.debug(f"Moved image from position {index + 1} to {index}")

    def move_image_down(self, index: int):
        """Move image to lower position (swap with right neighbor)"""
        if index < 0 or index >= self.images_count - 1:
            return

        # Swap with next position
        self.images[index], self.images[index + 1] = self.images[index + 1], self.images[index]

        # Update displays
        self.update_display(index)
        self.update_display(index + 1)
        self.enable_movement_buttons()

        logger.debug(f"Moved image from position {index + 1} to {index + 2}")

    def on_confirm_click(self):
        """Handle confirm button click - calculate 6-way TrueSkill ranking"""
        try:
            # Validate all images are loaded
            if any(img is None for img in self.images):
                QMessageBox.warning(self, "Error", "All 6 positions must have images.")
                return

            self.status_label.setText("Calculating 6-way TrueSkill ranking...")

            # Create TrueSkill environment
            env = trueskill.TrueSkill(draw_probability=0.0)

            # Get current ratings for all 6 images
            ratings = []
            for img in self.images:
                mu, sigma = TrueSkillAnnotationRecordTools.get_ts_values(img)
                ratings.append(env.create_rating(mu=mu, sigma=sigma))

            # Calculate new ratings using 6-way ranking
            # Order: position 0 is best (1st), position 5 is worst (6th)
            try:
                new_ratings = env.rate([
                    (ratings[0],),  # 1st place
                    (ratings[1],),  # 2nd place
                    (ratings[2],),  # 3rd place
                    (ratings[3],),  # 4th place
                    (ratings[4],),  # 5th place
                    (ratings[5],),  # 6th place
                    (ratings[6],),  # 7th place
                    (ratings[7],),   # 8th place
                    (ratings[8],),   # 9th place
                    (ratings[9],)   # 10th place

                ])
            except Exception as e:
                logger.warning(f"TrueSkill calculation failed, using mpmath backend: {e}")
                env = trueskill.TrueSkill(draw_probability=0.0, backend="mpmath")
                ratings = []
                for img in self.images:
                    mu, sigma = TrueSkillAnnotationRecordTools.get_ts_values(img)
                    ratings.append(env.create_rating(mu=mu, sigma=sigma))
                new_ratings = env.rate([(r,) for r in ratings])

            # Update ratings for all 6 images (respecting anchors)
            for i, img in enumerate(self.images):
                is_anchor = img.get_field_val("ankor", False)
                if not is_anchor:
                    data_manager.update_record_rating(img, new_ratings[i][0].mu, new_ratings[i][0].sigma)

            # Cache all pairwise relationships (15 pairs total)
            self.status_label.setText("Caching pairwise results...")

            self.status_label.setText("Ranking confirmed!")
            logger.info("6-way ranking completed successfully")

            # Update displays to show new ratings
            self.update_all_displays()

            # Auto-load next set if enabled
            if self.auto_load_checkbox.isChecked():
                self.load_random_six()
            else:
                self.confirm_btn.setEnabled(False)

        except Exception as e:
            logger.error(f"Error confirming ranking: {e}")
            QMessageBox.warning(self, "Error", f"Failed to confirm ranking: {e}")


