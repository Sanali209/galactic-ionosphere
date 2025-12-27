"""
List view context menu actions
"""
from typing import Optional
from loguru import logger
from PySide6.QtWidgets import QMessageBox, QInputDialog
from PySide6.QtCore import QModelIndex
import trueskill

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from anotation_tool import TrueSkillAnnotationRecordTools
from constants import DEFAULT_MU, MODEL_SIGMA
from ui_helpers import UIHelpers
from rating_helpers import RatingHelpers


class ListViewActions:
    """Actions for list view context menu operations"""

    def __init__(self, parent_widget, model, data_manager):
        """
        Args:
            parent_widget: The parent widget (ImageListRaterWidget)
            model: The ImageRatingModel instance
            data_manager: The centralized data manager
        """
        self.parent = parent_widget
        self.model = model
        self.data_manager = data_manager

    def rate_up(self, index: QModelIndex):
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
                self.data_manager.update_record_rating(winner, new_win.mu, new_win.sigma)
                self.data_manager.update_record_rating(loser, new_lose.mu, new_lose.sigma)

                logger.debug(f"Rated up: {winner.id} vs {loser.id}")

        except Exception as e:
            logger.error(f"Error rating up: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to rate up: {e}")

    def rate_down(self, index: QModelIndex):
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
                self.data_manager.update_record_rating(winner, new_win.mu, new_win.sigma)
                self.data_manager.update_record_rating(loser, new_lose.mu, new_lose.sigma)

                logger.debug(f"Rated down: {loser.id} vs {winner.id}")

        except Exception as e:
            logger.error(f"Error rating down: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to rate down: {e}")

    def reset_sigma(self, index: QModelIndex):
        """Reset sigma for selected item"""
        try:
            record = self.model.data(index)
            if record:
                self.data_manager.update_record_rating(
                    record,
                    record.get_field_val("avg_rating", DEFAULT_MU),
                    MODEL_SIGMA
                )
                logger.debug(f"Reset sigma for record {record.id}")
        except Exception as e:
            logger.error(f"Error resetting sigma: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to reset sigma: {e}")

    def adjust_mu(self, index: QModelIndex, amount: float):
        """Adjust mu for selected item"""
        try:
            record = self.model.data(index)
            if record:
                current_mu = record.get_field_val("avg_rating", DEFAULT_MU)
                new_mu = current_mu + amount
                self.data_manager.update_record_rating(
                    record,
                    new_mu,
                    record.get_field_val("trueskill_sigma", MODEL_SIGMA)
                )
                logger.debug(f"Adjusted mu for record {record.id} by {amount}")
        except Exception as e:
            logger.error(f"Error adjusting mu: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to adjust mu: {e}")

    def reset_all_sigma(self):
        """Reset sigma for all items"""
        try:
            for record in self.model.records:
                if not record.get_field_val("ankor"):
                    self.data_manager.update_record_rating(
                        record,
                        record.get_field_val("avg_rating", DEFAULT_MU),
                        MODEL_SIGMA
                    )
            logger.debug("Reset all sigma values")
        except Exception as e:
            logger.error(f"Error resetting all sigma: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to reset all sigma: {e}")

    def mark_as_anchor(self, index: QModelIndex):
        """Mark item as anchor"""
        try:
            record = self.model.data(index)
            if record:
                record.set_field_val("ankor", True)
                logger.debug(f"Marked record {record.id} as anchor")
        except Exception as e:
            logger.error(f"Error marking as anchor: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to mark as anchor: {e}")

    def remove_anchor(self, index: QModelIndex):
        """Remove anchor mark"""
        try:
            record = self.model.data(index)
            if record:
                record.set_field_val("ankor", False)
                logger.debug(f"Removed anchor mark from record {record.id}")
        except Exception as e:
            logger.error(f"Error removing anchor: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to remove anchor: {e}")

    def delete_from_list(self, index: QModelIndex):
        """Delete item from list"""
        try:
            record = self.model.data(index)
            if record:
                record.delete_rec()
                # Force data manager to reload the manual_voted_list from database
                self.data_manager.load_manual_voted_list()
                # Refresh data after deletion
                if hasattr(self.parent, 'refresh_data'):
                    self.parent.refresh_data()
                logger.debug(f"Deleted record {record.id} from list")
        except Exception as e:
            logger.error(f"Error deleting from list: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to delete from list: {e}")

    def solve_edges_for_item(self, index: QModelIndex):
        """Solve edges for the selected item"""
        try:
            record = self.model.data(index)
            if record:
                # Get the main widget from parent widgets
                parent = self.parent.parent()
                while parent and not hasattr(parent, 'solve_edges_with_threshold'):
                    parent = parent.parent()

                if parent and hasattr(parent, 'solve_edges_with_threshold'):
                    # Call the main widget's solve edges method with this specific record
                    parent.solve_edges_with_threshold(record=record)
                else:
                    UIHelpers.show_warning(
                        self.parent,
                        "Error",
                        "Cannot find main widget to perform edge solving"
                    )

                logger.debug(f"Solved edges for record {record.id}")
        except Exception as e:
            logger.error(f"Error solving edges for item: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to solve edges for item: {e}")

    def normalize_with_midpoint(self, index: QModelIndex):
        """Normalize ratings with the selected item as midpoint (5.0)"""
        try:
            record = self.model.data(index)
            if record:
                # Get the main widget from parent widgets
                parent = self.parent.parent()
                while parent and not hasattr(parent, 'normalize_ratings_with_midpoint'):
                    parent = parent.parent()

                if parent and hasattr(parent, 'normalize_ratings_with_midpoint'):
                    # Call the main widget's normalize method with this specific record as midpoint
                    parent.normalize_ratings_with_midpoint(record)
                else:
                    UIHelpers.show_warning(
                        self.parent,
                        "Error",
                        "Cannot find main widget to perform normalization"
                    )

                logger.debug(f"Normalized ratings with record {record.id} as midpoint")
        except Exception as e:
            logger.error(f"Error normalizing with midpoint: {e}")
            UIHelpers.show_warning(self.parent, "Error", f"Failed to normalize with midpoint: {e}")
