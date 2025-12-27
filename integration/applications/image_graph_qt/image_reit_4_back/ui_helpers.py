"""
UI helper functions for dialogs and user interactions
"""
from PySide6.QtWidgets import QMessageBox, QWidget
from typing import Optional


class UIHelpers:
    """Static helper methods for UI dialogs and interactions"""

    @staticmethod
    def show_confirmation_dialog(parent: Optional[QWidget], title: str, message: str) -> bool:
        """Show a Yes/No confirmation dialog

        Args:
            parent: Parent widget
            title: Dialog title
            message: Dialog message

        Returns:
            True if user clicked Yes, False otherwise
        """
        reply = QMessageBox.question(
            parent, title, message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes

    @staticmethod
    def show_info(parent: Optional[QWidget], title: str, message: str):
        """Show an information dialog

        Args:
            parent: Parent widget
            title: Dialog title
            message: Dialog message
        """
        QMessageBox.information(parent, title, message)

    @staticmethod
    def show_warning(parent: Optional[QWidget], title: str, message: str):
        """Show a warning dialog

        Args:
            parent: Parent widget
            title: Dialog title
            message: Dialog message
        """
        QMessageBox.warning(parent, title, message)

    @staticmethod
    def show_error(parent: Optional[QWidget], title: str, message: str):
        """Show an error dialog

        Args:
            parent: Parent widget
            title: Dialog title
            message: Dialog message
        """
        QMessageBox.critical(parent, title, message)
