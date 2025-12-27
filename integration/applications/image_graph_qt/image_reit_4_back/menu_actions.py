"""
Menu action handlers for the Image Rating Application
"""
from loguru import logger
from PySide6.QtWidgets import QFileDialog, QInputDialog, QProgressDialog, QApplication, QMainWindow, QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QCheckBox, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import Qt

from analitics_manager import analytics_manager
from data_manager import data_manager
from ui_helpers import UIHelpers
from anotation_tool import TrueSkillAnnotationRecordTools
from constants import MODEL_SIGMA, DEFAULT_MU
from model_config import model_sigma_config


class MenuActions:
    """Handles menu action operations"""

    def __init__(self, main_window: QMainWindow, main_widget):
        """
        Args:
            main_window: The main application window
            main_widget: The main ImageRatingWidget instance
            data_manager: The centralized data manager
            analytics_manager: The analytics manager instance
        """
        self.main_window = main_window
        self.main_widget = main_widget
        self.analytics_window = None

    def show_analytics_window(self):
        """Show analytics window"""
        from analitics_manager import AnalyticsWindow
        
        if self.analytics_window is None or not self.analytics_window.isVisible():
            self.analytics_window = AnalyticsWindow(analytics_manager, self.main_window)

        self.analytics_window.show()
        self.analytics_window.raise_()
        self.analytics_window.activateWindow()

    def add_folder_to_rating_job(self):
        """Add folder to rating job"""
        try:
            folder_path = QFileDialog.getExistingDirectory(self.main_window, "Select Folder with Images")
            if not folder_path:
                return

            total_images = len(data_manager.all_annotations)
            count, ok = QInputDialog.getInt(
                self.main_window,
                "Select Count",
                f"Enter number to add (1-{total_images}):",
                value=min(10, total_images)
            )

            if ok and count > 0:
                added_count = data_manager.add_folder_to_rating_job(folder_path, count)
                self.main_widget.status_label.setText(f"Added {added_count} images from folder.")

        except Exception as e:
            logger.error(f"Error adding folder: {e}")
            UIHelpers.show_warning(self.main_window, "Error", f"Failed to add folder: {e}")

    def clear_all_cache(self):
        """Clear all caches"""
        if not UIHelpers.show_confirmation_dialog(
            self.main_window,
            'Confirm Action',
            'This will clear all caches. Are you sure?'
        ):
            return

        try:
            self.data_manager.clear_all_caches()
            self.main_widget.status_label.setText("All caches have been cleared.")
            logger.info("All caches cleared by user")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            UIHelpers.show_warning(self.main_window, "Error", f"Failed to clear caches: {e}")

    def clean_cache_duplicates(self):
        """Clean duplicate and orphaned cache entries"""
        try:
            if not UIHelpers.show_confirmation_dialog(
                self.main_window,
                'Confirm Cache Cleanup',
                'This will remove orphaned and duplicate entries from all caches.\n\n'
                'The following will be cleaned:\n'
                '• Orphaned pair entries (for deleted records)\n'
                '• Inconsistent pairs (forward/reverse mismatches)\n'
                '• Orphaned rating cache entries\n\n'
                'This may take some time. Continue?'
            ):
                return

            # Show progress
            progress = QProgressDialog("Cleaning cache duplicates...", None, 0, 0, self.main_window)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()

            # Perform cleanup
            stats = data_manager.clean_cache_duplicates()

            progress.close()

            # Show results
            result_message = "Cache cleanup completed!\n\n"
            result_message += f"• Orphaned pair entries removed: {stats['orphaned_pairs']}\n"
            result_message += f"• Inconsistent pairs fixed: {stats['inconsistent_pairs']}\n"
            result_message += f"• Orphaned TrueSkill entries removed: {stats['orphaned_trueskill']}\n"
            result_message += f"• Orphaned item cache entries removed: {stats['orphaned_items']}\n"
            result_message += f"• Total cache entries cleaned: {stats['total_cleaned']}"

            self.main_widget.status_label.setText(f"Cache cleanup: {stats['total_cleaned']} entries removed.")
            logger.info(f"Cache cleanup completed: {stats}")

            UIHelpers.show_info(self.main_window, "Cleanup Complete", result_message)

        except Exception as e:
            logger.error(f"Error cleaning cache duplicates: {e}")
            UIHelpers.show_warning(self.main_window, "Error", f"Failed to clean cache duplicates: {e}")

    def set_all_sigma_to_default(self):
        """Set all sigma to default value"""
        TrueSkillAnnotationRecordTools.set_all_sigma_to_default(MODEL_SIGMA)
        self.main_widget.status_label.setText("Set all sigma to default value.")

    def set_all_mu_to_default(self):
        """Set all mu to default value"""
        TrueSkillAnnotationRecordTools.set_all_mu_to_default(DEFAULT_MU, MODEL_SIGMA)
        self.main_widget.status_label.setText("Set all mu to default value.")
    
    def configure_model_sigma(self):
        """Configure LLM model sigma settings"""
        try:
            dialog = QDialog(self.main_window)
            dialog.setWindowTitle("Configure Model Sigma")
            dialog.setMinimumWidth(400)
            layout = QVBoxLayout(dialog)
            
            # Info label
            info_label = QLabel(
                "Configure the sigma value used for new items added to the rating list.\n"
                "You can load sigma from a trained model's metadata or set it manually."
            )
            info_label.setWordWrap(True)
            layout.addWidget(info_label)
            
            # Current config display
            config_label = QLabel(model_sigma_config.get_config_summary())
            config_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(config_label)
            
            # Button to load from model
            load_btn = QPushButton("Load from Trained Model...")
            
            def load_from_model():
                model_dir = QFileDialog.getExistingDirectory(dialog, "Select Model Directory")
                if model_dir:
                    if model_sigma_config.load_from_model_metadata(model_dir):
                        sigma_spinbox.setValue(model_sigma_config.calculated_sigma)
                        error_label.setText(f"Error %: {model_sigma_config.model_error_percentage:.4f}")
                        config_label.setText(model_sigma_config.get_config_summary())
                        QMessageBox.information(dialog, "Success", 
                            f"Loaded sigma from model:\n"
                            f"Sigma: {model_sigma_config.calculated_sigma:.4f}\n"
                            f"Error %: {model_sigma_config.model_error_percentage:.4f}")
                    else:
                        QMessageBox.warning(dialog, "Error", 
                            "Failed to load model metadata.\n"
                            "Make sure the directory contains model_metadata.json")
            
            load_btn.clicked.connect(load_from_model)
            layout.addWidget(load_btn)
            
            # Sigma value spinbox
            sigma_layout = QHBoxLayout()
            sigma_layout.addWidget(QLabel("Model Sigma:"))
            sigma_spinbox = QDoubleSpinBox()
            sigma_spinbox.setRange(0.1, 10.0)
            sigma_spinbox.setDecimals(4)
            sigma_spinbox.setValue(model_sigma_config.calculated_sigma)
            sigma_layout.addWidget(sigma_spinbox)
            layout.addLayout(sigma_layout)
            
            # Error percentage display
            error_label = QLabel(f"Error %: {model_sigma_config.model_error_percentage:.4f}")
            layout.addWidget(error_label)
            
            # Checkbox to enable/disable sigma
            use_checkbox = QCheckBox("Use Model Sigma for New Items")
            use_checkbox.setChecked(model_sigma_config.use_llm_sigma)
            use_checkbox.setToolTip(
                "When enabled, new items will use the configured sigma value.\n"
                "When disabled, the default sigma from constants will be used."
            )
            layout.addWidget(use_checkbox)
            
            layout.addWidget(QLabel(""))  # Spacer
            
            # Model inference section
            inference_label = QLabel("Model Inference:")
            inference_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(inference_label)
            
            # Load model button
            load_model_btn = QPushButton("Load Model from HuggingFace")
            model_status_label = QLabel(
                f"Status: {'Loaded' if model_sigma_config.is_model_loaded() else 'Not loaded'}"
            )
            
            def load_model():
                if model_sigma_config.load_model_for_inference():
                    model_status_label.setText("Status: Loaded ✓")
                    model_status_label.setStyleSheet("color: green;")
                    config_label.setText(model_sigma_config.get_config_summary())
                    QMessageBox.information(dialog, "Success", 
                        "Model loaded successfully from HuggingFace!")
                else:
                    model_status_label.setText("Status: Load failed ✗")
                    model_status_label.setStyleSheet("color: red;")
                    QMessageBox.warning(dialog, "Error", 
                        "Failed to load model from HuggingFace.")
            
            load_model_btn.clicked.connect(load_model)
            layout.addWidget(load_model_btn)
            layout.addWidget(model_status_label)
            
            # Unload model button
            unload_model_btn = QPushButton("Unload Model")
            def unload_model():
                model_sigma_config.unload_model()
                model_status_label.setText("Status: Not loaded")
                model_status_label.setStyleSheet("")
                use_predictions_checkbox.setChecked(False)
                config_label.setText(model_sigma_config.get_config_summary())
            
            unload_model_btn.clicked.connect(unload_model)
            layout.addWidget(unload_model_btn)
            
            # Checkbox to enable predictions
            use_predictions_checkbox = QCheckBox("Use Model Predictions for New Items")
            use_predictions_checkbox.setChecked(model_sigma_config.use_predictions)
            use_predictions_checkbox.setToolTip(
                "When enabled, new items will be predicted using the loaded model.\n"
                "The predicted rating (1-10) will be denormalized to TrueSkill mu.\n"
                "Note: Normalization must be run first to store transform parameters."
            )
            layout.addWidget(use_predictions_checkbox)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            save_btn = QPushButton("Save")
            def save_config():
                model_sigma_config.calculated_sigma = sigma_spinbox.value()
                model_sigma_config.use_llm_sigma = use_checkbox.isChecked()
                model_sigma_config.use_predictions = use_predictions_checkbox.isChecked()
                model_sigma_config.save_config()
                
                sigma_status = "enabled" if use_checkbox.isChecked() else "disabled"
                pred_status = "enabled" if use_predictions_checkbox.isChecked() else "disabled"
                self.main_widget.status_label.setText(
                    f"Model sigma: {sigma_spinbox.value():.4f} ({sigma_status}), Predictions: {pred_status}"
                )
                logger.info(f"Model config saved: sigma={sigma_spinbox.value():.4f}, use_sigma={use_checkbox.isChecked()}, use_predictions={use_predictions_checkbox.isChecked()}")
                dialog.accept()
            
            save_btn.clicked.connect(save_config)
            button_layout.addWidget(save_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            
            dialog.exec()
            
        except Exception as e:
            logger.error(f"Error configuring model sigma: {e}")
            QMessageBox.warning(self.main_window, "Error", f"Failed to configure model sigma: {e}")
