"""
Data I/O Operations Module

This module contains operations for importing and exporting rating data.
"""

import json
from typing import Optional
from loguru import logger
from tqdm import tqdm
from PySide6.QtWidgets import QFileDialog, QMessageBox

from SLM.files_db.annotation_tool.annotation import AnnotationRecord
from SLM.files_db.components.File_record_wraper import FileRecord
from constants import MODEL_SIGMA, DEFAULT_MU


class DataIOOperations:
    """Static class for data import/export operations"""

    @staticmethod
    def export_data_json(parent_widget, data_manager) -> Optional[str]:
        """Export ratings data to JSON
        
        Args:
            parent_widget: Parent widget for dialogs
            data_manager: Data manager instance
            
        Returns:
            Status message or None if cancelled/failed
        """
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                parent_widget, 
                "Export Ratings as JSON", 
                "", 
                "JSON Files (*.json)"
            )
            if not file_path:
                return None

            export_data = {
                "job_scale_factor": data_manager.rating_job.get_field_val("transform_multiplier", 1.0),
                "records": []
            }

            manual_records = AnnotationRecord.find({
                'parent_id': data_manager.rating_job.id, 
                'manual': True
            })

            for rec in tqdm(manual_records, desc="Exporting manual records"):
                rec: AnnotationRecord
                md5 = rec.file.file_content_md5
                if not md5:
                    logger.warning(f"No MD5 for record {rec.id}. Skipping.")
                    continue

                record_data = {
                    "md5": md5,
                    "avg_rating": rec.get_field_val("avg_rating", DEFAULT_MU),
                    "trueskill_mu": rec.get_field_val("avg_rating", DEFAULT_MU),
                    "trueskill_sigma": rec.get_field_val("trueskill_sigma", MODEL_SIGMA)
                }
                export_data["records"].append(record_data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)

            logger.info(f"Exported {len(export_data['records'])} records to {file_path}")
            return f"Exported {len(export_data['records'])} manual records to {file_path}"

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to export data: {e}")
            return None

    @staticmethod
    def import_data_json(parent_widget, data_manager) -> Optional[str]:
        """Import ratings data from JSON
        
        Args:
            parent_widget: Parent widget for dialogs
            data_manager: Data manager instance
            
        Returns:
            Status message or None if cancelled/failed
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                parent_widget, 
                "Import Ratings from JSON", 
                "", 
                "JSON Files (*.json)"
            )
            if not file_path:
                return None

            # Clear existing manual flags
            manual_records = AnnotationRecord.find({
                'parent_id': data_manager.rating_job.id, 
                'manual': True
            })
            for rec in manual_records:
                rec.set_field_val("manual", None)

            with open(file_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            job_scale_factor = import_data.get("job_scale_factor", 1.0)
            data_manager.rating_job.set_field_val("transform_multiplier", job_scale_factor)

            imported_count = 0
            for record_data in tqdm(import_data.get("records", []), desc="Importing records"):
                md5 = record_data.get("md5")
                if not md5:
                    continue

                file = FileRecord.find_one({"file_content_md5": md5})
                if not file:
                    logger.warning(f"No file record found for MD5 {md5}. Skipping.")
                    continue

                rec = AnnotationRecord.find_one({
                    "file_id": file.id, 
                    "parent_id": data_manager.rating_job.id
                })
                if rec:
                    rec.set_field_val("avg_rating", record_data.get("avg_rating", DEFAULT_MU))
                    sigma = record_data.get("trueskill_sigma", MODEL_SIGMA)
                    rec.set_field_val("trueskill_sigma", sigma)
                    rec.set_field_val("manual", True)
                    imported_count += 1
                else:
                    logger.warning(f"No annotation record found for MD5 {md5}. Skipping.")

            # Refresh data
            data_manager.load_manual_voted_list()
            logger.info(f"Imported {imported_count} records from {file_path}")
            return f"Imported {imported_count} records from {file_path}"

        except Exception as e:
            logger.error(f"Error importing data: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to import data: {e}")
            return None

    @staticmethod
    def keep_100_rand_human_voted(parent_widget, data_manager) -> Optional[str]:
        """Keep only 100 random human voted items
        
        Args:
            parent_widget: Parent widget for dialogs
            data_manager: Data manager instance
            
        Returns:
            Status message or None if failed
        """
        try:
            import random
            
            query = {'parent_id': data_manager.rating_job.id, 'manual': {"$ne": None}}
            result = AnnotationRecord.find(query)
            
            if len(result) > 100:
                result_human_voted = random.sample(result, 100)
                # Remove manual flag from items not in the selected 100
                for item in result:
                    if item not in result_human_voted:
                        item.set_field_val("manual", None)
            
            data_manager.load_manual_voted_list()
            logger.info("Kept 100 random human voted items")
            return "Kept 100 random human voted items."

        except Exception as e:
            logger.error(f"Error keeping 100 random items: {e}")
            QMessageBox.warning(parent_widget, "Error", f"Failed to keep 100 random items: {e}")
            return None
