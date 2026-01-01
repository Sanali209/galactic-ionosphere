"""
Settings Pages Package

Re-exports all settings page classes for use in SettingsDialog.
"""
from uexplorer_src.ui.dialogs.settings_pages.general_page import GeneralSettingsPage
from uexplorer_src.ui.dialogs.settings_pages.thumbnail_page import ThumbnailSettingsPage
from uexplorer_src.ui.dialogs.settings_pages.ai_page import AISettingsPage
from uexplorer_src.ui.dialogs.settings_pages.search_page import SearchSettingsPage
from uexplorer_src.ui.dialogs.settings_pages.processing_page import ProcessingSettingsPage
from uexplorer_src.ui.dialogs.settings_pages.metadata_page import MetadataSettingsPage

__all__ = [
    "GeneralSettingsPage",
    "ThumbnailSettingsPage",
    "AISettingsPage",
    "SearchSettingsPage",
    "ProcessingSettingsPage",
    "MetadataSettingsPage",
]
