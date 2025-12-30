"""
DashboardViewModel - Aggregates system statistics for the Dashboard.

Fetches data from:
- MaintenanceService (DB counts)
- ProcessingPipeline (Queue sizes)
- FAISSService (Vector index status)
- DetectionService (Detection counts)
- TaskSystem (Recent tasks)
"""
from typing import Dict, Any, List, Optional
from PySide6.QtCore import Signal, QTimer
from loguru import logger
import asyncio
import time

from src.ui.mvvm.document_viewmodel import DocumentViewModel
from src.ui.cardview.models.card_item import CardItem

class DashboardViewModel(DocumentViewModel):
    """
    ViewModel for DashboardDocument.
    Aggregates system stats into 'CardItems' for display.
    """
    
    # Signals
    stats_updated = Signal()  # Data refreshed
    items_changed = Signal(list)  # For CardView (List[CardItem])
    
    def __init__(self, doc_id: str, locator=None):
        super().__init__(doc_id, locator)
        self._title = "Dashboard"
        
        # Data Cache
        self._stats: Dict[str, Any] = {}
        self._items: List[CardItem] = []
        
        # Auto-refresh timer (every 10s)
        self._refresh_timer = QTimer()
        self._refresh_timer.setInterval(10000)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start()
        
        # Initial load
        QTimer.singleShot(100, self.refresh)
        
        logger.debug("DashboardViewModel initialized")

    def refresh(self):
        """Trigger async data refresh."""
        asyncio.create_task(self._fetch_all_stats())

    async def _fetch_all_stats(self):
        """Fetch stats from all services."""
        try:
            new_items = []
            
            # 1. Core Counts (Maintenance/DB)
            await self._add_core_counts(new_items)
            
            # 2. Pipeline Status
            await self._add_pipeline_stats(new_items)
            
            # 3. Vector Search Status
            await self._add_vector_stats(new_items)
            
            # 4. Storage/Cache
            await self._add_storage_stats(new_items)
             
            # 5. Actions
            self._add_action_cards(new_items)

            self._items = new_items
            self.items_changed.emit(new_items)
            self.stats_updated.emit()
            logger.debug("Dashboard stats refreshed")
            
        except Exception as e:
            logger.error(f"Dashboard refresh failed: {e}")

    async def _add_core_counts(self, items: List[CardItem]):
        """Fetch and add core entity counts."""
        try:
            from src.ucorefs.models.file_record import FileRecord
            from src.ucorefs.tags.models import Tag
            from src.ucorefs.albums.models import Album
            
            file_count = await FileRecord.count()
            tag_count = await Tag.count()
            album_count = await Album.count()
            
            items.extend([
                self._create_stat_card("files", "Total Files", str(file_count), "library-books", "core"),
                self._create_stat_card("tags", "Total Tags", str(tag_count), "label", "core"),
                self._create_stat_card("albums", "Albums", str(album_count), "photo-album", "core")
            ])
        except Exception as e:
            logger.error(f"Failed to fetch core counts: {e}")

    async def _add_pipeline_stats(self, items: List[CardItem]):
        """Fetch processing pipeline queue sizes."""
        try:
            from src.ucorefs.processing.pipeline import ProcessingPipeline
            pipeline = self.locator.get_system(ProcessingPipeline)
            
            # Access internal sets if available (or add public getters later)
            pending_p2 = len(getattr(pipeline, '_phase2_pending', []))
            pending_p3 = len(getattr(pipeline, '_phase3_pending', []))
            
            items.append(self._create_progress_card(
                "pipeline", "Processing Queue", 
                f"{pending_p2 + pending_p3} Pending", 
                pending_p2 + pending_p3, 1000, # arbitrary max for progress visualization
                "pipeline"
            ))
        except KeyError:
            pass # Service not found

    async def _add_vector_stats(self, items: List[CardItem]):
        """Fetch FAISS index stats."""
        try:
            from src.ucorefs.vectors.faiss_service import FAISSIndexService
            faiss_service = self.locator.get_system(FAISSIndexService)
            
            stats = await faiss_service.get_index_stats()
            indexes = stats.get("indexes", {})
            
            for provider, info in indexes.items():
                size = info.get("size", 0)
                loaded = "Active" if info.get("loaded") else "Idle"
                items.append(self._create_stat_card(
                    f"vec_{provider}", f"Vector Index ({provider})", 
                    f"{size} vecs", "share-variant", "vector",
                    subtitle=loaded
                ))
        except KeyError:
            pass

    async def _add_storage_stats(self, items: List[CardItem]):
        """Fetch cache/storage stats."""
        # TODO: Implement actual disk usage check
        pass

    def _add_action_cards(self, items: List[CardItem]):
        """Add action buttons."""
        items.append(CardItem(
            id="action_optimize",
            title="Optimize Database",
            subtitle="Compact & Reindex",
            item_type="action",
            data={"action": "optimize_db", "icon": "database-check"}
        ))
        items.append(CardItem(
            id="action_recount",
            title="Recalculate Counts",
            subtitle="Fix Sync Issues",
            item_type="action",
            data={"action": "recount", "icon": "calculator"}
        ))

    def _create_stat_card(self, id: str, title: str, value: str, icon: str, group: str, subtitle: str = "") -> CardItem:
        return CardItem(
            id=id,
            title=title,
            subtitle=subtitle,
            item_type="stat",
            data={"value": value, "icon": icon, "group": group}
        )

    def _create_progress_card(self, id: str, title: str, value_text: str, current: int, total: int, group: str) -> CardItem:
        return CardItem(
            id=id,
            title=title,
            subtitle=value_text,
            item_type="progress",
            data={"current": current, "total": total, "group": group}
        )

    # Action Handlers
    def trigger_action(self, action_id: str):
        """Handle action card clicks."""
        logger.info(f"Dashboard action triggered: {action_id}")
        
        if action_id == "optimize_db":
            asyncio.create_task(self._run_maintenance_task("maintenance_database_optimization"))
        elif action_id == "recount":
            asyncio.create_task(self._run_maintenance_task("maintenance_background_verification"))

    async def _run_maintenance_task(self, task_name: str):
        """Run a maintenance task via TaskSystem."""
        try:
            from src.core.tasks.system import TaskSystem
            task_system = self.locator.get_system(TaskSystem)
            
            # Fire and forget (TaskSystem handles execution)
            await task_system.create_task(task_name, {}, priority=10)
            logger.info(f"Started maintenance task: {task_name}")
        except Exception as e:
            logger.error(f"Failed to start task {task_name}: {e}")
