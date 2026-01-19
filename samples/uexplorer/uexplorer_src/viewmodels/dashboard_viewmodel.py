"""
DashboardViewModel - Aggregates system statistics for the Dashboard.

Fetches data from:
- MaintenanceService (DB counts)
- ProcessingPipeline (Queue sizes)
- FAISSService (Vector index status)
- DetectionService (Detection counts)
- TaskSystem (Recent tasks)
"""
from typing import TYPE_CHECKING, Dict, Any, List, Optional
from PySide6.QtCore import Signal, QTimer
from loguru import logger
import asyncio
import time

from src.ui.mvvm.document_viewmodel import DocumentViewModel
from src.ui.cardview.models.card_item import CardItem

if TYPE_CHECKING:
    from src.core.service_locator import ServiceLocator

class DashboardViewModel(DocumentViewModel):
    """
    ViewModel for DashboardDocument.
    Aggregates system stats into 'CardItems' for display.
    """
    
    # Signals
    stats_updated = Signal()  # Data refreshed
    items_changed = Signal(list)  # For CardView (List[CardItem])
    
    def __init__(self, doc_id: str, locator: Optional["ServiceLocator"] = None) -> None:
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
        
        self.initialize_reactivity()
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
        """Fetch processing pipeline queue sizes via EngineProxy."""
        try:
            from src.core.engine.proxy import EngineProxy
            engine_proxy = self.locator.get_system(EngineProxy)
            
            if not engine_proxy:
                return
            
            async def _get_pipeline_stats():
                from src.core.locator import get_active_locator
                from src.ucorefs.processing.pipeline import ProcessingPipeline
                sl = get_active_locator()
                pipeline = sl.get_system(ProcessingPipeline)
                
                # Access internal sets (or add public getters later)
                pending_p2 = len(getattr(pipeline, '_phase2_pending', []))
                pending_p3 = len(getattr(pipeline, '_phase3_pending', []))
                return pending_p2, pending_p3
            
            future = engine_proxy.submit(_get_pipeline_stats())
            import asyncio
            pending_p2, pending_p3 = await asyncio.wrap_future(future)
            
            items.append(self._create_progress_card(
                "pipeline", "Processing Queue", 
                f"{pending_p2 + pending_p3} Pending", 
                pending_p2 + pending_p3, 1000, # arbitrary max for progress visualization
                "pipeline"
            ))
        except Exception as e:
            logger.debug(f"Pipeline stats not available: {e}")

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
            from src.core.engine.proxy import EngineProxy
            engine = self.locator.get_system(EngineProxy)
            
            # Fire and forget (Engine handles execution)
            engine.submit_task(task_name, priority=10)
            logger.info(f"Started maintenance task: {task_name}")
        except Exception as e:
            logger.error(f"Failed to start task {task_name}: {e}")

    # === Reactive SSOT Implementation ===

    @property
    def _event_bus(self):
        """Lazy access to EventBus."""
        from src.core.events import EventBus
        try:
            return self.locator.get_system(EventBus)
        except Exception:
            return None

    def initialize_reactivity(self):
        """Subscribe to database events."""
        bus = self._event_bus
        if bus:
             # Listen to all relevant updates
             bus.subscribe("db.file_records.updated", self._on_db_change)
             bus.subscribe("db.file_records.deleted", self._on_db_change)
             bus.subscribe("db.tags.updated", self._on_db_change)
             bus.subscribe("db.tags.deleted", self._on_db_change)
             logger.debug("DashboardViewModel: Reactivity initialized")

    def _on_db_change(self, data: dict):
        """
        Handle DB changes by scheduling a refresh.
        Debounced by the fact that refresh() is async and we just fire it.
        Ideally we should use a proper debounce if high volume.
        For now, simply relying on async execution loop.
        """
        # Simple debounce: check if already refreshing? 
        # Actually refresh() just spawns a task.
        # We can implement a dirty flag or just let it refresh.
        # To avoid spamming, let's delay it slightly and use QTimer if safe, 
        # or just call refresh() which is cheap enough (lazy loading not really, counts are DB queries).
        
        # Debounce logic: trigger singleShot timer (resets if called again)
        # Note: In Qt QTimer needs main thread. Bus callbacks might be async.
        # Safest is to just call refresh() but maybe throttle.
        self.refresh()
