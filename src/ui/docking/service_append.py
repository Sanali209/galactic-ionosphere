        return sidebar_map.get(sidebar, QtAds.SideBarRight)
    
    # BaseSystem interface compatibility
    async def initialize(self):
        """Initialize service (already done in __init__)."""
        pass
    
    async def shutdown(self):
        """Cleanup docking manager."""
        logger.info("DockingService shutting down")
