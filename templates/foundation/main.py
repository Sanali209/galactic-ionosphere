import sys
import asyncio
from PySide6.QtWidgets import QApplication
from qasync import QEventLoop

from src.core.locator import sl
from src.core.logging import setup_logging
from src.core.assets.manager import AssetManager
from src.core.journal.service import JournalService
from src.core.tasks.system import TaskSystem
from src.core.commands.bus import CommandBus
from src.core.database.manager import db_manager

from src.ui.main_window import MainWindow
from src.ui.viewmodels.main_viewmodel import MainViewModel
from src.ui.mvvm.provider import ViewModelProvider


async def main():
    # 1. Setup Logging
    setup_logging()
    
    # 2. Init Locator (Config, EventBus)
    sl.init()
    
    # 3. Connect Database
    # In a real app, read from config
    await db_manager.connect("localhost", 27017, "foundation_demo")
    
    # 4. Register Systems
    sl.register_system(CommandBus)
    sl.register_system(JournalService)
    sl.register_system(AssetManager)
    sl.register_system(TaskSystem)
    
    # 5. Start Systems
    await sl.start_all()
    
    # 6. Init UI
    app = QApplication(sys.argv)
    
    provider = ViewModelProvider(sl)
    main_vm = provider.get(MainViewModel)
    
    window = MainWindow(main_vm)

    window.show()
    
    # 7. Run Loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    try:
        await loop.run_forever()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await sl.stop_all()
        db_manager.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
