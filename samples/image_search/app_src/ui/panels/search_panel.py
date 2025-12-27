"""
Search panel for entering search queries.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                                QLineEdit, QPushButton, QSpinBox, QListWidget)
from PySide6.QtCore import Signal
from loguru import logger


class SearchPanel(QWidget):
    """
    Panel for image search interface.
    """
    search_requested = Signal(str, int)  # query, count
    
    def __init__(self, title: str, locator, parent=None):
        super().__init__(parent)
        self.locator = locator
        
        # Build UI
        self._setup_ui()
        
        # Load recent searches after UI is ready
        import asyncio
        asyncio.create_task(self._load_recent_searches())
    
    def _setup_ui(self):
        """Build the search panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Title
        title_label = QLabel("<b>Image Search</b>")
        layout.addWidget(title_label)
        
        # Search input
        layout.addWidget(QLabel("Search Query:"))
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter search terms...")
        self.query_input.returnPressed.connect(self._on_search_clicked)
        layout.addWidget(self.query_input)
        
        # Result count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Results:"))
        self.count_spinner = QSpinBox()
        self.count_spinner.setMinimum(1)
        self.count_spinner.setMaximum(100)
        self.count_spinner.setValue(20)
        count_layout.addWidget(self.count_spinner)
        count_layout.addStretch()
        layout.addLayout(count_layout)
        
        # Search button
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._on_search_clicked)
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.search_btn)
        
        layout.addSpacing(20)
        
        # Search history
        layout.addWidget(QLabel("<b>Recent Searches</b>"))
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        layout.addWidget(self.history_list)
        
        layout.addStretch()
    
    def _on_search_clicked(self):
        """Handle search button click."""
        query = self.query_input.text().strip()
        count = self.count_spinner.value()
        
        if not query:
            logger.warning("Empty search query")
            return
        
        logger.info(f"Search requested: '{query}' (count: {count})")
        self.search_requested.emit(query, count)
    
    def _on_history_clicked(self, item):
        """Handle history item click."""
        query = item.text()
        self.query_input.setText(query)
        self._on_search_clicked()
    
    def add_to_history(self, query: str):
        """Add query to history list."""
        items = [self.history_list.item(i).text() 
                for i in range(self.history_list.count())]
        
        if query not in items:
            self.history_list.insertItem(0, query)
            
            # Limit history to 10 items
            while self.history_list.count() > 10:
                self.history_list.takeItem(self.history_list.count() - 1)
    
    def set_searching(self, searching: bool):
        """Enable/disable UI during search."""
        self.search_btn.setEnabled(not searching)
        self.query_input.setEnabled(not searching)
        self.count_spinner.setEnabled(not searching)
        
        if searching:
            self.search_btn.setText("Searching...")
        else:
            self.search_btn.setText("Search")
    
    async def _load_recent_searches(self):
        """Load recent searches from MongoDB."""
        try:
            from app_src.models.search_history import SearchHistory
            
            # Get last 50 searches
            all_searches = await SearchHistory.find({}, limit=50)
            
            # Sort by timestamp descending and get unique queries
            all_searches.sort(key=lambda x: x.timestamp, reverse=True)
            unique_queries = []
            seen = set()
            
            for search in all_searches:
                if search.query not in seen:
                    unique_queries.append(search.query)
                    seen.add(search.query)
                    if len(unique_queries) >= 10:
                        break
            
            # Populate list
            for query in unique_queries:
                self.history_list.addItem(query)
            
            logger.info(f"✅ Loaded {len(unique_queries)} recent searches")
            
        except Exception as e:
            logger.debug(f"Could not load recent searches: {e}")
