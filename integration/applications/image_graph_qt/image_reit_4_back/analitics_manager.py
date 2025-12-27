import json
import time
from typing import Dict, Any, Tuple, List

import numpy as np
from PySide6.QtCore import QMutex
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget, QMainWindow, QVBoxLayout, QToolBar, QHBoxLayout, QPushButton, QGridLayout, \
    QLabel, QFileDialog, QMessageBox

from data_manager import data_manager
from constants import DEFAULT_MU, MODEL_SIGMA

# Try to import matplotlib for charts, fallback gracefully if not available
try:
    import matplotlib

    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    FigureCanvas = None
    Figure = None

from loguru import logger


class AnalyticsManager:
    """Manages analytics data calculation and storage"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self._comparison_count = 0
        self._mutex = QMutex()

    def get_rating_statistics(self) -> Dict[str, Any]:
        """Get current rating statistics"""
        self._mutex.lock()
        try:
            if not self.data_manager.manual_voted_list:
                return {"error": "No data available"}

            ratings = [rec.get_field_val("avg_rating", DEFAULT_MU)
                       for rec in self.data_manager.manual_voted_list]
            sigmas = [rec.get_field_val("trueskill_sigma", MODEL_SIGMA)
                      for rec in self.data_manager.manual_voted_list]

            return {
                "total_records": len(ratings),
                "mean_rating": np.mean(ratings),
                "std_rating": np.std(ratings),
                "min_rating": np.min(ratings),
                "max_rating": np.max(ratings),
                "mean_sigma": np.mean(sigmas),
                "min_sigma": np.min(sigmas),
                "max_sigma": np.max(sigmas),
                "comparison_count": self._comparison_count,
                "anchor_count": len([rec for rec in self.data_manager.manual_voted_list
                                     if rec.get_field_val("ankor", False)])
            }
        finally:
            self._mutex.unlock()

    def get_rating_distribution(self) -> Tuple[List[float], List[float]]:
        """Get rating distribution for histogram"""
        self._mutex.lock()
        try:
            ratings = [rec.get_field_val("avg_rating", DEFAULT_MU)
                       for rec in self.data_manager.manual_voted_list]

            if not ratings:
                return [], []

            hist, bins = np.histogram(ratings, bins=20)
            return hist.tolist(), bins.tolist()
        finally:
            self._mutex.unlock()

    def increment_comparison_count(self):
        """Increment the comparison counter"""
        self._mutex.lock()
        try:
            self._comparison_count += 1
        finally:
            self._mutex.unlock()


# Global analytics manager
analytics_manager = AnalyticsManager(data_manager)


class RatingDistributionChart(QWidget):
    """Chart showing rating distribution histogram"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Set basic style
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
        except:
            pass

    def update_chart(self, hist_data: Tuple[List[float], List[float]]):
        """Update histogram with rating distribution data"""
        hist, bins = hist_data
        if not hist or not bins:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Create histogram
        ax.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Rating Distribution')
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


class AnalyticsWindow(QMainWindow):
    """Separate window for analytics and statistics"""

    def __init__(self, analytics_manager: AnalyticsManager, parent=None):
        super().__init__(parent)
        self.analytics_manager = analytics_manager
        self.setWindowTitle("Image Rating Analytics")
        self.setGeometry(100, 100, 1000, 700)

        self.init_ui()
        self.setup_refresh_timer()

    def init_ui(self):
        """Initialize the analytics UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_analytics)
        toolbar.addAction(refresh_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_analytics)
        toolbar.addAction(export_action)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Rating distribution chart
        self.rating_dist_chart = RatingDistributionChart()
        main_layout.addWidget(self.rating_dist_chart)

        # Statistics summary
        self.stats_widget = self.create_statistics_summary()
        main_layout.addWidget(self.stats_widget)

        # Controls
        controls_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_analytics)
        controls_layout.addWidget(self.refresh_button)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_analytics)
        controls_layout.addWidget(self.export_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

    def create_statistics_summary(self) -> QWidget:
        """Create statistics summary widget"""
        widget = QWidget()
        layout = QGridLayout(widget)

        self.stats_labels = {}
        stats_to_show = [
            ("Total Records", "total_records"),
            ("Mean Rating", "mean_rating"),
            ("Std Dev", "std_rating"),
            ("Mean Sigma", "mean_sigma"),
            ("Comparisons", "comparison_count"),
            ("Anchors", "anchor_count")
        ]

        for i, (display_name, key) in enumerate(stats_to_show):
            label = QLabel(f"{display_name}:")
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: bold; color: blue;")

            row, col = i // 3, (i % 3) * 2
            layout.addWidget(label, row, col)
            layout.addWidget(value_label, row, col + 1)

            self.stats_labels[key] = value_label

        return widget

    def setup_refresh_timer(self):
        """Setup manual refresh (no auto-timer)"""
        pass  # No auto-refresh - only manual refresh

    def refresh_analytics(self):
        """Refresh all analytics data"""
        try:
            # Update statistics summary
            stats = self.analytics_manager.get_rating_statistics()
            self.update_statistics_display(stats)

            # Update chart
            self.update_chart()

        except Exception as e:
            logger.error(f"Error refreshing analytics: {e}")

    def update_statistics_display(self, stats: Dict[str, Any]):
        """Update the statistics display"""
        if "error" in stats:
            for label in self.stats_labels.values():
                label.setText("N/A")
            return

        for key, label in self.stats_labels.items():
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    label.setText(f"{value:.2f}")
                else:
                    label.setText(str(value))

    def update_chart(self):
        """Update the rating distribution chart"""
        hist_data = self.analytics_manager.get_rating_distribution()
        self.rating_dist_chart.update_chart(hist_data)

    def export_analytics(self):
        """Export analytics data to file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Analytics", "", "JSON Files (*.json)")
            if not file_path:
                return

            stats = self.analytics_manager.get_rating_statistics()
            hist_data = self.analytics_manager.get_rating_distribution()

            export_data = {
                "timestamp": time.time(),
                "statistics": stats,
                "rating_distribution": {
                    "histogram": hist_data[0],
                    "bins": hist_data[1]
                }
            }

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=4)

            QMessageBox.information(self, "Export Complete", f"Analytics exported to {file_path}")

        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            QMessageBox.warning(self, "Export Error", f"Failed to export analytics: {e}")
