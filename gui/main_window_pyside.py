"""
Main Window Module (PySide6)
============================

Main GUI window using PySide6 instead of tkinter.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QSpinBox, QCheckBox,
    QListWidget, QTextEdit, QTreeWidget, QTreeWidgetItem, QScrollArea,
    QGroupBox, QFileDialog, QMessageBox, QFrame, QInputDialog, QMenuBar, QMenu,
    QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QPoint, QSize
from PySide6.QtGui import QFont, QIcon, QCursor, QPalette, QColor, QPainter, QPixmap
from PySide6.QtWidgets import QStyleFactory
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
from datetime import datetime

from run_rj_loop import AVAILABLE_PLANS
from gui.plan_editor_pyside import PlanEditor, PlanEntry
from gui.config_manager_pyside import ConfigManager
from gui.launcher_pyside import RuneLiteLauncher
from gui.client_detector_pyside import ClientDetector
from gui.instance_manager_pyside import InstanceManager
from gui.statistics_pyside import StatisticsDisplay
from gui.logging_utils_pyside import LoggingUtils
from helpers.ipc import IPCClient
from utils.stats_monitor import StatsMonitor


class SimpleRecorderGUI(QMainWindow):
    """Main GUI application window using PySide6."""
    
    def __init__(self):
        """Initialize the main GUI window."""
        super().__init__()
        # Remove window title bar - menu bar will be topmost
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setWindowTitle("Simple Recorder - Plan Runner")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set minimum window size
        self.setMinimumSize(800, 600)
        
        # Track window state for dragging and resizing
        self._drag_position = None
        self._resize_edge = None  # Track which edge is being resized
        self._resize_start_pos = None
        self._resize_start_geometry = None
        self._resize_border_width = 8  # Width of resize border in pixels
        
        # Current theme
        self._current_theme = "dark"
        
        # Create custom top bar with menu bar and window controls
        self._create_custom_top_bar()
        
        # Setup file logging first (before anything else)
        self._setup_file_logging()
        
        # Initialize variables
        self._init_variables()
        
        # Initialize component managers
        self._init_components()
        
        # Create main widget and layout
        self._create_main_widget()
        
        # Install event filter on central widget and enable mouse tracking for cursor changes
        if self.centralWidget():
            self.centralWidget().installEventFilter(self)
            self.centralWidget().setMouseTracking(True)
            # Also install on all child widgets recursively
            self._install_cursor_tracking(self.centralWidget())
        
        # Apply initial theme (dark)
        QTimer.singleShot(100, lambda: self._apply_theme("Dark", "dark", QColor(53, 53, 53), QColor(61, 61, 61)))
        
        # Detect running instances on startup (after a short delay to ensure everything is initialized)
        QTimer.singleShot(1000, self._detect_running_instances_on_startup)
    
    def _create_custom_top_bar(self):
        """Create a custom top bar widget containing menu bar and window controls."""
        # Create a custom draggable menu bar
        class DraggableMenuBar(QMenuBar):
            def __init__(self, parent_window):
                super().__init__()
                self.parent_window = parent_window
                self.setMouseTracking(True)
            
            def mousePressEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton:
                    # Check if clicking on a menu item
                    action = self.actionAt(event.pos())
                    if action is not None:
                        # Clicking on menu - let parent handle it normally
                        super().mousePressEvent(event)
                        return
                    # Not clicking on menu - start dragging
                    self.parent_window._drag_position = event.globalPosition().toPoint() - self.parent_window.frameGeometry().topLeft()
                    event.accept()
                    return  # Don't call super() - we're handling it
                super().mousePressEvent(event)
            
            def mouseMoveEvent(self, event):
                if self.parent_window._drag_position is not None and event.buttons() == Qt.MouseButton.LeftButton:
                    # We're dragging - move the window
                    self.parent_window.move(event.globalPosition().toPoint() - self.parent_window._drag_position)
                    event.accept()
                    return  # Don't call super() - we're handling it
                super().mouseMoveEvent(event)
            
            def mouseReleaseEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton and self.parent_window._drag_position is not None:
                    self.parent_window._drag_position = None
                    event.accept()
                    return  # Don't call super() - we handled it
                super().mouseReleaseEvent(event)
        
        # Replace the default menu bar with our custom draggable one
        menubar = DraggableMenuBar(self)
        self.setMenuBar(menubar)
        
        # Make menu bar opaque so background color shows
        menubar.setAutoFillBackground(True)
        
        # Menu bar styling will be set when theme is applied - don't set hardcoded colors here
        # Initial styling will be applied via _apply_theme on startup
        
        # Don't install event filter - DraggableMenuBar handles mouse events directly
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("&New Plan...", self._on_new_plan, "Ctrl+N")
        file_menu.addAction("&Open Plan...", self._on_open_plan, "Ctrl+O")
        file_menu.addAction("&Save Plan", self._on_save_plan, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("&Preferences...", self._on_preferences)
        edit_menu.addSeparator()
        edit_menu.addAction("&Clear Log", self._on_clear_log)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("&Refresh Instances", self._on_refresh_instances, "F5")
        view_menu.addSeparator()
        view_menu.addAction("&Show Statistics", self._on_show_statistics)
        view_menu.addAction("&Show Plan Editor", self._on_show_plan_editor)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction("&Launch RuneLite Instances", self._on_launch_runelite)
        tools_menu.addAction("&Stop All Instances", self._on_stop_all_instances)
        tools_menu.addSeparator()
        tools_menu.addAction("&Detect Active Instances", self._on_detect_instances)
        
        # Themes menu
        themes_menu = menubar.addMenu("&Themes")
        self._create_themes_menu(themes_menu)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About", self._on_about)
        help_menu.addAction("&About Qt", QApplication.aboutQt)
        
        # Create a custom draggable top bar widget class
        class DraggableTopBar(QWidget):
            def __init__(self, parent_window):
                super().__init__()
                self.parent_window = parent_window
                self.setMouseTracking(True)
            
            def mousePressEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton:
                    # Check if clicking on window controls (right side)
                    if event.pos().x() > self.width() - 138:
                        super().mousePressEvent(event)
                        return
                    # Check if clicking on menu bar area - if so, let menu bar handle it first
                    # But if menu bar doesn't handle it (no action), we'll drag
                    menubar = self.parent_window.menuBar()
                    top_bar_global = self.mapToGlobal(event.pos())
                    menubar_local = menubar.mapFromGlobal(top_bar_global)
                    action = menubar.actionAt(menubar_local)
                    if action is not None:
                        # Clicking on a menu item - let menu bar handle it
                        super().mousePressEvent(event)
                        return
                    # Not clicking on menu or controls - start dragging
                    self.parent_window._drag_position = event.globalPosition().toPoint() - self.parent_window.frameGeometry().topLeft()
                    event.accept()
                    return  # Don't call super() - we're handling it
                super().mousePressEvent(event)
            
            def mouseMoveEvent(self, event):
                if self.parent_window._drag_position is not None and event.buttons() == Qt.MouseButton.LeftButton:
                    # We're dragging - move the window
                    self.parent_window.move(event.globalPosition().toPoint() - self.parent_window._drag_position)
                    event.accept()
                    return  # Don't call super() - we're handling it
                super().mouseMoveEvent(event)
            
            def mouseReleaseEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton and self.parent_window._drag_position is not None:
                    self.parent_window._drag_position = None
                    event.accept()
                    return  # Don't call super() - we handled it
                super().mouseReleaseEvent(event)
        
        # Also create a custom menu bar that allows dragging
        class DraggableMenuBar(QMenuBar):
            def __init__(self, parent_window):
                super().__init__()
                self.parent_window = parent_window
                self.setMouseTracking(True)
            
            def mousePressEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton:
                    # Check if clicking on a menu item
                    action = self.actionAt(event.pos())
                    if action is not None:
                        # Clicking on menu - let parent handle it normally
                        super().mousePressEvent(event)
                        return
                    # Not clicking on menu - start dragging
                    self.parent_window._drag_position = event.globalPosition().toPoint() - self.parent_window.frameGeometry().topLeft()
                    event.accept()
                    return  # Don't call super() - we're handling it
                super().mousePressEvent(event)
            
            def mouseMoveEvent(self, event):
                if self.parent_window._drag_position is not None and event.buttons() == Qt.MouseButton.LeftButton:
                    # We're dragging - move the window
                    self.parent_window.move(event.globalPosition().toPoint() - self.parent_window._drag_position)
                    event.accept()
                    return  # Don't call super() - we're handling it
                super().mouseMoveEvent(event)
            
            def mouseReleaseEvent(self, event):
                if event.button() == Qt.MouseButton.LeftButton and self.parent_window._drag_position is not None:
                    self.parent_window._drag_position = None
                    event.accept()
                    return  # Don't call super() - we handled it
                super().mouseReleaseEvent(event)
        
        top_bar = DraggableTopBar(self)
        top_bar.setFixedHeight(30)
        # Make top bar transparent so menu bar background shows through
        top_bar.setStyleSheet("background-color: transparent;")
        top_bar.setAutoFillBackground(False)
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(0)
        
        # Add menu bar (will expand)
        top_bar_layout.addWidget(menubar)
        
        # Add window controls
        controls_widget = self._create_window_controls()
        top_bar_layout.addWidget(controls_widget)
        
        top_bar.setLayout(top_bar_layout)
        
        # Store reference
        self._top_bar_widget = top_bar
        
        # Add top bar to the window using a custom layout approach
        # We'll add it to the main layout in _create_main_widget
        self._custom_top_bar = top_bar
    
    def _create_menu_bar(self):
        """Create the menu bar with File, Edit, View, Tools, and Help menus, plus window controls."""
        menubar = self.menuBar()
        
        # Style menu bar with a slightly different shade (lighter than main window)
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #3d3d3d;
                border-bottom: 1px solid #2a2a2a;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #4d4d4d;
            }
            QMenuBar::item:pressed {
                background-color: #5d5d5d;
            }
        """)
        
        # Install event filter for dragging (better than overriding methods)
        menubar.installEventFilter(self)
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("&New Plan...", self._on_new_plan, "Ctrl+N")
        file_menu.addAction("&Open Plan...", self._on_open_plan, "Ctrl+O")
        file_menu.addAction("&Save Plan", self._on_save_plan, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("&Preferences...", self._on_preferences)
        edit_menu.addSeparator()
        edit_menu.addAction("&Clear Log", self._on_clear_log)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("&Refresh Instances", self._on_refresh_instances, "F5")
        view_menu.addSeparator()
        view_menu.addAction("&Show Statistics", self._on_show_statistics)
        view_menu.addAction("&Show Plan Editor", self._on_show_plan_editor)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction("&Launch RuneLite Instances", self._on_launch_runelite)
        tools_menu.addAction("&Stop All Instances", self._on_stop_all_instances)
        tools_menu.addSeparator()
        tools_menu.addAction("&Detect Active Instances", self._on_detect_instances)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About", self._on_about)
        help_menu.addAction("&About Qt", QApplication.aboutQt)
        
        # Create a custom top bar widget that contains menu bar + controls
        top_bar = QWidget()
        top_bar.setFixedHeight(30)
        top_bar.setStyleSheet("background-color: #3d3d3d; border-bottom: 1px solid #2a2a2a;")
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(0)
        
        # Add menu bar (will expand to fill available space)
        top_bar_layout.addWidget(menubar)
        
        # Add window controls on the right
        controls_widget = self._create_window_controls()
        top_bar_layout.addWidget(controls_widget)
        
        top_bar.setLayout(top_bar_layout)
        
        # Install event filter on top bar for dragging
        top_bar.installEventFilter(self)
        top_bar.setMouseTracking(True)  # Enable mouse tracking for better event handling
        # Make sure top bar can receive mouse events
        top_bar.setAttribute(Qt.WidgetAttribute.WA_MouseTracking, True)
        self._top_bar_widget = top_bar
        
        # Store reference for adding to main layout
        self._custom_top_bar = top_bar
    
    def _create_window_controls(self):
        """Create the window control buttons (minimize, maximize, close)."""
        controls_widget = QWidget()
        controls_widget.setFixedHeight(30)  # Match menu bar height
        controls_widget.setFixedWidth(138)  # 46 * 3 buttons
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        
        # Common button style for dark theme
        button_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                min-width: 46px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.25);
            }
        """
        
        # Minimize button
        minimize_btn = QPushButton("−")
        minimize_btn.setFixedSize(46, 30)
        minimize_btn.setStyleSheet(button_style)
        minimize_btn.clicked.connect(self.showMinimized)
        minimize_btn.setToolTip("Minimize")
        
        # Maximize/Restore button
        self.maximize_btn = QPushButton("□")
        self.maximize_btn.setFixedSize(46, 30)
        self.maximize_btn.setStyleSheet(button_style)
        self.maximize_btn.clicked.connect(self._toggle_maximize)
        self.maximize_btn.setToolTip("Maximize")
        
        # Close button (X)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(46, 30)
        close_btn.setStyleSheet(button_style + """
            QPushButton:hover {
                background-color: #e81123;
                color: white;
            }
        """)
        close_btn.clicked.connect(self.close)
        close_btn.setToolTip("Close")
        
        controls_layout.addWidget(minimize_btn)
        controls_layout.addWidget(self.maximize_btn)
        controls_layout.addWidget(close_btn)
        controls_widget.setLayout(controls_layout)
        controls_widget.setStyleSheet("background-color: #3d3d3d;")
        
        return controls_widget
    
    def _add_window_controls_to_menu_bar(self, menubar):
        """Add minimize, maximize, and close buttons to the menu bar using setCornerWidget."""
        controls_widget = self._create_window_controls()
        # Make sure the widget is visible and properly sized
        controls_widget.show()  # Explicitly show it
        # Add the controls widget to the menu bar corner
        menubar.setCornerWidget(controls_widget, Qt.Corner.TopRightCorner)
        # Force update to ensure visibility
        menubar.update()
    
    def _toggle_maximize(self):
        """Toggle between maximized and normal window state."""
        if self.isMaximized():
            self.showNormal()
            self.maximize_btn.setText("□")
            self.maximize_btn.setToolTip("Maximize")
        else:
            self.showMaximized()
            self.maximize_btn.setText("❐")
            self.maximize_btn.setToolTip("Restore Down")
    
    def eventFilter(self, obj, event):
        """Event filter to handle menu bar dragging and cursor changes for resizing."""
        # Handle dragging on top bar widget (which contains menu bar and controls)
        if hasattr(self, '_top_bar_widget') and obj == self._top_bar_widget:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Check if clicking on window controls (right side of top bar)
                    if event.pos().x() > obj.width() - 138:  # 138 = width of controls (46*3)
                        return False  # Let controls handle it
                    
                    # Check if clicking on menu bar area (try to see if there's a menu action)
                    menubar = self.menuBar()
                    # Convert top bar local position to menu bar local position
                    top_bar_global = obj.mapToGlobal(event.pos())
                    menubar_local = menubar.mapFromGlobal(top_bar_global)
                    action = menubar.actionAt(menubar_local)
                    
                    if action is not None:
                        # Clicking on a menu item, let menu bar handle it
                        return False
                    
                    # Not clicking on menu or controls, start dragging
                    self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    return True  # We handled it - don't propagate
            elif event.type() == QEvent.Type.MouseMove:
                # Only handle dragging if we're actually in a drag state
                if self._drag_position is not None and event.buttons() == Qt.MouseButton.LeftButton:
                    self.move(event.globalPosition().toPoint() - self._drag_position)
                    return True  # We handled it - don't propagate
                return False  # Let other widgets handle mouse moves
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self._drag_position is not None:
                    self._drag_position = None
                    return True  # We handled the release
                return False  # Let other widgets handle release
        
        # Handle dragging on menu bar directly (fallback)
        if obj == self.menuBar():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Check if clicking on a menu item or button - if so, let menu bar handle it
                    action = self.menuBar().actionAt(event.pos())
                    # Also check if clicking on corner widget (window controls)
                    corner_widget = self.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
                    if corner_widget:
                        corner_rect = corner_widget.geometry()
                        if corner_rect.contains(event.pos()):
                            # Clicking on window controls, let them handle it
                            return False
                    if action is not None:
                        # Clicking on a menu, let menu bar handle it normally
                        return False
                    # Not clicking on menu or controls, start dragging
                    self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    event.accept()
                    return True  # We handled it
            elif event.type() == QEvent.Type.MouseMove:
                # Only handle dragging if we're actually in a drag state
                if self._drag_position is not None and event.buttons() == Qt.MouseButton.LeftButton:
                    self.move(event.globalPosition().toPoint() - self._drag_position)
                    event.accept()
                    return True  # We handled it
                # Otherwise, let menu bar handle mouse moves (for menu hover effects)
                return False
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_position = None
                # Always let menu bar handle release events
                return False
            # For all other events, let menu bar handle them
            return False
        
        # Handle cursor changes on other widgets (but not menu bar or top bar)
        if isinstance(obj, QWidget) and event.type() == QEvent.Type.MouseMove:
            # Skip menu bar and top bar - they handle their own dragging
            if obj == self.menuBar() or (hasattr(self, '_top_bar_widget') and obj == self._top_bar_widget):
                return False
            
            if not self.isMaximized() and not self._resize_edge:
                # Convert widget-local position to main window coordinates
                if obj == self:
                    pos = event.pos()
                else:
                    # Map from widget coordinates to main window coordinates
                    global_pos = obj.mapToGlobal(event.pos())
                    pos = self.mapFromGlobal(global_pos)
                
                edge = self._get_resize_edge(pos)
                if edge:
                    cursor = self._get_cursor_for_edge(edge)
                    self.setCursor(cursor)
                    # Set cursor on the widget that received the event
                    obj.setCursor(cursor)
                    # Also set on all visible child widgets
                    for child in self.findChildren(QWidget):
                        if child.isVisible() and child != obj and child != self.menuBar():
                            child.setCursor(cursor)
                    return False  # Let the event continue
                else:
                    arrow_cursor = QCursor(Qt.CursorShape.ArrowCursor)
                    self.setCursor(arrow_cursor)
                    obj.setCursor(arrow_cursor)
        # Handle dragging on custom top bar if it exists
        elif hasattr(self, '_top_bar_widget') and obj == self._top_bar_widget:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if event.buttons() == Qt.MouseButton.LeftButton and self._drag_position is not None:
                    self.move(event.globalPosition().toPoint() - self._drag_position)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_position = None
        # Let other events pass through normally
        return super().eventFilter(obj, event)
    
    def _install_cursor_tracking(self, widget: QWidget):
        """Recursively install event filter and enable mouse tracking on widget and its children."""
        widget.installEventFilter(self)
        widget.setMouseTracking(True)
        for child in widget.findChildren(QWidget):
            if child != widget:
                child.installEventFilter(self)
                child.setMouseTracking(True)
    
    def _get_resize_edge(self, pos: QPoint) -> Optional[str]:
        """Determine which edge/corner the mouse is near for resizing. Returns None if not near an edge."""
        # Use global position relative to window
        global_pos = self.mapToGlobal(pos)
        window_rect = self.geometry()
        local_x = pos.x()
        local_y = pos.y()
        width, height = self.width(), self.height()
        border = self._resize_border_width
        
        # Check corners first (they take priority)
        if local_x < border and local_y < border:
            return "top-left"
        elif local_x >= width - border and local_y < border:
            return "top-right"
        elif local_x < border and local_y >= height - border:
            return "bottom-left"
        elif local_x >= width - border and local_y >= height - border:
            return "bottom-right"
        # Check edges
        elif local_x < border:
            return "left"
        elif local_x >= width - border:
            return "right"
        elif local_y < border:
            return "top"
        elif local_y >= height - border:
            return "bottom"
        return None
    
    def _get_cursor_for_edge(self, edge: str) -> QCursor:
        """Get the appropriate cursor for the given resize edge."""
        if edge in ["top-left", "bottom-right"]:
            return QCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edge in ["top-right", "bottom-left"]:
            return QCursor(Qt.CursorShape.SizeBDiagCursor)
        elif edge in ["left", "right"]:
            return QCursor(Qt.CursorShape.SizeHorCursor)
        elif edge in ["top", "bottom"]:
            return QCursor(Qt.CursorShape.SizeVerCursor)
        return QCursor(Qt.CursorShape.ArrowCursor)
    
    def mousePressEvent(self, event):
        """Handle mouse press events for window resizing."""
        # Don't resize if window is maximized
        if self.isMaximized():
            super().mousePressEvent(event)
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._get_resize_edge(event.pos())
            if edge:
                # Start resizing
                self._resize_edge = edge
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_geometry = self.geometry()
                # Set cursor immediately
                self.setCursor(self._get_cursor_for_edge(edge))
                # Also set on all child widgets
                for child in self.findChildren(QWidget):
                    if child.isVisible():
                        child.setCursor(self._get_cursor_for_edge(edge))
                event.accept()
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for window resizing and cursor changes."""
        # Don't resize if window is maximized
        if self.isMaximized():
            super().mouseMoveEvent(event)
            return
        
        # Check if we're currently resizing
        if self._resize_edge and event.buttons() == Qt.MouseButton.LeftButton:
            # Set cursor for current resize operation
            cursor = self._get_cursor_for_edge(self._resize_edge)
            self.setCursor(cursor)
            # Also set on all child widgets to ensure visibility
            for child in self.findChildren(QWidget):
                if child.isVisible():
                    child.setCursor(cursor)
            
            # Calculate new geometry based on resize edge
            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self._resize_start_pos
            
            x, y, w, h = (
                self._resize_start_geometry.x(),
                self._resize_start_geometry.y(),
                self._resize_start_geometry.width(),
                self._resize_start_geometry.height(),
            )
            
            if "left" in self._resize_edge:
                x += delta.x()
                w -= delta.x()
                if w < self.minimumWidth():
                    w = self.minimumWidth()
                    x = self._resize_start_geometry.x() + self._resize_start_geometry.width() - w
            elif "right" in self._resize_edge:
                w += delta.x()
                if w < self.minimumWidth():
                    w = self.minimumWidth()
            
            if "top" in self._resize_edge:
                y += delta.y()
                h -= delta.y()
                if h < self.minimumHeight():
                    h = self.minimumHeight()
                    y = self._resize_start_geometry.y() + self._resize_start_geometry.height() - h
            elif "bottom" in self._resize_edge:
                h += delta.y()
                if h < self.minimumHeight():
                    h = self.minimumHeight()
            
            self.setGeometry(x, y, w, h)
            event.accept()
            return
        
        # Update cursor based on mouse position (when not resizing)
        edge = self._get_resize_edge(event.pos())
        if edge:
            cursor = self._get_cursor_for_edge(edge)
            self.setCursor(cursor)
            # Also set cursor on central widget and its children to ensure it's visible
            if self.centralWidget():
                self.centralWidget().setCursor(cursor)
                for child in self.centralWidget().findChildren(QWidget):
                    if child.isVisible():
                        child.setCursor(cursor)
        else:
            arrow_cursor = QCursor(Qt.CursorShape.ArrowCursor)
            self.setCursor(arrow_cursor)
            # Reset cursor on central widget and its children
            if self.centralWidget():
                self.centralWidget().setCursor(arrow_cursor)
                for child in self.centralWidget().findChildren(QWidget):
                    if child.isVisible():
                        child.setCursor(arrow_cursor)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop resizing."""
        if event.button() == Qt.MouseButton.LeftButton and self._resize_edge:
            self._resize_edge = None
            self._resize_start_pos = None
            self._resize_start_geometry = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)
    
    def _on_new_plan(self):
        """Handle New Plan menu action."""
        if hasattr(self, 'plan_editor'):
            self.plan_editor.new_plan()
    
    def _on_open_plan(self):
        """Handle Open Plan menu action."""
        if hasattr(self, 'plan_editor'):
            self.plan_editor.open_plan()
    
    def _on_save_plan(self):
        """Handle Save Plan menu action."""
        if hasattr(self, 'plan_editor'):
            self.plan_editor.save_plan()
    
    def _on_preferences(self):
        """Handle Preferences menu action."""
        if hasattr(self, 'config_manager'):
            self.config_manager.show()
    
    def _on_clear_log(self):
        """Handle Clear Log menu action."""
        if hasattr(self, 'log_text'):
            self.log_text.clear()
    
    def _on_refresh_instances(self):
        """Handle Refresh Instances menu action."""
        if hasattr(self, 'client_detector'):
            self.client_detector.detect_running_clients(
                self.create_instance_tab_wrapper,
                self._remove_detected_instance,
                self._log_message
            )
    
    def _on_show_statistics(self):
        """Handle Show Statistics menu action."""
        if hasattr(self, 'statistics_display'):
            self.statistics_display.show()
    
    def _on_show_plan_editor(self):
        """Handle Show Plan Editor menu action."""
        if hasattr(self, 'plan_editor'):
            self.plan_editor.show()
    
    def _on_launch_runelite(self):
        """Handle Launch RuneLite Instances menu action."""
        if hasattr(self, 'launcher') and hasattr(self, 'launch_button'):
            self.launch_button.click()
    
    def _on_stop_all_instances(self):
        """Handle Stop All Instances menu action."""
        if hasattr(self, 'stop_button'):
            self.stop_button.click()
    
    def _on_detect_instances(self):
        """Handle Detect Active Instances menu action."""
        if hasattr(self, 'detect_now_button'):
            self.detect_now_button.click()
    
    def _on_about(self):
        """Handle About menu action."""
        QMessageBox.about(
            self,
            "About Simple Recorder",
            "Simple Recorder - Plan Runner\n\n"
            "A GUI application for managing RuneLite bot instances and plans.\n\n"
            "Version 1.0"
        )
    
    def _create_themes_menu(self, themes_menu: QMenu):
        """Create the Themes menu with color circle icons."""
        # Define all themes with their color palettes
        # Each theme has: variant (dark/light), base color, and menu bar color (slightly different shade)
        themes = {
            "Dark": {"variant": "dark", "color": QColor(53, 53, 53), "menubar_color": QColor(61, 61, 61)},
            "Light Dark": {"variant": "light", "color": QColor(80, 80, 80), "menubar_color": QColor(95, 95, 95)},
            "White": {"variant": "light", "color": QColor(240, 240, 240), "menubar_color": QColor(250, 250, 250)},
            "Blue": {"variant": "dark", "color": QColor(30, 60, 120), "menubar_color": QColor(40, 75, 140)},
            "Light Blue": {"variant": "light", "color": QColor(100, 150, 220), "menubar_color": QColor(120, 170, 240)},
            "Green": {"variant": "dark", "color": QColor(30, 100, 60), "menubar_color": QColor(40, 120, 75)},
            "Light Green": {"variant": "light", "color": QColor(100, 200, 150), "menubar_color": QColor(120, 220, 170)},
            "Red": {"variant": "dark", "color": QColor(120, 30, 30), "menubar_color": QColor(140, 40, 40)},
            "Light Red": {"variant": "light", "color": QColor(220, 100, 100), "menubar_color": QColor(240, 120, 120)},
            "Orange": {"variant": "dark", "color": QColor(150, 80, 30), "menubar_color": QColor(170, 100, 50)},
            "Light Orange": {"variant": "light", "color": QColor(255, 180, 100), "menubar_color": QColor(255, 200, 130)},
            "Yellow": {"variant": "dark", "color": QColor(150, 150, 30), "menubar_color": QColor(170, 170, 50)},
            "Light Yellow": {"variant": "light", "color": QColor(255, 255, 150), "menubar_color": QColor(255, 255, 180)},
            "Purple": {"variant": "dark", "color": QColor(100, 30, 120), "menubar_color": QColor(120, 50, 140)},
            "Light Purple": {"variant": "light", "color": QColor(200, 150, 220), "menubar_color": QColor(220, 170, 240)},
        }
        
        for theme_name, theme_info in themes.items():
            # Create a colored circle icon
            icon = self._create_color_circle_icon(theme_info["color"], 16)
            action = themes_menu.addAction(icon, theme_name)
            # Use a lambda with default arguments to capture the values correctly
            variant = theme_info["variant"]
            color = QColor(theme_info["color"])  # Create a copy to avoid reference issues
            menubar_color = QColor(theme_info["menubar_color"])  # Menu bar color
            action.triggered.connect(lambda checked, name=theme_name, var=variant, col=color, mb_col=menubar_color: 
                                    self._apply_theme(name, var, col, mb_col))
    
    def _create_color_circle_icon(self, color: QColor, size: int = 16) -> QIcon:
        """Create a circular icon with the given color."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, size - 4, size - 4)
        painter.end()
        return QIcon(pixmap)
    
    def _apply_theme(self, theme_name: str, variant: str, base_color: QColor, menubar_color: QColor):
        """Apply a theme to the application."""
        app = QApplication.instance()
        if app is None:
            return
        
        # Use Fusion style
        app.setStyle(QStyleFactory.create("Fusion"))
        
        # Create palette based on theme
        palette = QPalette()
        
        if variant == "dark":
            # Dark variant - darker backgrounds, lighter text
            window_bg = base_color.darker(150) if base_color.lightness() > 50 else base_color
            window_text = QColor(255, 255, 255) if window_bg.lightness() < 128 else QColor(0, 0, 0)
            base_bg = window_bg.darker(120)
            button_bg = window_bg
            highlight = base_color.lighter(130)
        else:
            # Light variant - lighter backgrounds, darker text
            window_bg = base_color.lighter(150) if base_color.lightness() < 128 else base_color
            window_text = QColor(0, 0, 0) if window_bg.lightness() > 128 else QColor(255, 255, 255)
            base_bg = window_bg.lighter(110)
            button_bg = window_bg
            highlight = base_color.darker(120)
        
        # Calculate menu bar colors (slightly different shade from window)
        if variant == "dark":
            # For dark themes, menu bar is slightly lighter
            mb_bg = menubar_color if menubar_color.lightness() < window_bg.lightness() else menubar_color.darker(110)
            mb_text = QColor(255, 255, 255) if mb_bg.lightness() < 128 else QColor(0, 0, 0)
        else:
            # For light themes, menu bar is slightly darker
            mb_bg = menubar_color if menubar_color.lightness() > window_bg.lightness() else menubar_color.lighter(110)
            mb_text = QColor(0, 0, 0) if mb_bg.lightness() > 128 else QColor(255, 255, 255)
        
        # Window (background)
        palette.setColor(QPalette.ColorRole.Window, window_bg)
        palette.setColor(QPalette.ColorRole.WindowText, window_text)
        
        # Base (input fields background)
        palette.setColor(QPalette.ColorRole.Base, base_bg)
        palette.setColor(QPalette.ColorRole.AlternateBase, window_bg)
        
        # Text
        palette.setColor(QPalette.ColorRole.Text, window_text)
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        
        # Button
        palette.setColor(QPalette.ColorRole.Button, button_bg)
        palette.setColor(QPalette.ColorRole.ButtonText, window_text)
        
        # Highlight (selected items)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        # Tooltip
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0) if variant == "dark" else QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255) if variant == "dark" else QColor(0, 0, 0))
        
        # Disabled
        disabled_color = QColor(127, 127, 127)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, disabled_color)
        
        # Link
        palette.setColor(QPalette.ColorRole.Link, highlight)
        palette.setColor(QPalette.ColorRole.LinkVisited, highlight.darker(120))
        
        # Apply the palette
        app.setPalette(palette)
        
        # Update output text widgets to use theme-aware colors
        # This ensures log text is readable with the current theme
        if hasattr(self, 'log_text') and self.log_text:
            log_palette = self.log_text.palette()
            log_palette.setColor(QPalette.ColorRole.Base, base_bg)
            log_palette.setColor(QPalette.ColorRole.Text, window_text)
            self.log_text.setPalette(log_palette)
        
        # Update instance output widgets if they exist
        if hasattr(self, 'instance_tabs'):
            for instance_name, instance_tab in self.instance_tabs.items():
                if hasattr(instance_tab, 'output_text') and instance_tab.output_text:
                    instance_palette = instance_tab.output_text.palette()
                    instance_palette.setColor(QPalette.ColorRole.Base, base_bg)
                    instance_palette.setColor(QPalette.ColorRole.Text, window_text)
                    instance_tab.output_text.setPalette(instance_palette)
        
        # Update menu bar styling with theme-specific menu bar color
        menubar = self.menuBar()
        mb_border = mb_bg.darker(120) if variant == "dark" else mb_bg.lighter(120)
        
        # Set menu bar palette to ensure background color is applied
        menubar.setAutoFillBackground(True)
        mb_palette = menubar.palette()
        mb_palette.setColor(menubar.backgroundRole(), mb_bg)
        mb_palette.setColor(menubar.foregroundRole(), mb_text)
        menubar.setPalette(mb_palette)
        
        # Apply stylesheet for menu bar - use !important to override any other styles
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {mb_bg.name()} !important;
                border: none !important;
                border-bottom: 1px solid {mb_border.name()} !important;
                padding: 2px;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 8px;
                color: {mb_text.name()};
            }}
            QMenuBar::item:selected {{
                background-color: {highlight.name()};
            }}
            QMenuBar::item:pressed {{
                background-color: {highlight.darker(110).name()};
            }}
        """)
        
        # Update custom top bar styling - set background to menu bar color so it matches
        if hasattr(self, '_custom_top_bar'):
            self._custom_top_bar.setStyleSheet(f"background-color: {mb_bg.name()}; border: none; border-bottom: 1px solid {mb_border.name()};")
            self._custom_top_bar.setAutoFillBackground(True)
            # Also set palette for top bar
            top_bar_palette = self._custom_top_bar.palette()
            top_bar_palette.setColor(self._custom_top_bar.backgroundRole(), mb_bg)
            self._custom_top_bar.setPalette(top_bar_palette)
        
        self._current_theme = theme_name
        if hasattr(self, 'log_callback'):
            self.log_callback(f"Theme changed to: {theme_name}", "info")
    
    def _get_bot_project_root(self) -> Path:
        """Return the bot_runelite_IL project root (directory containing gui/, credentials/, etc.)."""
        return Path(__file__).resolve().parent.parent

    def _get_runelite_project_root(self) -> Path:
        """Return the RuneLite project root (sibling of bot_runelite_IL, e.g. .../repos/runelite)."""
        return self._get_bot_project_root().parent / "runelite"

    def _update_derived_config_paths(self):
        """Set projectDir, baseDir, exportsBase, credentialsDir from bot/runelite layout (not stored in config)."""
        bot_root = self._get_bot_project_root()
        self.config_vars["projectDir"] = str(self._get_runelite_project_root())
        self.config_vars["baseDir"] = str(bot_root / "instances")
        self.config_vars["exportsBase"] = str(bot_root / "exports")
        self.config_vars["credentialsDir"] = str(bot_root / "credentials")

    def _init_variables(self):
        """Initialize all variables."""
        # RuneLite launcher variables
        self.selected_credentials = []
        self.runelite_process = None
        self.base_port = 17000
        self.launch_delay = 0
        self._closing_async = False  # Flag to track async close in progress
        
        # Configuration variables. projectDir, baseDir, exportsBase, credentialsDir are always
        # derived (runelite = sibling of bot root; others under bot root). See _update_derived_config_paths.
        bot_root = self._get_bot_project_root()
        runelite_root = bot_root.parent / "runelite"
        self.config_vars = {
            "projectDir": str(runelite_root),
            "baseDir": str(bot_root / "instances"),
            "exportsBase": str(bot_root / "exports"),
            "credentialsDir": str(bot_root / "credentials"),
            "autoDetect": True
        }
        
        # Path entries (will be populated when widgets are created)
        self.path_entries = {}
        
        # Instance management
        self.instance_tabs = {}
        self.instance_ports = {}
        self.detected_clients = {}
        self.client_detection_running = False
        self.instances_notebook = None  # Will be created in _create_instances_tab()
        
        # Base completion patterns
        self.base_completion_patterns = [
            'phase: done',
            'status: done', 'plan done', 'execution done',
            'plan completed', 'execution completed', 'finished successfully',
            'task completed', 'mission accomplished', 'objective complete'
        ]
        
        # Initialize skill icons dict
        self.skill_icons = {}
        
        # Stats monitors
        self.stats_monitors = {}
    
    def _setup_file_logging(self):
        """Setup file logging to capture all logs even if GUI crashes."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path(__file__).resolve().parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"gui_{timestamp}.log"
            
            # Setup file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
            
            # Log startup
            logging.info(f"GUI started, logging to: {log_file}")
            self._log_file_path = log_file
        except Exception as e:
            # If logging setup fails, at least try to print
            print(f"Failed to setup file logging: {e}")
            self._log_file_path = None
    
    def _init_components(self):
        """Initialize all component managers."""
        # No launch-config.json; config manager uses QSettings for the single preference (auto-detect)
        self.config_file = None
        
        # Initialize logging - log text widget will be created in _create_main_widget
        self.log_text = None
        
        # Initialize config manager
        self.config_manager = ConfigManager(
            None,
            self.config_vars,
            self.path_entries
        )
        
        # Initialize statistics (needs skill icons)
        self.statistics = StatisticsDisplay(
            self,
            self.instance_tabs,
            self.instance_ports,
            self.skill_icons,
            self.stats_monitors,
            self._log_message,
            self._get_credential_name,  # Pass credential name getter
            self._log_message_to_instance  # Pass instance logging callback
        )
        # Load skill icons
        self.skill_icons = self.statistics.load_skill_icons()
        
        # Initialize launcher (will be fully initialized in _create_main_widget)
        self.launcher = None
        
        # Initialize client detector (will be fully initialized in _create_main_widget)
        self.client_detector = None
        
        # Initialize instance manager (will be fully initialized in _create_main_widget)
        self.instance_manager = None
    
    def _create_main_widget(self):
        """Create the main widget and all UI components."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Add custom top bar if it exists (contains menu bar + window controls)
        if hasattr(self, '_custom_top_bar'):
            main_layout.addWidget(self._custom_top_bar)
        
        # Create tab widget
        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)
        
        # Create Client tab with sub-tabs
        self._create_client_tab()
        
        # Create Instances tab with sub-notebook for instance tabs
        self._create_instances_tab()
        
        # Load configuration first (before initializing components that need config)
        self._load_configuration()
        
        # Initialize component managers that need widgets
        self._init_widget_dependent_components()
    
    def _create_client_tab(self):
        """Create the Client tab with sub-tabs (Launcher, Setup & Configuration, Output)."""
        # Client tab widget
        client_tab = QWidget()
        
        # Sub-tab widget for Client tab
        client_sub_tabs = QTabWidget()
        client_sub_tabs.addTab(self._create_launcher_tab(), "Launcher")
        client_sub_tabs.addTab(self._create_setup_config_tab(), "Setup & Configuration")
        client_sub_tabs.addTab(self._create_output_tab(), "Output")
        
        # Layout for client tab
        client_layout = QVBoxLayout(client_tab)
        client_layout.setContentsMargins(0, 0, 0, 0)
        client_layout.addWidget(client_sub_tabs)
        
        # Add to main notebook
        self.notebook.addTab(client_tab, "Client")
    
    def _create_instances_tab(self):
        """Create the Instances tab with a sub-notebook for instance tabs."""
        # Instances tab widget
        instances_tab = QWidget()
        
        # Sub-notebook for instance tabs (will be populated by instance manager)
        self.instances_notebook = QTabWidget()
        
        # Layout for instances tab
        instances_layout = QVBoxLayout(instances_tab)
        instances_layout.setContentsMargins(0, 0, 0, 0)
        instances_layout.addWidget(self.instances_notebook)
        
        # Add to main notebook (next to Client tab)
        self.notebook.addTab(instances_tab, "Instances")
    
    def _create_launcher_tab(self):
        """Create the Launcher sub-tab."""
        launcher_widget = QWidget()
        main_layout = QHBoxLayout(launcher_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side: Scrollable content
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Credential Selection
        creds_group = QGroupBox("Credential Selection")
        creds_layout = QHBoxLayout()
        
        # Available credentials
        creds_left_layout = QVBoxLayout()
        creds_left_layout.addWidget(QLabel("Available Credentials:"))
        self.credentials_listbox = QListWidget()
        self.credentials_listbox.setSelectionMode(QListWidget.MultiSelection)
        creds_left_layout.addWidget(self.credentials_listbox)
        
        # Buttons
        cred_buttons_layout = QVBoxLayout()
        self.add_cred_button = QPushButton(">")
        self.remove_cred_button = QPushButton("<")
        self.move_up_button = QPushButton("↑")
        self.move_down_button = QPushButton("↓")
        self.clear_creds_button = QPushButton("Clear")
        cred_buttons_layout.addWidget(self.add_cred_button)
        cred_buttons_layout.addWidget(self.remove_cred_button)
        cred_buttons_layout.addWidget(self.move_up_button)
        cred_buttons_layout.addWidget(self.move_down_button)
        cred_buttons_layout.addWidget(self.clear_creds_button)
        cred_buttons_layout.addStretch()
        
        # Selected credentials
        creds_right_layout = QVBoxLayout()
        creds_right_layout.addWidget(QLabel("Selected Credentials (Launch Order):"))
        self.selected_credentials_listbox = QListWidget()
        creds_right_layout.addWidget(self.selected_credentials_listbox)
        
        creds_layout.addLayout(creds_left_layout)
        creds_layout.addLayout(cred_buttons_layout)
        creds_layout.addLayout(creds_right_layout)
        creds_group.setLayout(creds_layout)
        left_layout.addWidget(creds_group)
        
        # Launch Controls
        launch_controls_group = QGroupBox("Launch Controls")
        launch_controls_layout = QVBoxLayout()
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        self.launch_button = QPushButton("Launch RuneLite Instances")
        self.stop_button = QPushButton("Stop All Instances")
        buttons_layout.addWidget(self.launch_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addStretch()
        
        launch_controls_layout.addLayout(buttons_layout)
        
        launch_controls_group.setLayout(launch_controls_layout)
        left_layout.addWidget(launch_controls_group)
        
        left_layout.addStretch()
        left_scroll.setWidget(left_content)
        
        # Right side: Instance Manager
        instance_manager_group = QGroupBox("Instance Manager")
        instance_manager_layout = QVBoxLayout()
        
        # Add Detect Active Instances button at the top of Instance Manager
        self.detect_now_button = QPushButton("Detect Active Instances")
        instance_manager_layout.addWidget(self.detect_now_button)
        
        instance_manager_layout.addWidget(QLabel("Active Instances:"))
        
        self.instance_tree = QTreeWidget()
        self.instance_tree.setHeaderLabels(["", "Credential", "Port"])  # First column for icon
        self.instance_tree.setColumnWidth(0, 30)  # Icon column
        self.instance_tree.setColumnWidth(1, 200)  # Credential
        self.instance_tree.setColumnWidth(2, 80)   # Port
        
        # Connect item clicked signal to handle navigation
        self.instance_tree.itemClicked.connect(self._on_instance_tree_item_clicked)
        
        instance_manager_layout.addWidget(self.instance_tree)
        
        instance_manager_group.setLayout(instance_manager_layout)
        
        # Add to main layout
        main_layout.addWidget(left_scroll, 2)  # Left side gets 2/3 of space
        main_layout.addWidget(instance_manager_group, 1)  # Right side gets 1/3
        
        return launcher_widget
    
    def _create_setup_config_tab(self):
        """Create the Setup & Configuration sub-tab."""
        setup_widget = QWidget()
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        setup_content = QWidget()
        setup_layout = QVBoxLayout(setup_content)
        setup_layout.setSpacing(10)
        setup_layout.setContentsMargins(10, 10, 10, 10)
        
        # Setup & Configuration group
        config_group = QGroupBox("Setup & Configuration")
        config_layout = QGridLayout()
        config_layout.setSpacing(5)
        
        row = 0
        # projectDir, baseDir, exportsBase, credentialsDir are auto-derived; Java 11 via Gradle toolchain
        
        # Auto-detect checkbox
        self.auto_detect_checkbox = QCheckBox("Auto-detect paths on startup")
        self.auto_detect_checkbox.setChecked(self.config_vars["autoDetect"])
        config_layout.addWidget(self.auto_detect_checkbox, row, 0, 1, 2)
        row += 1
        
        # Config buttons
        config_buttons_layout = QHBoxLayout()
        auto_detect_paths_button = QPushButton("Auto-detect Paths")
        save_config_button = QPushButton("Save Config")
        self.setup_deps_button = QPushButton("Setup Dependencies")
        config_buttons_layout.addWidget(auto_detect_paths_button)
        config_buttons_layout.addWidget(save_config_button)
        config_buttons_layout.addWidget(self.setup_deps_button)
        config_buttons_layout.addStretch()
        config_layout.addLayout(config_buttons_layout, row, 0, 1, 3)
        
        config_group.setLayout(config_layout)
        setup_layout.addWidget(config_group)
        setup_layout.addStretch()
        
        setup_scroll.setWidget(setup_content)
        
        setup_widget_layout = QVBoxLayout(setup_widget)
        setup_widget_layout.setContentsMargins(0, 0, 0, 0)
        setup_widget_layout.addWidget(setup_scroll)
        
        return setup_widget
    
    def _create_output_tab(self):
        """Create the Output sub-tab."""
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        
        # Ensure the text widget uses theme-aware colors
        # The palette will be applied via QApplication.setPalette(), but we also
        # set the background to match the Base color for better contrast
        palette = self.log_text.palette()
        app_palette = QApplication.instance().palette()
        palette.setColor(QPalette.ColorRole.Base, app_palette.color(QPalette.ColorRole.Base))
        palette.setColor(QPalette.ColorRole.Text, app_palette.color(QPalette.ColorRole.Text))
        self.log_text.setPalette(palette)
        
        output_layout.addWidget(self.log_text)
        
        return output_widget
    
    def _init_widget_dependent_components(self):
        """Initialize components that depend on widgets being created."""
        # Initialize launcher
        self.launcher = RuneLiteLauncher(
            self,
            self.config_vars,
            None,  # base_port_spinbox - removed from UI
            None,  # launch_delay_spinbox - removed from UI
            self.credentials_listbox,
            self.selected_credentials_listbox,
            self.selected_credentials,
            self._log_message,
            None,  # instance_count_label - removed from UI
            self.launch_button,
            self.create_instance_tab_wrapper,
            self.instance_ports,  # Pass reference to instance_ports dict
            self._cleanup_all_instances  # Pass cleanup callback
        )
        
        # Initialize client detector
        # Get project directory - use config if set, otherwise use the directory containing gui_pyside.py
        project_dir_str = self.config_vars.get("projectDir", "")
        if project_dir_str:
            project_dir = Path(project_dir_str)
        else:
            # Default to the directory containing gui_pyside.py (project root)
            # This file is in gui/, so go up one level
            project_dir = Path(__file__).resolve().parent.parent
        
        # Get base directory (where instances are stored) - use config if set
        base_dir_str = self.config_vars.get("baseDir", "")
        if base_dir_str:
            base_dir = Path(base_dir_str)
        else:
            # Default to project_dir/instances
            base_dir = project_dir / "instances"
        
        credentials_dir_str = self.config_vars.get("credentialsDir", "")
        if credentials_dir_str:
            credentials_dir = Path(credentials_dir_str)
        else:
            credentials_dir = project_dir / "credentials"
        self.client_detector = ClientDetector(
            self,
            self.detected_clients,
            self.client_detection_running,
            None,  # No detection status label needed
            project_dir=project_dir,
            base_dir=base_dir,
            credentials_dir=credentials_dir
        )
        
        # Initialize instance manager (use instances_notebook instead of main notebook)
        self.instance_manager = InstanceManager(
            self,
            self.instances_notebook,  # Use instances sub-notebook instead of main notebook
            self.instance_tabs,
            self.instance_ports,
            self.detected_clients,
            self.skill_icons,
            self.stats_monitors,
            self.base_completion_patterns,
            self.selected_credentials,
            self._log_message,
            self.statistics.stop_statistics_timer,
            self._update_plan_details_wrapper,
            self._update_parameter_widgets_wrapper,
            self.statistics.update_stats_text,
            self.statistics.start_stats_monitor,
            self._get_credential_name,  # Pass credential name getter
            self._log_message_to_instance  # Pass instance logging callback
        )
        
        # Connect signals
        self._connect_signals()
    
    def _connect_signals(self):
        """Connect all signals and slots."""
        # Launcher buttons
        self.launch_button.clicked.connect(lambda: self.launcher.launch_runelite(
            save_config_callback=lambda: self.config_manager.save_config(
                base_port_var=None,
                launch_delay_var=None,
                log_callback=self._log_message
            )
        ))
        self.stop_button.clicked.connect(lambda: self.launcher.stop_runelite(self._stop_all_instances))

        # Credential buttons
        self.add_cred_button.clicked.connect(self.launcher.add_credential)
        self.remove_cred_button.clicked.connect(self.launcher.remove_credential)
        self.move_up_button.clicked.connect(self.launcher.move_credential_up)
        self.move_down_button.clicked.connect(self.launcher.move_credential_down)
        self.clear_creds_button.clicked.connect(self.launcher.clear_credentials)
        
        # Detect Now button
        self.detect_now_button.clicked.connect(
            lambda: self.client_detector.detect_running_clients(
                self.create_instance_tab_wrapper,
                self._remove_detected_instance,
                self._log_message
            )
        )
        
        # Setup dependencies button
        self.setup_deps_button.clicked.connect(self.launcher.setup_dependencies)
        
        # Populate credentials
        self.launcher.populate_credentials()
    
    def _load_configuration(self):
        """Load configuration after widgets are created."""
        self.config_manager.load_config(
            base_port_var=None,  # Removed from UI
            launch_delay_var=None,  # Removed from UI
            log_callback=self._log_message
        )
        self._update_derived_config_paths()  # baseDir, exportsBase, credentialsDir from bot root
    
    def _log_message(self, message: str, level: str = 'info'):
        """Wrapper for logging messages to the main log text widget and file."""
        # Also log to file
        try:
            if level == 'error':
                logging.error(message)
            elif level == 'warning':
                logging.warning(message)
            else:
                logging.info(message)
        except Exception:
            pass  # Don't crash if file logging fails
        
        # Log to GUI widget
        if self.log_text:
            try:
                LoggingUtils.log_message(self.log_text, message, level)
            except Exception as e:
                try:
                    logging.error(f"Error logging to GUI widget: {e}", exc_info=True)
                except:
                    pass
    
    def _log_message_to_instance(self, instance_name: str, message: str, level: str = 'info'):
        """Wrapper for logging messages to an instance's output text widget."""
        if instance_name in self.instance_tabs:
            instance_tab = self.instance_tabs[instance_name]
            if hasattr(instance_tab, 'output_text') and instance_tab.output_text:
                from gui.logging_utils_pyside import LoggingUtils
                LoggingUtils.log_message(instance_tab.output_text, message, level)
    
    def _stop_all_instances(self):
        """Stop all running instances."""
        self._log_message("_stop_all_instances called", 'info')
        if self.instance_manager:
            for instance_name in list(self.instance_tabs.keys()):
                try:
                    instance_tab = self.instance_tabs[instance_name]
                    if hasattr(instance_tab, 'is_running') and instance_tab.is_running:
                        self.instance_manager.stop_plans_for_instance(instance_name)
                        self._log_message(f"Stopped plans for {instance_name}", 'info')
                    else:
                        self._log_message(f"Instance {instance_name} is not running", 'info')
                except Exception as e:
                    self._log_message(f"Error stopping {instance_name}: {str(e)}", 'error')
        else:
            self._log_message("Instance manager not available", 'warning')

    def create_instance_tab_wrapper(self, instance_name: str, port: int):
        """Wrapper to create instance tab using instance manager."""
        self._log_message(f"create_instance_tab_wrapper called for {instance_name} on port {port}", 'info')
        
        # Store port mapping and update display regardless of tab creation
        self.instance_ports[instance_name] = port
        self._log_message(f"Stored port {port} for {instance_name}. instance_ports now has {len(self.instance_ports)} entries: {dict(self.instance_ports)}", 'info')
        
        # Try to create instance tab if manager exists
        result = None
        if self.instance_manager:
            try:
                self._log_message(f"Creating instance tab: {instance_name} on port {port}", 'info')
                result = self.instance_manager.create_instance_tab(
                    instance_name, port,
                    browse_directory_callback=self._browse_directory_wrapper,
                    populate_sequences_callback=self._populate_sequences_wrapper,
                    save_sequence_callback=self._save_sequence_wrapper,
                    load_sequence_callback=self._load_sequence_wrapper,
                    delete_sequence_callback=self._delete_sequence_wrapper,
                    add_plan_callback=self._add_plan_wrapper,
                    remove_plan_callback=self._remove_plan_wrapper,
                    move_plan_up_callback=self._move_plan_up_wrapper,
                    move_plan_down_callback=self._move_plan_down_wrapper,
                    clear_plans_callback=self._clear_plans_wrapper,
                    update_plan_details_callback=self._update_plan_details_wrapper,
                    update_parameter_widgets_callback=self._update_parameter_widgets_wrapper,
                    add_rule_callback=self._add_rule_wrapper,
                    clear_params_callback=self._clear_params_wrapper,
                    clear_rules_callback=self._clear_rules_wrapper,
                    start_plans_callback=self._start_plans_wrapper,
                    stop_plans_callback=self._stop_plans_wrapper
                )
                # Store in instance_tabs if not already there (should already be stored by create_instance_tab)
                if instance_name not in self.instance_tabs:
                    self.instance_tabs[instance_name] = result
                    self._log_message(f"Stored instance tab in instance_tabs: {instance_name}", 'info')
                else:
                    self._log_message(f"Instance tab already in instance_tabs: {instance_name}", 'info')
                self._log_message(f"Instance tab creation attempted for {instance_name}", 'info')
            except Exception as e:
                self._log_message(f"Error creating instance tab: {str(e)}", 'error')
                import traceback
                self._log_message(f"Traceback: {traceback.format_exc()}", 'error')
        else:
            self._log_message("Instance manager not available, skipping tab creation", 'warning')
        
        # Always update instance manager display, even if tab creation failed
        self._update_instance_manager_display(instance_name, port)
        
        return result
    
    def _update_instance_manager_display(self, instance_name: str, port: int):
        """Update the instance manager treeview with a new instance."""
        self._log_message(f"_update_instance_manager_display called for {instance_name} on port {port}", 'info')
        
        if not hasattr(self, 'instance_tree'):
            self._log_message(f"instance_tree attribute not found", 'error')
            return
            
        if self.instance_tree is None:
            self._log_message(f"instance_tree is None", 'error')
            return
        
        # Check if instance already exists in tree
        root = self.instance_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            # Check item data (stored in column 0) for instance name
            stored_name = item.data(0, Qt.ItemDataRole.UserRole)
            if stored_name == instance_name:
                self._log_message(f"Instance {instance_name} already in tree, updating", 'info')
                item.setText(1, self._get_credential_name(instance_name))  # Column 1: credential
                item.setText(2, str(port))  # Column 2: port
                return
        
        # Find the credential file name for this instance
        credential_name = self._get_credential_name(instance_name)
        
        # Add to treeview
        item = QTreeWidgetItem(self.instance_tree)
        # Column 0: Icon (external link icon - standard "open in new tab" symbol)
        item.setText(0, "↗")  # Northeast arrow - standard external link icon
        item.setToolTip(0, f"Click to navigate to {instance_name} tab")
        # Column 1: Credential
        item.setText(1, credential_name)
        # Column 2: Port
        item.setText(2, str(port))
        
        # Store instance name in item data for easy retrieval
        item.setData(0, Qt.ItemDataRole.UserRole, instance_name)
        
        self._log_message(f"Added {instance_name} to instance manager (credential: {credential_name}, port: {port})", 'success')
    
    def _get_credential_name(self, instance_name: str) -> str:
        """Get credential name for an instance."""
        # For detected instances, extract port from name and get credential from detector
        if instance_name.startswith("detected_"):
            try:
                port = int(instance_name.replace("detected_", ""))
                credential = self.client_detector.get_credential_for_port(port)
                return credential if credential != "Unknown" else "Detected"
            except ValueError:
                return "Detected"
        
        for cred in self.selected_credentials:
            cred_username = cred.replace('.properties', '')
            if cred_username == instance_name:
                return cred_username
        return "Unknown"
    
    def _remove_detected_instance(self, instance_name: str):
        """Remove a detected instance (called by client detector)."""
        self._log_message(f"Removing detected instance: {instance_name}", 'info')
        
        # Remove from instance manager tab if it exists
        if self.instance_manager and hasattr(self.instance_manager, 'remove_instance_tab'):
            try:
                self.instance_manager.remove_instance_tab(instance_name)
            except Exception as e:
                self._log_message(f"Error removing tab for {instance_name}: {str(e)}", 'warning')
        
        # Remove from display (which also removes from instance_ports and instance_tabs)
        self._remove_instance_from_display(instance_name)
    
    def _remove_instance_from_display(self, instance_name: str):
        """Remove an instance from the instance manager treeview."""
        if not hasattr(self, 'instance_tree') or self.instance_tree is None:
            return
        
        # Find and remove the item
        root = self.instance_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            # Check item data (stored in column 0) for instance name
            stored_name = item.data(0, Qt.ItemDataRole.UserRole)
            if stored_name == instance_name:
                root.removeChild(item)
                break
        
        # Also remove from instance_ports
        if instance_name in self.instance_ports:
            del self.instance_ports[instance_name]
        
        # Also remove from instance_tabs
        if instance_name in self.instance_tabs:
            del self.instance_tabs[instance_name]
    
    def _on_instance_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle clicks on instance tree items to navigate to instance tabs."""
        try:
            # Only navigate if clicking on the icon column (column 0)
            if column == 0:
                instance_name = item.data(0, Qt.ItemDataRole.UserRole)
                if not instance_name:
                    # Fallback: get from column 1 (instance name)
                    instance_name = item.text(1)
                
                if instance_name:
                    # Use QTimer.singleShot(0, ...) to defer navigation to next event loop iteration
                    # This prevents blocking the click handler
                    QTimer.singleShot(0, lambda: self._navigate_to_instance_tab(instance_name))
        except Exception as e:
            error_msg = f"Error handling instance tree click: {str(e)}"
            self._log_message(error_msg, 'error')
            logging.error(error_msg, exc_info=True)
            import traceback
            logging.error(traceback.format_exc())
    
    def _navigate_to_instance_tab(self, instance_name: str):
        """Navigate to a specific instance tab."""
        try:
            self._log_message(f"Navigating to instance tab: {instance_name}", 'info')
            logging.info(f"Navigating to instance tab: {instance_name}")
            
            # First, switch to the Instances tab in the main notebook
            instances_tab_found = False
            for i in range(self.notebook.count()):
                if self.notebook.tabText(i) == "Instances":
                    self.notebook.setCurrentIndex(i)
                    instances_tab_found = True
                    logging.info(f"Switched to Instances tab")
                    break
            
            if not instances_tab_found:
                error_msg = "Instances tab not found in main notebook"
                self._log_message(error_msg, 'error')
                logging.error(error_msg)
                return
            
            # Then, switch to the specific instance tab in the instances notebook
            if not hasattr(self, 'instances_notebook'):
                error_msg = "instances_notebook attribute not found"
                self._log_message(error_msg, 'error')
                logging.error(error_msg)
                return
            
            if self.instances_notebook is None:
                error_msg = "instances_notebook is None"
                self._log_message(error_msg, 'error')
                logging.error(error_msg)
                return
            
            for i in range(self.instances_notebook.count()):
                if self.instances_notebook.tabText(i) == instance_name:
                    self.instances_notebook.setCurrentIndex(i)
                    self._log_message(f"Switched to instance tab: {instance_name}", 'success')
                    logging.info(f"Switched to instance tab: {instance_name}")
                    return
            
            self._log_message(f"Instance tab not found: {instance_name}", 'warning')
            logging.warning(f"Instance tab not found: {instance_name}")
        except Exception as e:
            error_msg = f"Error navigating to instance tab {instance_name}: {str(e)}"
            self._log_message(error_msg, 'error')
            logging.error(error_msg, exc_info=True)
            import traceback
            logging.error(traceback.format_exc())
    
    def _cleanup_all_instances(self):
        """Remove all instances from display and dictionaries after stopping."""
        self._log_message("_cleanup_all_instances called", 'info')
        
        # Get list of all instance names before removing
        instance_names = list(self.instance_ports.keys())
        self._log_message(f"Cleaning up {len(instance_names)} instances: {instance_names}", 'info')
        
        # Remove each instance from display and dictionaries
        for instance_name in instance_names:
            # Remove tab from notebook first (before removing from dictionaries)
            if self.instance_manager and hasattr(self.instance_manager, 'remove_instance_tab'):
                try:
                    self.instance_manager.remove_instance_tab(instance_name)
                except Exception as e:
                    self._log_message(f"Error removing tab for {instance_name}: {str(e)}", 'warning')
                    import traceback
                    self._log_message(f"Traceback: {traceback.format_exc()}", 'error')
            
            # Remove from display (this also removes from instance_ports and instance_tabs)
            self._remove_instance_from_display(instance_name)
        
        self._log_message(f"Cleaned up {len(instance_names)} instances", 'success')
    
    def _update_plan_details_wrapper(self, instance_name: str, widget):
        """Wrapper for updating plan details from either listbox or tree widget."""
        if not self.instance_manager:
            return
        
        plan_id = None
        current_item = None
        
        # Try to get current item - handle both QListWidget and QTreeWidget
        try:
            current_item = widget.currentItem()
        except AttributeError:
            return
        
        if not current_item:
            return
        
        # Check if widget is QTreeWidget (Available Plans)
        if hasattr(current_item, 'data') and hasattr(current_item, 'childCount'):
            # Skip if clicking on a directory item (only allow leaf items)
            if current_item.childCount() > 0:
                return
            
            # Get plan_id from item data
            plan_id = current_item.data(0, Qt.ItemDataRole.UserRole)
        
        # Check if widget is QListWidget (Selected Plans)
        elif hasattr(current_item, 'text'):
            # Extract plan_id from display text (format: "Label (plan_id)")
            display_text = current_item.text()
            if '(' in display_text and ')' in display_text:
                plan_id = display_text.split('(')[-1].rstrip(')')
            else:
                # Fallback: try to find plan by label
                for pid, plan_class in AVAILABLE_PLANS.items():
                    label = getattr(plan_class, 'label', pid.replace('_', ' ').title())
                    if label == display_text:
                        plan_id = pid
                        break
        
        if not plan_id:
            return
        
        # Get plan class and description
        plan_class = AVAILABLE_PLANS.get(plan_id)
        if not plan_class:
            return
        
        # Get description from plan class
        description = getattr(plan_class, 'description', 'No description available.')
        
        # Update description text in the GUI
        instance_tab = self.instance_manager.instance_tabs.get(instance_name)
        if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
            plan_runner_tab = instance_tab.plan_runner_tab
            if hasattr(plan_runner_tab, 'description_text'):
                plan_runner_tab.description_text.setPlainText(description)
    
    def _update_parameter_widgets_wrapper(self, instance_name: str, listbox):
        """Wrapper for updating parameter widgets."""
        pass
    
    def _browse_directory_wrapper(self, instance_name: str, dir_edit):
        """Wrapper for browsing directory."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        current_dir = dir_edit.text()
        directory = QFileDialog.getExistingDirectory(
            self,
            f"Select Session Directory for {instance_name}",
            current_dir if current_dir else str(Path.home())
        )
        if directory:
            dir_edit.setText(directory)
    
    def _populate_sequences_wrapper(self, instance_name: str, listbox):
        """Wrapper for populating sequences list."""
        from pathlib import Path
        try:
            sequences_dir = Path(__file__).resolve().parent.parent / "plan_sequences"
            if sequences_dir.exists():
                listbox.clear()
                for seq_file in sorted(sequences_dir.glob("*.json")):
                    listbox.addItem(seq_file.stem)
        except Exception as e:
            self._log_message(f"Error populating sequences list: {e}", 'error')
    
    def _save_sequence_wrapper(self, instance_name: str):
        """Wrapper for saving sequence."""
        self._log_message("Sequence saving not yet implemented", 'info')
    
    def _load_sequence_wrapper(self, instance_name: str, listbox):
        """Wrapper for loading sequence."""
        self._log_message("Sequence loading not yet implemented", 'info')
    
    def _delete_sequence_wrapper(self, instance_name: str, listbox):
        """Wrapper for deleting sequence."""
        self._log_message("Sequence deletion not yet implemented", 'info')
    
    def _add_plan_wrapper(self, instance_name: str, available_tree, selected_listbox):
        """Wrapper for adding plan to selection."""
        if self.instance_manager:
            self.instance_manager._add_plan_to_selection(instance_name, available_tree, selected_listbox)
    
    def _remove_plan_wrapper(self, instance_name: str, selected_listbox):
        """Wrapper for removing plan from selection."""
        if self.instance_manager:
            self.instance_manager._remove_plan_from_selection(instance_name, selected_listbox)
    
    def _move_plan_up_wrapper(self, instance_name: str, selected_listbox):
        """Wrapper for moving plan up."""
        if self.instance_manager:
            self.instance_manager._move_plan_up(instance_name, selected_listbox)
    
    def _move_plan_down_wrapper(self, instance_name: str, selected_listbox):
        """Wrapper for moving plan down."""
        if self.instance_manager:
            self.instance_manager._move_plan_down(instance_name, selected_listbox)
    
    def _clear_plans_wrapper(self, instance_name: str, selected_listbox):
        """Wrapper for clearing all plans."""
        if self.instance_manager:
            self.instance_manager._clear_selected_plans(instance_name, selected_listbox)
    
    def _add_rule_wrapper(self, instance_name: str, selected_listbox, rule_type_combo, rule_data):
        """Wrapper for adding rule."""
        if self.instance_manager:
            self.instance_manager._add_rule_inline_advanced(instance_name, selected_listbox, rule_type_combo, rule_data)
    
    def _clear_params_wrapper(self, instance_name: str, selected_listbox):
        """Wrapper for clearing parameters."""
        if self.instance_manager:
            self.instance_manager._clear_plan_parameters(instance_name, selected_listbox)
    
    def _clear_rules_wrapper(self, instance_name: str, selected_listbox):
        """Wrapper for clearing rules."""
        if self.instance_manager:
            self.instance_manager._clear_plan_rules(instance_name, selected_listbox)
    
    def _start_plans_wrapper(self, instance_name: str, session_dir: str, port: int):
        """Wrapper for starting plans."""
        self._log_message(f"Starting plans for {instance_name}", 'info')
        if self.instance_manager:
            self.instance_manager.start_plans_for_instance(instance_name, session_dir, port)
    
    def _stop_plans_wrapper(self, instance_name: str):
        """Wrapper for stopping plans."""
        self._log_message(f"Stopping plans for {instance_name}", 'info')
        if self.instance_manager:
            self.instance_manager.stop_plans_for_instance(instance_name)
    
    def customEvent(self, event):
        """Handle custom events posted from background threads."""
        # Handle log-from-thread events (launcher maximize thread); just run callback, no extra log
        if getattr(event, 'log_from_thread', False) and hasattr(event, 'callback'):
            try:
                event.callback()
            except Exception as e:
                self._log_message(f"Error in log-from-thread callback: {str(e)}", 'error')
            return
        # Handle cleanup events from launcher
        if hasattr(event, 'callback') and not hasattr(event, 'delay_ms'):
            try:
                self._log_message("Processing cleanup event from background thread", 'info')
                event.callback()
                self._log_message("Cleanup event processed successfully", 'info')
            except Exception as e:
                self._log_message(f"Error in custom event callback: {str(e)}", 'error')
                import traceback
                self._log_message(f"Traceback: {traceback.format_exc()}", 'error')
        # Handle create tabs events from launcher (with delay)
        elif hasattr(event, 'callback') and hasattr(event, 'delay_ms'):
            try:
                self._log_message(f"Processing create tabs event from background thread (delay: {event.delay_ms}ms)", 'info')
                # Use QTimer on main thread to delay the callback
                from PySide6.QtCore import QTimer
                QTimer.singleShot(event.delay_ms, event.callback)
                self._log_message("Create tabs event scheduled successfully", 'info')
            except Exception as e:
                self._log_message(f"Error in create tabs event: {str(e)}", 'error')
                import traceback
                self._log_message(f"Traceback: {traceback.format_exc()}", 'error')
        # Handle close window events from stop thread
        # Check if it's a CloseWindowEvent (has no callback attribute)
        elif not hasattr(event, 'callback'):
            try:
                self._log_message("Processing close window event from background thread", 'info')
                # Don't reuse the original close_event - it's been deleted
                # Just call close() directly which will trigger closeEvent again
                # But this time we'll skip the check since instances are already stopped
                # and _closing_async flag is set
                self.close()
                self._log_message("Close window event processed successfully", 'info')
            except Exception as e:
                self._log_message(f"Error in close window event: {str(e)}", 'error')
                import traceback
                self._log_message(f"Traceback: {traceback.format_exc()}", 'error')
        else:
            super().customEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Check if there are running instances
        has_running_instances = False
        try:
            import psutil
            base_dir = self.config_vars.get("baseDir", "")
            if base_dir:
                pid_file = Path(base_dir) / "runelite-pids.txt"
            else:
                project_dir = Path(__file__).resolve().parent.parent
                pid_file = project_dir / "instances" / "runelite-pids.txt"
            
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        lines = f.readlines()
                        # Quick check - avoid slow cmdline() call that can freeze UI
                        for line in lines[:10]:  # Limit to first 10 PIDs for speed
                            parts = line.strip().split(',')
                            if len(parts) >= 1:
                                try:
                                    pid = int(parts[0])
                                    # Quick existence check only - don't call cmdline() which is slow
                                    if psutil.pid_exists(pid):
                                        try:
                                            proc = psutil.Process(pid)
                                            # Just check process name, don't check cmdline (which is slow)
                                            if 'java' in proc.name().lower():
                                                has_running_instances = True
                                                break
                                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                                            pass
                                except (ValueError, IndexError):
                                    pass
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"Error checking running processes: {e}")
        
        if has_running_instances:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Stop All Instances?",
                "There are running instances. Do you want to stop all instances before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.StandardButton.Yes:
                # Mark that we're closing asynchronously
                self._closing_async = True
                # Defer stop operations to prevent blocking the UI
                # Use QTimer to schedule on next event loop iteration
                QTimer.singleShot(0, lambda: self._stop_all_instances_and_close(event))
                event.ignore()  # Don't close yet, wait for async stop
                return
        
        # No running instances, proceed with normal close
        self._perform_cleanup()
        event.accept()
    
    def _stop_all_instances_and_close(self, close_event):
        """Stop all instances asynchronously, then close the window."""
        # Store reference to self for use in thread
        window_ref = self
        
        def stop_in_thread():
            try:
                # Stop all instances (these may block, so run in thread)
                window_ref._stop_all_instances()
                # Also stop via launcher (this already runs in a thread internally)
                if window_ref.launcher:
                    window_ref.launcher.stop_runelite(window_ref._cleanup_all_instances)
                
                # Perform cleanup
                window_ref._perform_cleanup()
                
                # Close the window after operations complete (on main thread)
                # Use QApplication.postEvent instead of QTimer since we're in a background thread
                # Don't reuse the close_event - it gets deleted. Just call close() directly.
                from PySide6.QtWidgets import QApplication
                from PySide6.QtCore import QEvent
                
                class CloseWindowEvent(QEvent):
                    def __init__(self):
                        super().__init__(QEvent.Type.User)
                
                close_evt = CloseWindowEvent()
                QApplication.postEvent(window_ref, close_evt)
            except Exception as e:
                logging.error(f"Error stopping instances: {e}")
                # Close anyway after error
                from PySide6.QtWidgets import QApplication
                from PySide6.QtCore import QEvent
                
                class CloseWindowEvent(QEvent):
                    def __init__(self):
                        super().__init__(QEvent.Type.User)
                
                close_evt = CloseWindowEvent()
                QApplication.postEvent(window_ref, close_evt)
        
        # Run stop operations in a background thread to prevent blocking UI
        import threading
        thread = threading.Thread(target=stop_in_thread, daemon=True)
        thread.start()
    
    def _perform_cleanup(self):
        """Perform cleanup operations before closing."""
        # Stop all running plans
        if self.instance_manager:
            for instance_name in list(self.instance_tabs.keys()):
                instance_tab = self.instance_tabs.get(instance_name)
                if instance_tab and hasattr(instance_tab, 'plan_runner_tab'):
                    plan_runner_tab = instance_tab.plan_runner_tab
                    if hasattr(plan_runner_tab, 'is_running') and plan_runner_tab.is_running:
                        try:
                            self.instance_manager.stop_plans_for_instance(instance_name)
                        except Exception as e:
                            logging.error(f"Error stopping plans for {instance_name}: {e}")
        
        # Stop all stats monitors
        for username in list(self.stats_monitors.keys()):
            try:
                self.statistics.stop_stats_monitor(username)
            except Exception as e:
                logging.error(f"Error stopping stats monitor for {username}: {e}")
        
        # Stop client detection
        if self.client_detector:
            try:
                self.client_detector.stop_client_detection()
            except Exception as e:
                logging.error(f"Error stopping client detection: {e}")
        
        # Save configuration
        if self.config_manager:
            try:
                self.config_manager.save_config(
                    base_port_var=None,  # Removed from UI
                    launch_delay_var=None,  # Removed from UI
                    log_callback=self._log_message
                )
            except Exception as e:
                logging.error(f"Error saving configuration: {e}")
    
    def _detect_running_instances_on_startup(self):
        """Detect running RuneLite instances on startup and populate instance manager."""
        try:
            import psutil
            
            # Use baseDir from config (where instances are stored)
            base_dir_str = self.config_vars.get("baseDir", "")
            if base_dir_str:
                base_dir = Path(base_dir_str)
            else:
                # Fallback to projectDir/instances
                project_dir_str = self.config_vars.get("projectDir", "")
                if project_dir_str:
                    base_dir = Path(project_dir_str) / "instances"
                else:
                    # Final fallback
                    base_dir = Path(__file__).resolve().parent.parent / "instances"
            
            # Read PID file
            pid_file = base_dir / "runelite-pids.txt"
            if not pid_file.exists():
                self._log_message("No PID file found, no running instances to detect", 'info')
                return
            
            detected_instances = []
            
            with open(pid_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # PID file format: pid,instance_index,port,instance_dir
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            pid = int(parts[0])
                            inst_index = int(parts[1])
                            inst_port = int(parts[2])
                            inst_dir = Path(parts[3])
                            
                            # Check if process is still running
                            if psutil.pid_exists(pid):
                                try:
                                    proc = psutil.Process(pid)
                                    if 'java' in proc.name().lower():
                                        # Check if it's a RuneLite process
                                        cmdline = proc.cmdline()
                                        if cmdline and any('runelite' in arg.lower() for arg in cmdline):
                                            # Try to get credential name from instance directory
                                            credential_name = "Unknown"
                                            cred_file = inst_dir / ".runelite" / "credentials.properties"
                                            if cred_file.exists():
                                                try:
                                                    with open(cred_file, 'r') as cf:
                                                        for cred_line in cf:
                                                            cred_line = cred_line.strip()
                                                            if cred_line.startswith('JX_DISPLAY_NAME='):
                                                                username = cred_line.split('=', 1)[1].strip()
                                                                # Try to match with available credentials
                                                                creds_dir_str = self.config_vars.get("credentialsDir", "")
                                                                if creds_dir_str:
                                                                    creds_dir = Path(creds_dir_str)
                                                                else:
                                                                    creds_dir = Path(__file__).resolve().parent.parent / "credentials"
                                                                
                                                                if creds_dir.exists():
                                                                    for cred_file_path in creds_dir.glob("*.properties"):
                                                                        if cred_file_path.stem.lower() == username.lower():
                                                                            credential_name = cred_file_path.stem
                                                                            break
                                                                        # Also check JX_DISPLAY_NAME in credential file
                                                                        try:
                                                                            with open(cred_file_path, 'r') as cf2:
                                                                                for cf_line in cf2:
                                                                                    if cf_line.startswith('JX_DISPLAY_NAME='):
                                                                                        cred_username = cf_line.split('=', 1)[1].strip()
                                                                                        if cred_username.lower() == username.lower():
                                                                                            credential_name = cred_file_path.stem
                                                                                            break
                                                                        except Exception:
                                                                            continue
                                                                else:
                                                                    credential_name = username
                                                                break
                                                except Exception:
                                                    pass
                                            
                                            # Use detected port name format
                                            instance_name = f"detected_{inst_port}"
                                            
                                            detected_instances.append({
                                                'name': instance_name,
                                                'port': inst_port,
                                                'credential': credential_name,
                                                'pid': pid
                                            })
                                            self._log_message(f"Detected running instance on port {inst_port} (credential: {credential_name})", 'info')
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (ValueError, IndexError):
                            continue
            
            if not detected_instances:
                self._log_message("No running RuneLite instances found", 'info')
                return
            
            self._log_message(f"Found {len(detected_instances)} running RuneLite instance(s)", 'info')
            
            # Create instance tabs for detected instances
            for inst_info in detected_instances:
                instance_name = inst_info['name']
                port = inst_info['port']
                
                # Add to instance ports
                self.instance_ports[instance_name] = port
                
                # Store credential mapping in client detector
                if self.client_detector:
                    self.client_detector.port_to_credential[port] = inst_info['credential']
                    self.client_detector.detected_clients[instance_name] = port
                
                # Create instance tab
                if self.instance_manager:
                    try:
                        self.instance_manager.create_instance_tab(
                            instance_name, port,
                            browse_directory_callback=self._browse_directory_wrapper,
                            populate_sequences_callback=self._populate_sequences_wrapper,
                            save_sequence_callback=self._save_sequence_wrapper,
                            load_sequence_callback=self._load_sequence_wrapper,
                            delete_sequence_callback=self._delete_sequence_wrapper,
                            add_plan_callback=self._add_plan_wrapper,
                            remove_plan_callback=self._remove_plan_wrapper,
                            move_plan_up_callback=self._move_plan_up_wrapper,
                            move_plan_down_callback=self._move_plan_down_wrapper,
                            clear_plans_callback=self._clear_plans_wrapper,
                            update_plan_details_callback=self._update_plan_details_wrapper,
                            update_parameter_widgets_callback=self._update_parameter_widgets_wrapper,
                            add_rule_callback=self._add_rule_wrapper,
                            clear_params_callback=self._clear_params_wrapper,
                            clear_rules_callback=self._clear_rules_wrapper,
                            start_plans_callback=self._start_plans_wrapper,
                            stop_plans_callback=self._stop_plans_wrapper
                        )
                        # Update instance manager display
                        self._update_instance_manager_display(instance_name, port)
                        self._log_message(f"Created tab for detected instance: {instance_name}", 'success')
                    except Exception as e:
                        self._log_message(f"Error creating tab for detected instance {instance_name}: {e}", 'error')
                        import traceback
                        self._log_message(f"Traceback: {traceback.format_exc()}", 'error')
            
            if detected_instances:
                self._log_message(f"Detected and loaded {len(detected_instances)} running instance(s)", 'success')
        except Exception as e:
            self._log_message(f"Error detecting running instances on startup: {e}", 'error')
            import traceback
            self._log_message(f"Traceback: {traceback.format_exc()}", 'error')