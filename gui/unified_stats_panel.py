"""
Unified Stats Panel Widget

A single widget that combines Skills, Inventory, and Equipment displays
with tab buttons to switch between them.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget,
    QLabel, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt
from pathlib import Path
import logging

from gui.inventory_panel import InventoryPanel
from gui.equipment_panel import EquipmentPanel


class UnifiedStatsPanel(QWidget):
    """Unified widget that displays Skills, Inventory, or Equipment based on selected tab."""
    
    def __init__(self, parent=None):
        """Initialize unified stats panel."""
        super().__init__(parent)
        
        self.current_view = "Skills"  # Default view
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        # Create tab buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        
        self.skills_btn = QPushButton("Skills")
        self.inventory_btn = QPushButton("Inventory")
        self.equipment_btn = QPushButton("Equipment")
        
        # Style buttons
        self.skills_btn.setCheckable(True)
        self.inventory_btn.setCheckable(True)
        self.equipment_btn.setCheckable(True)
        
        # Set Skills as default selected
        self.skills_btn.setChecked(True)
        
        # Connect button signals
        self.skills_btn.clicked.connect(lambda: self.switch_view("Skills"))
        self.inventory_btn.clicked.connect(lambda: self.switch_view("Inventory"))
        self.equipment_btn.clicked.connect(lambda: self.switch_view("Equipment"))
        
        button_layout.addWidget(self.skills_btn)
        button_layout.addWidget(self.inventory_btn)
        button_layout.addWidget(self.equipment_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # Create stacked widget to hold different views
        self.stacked_widget = QStackedWidget()
        
        # Skills view - scrollable list
        skills_scroll = QScrollArea()
        skills_scroll.setWidgetResizable(True)
        self.skills_content = QWidget()
        self.skills_content_layout = QVBoxLayout(self.skills_content)
        skills_scroll.setWidget(self.skills_content)
        self.stacked_widget.addWidget(skills_scroll)  # Index 0
        
        # Inventory view - panel with background
        self.inventory_panel = InventoryPanel()
        self.stacked_widget.addWidget(self.inventory_panel)  # Index 1
        
        # Equipment view - panel with background
        self.equipment_panel = EquipmentPanel()
        self.stacked_widget.addWidget(self.equipment_panel)  # Index 2
        
        main_layout.addWidget(self.stacked_widget, 1)
        
        # Set maximum width based on background image size (240px) + some padding
        self.setMaximumWidth(260)
        
        # Set size policy - don't expand vertically, prefer fixed width
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        
        # Set initial view
        self.stacked_widget.setCurrentIndex(0)  # Skills
    
    def switch_view(self, view_name: str):
        """Switch to the specified view."""
        if view_name == "Skills":
            self.current_view = "Skills"
            self.stacked_widget.setCurrentIndex(0)
            self.skills_btn.setChecked(True)
            self.inventory_btn.setChecked(False)
            self.equipment_btn.setChecked(False)
        elif view_name == "Inventory":
            self.current_view = "Inventory"
            self.stacked_widget.setCurrentIndex(1)
            self.skills_btn.setChecked(False)
            self.inventory_btn.setChecked(True)
            self.equipment_btn.setChecked(False)
        elif view_name == "Equipment":
            self.current_view = "Equipment"
            self.stacked_widget.setCurrentIndex(2)
            self.skills_btn.setChecked(False)
            self.inventory_btn.setChecked(False)
            self.equipment_btn.setChecked(True)
    
    def get_skills_layout(self):
        """Get the layout for skills content."""
        return self.skills_content_layout
    
    def get_inventory_panel(self):
        """Get the inventory panel widget."""
        return self.inventory_panel
    
    def get_equipment_panel(self):
        """Get the equipment panel widget."""
        return self.equipment_panel
