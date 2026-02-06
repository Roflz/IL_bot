"""
Inventory Panel Widget
=====================

Custom widget that displays the inventory background image with item icons overlaid
at their correct grid positions.
"""

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter, QImage
from typing import Dict, Optional, List
from pathlib import Path
import logging
import base64
from io import BytesIO


class InventoryPanel(QWidget):
    """Widget that displays inventory background with item icons overlaid."""
    
    # Inventory grid: 4 columns x 7 rows = 28 slots
    GRID_COLUMNS = 4
    GRID_ROWS = 7
    TOTAL_SLOTS = 28
    
    def __init__(self, parent=None):
        """Initialize inventory panel."""
        super().__init__(parent)
        self.background_pixmap: Optional[QPixmap] = None
        self.item_icons: Dict[int, QPixmap] = {}  # slot_index -> icon pixmap
        self.item_data: Dict[int, Dict] = {}  # slot_index -> {name, quantity, iconBase64}
        self.slot_size = 0  # Will be calculated based on background image size
        self.grid_start_x = 0  # Will be calculated based on background image
        self.grid_start_y = 0  # Will be calculated based on background image
        self.grid_spacing_x = 0  # Will be calculated based on background image
        self.grid_spacing_y = 0  # Will be calculated based on background image
        
        # Set size policy to prefer the background image size
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        self._load_background()
        self._calculate_grid_positions()
        
        # Set minimum size based on background
        if self.background_pixmap:
            self.setMinimumSize(self.background_pixmap.size() / 2)
    
    def _load_background(self):
        """Load the inventory background image."""
        try:
            # Try to load from gui/assets directory
            assets_dir = Path(__file__).resolve().parent / "assets"
            bg_path = assets_dir / "inventory_background.png"
            
            if bg_path.exists():
                self.background_pixmap = QPixmap(str(bg_path))
                logging.info(f"[GUI] Loaded inventory background from {bg_path}")
            else:
                # Create a placeholder if image doesn't exist
                logging.warning(f"[GUI] Inventory background not found at {bg_path}, using placeholder")
                self.background_pixmap = self._create_placeholder(200, 350)  # Approximate size
        except Exception as e:
            logging.error(f"[GUI] Error loading inventory background: {e}")
            self.background_pixmap = self._create_placeholder(200, 350)
    
    def _create_placeholder(self, width: int, height: int) -> QPixmap:
        """Create a placeholder pixmap if background image is missing."""
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.darkGray)
        return pixmap
    
    def _calculate_grid_positions(self):
        """Calculate grid positions based on background image size."""
        if not self.background_pixmap:
            return
        
        bg_width = self.background_pixmap.width()
        bg_height = self.background_pixmap.height()
        
        # Inventory grid placement:
        # Left border: 24px, Right border: 24px
        # Available width: bg_width - 24 - 24 = bg_width - 48
        # 4 columns, so each column = (bg_width - 48) / 4
        left_border = 24
        right_border = 24
        available_width = bg_width - left_border - right_border
        self.grid_spacing_x = available_width // self.GRID_COLUMNS  # 48px for 240px image
        
        # Grid starts at left border
        self.grid_start_x = left_border
        
        # Height placement:
        # Top border: 34px, Bottom border: 34px
        # Available height: bg_height - 34 - 34 = bg_height - 68
        # 8 rows (actually 7 rows for inventory, but using 8 for calculation), so each row = (bg_height - 68) / 8
        top_border = 34
        bottom_border = 34
        available_height = bg_height - top_border - bottom_border
        self.grid_spacing_y = available_height // self.GRID_ROWS  # Divide by 7 rows for inventory
        
        # Grid starts at top border
        self.grid_start_y = top_border
        
        # Calculate base slot size, then scale by 1.5x
        # Base slot size is the smaller of the two spacings (to keep items square)
        base_slot_size = min(self.grid_spacing_x, self.grid_spacing_y) - 2  # 2px padding
        # Scale by 1.5x
        self.slot_size = int(base_slot_size * 1)
        # Clamp to reasonable bounds (at least 16px, at most 60px)
        self.slot_size = max(16, min(60, self.slot_size))
        
        logging.debug(f"[GUI] Inventory grid: bg_size=({bg_width}x{bg_height}), "
                     f"start=({self.grid_start_x}, {self.grid_start_y}), "
                     f"spacing=({self.grid_spacing_x}, {self.grid_spacing_y}), slot_size={self.slot_size}")
    
    def slot_to_grid_position(self, slot_index: int) -> tuple:
        """Convert slot index (0-27) to grid (row, col) position.
        
        OSRS inventory layout:
        Row 0: slots 0, 1, 2, 3
        Row 1: slots 4, 5, 6, 7
        ...
        Row 6: slots 24, 25, 26, 27
        """
        row = slot_index // self.GRID_COLUMNS
        col = slot_index % self.GRID_COLUMNS
        return (row, col)
    
    def grid_to_pixel_position(self, row: int, col: int) -> tuple:
        """Convert grid (row, col) to pixel (x, y) position.
        
        Items are centered in their column quadrants and row segments.
        Column 0 center: 24 + 24 = 48px
        Column 1 center: 72 + 24 = 96px
        Column 2 center: 120 + 24 = 144px
        Column 3 center: 168 + 24 = 192px
        
        Row centers are calculated similarly, centered in each row segment.
        """
        # Calculate center of column quadrant
        column_center_x = self.grid_start_x + (col * self.grid_spacing_x) + (self.grid_spacing_x // 2)
        # Center item on the column center
        x = column_center_x - (self.slot_size // 2)
        
        # Calculate center of row segment
        row_center_y = self.grid_start_y + (row * self.grid_spacing_y) + (self.grid_spacing_y // 2)
        # Center item on the row center
        y = row_center_y - (self.slot_size // 2)
        
        return (x, y)
    
    def decode_base64_icon(self, icon_base64: str) -> Optional[QPixmap]:
        """Decode base64 icon string to QPixmap."""
        try:
            from PIL import Image
            from PIL.ImageQt import ImageQt
            
            # Decode base64
            image_data = base64.b64decode(icon_base64)
            # Load image from bytes
            img = Image.open(BytesIO(image_data))
            # Resize to slot size
            img = img.resize((self.slot_size, self.slot_size), Image.Resampling.LANCZOS)
            # Convert to QPixmap
            qt_img = ImageQt(img)
            pixmap = QPixmap.fromImage(qt_img)
            return pixmap
        except Exception as e:
            logging.debug(f"[GUI] Error decoding base64 icon: {e}")
            return None
    
    def set_items(self, items: List[Dict]):
        """Set inventory items to display.
        
        Args:
            items: List of item dicts with keys: slot (0-27), id, itemName, quantity, iconBase64
        """
        self.item_icons.clear()
        self.item_data.clear()
        
        for item in items:
            slot = item.get('slot')
            if slot is None or slot < 0 or slot >= self.TOTAL_SLOTS:
                continue
            
            item_id = item.get('id', -1)
            if item_id == -1:
                continue  # Empty slot
            
            # Store item data
            self.item_data[slot] = {
                'name': item.get('itemName', 'Unknown'),
                'quantity': item.get('quantity', 0),
                'id': item_id
            }
            
            # Decode and store icon
            icon_base64 = item.get('iconBase64')
            if icon_base64:
                icon_pixmap = self.decode_base64_icon(icon_base64)
                if icon_pixmap:
                    self.item_icons[slot] = icon_pixmap
        
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the background and item icons."""
        if not self.background_pixmap:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget size
        widget_width = self.width()
        widget_height = self.height()
        bg_width = self.background_pixmap.width()
        bg_height = self.background_pixmap.height()
        
        # Calculate scaling to maintain aspect ratio
        scale_x = widget_width / bg_width
        scale_y = widget_height / bg_height
        scale = min(scale_x, scale_y)  # Use smaller scale to fit within widget
        
        # Calculate scaled background size
        scaled_bg_width = int(bg_width * scale)
        scaled_bg_height = int(bg_height * scale)
        
        # Center the background if widget is larger
        bg_x = (widget_width - scaled_bg_width) // 2
        bg_y = (widget_height - scaled_bg_height) // 2
        
        # Draw background (scaled to maintain aspect ratio)
        scaled_bg = self.background_pixmap.scaled(
            scaled_bg_width, scaled_bg_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(bg_x, bg_y, scaled_bg)
        
        # Draw item icons at their grid positions (relative to scaled background)
        for slot, icon_pixmap in self.item_icons.items():
            row, col = self.slot_to_grid_position(slot)
            x, y = self.grid_to_pixel_position(row, col)
            
            # Scale positions relative to background
            scaled_x = bg_x + int(x * scale)
            scaled_y = bg_y + int(y * scale)
            scaled_size = int(self.slot_size * scale)
            
            # Draw icon
            scaled_icon = icon_pixmap.scaled(
                scaled_size, scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.drawPixmap(scaled_x, scaled_y, scaled_icon)
    
    def sizeHint(self):
        """Return preferred widget size."""
        if self.background_pixmap:
            # Return background size as preferred size
            size = self.background_pixmap.size()
            return size
        return super().sizeHint()
    
    def minimumSizeHint(self):
        """Return minimum widget size."""
        if self.background_pixmap:
            # Allow some scaling down but maintain aspect ratio
            size = self.background_pixmap.size()
            return size / 2
        return super().minimumSizeHint()
    
    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        self.update()  # Repaint when resized
