"""
Equipment Panel Widget
======================

Custom widget that displays the equipment background image with item icons overlaid
at their correct equipment slot positions.
"""

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter
from typing import Dict, Optional, List
from pathlib import Path
import logging
import base64
from io import BytesIO


class EquipmentPanel(QWidget):
    """Widget that displays equipment background with item icons overlaid."""
    
    # Equipment slot positions (absolute pixel positions from top-left of image)
    # Each slot is 38x38 pixels
    # Y positions: HEAD=38, AMULET=77, BODY=117, LEGS=156, BOOTS=197
    SLOT_POSITIONS = {
        'HEAD': (120, 38),      # Top center (x = width/2 = 120)
        'CAPE': (72, 77),       # Left side, same row as amulet
        'AMULET': (168, 77),    # Right side
        'WEAPON': (72, 117),    # Left side, same row as body
        'BODY': (120, 117),     # Center
        'SHIELD': (168, 117),   # Right side, same row as body
        'LEGS': (120, 156),     # Center, below body
        'GLOVES': (72, 156),    # Left side, same row as legs
        'BOOTS': (120, 197),    # Center, bottom
        'RING': (168, 156),     # Right side, same row as legs
        'AMMO': (168, 38),      # Top right, same row as head
    }
    
    SLOT_SIZE = 38  # Each equipment slot is 38x38 pixels
    
    def __init__(self, parent=None):
        """Initialize equipment panel."""
        super().__init__(parent)
        self.background_pixmap: Optional[QPixmap] = None
        self.item_icons: Dict[str, QPixmap] = {}  # slot_name -> icon pixmap
        self.item_data: Dict[str, Dict] = {}  # slot_name -> {name, quantity, iconBase64}
        self.slot_size = self.SLOT_SIZE  # Use fixed slot size of 38x38
        
        # Set size policy to prefer the background image size
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        self._load_background()
        
        # Set minimum size based on background
        if self.background_pixmap:
            self.setMinimumSize(self.background_pixmap.size() / 2)
    
    def _load_background(self):
        """Load the equipment background image."""
        try:
            # Try to load from gui/assets directory
            assets_dir = Path(__file__).resolve().parent / "assets"
            bg_path = assets_dir / "equipment_background.png"
            
            if bg_path.exists():
                self.background_pixmap = QPixmap(str(bg_path))
                logging.info(f"[GUI] Loaded equipment background from {bg_path}")
            else:
                # Create a placeholder if image doesn't exist
                logging.warning(f"[GUI] Equipment background not found at {bg_path}, using placeholder")
                self.background_pixmap = self._create_placeholder(200, 350)  # Approximate size
        except Exception as e:
            logging.error(f"[GUI] Error loading equipment background: {e}")
            self.background_pixmap = self._create_placeholder(200, 350)
    
    def _create_placeholder(self, width: int, height: int) -> QPixmap:
        """Create a placeholder pixmap if background image is missing."""
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.darkGray)
        return pixmap
    
    def slot_to_pixel_position(self, slot_name: str) -> tuple:
        """Convert slot name to pixel (x, y) position.
        
        Uses exact pixel positions from top-left of image.
        Y positions are the TOP of the slot (not the center).
        """
        if not self.background_pixmap:
            return (0, 0)
        
        if slot_name not in self.SLOT_POSITIONS:
            logging.warning(f"[GUI] Unknown equipment slot: {slot_name}")
            return (0, 0)
        
        # Get absolute position (x, y) from top-left of image
        # Y is the TOP of the slot
        x, y = self.SLOT_POSITIONS[slot_name]
        
        # Center horizontally, but use Y as-is (top edge of icon at Y position)
        x -= self.slot_size // 2
        
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
        """Set equipment items to display.
        
        Args:
            items: List of item dicts with keys: slot (e.g., 'HEAD'), id, name, quantity, iconBase64
        """
        self.item_icons.clear()
        self.item_data.clear()
        
        for item in items:
            slot_name = item.get('slot', '').upper()
            if not slot_name or slot_name not in self.SLOT_POSITIONS:
                continue
            
            item_id = item.get('id', -1)
            if item_id == -1:
                continue  # Empty slot
            
            # Store item data
            self.item_data[slot_name] = {
                'name': item.get('name', 'Unknown'),
                'quantity': item.get('quantity', 0),
                'id': item_id
            }
            
            # Decode and store icon
            icon_base64 = item.get('iconBase64')
            if icon_base64:
                icon_pixmap = self.decode_base64_icon(icon_base64)
                if icon_pixmap:
                    self.item_icons[slot_name] = icon_pixmap
        
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
        
        # Draw item icons at their slot positions (relative to scaled background)
        for slot_name, icon_pixmap in self.item_icons.items():
            x, y = self.slot_to_pixel_position(slot_name)
            
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
