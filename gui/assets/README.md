# GUI Assets Directory

This directory contains background images for the inventory and equipment panels.

## Required Images

### 1. `inventory_background.png`
- **Description**: Background image showing the inventory grid (4x7 slots, 28 total slots)
- **Source**: Screenshot from RuneLite showing the inventory panel
- **Recommended size**: Approximately 200x350 pixels (or similar aspect ratio)
- **Content**: Should show the empty inventory grid slots on the right side of the equipment/inventory panel

### 2. `equipment_background.png`
- **Description**: Background image showing the equipment slots arranged in a human figure outline
- **Source**: Screenshot from RuneLite showing the equipment panel
- **Recommended size**: Approximately 200x350 pixels (or similar aspect ratio)
- **Content**: Should show the equipment slots (head, cape, amulet, weapon, body, shield, legs, gloves, boots, ring, ammo) arranged on a human figure outline

## How to Create These Images

1. **Take Screenshots**:
   - Open RuneLite and ensure you have an empty inventory and no equipment
   - Open the Equipment tab (or the combined Equipment/Inventory view)
   - Take a screenshot of:
     - The entire equipment/inventory panel
     - Or crop to show just the inventory grid (for inventory_background.png)
     - Or crop to show just the equipment slots (for equipment_background.png)

2. **Save Images**:
   - Save as PNG format
   - Name them exactly as specified above
   - Place them in this `gui/assets/` directory

3. **Image Requirements**:
   - PNG format (supports transparency if needed)
   - Clear, high-quality images
   - Empty slots (no items) for accurate positioning
   - Consistent lighting/colors

## Fallback Behavior

If these images are not found, the widgets will create placeholder gray backgrounds. However, item positioning will not be accurate without the proper background images.

## Positioning

The widgets calculate item positions based on the background image dimensions:
- **Inventory**: Items are positioned on a 4x7 grid (columns x rows)
- **Equipment**: Items are positioned at predefined percentages of the image dimensions

If your background images have different dimensions or layouts, you may need to adjust the positioning calculations in:
- `gui/inventory_panel.py` - `_calculate_grid_positions()` method
- `gui/equipment_panel.py` - `SLOT_POSITIONS` dictionary
