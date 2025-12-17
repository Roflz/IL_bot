#!/usr/bin/env python3
"""
Simple collision map generator.
Creates a single comprehensive PNG image of the collected collision data.
"""
import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CollisionDataFlag constants from RuneLite API
BLOCK_N  = 2        # CollisionDataFlag.BLOCK_MOVEMENT_NORTH
BLOCK_E  = 8        # CollisionDataFlag.BLOCK_MOVEMENT_EAST  
BLOCK_S  = 32       # CollisionDataFlag.BLOCK_MOVEMENT_SOUTH
BLOCK_W  = 128      # CollisionDataFlag.BLOCK_MOVEMENT_WEST
BLOCK_NE = 4        # CollisionDataFlag.BLOCK_MOVEMENT_NORTH_EAST
BLOCK_SE = 16       # CollisionDataFlag.BLOCK_MOVEMENT_SOUTH_EAST
BLOCK_SW = 64       # CollisionDataFlag.BLOCK_MOVEMENT_SOUTH_WEST
BLOCK_NW = 1        # CollisionDataFlag.BLOCK_MOVEMENT_NORTH_WEST
BLOCK_OBJECT = 256  # CollisionDataFlag.BLOCK_MOVEMENT_OBJECT
BLOCK_FULL = 2359552  # CollisionDataFlag.BLOCK_MOVEMENT_FULL

# Internal edge constants
N = 0x1
E = 0x2
S = 0x4
W = 0x8
NE = 0x10
SE = 0x20
SW = 0x40
NW = 0x80
FULL = 0x100

# === helpers ===
DIR = {
    'N': {'bit': 2,   'dx': 0,  'dy': +1, 'opp_bit': 32},   # RL flags for A (cardinals)
    'E': {'bit': 8,   'dx': +1, 'dy': 0,  'opp_bit': 128},
    'S': {'bit': 32,  'dx': 0,  'dy': -1, 'opp_bit': 2},
    'W': {'bit': 128, 'dx': -1, 'dy': 0,  'opp_bit': 8},
}
DIAG = {
    'NE': {'bit': 4,  'dx': +1, 'dy': +1, 'opp_bit': 64},   # B (diagonals)
    'SE': {'bit': 16, 'dx': +1, 'dy': -1, 'opp_bit': 1},
    'SW': {'bit': 64, 'dx': -1, 'dy': -1, 'opp_bit': 4},
    'NW': {'bit': 1,  'dx': -1, 'dy': +1, 'opp_bit': 16},
}


def mask_from_flags(flags: int):
    """Convert CollisionDataFlag to internal edge mask."""
    m = 0
    if flags & BLOCK_N:  m |= N
    if flags & BLOCK_E:  m |= E
    if flags & BLOCK_S:  m |= S
    if flags & BLOCK_W:  m |= W
    if flags & BLOCK_NE: m |= NE
    if flags & BLOCK_SE: m |= SE
    if flags & BLOCK_SW: m |= SW
    if flags & BLOCK_NW: m |= NW
    if flags & BLOCK_OBJECT: m |= FULL  # Treat OBJECT as full block
    if flags & BLOCK_FULL: m |= FULL
    return m


def draw_wall_lines(draw, x, y, tile_size, orientation_a, orientation_b, is_real_door=False):
    """Draw wall lines based on movement blocking flags."""
    line_color = (0, 255, 255) if is_real_door else (255, 255, 255)
    line_width = 2

    # Draw ONLY cardinals
    if orientation_a & 2:   draw.line([x, y, x + tile_size, y], fill=line_color, width=line_width)              # N
    if orientation_a & 8:   draw.line([x + tile_size, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)  # E
    if orientation_a & 32:  draw.line([x, y + tile_size, x + tile_size, y + tile_size], fill=line_color, width=line_width)  # S
    if orientation_a & 128: draw.line([x, y, x, y + tile_size], fill=line_color, width=line_width)              # W

    # Intentionally skip NE/SE/SW/NW drawing


def load_collision_data():
    """Load ALL collision data from cache."""
    # Use the script's directory as the base path
    script_dir = Path(__file__).parent
    cache_file = script_dir / "collision_cache" / "collision_map_debug.json"
    print(f"[DEBUG] Loading from: {cache_file.absolute()}")
    if not cache_file.exists():
        print("ERROR: No collision cache found. Run collision mapping first.")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        all_collision_data = data.get("collision_data", {})
        print(f"[DEBUG] Total collision tiles in cache: {len(all_collision_data)}")
        
        print(f"[DEBUG] Using ALL collision data: {len(all_collision_data)} tiles")
        return all_collision_data
    except Exception as e:
        print(f"ERROR: Could not load cache: {e}")
        return None


def is_water_tile(x, y, plane, water_data):
    """Check if a tile is water."""
    return f"{x},{y},{plane}" in water_data


def generate_single_map():
    """Generate a single comprehensive collision map with embedded legend using ALL tiles."""
    print("SIMPLE COLLISION MAP GENERATOR")
    print("=" * 50)
    
    # Load ALL collision data
    collision_data = load_collision_data()
    if not collision_data:
        return False
    
    print(f"Loaded {len(collision_data)} collision tiles")
    
    # Determine map bounds
    all_x = [tile['x'] for tile in collision_data.values()]
    all_y = [tile['y'] for tile in collision_data.values()]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1
    
    print(f"Map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print(f"Map size: {width_tiles}x{height_tiles}")
    
    # Create the map image with space for legend
    tile_size = 16  # 2x2 pixels per tile for better wall visualization
    map_width = width_tiles * tile_size
    map_height = height_tiles * tile_size
    
    # Add space for legend on the right
    legend_width = 200
    padding = 20
    
    img_width = map_width + legend_width + (padding * 3)
    img_height = max(map_height, 300) + (padding * 2)  # Ensure enough height for legend
    
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))  # Black background
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        small_font = ImageFont.truetype("arial.ttf", 8)
        legend_font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        small_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw collision tiles
    map_start_x = padding
    map_start_y = padding
    
    # Collect wall tiles to draw orientation lines after all tiles are drawn
    wall_tiles = []  # (draw_x, draw_y, tile_size, orientation_a, orientation_b, is_real_door)
    
    # First pass: collect door clearing information without modifying the dictionary
    door_clears = {}  # tile_id -> mask_override
    
    for key, tile_data in collision_data.items():
        x, y, p = tile_data['x'], tile_data['y'], tile_data['p']
        
        # Check for door and clear appropriate bits
        door_info = tile_data.get('door')
        if door_info and isinstance(door_info, dict) and 'orientationA' in door_info:
            is_real_door = door_info.get('actions', []) and door_info['actions'][0] == 'Open'
            if is_real_door:
                # Get flags and compute mask
                flags = tile_data.get('flags', 0)
                m = mask_from_flags(flags)
                
                # Clear door bits from current tile
                orientation_a = door_info.get('orientationA', 0)
                orientation_b = door_info.get('orientationB', 0)
                
                # Clear cardinal edges
                for name, info in DIR.items():
                    if orientation_a & info['bit']:
                        if info['bit'] == 2:   m &= ~N
                        if info['bit'] == 8:   m &= ~E
                        if info['bit'] == 32:  m &= ~S
                        if info['bit'] == 128: m &= ~W
                        # Clear opposite edge on neighbor
                        nx, ny = x + info['dx'], y + info['dy']
                        nid = f"{nx},{ny},{p}"
                        if nid in collision_data:
                            nflags = collision_data[nid].get('flags', 0)
                            nm = mask_from_flags(nflags)
                            if info['opp_bit'] == 2:   nm &= ~N
                            if info['opp_bit'] == 8:   nm &= ~E
                            if info['opp_bit'] == 32:  nm &= ~S
                            if info['opp_bit'] == 128: nm &= ~W
                            door_clears[nid] = nm
                
                # Clear diagonal edges
                for name, info in DIAG.items():
                    if orientation_b & info['bit']:
                        if info['bit'] == 4:   m &= ~NE
                        if info['bit'] == 16:  m &= ~SE
                        if info['bit'] == 64:  m &= ~SW
                        if info['bit'] == 1:   m &= ~NW
                        nx, ny = x + info['dx'], y + info['dy']
                        nid = f"{nx},{ny},{p}"
                        if nid in collision_data:
                            nflags = collision_data[nid].get('flags', 0)
                            nm = mask_from_flags(nflags)
                            if info['opp_bit'] == 4:   nm &= ~NE
                            if info['opp_bit'] == 16:  nm &= ~SE
                            if info['opp_bit'] == 64:  nm &= ~SW
                            if info['opp_bit'] == 1:   nm &= ~NW
                            door_clears[nid] = nm
                
                # Store the cleared mask for the door tile itself
                door_clears[key] = m
    
    # Second pass: draw tiles using the collected door clearing information
    tiles_drawn = 0
    for key, tile_data in collision_data.items():
        x, y, p = tile_data['x'], tile_data['y'], tile_data['p']
        
        # Calculate position on image
        draw_x = map_start_x + (x - min_x) * tile_size
        draw_y = map_start_y + map_height - ((y - min_y) * tile_size) - tile_size  # Invert Y
        
        # Get flags and compute mask
        flags = tile_data.get('flags', 0)
        m = mask_from_flags(flags)
        
        # Use door clearing information if available
        if key in door_clears:
            m = door_clears[key]
        
        # Determine color and symbol based on flags
        color = (0, 0, 0)  # Default black
        symbol = ""
        
        # Check for door/ladder markers first (these take priority)
        door_info = tile_data.get('door')
        if door_info and isinstance(door_info, dict) and 'orientationA' in door_info:
            color = (0, 0, 0)  # Black background for doors
            symbol = "D"  # D for door
        elif tile_data.get('ladderUp', False):
            color = (0, 0, 0)  # Black background
            symbol = "U"  # U for ladder up
        elif tile_data.get('ladderDown', False):
            color = (0, 0, 0)  # Black background
            symbol = "↓"  # ↓ for ladder down
        # Check for full tile blocked (after doors/ladders)
        elif m & FULL:
            # Check if it's an object vs solid block
            if flags & BLOCK_OBJECT:
                color = (0, 0, 255)  # Blue background for objects
                symbol = "O"  # O for object
            else:
                color = (255, 0, 0)  # Red background for solid blocks
                symbol = "#"  # # for solid
        else:
            # Default walkable tile
            color = (0, 0, 0)  # Black background
            symbol = "."
        
        # Draw tile background
        draw.rectangle([draw_x, draw_y, draw_x + tile_size, draw_y + tile_size], fill=color)
        
        # Add symbol
        if symbol:
            draw.text((draw_x + 1, draw_y + 1), symbol, fill=(255, 255, 255), font=small_font)
        
        
        # Store for wall line drawing if there are blocking edges
        if m & (N | E | S | W | NE | SE | SW | NW):
            # Convert mask back to orientation format for drawing
            orientation_a = 0
            orientation_b = 0
            
            # Convert internal mask to orientation format (using RuneLite API values)
            if m & N: orientation_a |= 2   # BLOCK_MOVEMENT_NORTH
            if m & E: orientation_a |= 8   # BLOCK_MOVEMENT_EAST
            if m & S: orientation_a |= 32  # BLOCK_MOVEMENT_SOUTH
            if m & W: orientation_a |= 128 # BLOCK_MOVEMENT_WEST
            
            if m & NE: orientation_b |= 4   # BLOCK_MOVEMENT_NORTH_EAST
            if m & SE: orientation_b |= 16  # BLOCK_MOVEMENT_SOUTH_EAST
            if m & SW: orientation_b |= 64  # BLOCK_MOVEMENT_SOUTH_WEST
            if m & NW: orientation_b |= 1   # BLOCK_MOVEMENT_NORTH_WEST
            
            # Check if this tile has a door with orientation data
            door_info = tile_data.get('door')
            is_real_door = door_info and isinstance(door_info, dict) and 'orientationA' in door_info
            wall_tiles.append((draw_x, draw_y, tile_size, orientation_a, orientation_b, is_real_door))
        
        tiles_drawn += 1
    
    # Draw wall orientation lines AFTER all tiles are drawn (so they appear on top)
    print(f"[DEBUG] Drawing {len(wall_tiles)} wall orientation lines...")
    for draw_x, draw_y, tile_size, orientation_a, orientation_b, is_real_door in wall_tiles:
        draw_wall_lines(draw, draw_x, draw_y, tile_size, orientation_a, orientation_b, is_real_door)
    
    # Draw border around map
    draw.rectangle([map_start_x - 2, map_start_y - 2, 
                   map_start_x + map_width + 2, map_start_y + map_height + 2], 
                  outline=(100, 100, 100), width=2)
    
    # Draw embedded legend
    legend_x = map_start_x + map_width + padding
    legend_y = map_start_y
    
    # Legend title
    draw.text((legend_x, legend_y), "RuneScape Collision Map", fill=(255, 255, 255), font=title_font)
    legend_y += 30
    
    # Legend items
    def add_legend_item(text, color, y):
        # Draw colored square
        draw.rectangle([legend_x, y, legend_x + 20, y + 15], fill=color, outline=(255, 255, 255))
        # Draw text
        draw.text((legend_x + 25, y + 2), text, fill=(255, 255, 255), font=legend_font)
        return y + 20
    
    legend_y = add_legend_item("Walkable (.)", (0, 0, 0), legend_y)
    legend_y = add_legend_item("Solid block (#)", (255, 0, 0), legend_y)
    legend_y = add_legend_item("Object (O)", (0, 0, 255), legend_y)
    legend_y = add_legend_item("Door (D) - Cyan lines", (0, 255, 255), legend_y)
    legend_y = add_legend_item("Ladder up (U)", (0, 0, 0), legend_y)
    legend_y = add_legend_item("Ladder down (↓)", (0, 0, 0), legend_y)
    legend_y = add_legend_item("Wall edges (from flags)", (255, 255, 255), legend_y)
    
    # Add statistics
    legend_y += 20
    draw.text((legend_x, legend_y), "Statistics:", fill=(255, 255, 255), font=legend_font)
    legend_y += 20
    
    # Count tile types based on flags
    solid_block_count = 0
    object_count = 0
    door_count = 0
    ladder_up_count = 0
    ladder_down_count = 0
    wall_edge_count = 0
    
    for tile in collision_data.values():
        flags = tile.get('flags', 0)
        m = mask_from_flags(flags)
        
        if tile.get('door'):
            door_count += 1
        elif tile.get('ladderUp', False):
            ladder_up_count += 1
        elif tile.get('ladderDown', False):
            ladder_down_count += 1
        elif m & FULL:
            if flags & BLOCK_OBJECT:
                object_count += 1
            else:
                solid_block_count += 1
        
        # Count tiles with wall edges
        if m & (N | E | S | W | NE | SE | SW | NW):
            wall_edge_count += 1
    
    draw.text((legend_x, legend_y), f"Tiles: {len(collision_data):,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Solid blocks: {solid_block_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Objects: {object_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Doors: {door_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Ladders up: {ladder_up_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Ladders down: {ladder_down_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Wall edges: {wall_edge_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Size: {width_tiles}x{height_tiles}", fill=(255, 255, 255), font=legend_font)
    
    # Save the map with embedded legend
    script_dir = Path(__file__).parent
    output_file = script_dir / "collision_cache" / "detailed_collision_map_debug.png"
    
    print(f"[DEBUG] Saving PNG to: {output_file}")
    print(f"[DEBUG] Image size: {img.size[0]}x{img.size[1]} pixels")
    print(f"[DEBUG] Tiles to draw: {tiles_drawn}")
    
    img.save(output_file)
    print(f"Map with embedded legend saved to: {output_file}")
    print(f"Tiles drawn: {tiles_drawn}")
    
    # Verify the file was created
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"[DEBUG] PNG file created successfully, size: {file_size:,} bytes")
    else:
        print(f"[ERROR] PNG file was not created!")
    
    return True




def main():
    print("SIMPLE COLLISION MAP GENERATOR")
    print("=" * 50)
    print("This will generate a comprehensive collision map using ALL tiles from the cache.")
    print()
    
    try:
        success = generate_single_map()
        if success:
            print("\n[SUCCESS] Collision map with embedded legend created successfully!")
            print("File created:")
            print("  - detailed_collision_map_debug.png (includes legend and statistics)")
        else:
            print("\n[ERROR] Failed to create collision map!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
