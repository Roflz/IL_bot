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


def draw_wall_lines(draw, x, y, tile_size, orientation_a, orientation_b):
    """Draw wall lines based on orientation."""
    # Wall orientation values:
    # 1 = West, 2 = North, 4 = East, 8 = South
    # 16 = North-west, 32 = North-east, 64 = South-east, 128 = South-west
    
    line_color = (255, 255, 0)  # Yellow for wall lines
    line_width = 3  # Thicker lines for better visibility
    
    # Draw orientation A
    if orientation_a & 1:  # West
        draw.line([x, y, x, y + tile_size], fill=line_color, width=line_width)
    if orientation_a & 2:  # North
        draw.line([x, y, x + tile_size, y], fill=line_color, width=line_width)
    if orientation_a & 4:  # East
        draw.line([x + tile_size, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    if orientation_a & 8:  # South
        draw.line([x, y + tile_size, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    
    # Draw orientation B
    if orientation_b & 1:  # West
        draw.line([x, y, x, y + tile_size], fill=line_color, width=line_width)
    if orientation_b & 2:  # North
        draw.line([x, y, x + tile_size, y], fill=line_color, width=line_width)
    if orientation_b & 4:  # East
        draw.line([x + tile_size, y, x + tile_size, y + tile_size], fill=line_color, width=line_width)
    if orientation_b & 8:  # South
        draw.line([x, y + tile_size, x + tile_size, y + tile_size], fill=line_color, width=line_width)


def load_collision_data():
    """Load collision data from cache."""
    # Use the script's directory as the base path
    script_dir = Path(__file__).parent
    cache_file = script_dir / "collision_cache" / "collision_map.json"
    print(f"[DEBUG] Loading from: {cache_file.absolute()}")
    if not cache_file.exists():
        print("ERROR: No collision cache found. Run collision mapping first.")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return data.get("collision_data", {})
    except Exception as e:
        print(f"ERROR: Could not load cache: {e}")
        return None


def is_water_tile(x, y, plane, water_data):
    """Check if a tile is water."""
    return f"{x},{y},{plane}" in water_data


def generate_single_map():
    """Generate a single comprehensive collision map with embedded legend."""
    print("SIMPLE COLLISION MAP GENERATOR")
    print("=" * 50)
    
    # Load collision data
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
    
    tiles_drawn = 0
    for key, tile_data in collision_data.items():
        x, y, p = tile_data['x'], tile_data['y'], tile_data['p']
        
        # Calculate position on image
        draw_x = map_start_x + (x - min_x) * tile_size
        draw_y = map_start_y + map_height - ((y - min_y) * tile_size) - tile_size  # Invert Y
        
        # Determine color and symbol
        color = (0, 0, 0)  # Default black
        symbol = ""
        
        if tile_data.get('door'):
            # Draw wall as lines based on orientation instead of filling entire tile
            door_info = tile_data.get('door', {})
            orientation_a = door_info.get('orientationA', 0)
            orientation_b = door_info.get('orientationB', 0)
            passable = door_info.get('passable', False)
            
            # Don't fill the entire tile - just draw wall lines on appropriate sides
            if not passable:
                # Draw wall lines on the sides that are blocked
                draw_wall_lines(draw, draw_x, draw_y, tile_size, orientation_a, orientation_b)
                # Fill the tile with a light background to show it's walkable
                draw.rectangle([draw_x, draw_y, draw_x + tile_size, draw_y + tile_size], fill=(64, 64, 64))  # Dark gray background
                symbol = "W"
            else:
                # Passable walls (bridges, doors) - just show as walkable
                draw.rectangle([draw_x, draw_y, draw_x + tile_size, draw_y + tile_size], fill=(0, 255, 0))  # Green for walkable
                symbol = "P"
            
            # Don't set color since we're not using it for filling
            color = None
        elif tile_data.get('solid', False):
            color = (255, 0, 0)    # Red
            symbol = "#"
        elif tile_data.get('object', False):
            color = (0, 0, 255)    # Blue
            symbol = "O"
        elif tile_data.get('walkable', False):
            color = (0, 255, 0)    # Green
            symbol = "."
        else:
            color = (128, 128, 128)  # Gray
            symbol = "?"
        
        # Draw tile (only if not a door, since doors handle their own drawing)
        if not tile_data.get('door'):
            draw.rectangle([draw_x, draw_y, draw_x + tile_size, draw_y + tile_size], fill=color)
        
        # Add symbol
        if symbol:
            draw.text((draw_x + 1, draw_y + 1), symbol, fill=(255, 255, 255), font=small_font)
        
        tiles_drawn += 1
    
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
    
    legend_y = add_legend_item("Walkable (.)", (0, 255, 0), legend_y)
    legend_y = add_legend_item("Solid (#)", (255, 0, 0), legend_y)
    legend_y = add_legend_item("Object (O)", (0, 0, 255), legend_y)
    legend_y = add_legend_item("Door (D)", (255, 255, 0), legend_y)
    legend_y = add_legend_item("Unknown (?)", (128, 128, 128), legend_y)
    legend_y = add_legend_item("No Data", (0, 0, 0), legend_y)
    
    # Add statistics
    legend_y += 20
    draw.text((legend_x, legend_y), "Statistics:", fill=(255, 255, 255), font=legend_font)
    legend_y += 20
    
    # Count tile types
    walkable_count = sum(1 for tile in collision_data.values() if tile.get('walkable', False))
    solid_count = sum(1 for tile in collision_data.values() if tile.get('solid', False))
    object_count = sum(1 for tile in collision_data.values() if tile.get('object', False))
    door_count = sum(1 for tile in collision_data.values() if tile.get('door'))
    
    draw.text((legend_x, legend_y), f"Tiles: {len(collision_data):,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Walkable: {walkable_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Solid: {solid_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Objects: {object_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Doors: {door_count:,}", fill=(255, 255, 255), font=legend_font)
    legend_y += 15
    draw.text((legend_x, legend_y), f"Size: {width_tiles}x{height_tiles}", fill=(255, 255, 255), font=legend_font)
    
    # Save the map with embedded legend
    script_dir = Path(__file__).parent
    output_file = script_dir / "collision_cache" / "detailed_collision_map.png"
    
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
    print("This will generate a single comprehensive collision map with embedded legend.")
    print()
    
    try:
        success = generate_single_map()
        if success:
            print("\n[SUCCESS] Collision map with embedded legend created successfully!")
            print("File created:")
            print("  - detailed_collision_map.png (includes legend and statistics)")
        else:
            print("\n[ERROR] Failed to create collision map!")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
