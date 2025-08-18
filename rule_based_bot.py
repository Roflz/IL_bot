import time
import pyautogui
import json
import os

# Absolute path to the latest game state JSON exported by RuneLite plugin
GAME_STATE_PATH = r'C:\temp\runelite_gamestate.json'

# Hardcoded screen coordinates for tree and bank (update as needed)
TREE_POS = (800, 400)
BANK_POS = (1000, 200)

# Inventory slot count for OSRS
INVENTORY_SIZE = 28
LOG_ITEM_IDS = {1511, 1521, 1519, 1517, 1515, 1513}  # Normal, oak, willow, maple, yew, magic logs

# Helper to read game state

def read_game_state():
    if os.path.exists(GAME_STATE_PATH):
        try:
            with open(GAME_STATE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading game state: {e}")
    return None

def inventory_is_full(state):
    inv = state.get('inventory', [])
    return len(inv) >= INVENTORY_SIZE

def has_logs(state):
    inv = state.get('inventory', [])
    for item in inv:
        if str(item.get('id')) in map(str, LOG_ITEM_IDS):
            return True
    return False

def at_tree(state):
    # Placeholder: check if player is near tree (use player position and tree position)
    # For now, always return False to force movement
    return False

def at_bank(state):
    # Placeholder: check if player is near bank
    return False

def find_player_position(state):
    # Try common keys for player position
    for key in ['player', 'localPlayer', 'self']:
        if key in state:
            p = state[key]
            if 'x' in p and 'y' in p:
                return p['x'], p['y']
            if 'world_x' in p and 'world_y' in p:
                return p['world_x'], p['world_y']
    # Fallback: try top-level keys
    if 'x' in state and 'y' in state:
        return state['x'], state['y']
    if 'world_x' in state and 'world_y' in state:
        return state['world_x'], state['world_y']
    return None, None

def find_trees(state):
    trees = []
    for obj in state.get('objects', []):
        if obj.get('name', '').lower() == 'tree':
            # Try both x/y and world_x/world_y
            if 'x' in obj and 'y' in obj:
                trees.append({'x': obj['x'], 'y': obj['y']})
            elif 'world_x' in obj and 'world_y' in obj:
                trees.append({'x': obj['world_x'], 'y': obj['world_y']})
    return trees

def main():
    print("Bot diagnostic: Finding closest tree each tick. Press Ctrl+C to stop.")
    while True:
        state = read_game_state()
        if not state:
            print("No game state available. Waiting...")
            time.sleep(5)
            continue
        px, py = find_player_position(state)
        if px is None or py is None:
            print("Player position not found in state.")
            time.sleep(5)
            continue
        print(f"Player position: x={px}, y={py}")
        trees = find_trees(state)
        if not trees:
            print("No trees found in state.")
            time.sleep(5)
            continue
        min_dist = float('inf')
        closest_tree = None
        for i, tree in enumerate(trees):
            tx, ty = tree['x'], tree['y']
            dist = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
            print(f"Tree {i}: x={tx}, y={ty}, distance={dist:.2f}")
            if dist < min_dist:
                min_dist = dist
                closest_tree = tree
        if closest_tree:
            print(f"Closest tree: x={closest_tree['x']}, y={closest_tree['y']}, distance={min_dist:.2f}")
        print("---")
        time.sleep(5)

if __name__ == '__main__':
    main() 