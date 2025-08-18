import numpy as np
from collections import Counter, defaultdict
from typing import Dict, Optional
import hashlib

def stable_hash(name):
    if not isinstance(name, str):
        name = str(name)
    return int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % 100000

# Add a safe float conversion helper
def safe_float(val, default=0.0):
    try:
        if val == "" or val is None:
            return default
        return float(val)
    except Exception:
        return default

def extract_state_features(state: dict, n_npcs=10, n_objects=10, n_widgets=10, prev_state: Optional[dict]=None) -> np.ndarray:
    """
    Extract focused features from a game state dict for sapphire ring crafting.
    Optimized for bank → furnace → crafting workflow with 85% feature reduction.
    """
    features = []
    
    # Player features (8) - Essential for movement and positioning
    player = state.get('player', {})
    px, py = player.get('world_x', 0), player.get('world_y', 0)
    features.extend([
        px, py, player.get('plane', 0), player.get('health', 0), player.get('animation', 0),
        player.get('run_energy', 0), player.get('prayer', 0), player.get('special_attack', 0)
    ])
    
    # Camera features (5) - Essential for movement and interaction
    features.extend([
        state.get('camera_x', 0), state.get('camera_y', 0), state.get('camera_z', 0),
        state.get('camera_pitch', 0), state.get('camera_yaw', 0)
    ])
    
    # Inventory (56) - 28 slots × 2 (id, quantity) - Essential for crafting materials
    inventory = state.get('inventory', [])
    for i in range(28):
        if i < len(inventory):
            features.append(inventory[i].get('id', -1))
            features.append(inventory[i].get('quantity', 0))
        else:
            features.extend([-1, 0])
    
    # Bank state (1) - Is bank open or closed?
    bank_open = 0
    widgets = state.get('widgets', [])
    for widget in widgets:
        if 'bank' in widget.get('text', '').lower() or widget.get('id') in [12, 13]:  # Common bank widget IDs
            bank_open = 1
            break
    features.append(bank_open)
    
    # Bank contents (simplified) - Just count of key items
    bank_items = state.get('bank', [])
    sapphire_count = 0
    gold_bar_count = 0
    ring_count = 0
    for item in bank_items:
        item_id = item.get('id', -1)
        quantity = item.get('quantity', 0)
        if item_id == 1603:  # Sapphire
            sapphire_count += quantity
        elif item_id == 2357:  # Gold bar
            gold_bar_count += quantity
        elif item_id in [1607, 1609, 1611]:  # Sapphire rings/necklaces/amulets
            ring_count += quantity
    features.extend([sapphire_count, gold_bar_count, ring_count])
    
    # Game Objects - Focus on banks and furnaces, plus nearby objects
    objects = state.get('game_objects', [])
    bank_booths = []
    furnaces = []
    nearby_objects = []
    
    for obj in objects:
        obj_name = obj.get('name', '').lower()
        obj_x, obj_y = obj.get('x', 0), obj.get('y', 0)
        distance = (obj_x - px)**2 + (obj_y - py)**2
        
        if 'bank' in obj_name or 'booth' in obj_name:
            bank_booths.append(obj)
        elif 'furnace' in obj_name:
            furnaces.append(obj)
        elif distance < 10000:  # Within reasonable distance
            nearby_objects.append(obj)
    
    # Bank booths (closest 3) - 3 × 5 (id, x, y, plane, name_hash)
    bank_booths.sort(key=lambda o: (o.get('x', 0)-px)**2 + (o.get('y', 0)-py)**2)
    for i in range(3):
        if i < len(bank_booths):
            obj = bank_booths[i]
            features.extend([obj.get('id', -1), obj.get('x', 0), obj.get('y', 0), obj.get('plane', 0), stable_hash(obj.get('name', ''))])
        else:
            features.extend([-1, 0, 0, 0, 0])
    
    # Furnaces (closest 3) - 3 × 5 (id, x, y, plane, name_hash)
    furnaces.sort(key=lambda o: (o.get('x', 0)-px)**2 + (o.get('y', 0)-py)**2)
    for i in range(3):
        if i < len(furnaces):
            obj = furnaces[i]
            features.extend([obj.get('id', -1), obj.get('x', 0), obj.get('y', 0), obj.get('plane', 0), stable_hash(obj.get('name', ''))])
        else:
            features.extend([-1, 0, 0, 0, 0])
    
    # Nearby objects (closest 10) - 10 × 5 (id, x, y, plane, name_hash)
    nearby_objects.sort(key=lambda o: (o.get('x', 0)-px)**2 + (o.get('y', 0)-py)**2)
    for i in range(10):
        if i < len(nearby_objects):
            obj = nearby_objects[i]
            features.extend([obj.get('id', -1), obj.get('x', 0), obj.get('y', 0), obj.get('plane', 0), stable_hash(obj.get('name', ''))])
        else:
            features.extend([-1, 0, 0, 0, 0])
    
    # NPCs - Only bankers (closest 5) - 5 × 6 (id, x, y, plane, health, name_hash)
    npcs = state.get('npcs', [])
    bankers = [npc for npc in npcs if 'bank' in npc.get('name', '').lower()]
    bankers.sort(key=lambda n: (n.get('x', 0)-px)**2 + (n.get('y', 0)-py)**2)
    for i in range(5):
        if i < len(bankers):
            npc = bankers[i]
            features.extend([npc.get('id', -1), npc.get('x', 0), npc.get('y', 0), npc.get('plane', 0), npc.get('health', 0), stable_hash(npc.get('name', ''))])
        else:
            features.extend([-1, 0, 0, 0, 0, 0])
    
    # Minimap (4) - Essential for navigation
    minimap = state.get('minimap_world_info', {})
    features.append(minimap.get('base_x', 0))
    features.append(minimap.get('base_y', 0))
    features.append(state.get('minimap_rotation', 0))
    features.append(state.get('minimap_zoom', 0))
    
    # Skills - Only crafting (2) - level and xp
    skills = state.get('skills', {})
    crafting_data = skills.get('crafting', {})
    features.append(safe_float(crafting_data.get('level', 0)))
    features.append(safe_float(crafting_data.get('xp', 0)))
    
    # Chatbox (1) - Just the last message hash
    chatbox = state.get('chatbox', '')
    if not chatbox:
        # Try to extract from widgets
        for w in widgets:
            if 'Press Enter to Chat' in w.get('text', ''):
                chatbox = w.get('text', '')
                break
    if isinstance(chatbox, list):
        chatbox_str = '\n'.join(map(str, chatbox))
    else:
        chatbox_str = str(chatbox)
    features.append(stable_hash(chatbox_str))
    
    # Tabs (13) - All tabs active/inactive
    tab_names = ['combat', 'skills', 'quests', 'inventory', 'equipment', 'prayer', 'magic', 
                'clan', 'friends', 'account', 'settings', 'emotes', 'music']
    for tab_name in tab_names:
        features.append(1 if state.get('currentTab', 0) == tab_names.index(tab_name) else 0)
    
    # Previous state features (if provided) - recursive but without prev_state to avoid infinite recursion
    if prev_state is not None:
        prev_feats = extract_state_features(prev_state, None)
        features.extend(prev_feats)
    
    return np.array(features, dtype=np.float32)

# TODO: Integrate n_npcs, n_objects, n_widgets as GUI-configurable parameters
# TODO: Use a fixed vocabulary for histograms for full one-hot encoding if desired

from typing import Dict

def extract_action_features(action: Dict, prev_action: Optional[Dict]=None) -> np.ndarray:
    """
    Extract focused action features for sapphire ring crafting.
    Includes all essential mouse/keyboard input for natural movement.
    """
    features = []
    
    # Mouse position (2)
    features.append(safe_float(action.get('xw_norm', 0)))
    features.append(safe_float(action.get('yw_norm', 0)))
    
    # Event type (5) - One-hot encoded
    features.extend(action.get('event_type_onehot', [0,0,0,0,0]))
    
    # Active keys (13) - One-hot encoded
    features.extend(action.get('active_keys_onehot', [0]*13))
    
    # Scroll (2)
    features.append(safe_float(action.get('scroll_dx', 0)))
    features.append(safe_float(action.get('scroll_dy', 0)))
    
    # Timing (1)
    features.append(safe_float(action.get('dt', 0)))
    
    # Click type (1) - 0=none, 1=left, 2=right, 3=middle
    btn = action.get('btn', '').lower()
    if btn == 'left':
        features.append(1)
    elif btn == 'right':
        features.append(2)
    elif btn == 'middle':
        features.append(3)
    else:
        features.append(0)
    
    # Key code (1)
    features.append(safe_float(action.get('key_code', 0)))
    
    # Previous action features (if provided)
    if prev_action is not None:
        prev_feats = extract_action_features(prev_action, None)
        features.extend(prev_feats)
    
    return np.array(features, dtype=np.float32) 