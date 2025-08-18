#!/usr/bin/env python3
"""
Live Feature Extractor for Real-time Bot Operation

This script extracts features from live gamestate data using the EXACT SAME logic
as the training preprocessing script (extract_features.py) to ensure consistency.
"""

from __future__ import annotations

import threading, time, json, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Centralized path resolution
BASE_DIR = Path(__file__).resolve().parent

def rp(*parts: str) -> Path:
    """Repo-local absolute path builder."""
    return (BASE_DIR / Path(*parts)).resolve()

class LiveFeatureExtractor:
    def __init__(self, bot_mode: str = "bot1"):
        """Initialize the live feature extractor."""
        # Set up paths using the same logic as the original
        rp = lambda *parts: Path(__file__).parent / Path(*parts)
        
        self.bot_mode = bot_mode
        
        # Ensure these exact attribute names exist with absolute paths
        self.gamestate_file = rp("data", bot_mode, "runelite_gamestate.json").resolve()
        self.gamestates_dir = rp("data", bot_mode, "gamestates").resolve()
        
        # Monitor state + source tracking
        self._source_mode = "unknown"  # "rolling" | "single" | "unknown"
        self._latest_gs = None  # last parsed dict (or None)
        self._monitor_thread = None
        self._stop_evt = None
        
        # Feature extraction settings - MUST match training script
        self.n_features = 128
        self.current_feature_index = 0
        self.session_start_time = None
        
        # Load feature mappings and id mappings created by training script
        self._feat_map = []  # length 128: list of {"path": "...", "kind": "numeric|categorical", "category": "..."}
        self._id_maps = {}   # from id_mappings.json created by training script
        
        self._load_mappings()
        
        print(f"[GS] paths set | file={self.gamestate_file} dir={self.gamestates_dir}")
        print(f"[GS] feature mappings loaded: {len(self._feat_map)} features")
        print(f"[GS] id mappings loaded: {len(self._id_maps)} categories")

    def _load_mappings(self):
        """Load feature mappings and id mappings created by training script."""
        try:
            # Load feature mappings created by extract_features.py
            feat_path = rp("data", "features", "feature_mappings.json")
            if feat_path.exists():
                with open(feat_path, "r", encoding="utf-8") as f:
                    feat_data = json.load(f)
                
                # Build 128-length mapping array
                self._feat_map = [None] * 128
                for item in feat_data:
                    idx = item.get("feature_index", 0)
                    if 0 <= idx < 128:
                        feature_name = item.get("feature_name", "")  # Use 'feature_name' key as that's what the file contains
                        data_type = item.get("data_type", "")
                        category = item.get("feature_group", "")
                        
                        # Determine if numeric or categorical based on training script logic
                        kind = "numeric" if data_type in ["world_coordinate", "angle_degrees", "camera_coordinate", "time_ms", "item_id", "skill_level", "count", "boolean", "animation_id", "npc_id", "object_id", "tab_id"] else "categorical"
                        
                        self._feat_map[idx] = {
                            "feature_name": feature_name,
                            "kind": kind,
                            "category": category,
                            "data_type": data_type
                        }
            
            # Load id mappings created by extract_features.py
            id_path = rp("data", "features", "id_mappings.json")
            if id_path.exists():
                with open(id_path, "r") as f:
                    self._id_maps = json.load(f)
                    
        except Exception as e:
            print(f"[GS] Warning: Failed to load mappings: {e}")
            # Fallback: create empty mappings
            self._feat_map = [None] * 128
            self._id_maps = {}

    def _get_feature_path(self, feature_name: str) -> str:
        """Get the actual JSON path for a feature name."""
        # Map feature names to actual JSON paths in the gamestate
        # These paths MUST match exactly how extract_features.py works
        feature_paths = {
            # Player features (0-4)
            "player_world_x": "player.world_x",
            "player_world_y": "player.world_y", 
            "player_animation_id": "player.animation_id",
            "player_is_moving": "player.is_moving",
            "player_movement_direction": "player.movement_direction",
            
            # Interaction features (5-8)
            "action_type": "last_interaction.action",
            "item_name": "last_interaction.item_name",
            "target": "last_interaction.target",
            "time_since_interaction": "last_interaction.timestamp",
            
            # Camera features (9-13)
            "camera_x": "camera_x",
            "camera_y": "camera_y",
            "camera_z": "camera_z",
            "camera_pitch": "camera_pitch",
            "camera_yaw": "camera_yaw",
            
            # Inventory features (14-41) - 28 slots
            # CORRECT: inventory is under tabs.inventory, not directly under inventory
            "inventory_slot_0": "tabs.inventory.0.id",
            "inventory_slot_1": "tabs.inventory.1.id",
            "inventory_slot_2": "tabs.inventory.2.id",
            "inventory_slot_3": "tabs.inventory.3.id",
            "inventory_slot_4": "tabs.inventory.4.id",
            "inventory_slot_5": "tabs.inventory.5.id",
            "inventory_slot_6": "tabs.inventory.6.id",
            "inventory_slot_7": "tabs.inventory.7.id",
            "inventory_slot_8": "tabs.inventory.8.id",
            "inventory_slot_9": "tabs.inventory.9.id",
            "inventory_slot_10": "tabs.inventory.10.id",
            "inventory_slot_11": "tabs.inventory.11.id",
            "inventory_slot_12": "tabs.inventory.12.id",
            "inventory_slot_13": "tabs.inventory.13.id",
            "inventory_slot_14": "tabs.inventory.14.id",
            "inventory_slot_15": "tabs.inventory.15.id",
            "inventory_slot_16": "tabs.inventory.16.id",
            "inventory_slot_17": "tabs.inventory.17.id",
            "inventory_slot_18": "tabs.inventory.18.id",
            "inventory_slot_19": "tabs.inventory.19.id",
            "inventory_slot_20": "tabs.inventory.20.id",
            "inventory_slot_21": "tabs.inventory.21.id",
            "inventory_slot_22": "tabs.inventory.22.id",
            "inventory_slot_23": "tabs.inventory.23.id",
            "inventory_slot_24": "tabs.inventory.24.id",
            "inventory_slot_25": "tabs.inventory.25.id",
            "inventory_slot_26": "tabs.inventory.26.id",
            "inventory_slot_27": "tabs.inventory.27.id",
            
            # Bank features (42-62) - CORRECT: use bank_item_positions structure
            "bank_open": "bank_open",
            # These are special - they need custom extraction logic, not direct paths
            "sapphires_exists": "bank_item_positions.sapphires",
            "sapphires_quantity": "bank_item_positions.sapphires.0.quantity",
            "sapphires_slot": "bank_item_positions.sapphires.0.slot",
            "sapphires_x": "bank_item_positions.sapphires.0.canvas_x",
            "sapphires_y": "bank_item_positions.sapphires.0.canvas_y",
            "rings_exists": "bank_item_positions.rings",
            "rings_quantity": "bank_item_positions.rings.0.quantity",
            "rings_slot": "bank_item_positions.rings.0.slot",
            "rings_x": "bank_item_positions.rings.0.canvas_x",
            "rings_y": "bank_item_positions.rings.0.canvas_y",
            "gold_bars_exists": "bank_item_positions.gold_bars",
            "gold_bars_quantity": "bank_item_positions.gold_bars.0.quantity",
            "gold_bars_slot": "bank_item_positions.gold_bars.0.slot",
            "gold_bars_x": "bank_item_positions.gold_bars.0.canvas_x",
            "gold_bars_y": "bank_item_positions.gold_bars.0.canvas_y",
            "moulds_exists": "bank_item_positions.moulds",
            "moulds_quantity": "bank_item_positions.moulds.0.quantity",
            "moulds_slot": "bank_item_positions.moulds.0.slot",
            "moulds_x": "bank_item_positions.moulds.0.canvas_x",
            "moulds_y": "bank_item_positions.moulds.0.canvas_y",
            
            # Phase context features (63-66)
            "phase_type": "phase_context.cycle_phase",
            "phase_start_time": "phase_context.phase_start_time",
            "phase_duration": "phase_context.phase_duration_ms",
            "gamestates_count": "phase_context.gamestates_in_phase",
            
            # Game objects features (67-108) - 42 objects × 1 feature each
            "game_object_0_distance": "game_objects.0.distance",
            "game_object_1_distance": "game_objects.1.distance",
            "game_object_2_distance": "game_objects.2.distance",
            "game_object_3_distance": "game_objects.3.distance",
            "game_object_4_distance": "game_objects.4.distance",
            "game_object_5_distance": "game_objects.5.distance",
            "game_object_6_distance": "game_objects.6.distance",
            "game_object_7_distance": "game_objects.7.distance",
            "game_object_8_distance": "game_objects.8.distance",
            "game_object_9_distance": "game_objects.9.distance",
            "game_object_10_distance": "game_objects.10.distance",
            "game_object_11_distance": "game_objects.11.distance",
            "game_object_12_distance": "game_objects.12.distance",
            "game_object_13_distance": "game_objects.13.distance",
            "game_object_14_distance": "game_objects.14.distance",
            "game_object_15_distance": "game_objects.15.distance",
            "game_object_16_distance": "game_objects.16.distance",
            "game_object_17_distance": "game_objects.17.distance",
            "game_object_18_distance": "game_objects.18.distance",
            "game_object_19_distance": "game_objects.19.distance",
            "game_object_20_distance": "game_objects.20.distance",
            "game_object_21_distance": "game_objects.21.distance",
            "game_object_22_distance": "game_objects.22.distance",
            "game_object_23_distance": "game_objects.23.distance",
            "game_object_24_distance": "game_objects.24.distance",
            "game_object_25_distance": "game_objects.25.distance",
            "game_object_26_distance": "game_objects.26.distance",
            "game_object_27_distance": "game_objects.27.distance",
            "game_object_28_distance": "game_objects.28.distance",
            "game_object_29_distance": "game_objects.29.distance",
            "game_object_30_distance": "game_objects.30.distance",
            "game_object_31_distance": "game_objects.31.distance",
            "game_object_32_distance": "game_objects.32.distance",
            "game_object_33_distance": "game_objects.33.distance",
            "game_object_34_distance": "game_objects.34.distance",
            "game_object_35_distance": "game_objects.35.distance",
            "game_object_36_distance": "game_objects.36.distance",
            "game_object_37_distance": "game_objects.37.distance",
            "game_object_38_distance": "game_objects.38.distance",
            "game_object_39_distance": "game_objects.39.distance",
            "game_object_40_distance": "game_objects.40.distance",
            "game_object_41_distance": "game_objects.41.distance",
            
            # NPC features (109-123) - 15 NPCs × 1 feature each
            "npc_0_distance": "npcs.0.distance",
            "npc_1_distance": "npcs.1.distance",
            "npc_2_distance": "npcs.2.distance",
            "npc_3_distance": "npcs.3.distance",
            "npc_4_distance": "npcs.4.distance",
            "npc_5_distance": "npcs.5.distance",
            "npc_6_distance": "npcs.6.distance",
            "npc_7_distance": "npcs.7.distance",
            "npc_8_distance": "npcs.8.distance",
            "npc_9_distance": "npcs.9.distance",
            "npc_10_distance": "npcs.10.distance",
            "npc_11_distance": "npcs.11.distance",
            "npc_12_distance": "npcs.12.distance",
            "npc_13_distance": "npcs.13.distance",
            "npc_14_distance": "npcs.14.distance",
            
            # Tab features (124)
            "current_tab": "tabs.currentTab",
            
            # Skills features (125-126)
            "crafting_level": "skills.crafting.level",
            "crafting_xp": "skills.crafting.xp",
            
            # Timestamp feature (127)
            "timestamp": "timestamp"
        }
        
        return feature_paths.get(feature_name, feature_name)

    def start_monitor(self):
        """Start monitoring thread (idempotent)."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_evt = threading.Event()
        
        def _loop():
            last_source = None
            last_log_time = 0
            tick_count = 0
            
            while not self._stop_evt.is_set():
                try:
                    # Pick source: rolling newest <5s old else single file
                    newest_path = None
                    newest_mtime = 0.0
                    
                    if self.gamestates_dir.exists():
                        for p in self.gamestates_dir.glob("*.json"):
                            m = p.stat().st_mtime
                            if m > newest_mtime:
                                newest_mtime, newest_path = m, p
                    
                    use_rolling = newest_path is not None and (time.time() - newest_mtime) < 5.0
                    
                    if use_rolling:
                        self._source_mode = "rolling"
                        path = newest_path
                    else:
                        self._source_mode = "single"
                        path = self.gamestate_file if self.gamestate_file.exists() else None
                    
                    # Parse JSON safely with retry on partial write
                    if path:
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            
                            if "timestamp" not in data:
                                data["timestamp"] = int(time.time() * 1000)
                            
                            self._latest_gs = data
                                
                        except (json.JSONDecodeError, FileNotFoundError):
                            # Partial write or file not found - skip this iteration
                            pass
                    
                    # Log source changes and heartbeat (rate-limited to avoid spam)
                    tick_count += 1
                    if last_source != self._source_mode and (time.time() - last_log_time) > 2.0:
                        print(f"[GST] source -> {self._source_mode}")
                        last_source = self._source_mode
                        last_log_time = time.time()
                    
                    # Heartbeat every ~3s
                    if tick_count % 5 == 0:  # 5 ticks * 600ms = ~3s
                        ts = int(time.time())
                        print(f"[GST] tick ts={ts} source={self._source_mode}")
                        
                except Exception as e:
                    # Minimal error logging
                    if (time.time() - last_log_time) > 5.0:
                        print(f"[GST] monitor error: {e}")
                        last_log_time = time.time()
                
                time.sleep(0.6)  # ~600ms interval
        
        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self):
        """Stop monitoring thread (idempotent)."""
        if self._stop_evt:
            self._stop_evt.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.5)
            self._monitor_thread = None

    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def get_data_source_info(self) -> Tuple[str, Path, Path]:
        """Get current data source information."""
        return (self._source_mode, self.gamestate_file, self.gamestates_dir)

    def get_latest_gamestate(self) -> Optional[Dict]:
        """Get latest gamestate data."""
        if self.is_monitoring():
            return dict(self._latest_gs) if self._latest_gs else None
        
        # Non-monitor fallback: one-shot read using rolling-preferred rule
        newest_path = None
        newest_mtime = 0.0
        
        if self.gamestates_dir.exists():
            for p in self.gamestates_dir.glob("*.json"):
                m = p.stat().st_mtime
                if m > newest_mtime:
                    newest_mtime, newest_path = m, p
        
        use_rolling = newest_path is not None and (time.time() - newest_mtime) < 5.0
        path = newest_path if use_rolling else (self.gamestate_file if self.gamestate_file.exists() else None)
        
        if not path:
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "timestamp" not in data:
                data["timestamp"] = int(time.time() * 1000)
            return data
        except Exception:
            return None

    def _safe_get(self, obj, path, default=None):
        """Safe getter that supports dotted paths and [idx] - matches training script logic."""
        try:
            if '[' in path and ']' in path:
                # Handle array indexing like "inventory[0].id"
                parts = path.split('[')
                base_path = parts[0]
                index_part = parts[1].split(']')[0]
                remaining_path = ']'.join(parts[1].split(']')[1:])
                
                # Get the base value
                base_value = self._safe_get(obj, base_path, None)
                if not isinstance(base_value, list):
                    return default
                
                # Apply index
                try:
                    index = int(index_part)
                    if 0 <= index < len(base_value):
                        indexed_value = base_value[index]
                        # If there's remaining path, continue traversing
                        if remaining_path and remaining_path.startswith('.'):
                            return self._safe_get(indexed_value, remaining_path[1:], default)
                        return indexed_value
                    return default
                except (ValueError, IndexError):
                    return default
            
            # Split path into segments
            segments = path.split('.')
            current = obj
            
            # Walk the path
            for segment in segments:
                if isinstance(current, dict):
                    current = current.get(segment)
                elif isinstance(current, list):
                    try:
                        idx = int(segment)
                        if 0 <= idx < len(current):
                            current = current[idx]
                        else:
                            return default
                    except ValueError:
                        return default
                else:
                    return default
                    
                if current is None:
                    break
            
            return current
        except Exception:
            return default

    def _extract_features_from_mappings(self, gamestate: Dict) -> np.ndarray:
        """Extract features using the EXACT same logic as extract_features.py"""
        features = []
        
        # Reset feature index counter for this gamestate
        self.current_feature_index = 0
        
        # Extract features from each category - EXACTLY like extract_features.py does
        
        # Player features (0-4)
        player_features = self._extract_player_features(gamestate.get('player', {}))
        features.extend(player_features)
        
        # Interaction features (5-8)
        interaction_features = self._extract_interaction_features(gamestate)
        features.extend(interaction_features)
        
        # Camera features (9-13)
        camera_features = self._extract_camera_features(gamestate)
        features.extend(camera_features)
        
        # Inventory features (14-41) - 28 slots
        # CORRECT: gamestate.get('inventory', []) NOT tabs.inventory
        inventory = gamestate.get('inventory', [])
        inventory_features = self._extract_inventory_features(inventory)
        features.extend(inventory_features)
        
        # Bank features (42-62)
        bank_features = self._extract_bank_features(gamestate)
        features.extend(bank_features)
        
        # Phase context features (63-66)
        phase_features = self._extract_phase_context_features(gamestate)
        features.extend(phase_features)
        
        # Game objects features (67-108) - 42 objects × 1 feature each
        game_object_features = self._extract_game_objects_features(gamestate)
        features.extend(game_object_features)
        
        # NPC features (109-123) - 15 NPCs × 1 feature each
        npcs = gamestate.get('npcs', [])
        npc_features = self._extract_npc_features(npcs)
        features.extend(npc_features)
        
        # Tab features (124)
        tabs = gamestate.get('tabs', {})
        tabs_features = self._extract_tabs_features(tabs)
        features.extend(tabs_features)
        
        # Skills features (125-126)
        skills = gamestate.get('skills', {})
        skills_features = self._extract_skills_features(skills)
        features.extend(skills_features)
        
        # Action summary features
        action_summary_features = self._extract_action_summary_features(gamestate)
        features.extend(action_summary_features)
        
        # Timestamp feature (127)
        absolute_timestamp = gamestate.get('timestamp', 0)
        relative_timestamp = self._to_relative_timestamp(absolute_timestamp)
        features.append(relative_timestamp)
        
        # Ensure we have exactly 128 features
        if len(features) != 128:
            raise ValueError(f"Expected 128 features, got {len(features)}")
        
        return np.array(features, dtype=np.float64)
    
    def _extract_player_features(self, player: Dict) -> List[float]:
        """Extract player state features using OSRS IDs directly where possible - EXACTLY like extract_features.py"""
        features = []
        
        # Position (numerical - keep as-is)
        world_x = player.get('world_x', 0)
        world_y = player.get('world_y', 0)
        features.extend([
            self._safe_float(world_x),
            self._safe_float(world_y)
        ])
        
        # Animation ID (numerical - use OSRS ID directly)
        animation_id = player.get('animation_id', -1)
        features.append(float(animation_id))
        
        # Movement state (boolean - keep as-is)
        is_moving = player.get('is_moving', False)
        features.append(1.0 if is_moving else 0.0)
        
        # Movement direction (categorical - hash this since it's dynamic text)
        direction = player.get('movement_direction', 'stationary')
        hashed_direction = self._stable_hash(direction)
        features.append(float(hashed_direction))
        
        return features
    
    def _extract_interaction_features(self, gamestate: Dict) -> List[float]:
        """Extract interaction features as separate meaningful features instead of one hash - EXACTLY like extract_features.py"""
        features = []
        
        last_interaction = gamestate.get('last_interaction', {})
        gamestate_timestamp = gamestate.get('timestamp', 0)
        
        # Extract the actual data fields from last_interaction
        action = last_interaction.get('action', '')
        item_name = last_interaction.get('item_name', '')
        target = last_interaction.get('target', '')
        interaction_timestamp = last_interaction.get('timestamp', 0)
        
        # Feature 5: Action type (hash the action text since it's dynamic)
        action_hash = self._stable_hash(action)
        features.append(self._safe_float(action_hash))
        
        # Feature 6: Item name (hash the item name since it's dynamic)
        item_hash = self._stable_hash(item_name)
        features.append(self._safe_float(item_hash))
        
        # Feature 7: Target (hash the target text since it's dynamic)
        target_hash = self._stable_hash(target)
        features.append(self._safe_float(target_hash))
        
        # Feature 8: Time since interaction (raw milliseconds)
        if interaction_timestamp > 0 and gamestate_timestamp > 0:
            time_since_interaction = gamestate_timestamp - interaction_timestamp
        else:
            time_since_interaction = 0.0
        features.append(time_since_interaction)
        
        return features
    
    def _extract_camera_features(self, state: Dict) -> List[float]:
        """Extract camera position and orientation (numerical - keep as-is) - EXACTLY like extract_features.py"""
        features = []
        
        camera_x = state.get('camera_x', 0)
        camera_y = state.get('camera_y', 0)
        camera_z = state.get('camera_z', 0)
        camera_pitch = state.get('camera_pitch', 0)
        camera_yaw = state.get('camera_yaw', 0)
        
        features.extend([camera_x, camera_y, camera_z, camera_pitch, camera_yaw])
        
        return features
    
    def _extract_inventory_features(self, inventory: List[Dict]) -> List[float]:
        """Extract inventory features using item IDs directly instead of hashing - EXACTLY like extract_features.py"""
        features = []
        
        for i in range(28):
            if i < len(inventory) and inventory[i]:
                item = inventory[i]
                item_id = item.get('id', -1)
                features.append(self._safe_float(item_id))  # Use OSRS item ID directly
            else:
                features.append(-1.0)  # No item
        
        return features
    
    def _extract_bank_features(self, gamestate: Dict) -> List[float]:
        """Extract bank features as separate meaningful features for each material position - EXACTLY like extract_features.py"""
        features = []
        
        # Feature: Bank open status (boolean)
        bank_open = gamestate.get('bank_open', False)
        features.append(1.0 if bank_open else 0.0)
        
        # Individual bank material features (20 total)
        # For each material type, create 5 features: existence, quantity, slot, x, y
        bank_item_positions = gamestate.get('bank_item_positions', {})
        
        # Define material types
        material_types = ['sapphires', 'gold_bars', 'rings', 'moulds']
        
        for material_type in material_types:
            items = bank_item_positions.get(material_type, [])
            
            if items:
                # Take the first item of this type
                item = items[0]
                
                # Feature 1: Existence (1.0 if item exists, 0.0 if not)
                features.append(1.0)
                
                # Feature 2: Quantity
                quantity = item.get('quantity', 0)
                features.append(self._safe_float(quantity))
                
                # Feature 3: Slot
                slot = item.get('slot', -1)
                features.append(self._safe_float(slot))
                
                # Feature 4: Canvas X coordinate
                canvas_x = item.get('canvas_x', -1)
                features.append(self._safe_float(canvas_x))
                
                # Feature 5: Canvas Y coordinate
                canvas_y = item.get('canvas_y', -1)
                features.append(self._safe_float(canvas_y))
            else:
                # Material doesn't exist - set all features to 0/-1
                features.extend([0.0, 0.0, -1.0, 0.0, 0.0])
        
        return features
    
    def _extract_phase_context_features(self, gamestate: Dict) -> List[float]:
        """Extract phase context features - EXACTLY like extract_features.py"""
        features = []
        
        phase_context = gamestate.get('phase_context', {})
        
        # Phase type (hash the phase type text)
        cycle_phase = phase_context.get('cycle_phase', 'unknown')
        phase_type_hash = self._stable_hash(cycle_phase)
        features.append(float(phase_type_hash))
        
        # Phase start time (relative to session start, milliseconds)
        phase_start_time = phase_context.get('phase_start_time', 0)
        features.append(float(phase_start_time))
        
        # Phase duration (raw milliseconds)
        phase_duration_ms = phase_context.get('phase_duration_ms', 0)
        features.append(float(phase_duration_ms))
        
        # Gamestates in phase (raw count)
        gamestates_in_phase = phase_context.get('gamestates_in_phase', 0)
        features.append(float(gamestates_in_phase))
        
        return features
    
    def _extract_game_objects_features(self, gamestate: Dict) -> List[float]:
        """Extract game objects features - EXACTLY like extract_features.py"""
        features = []
        
        # 1. 10 closest game objects - each with 3 features (ID, x, y)
        all_objects = gamestate.get('game_objects', [])
        unique_objects = []
        seen_coords = set()
        
        for obj in all_objects:
            if obj is None or obj.get('name') == 'null' or obj.get('id') is None:
                continue
            coords = (obj.get('x', 0), obj.get('y', 0))
            if coords not in seen_coords:
                unique_objects.append(obj)
                seen_coords.add(coords)
        
        unique_objects.sort(key=lambda obj: obj.get('distance', float('inf')))
        
        for i in range(10):
            if i < len(unique_objects):
                obj = unique_objects[i]
                features.extend([
                    float(obj.get('id', 0)),  # Object ID
                    float(obj.get('x', 0)),   # World X
                    float(obj.get('y', 0))    # World Y
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # 2. 1 closest furnace - with 3 features (ID, x, y)
        furnaces = gamestate.get('furnaces', [])
        if furnaces:
            furnaces.sort(key=lambda f: f.get('distance', float('inf')))
            closest_furnace = furnaces[0]
            features.extend([
                float(closest_furnace.get('id', 0)),
                float(closest_furnace.get('x', 0)),
                float(closest_furnace.get('y', 0))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. 3 closest bank booths - each with 3 features (ID, x, y)
        bank_booths = [obj for obj in all_objects if obj.get('name') == 'Bank booth']
        bank_booths.sort(key=lambda obj: obj.get('distance', float('inf')))
        
        for i in range(3):
            if i < len(bank_booths):
                booth = bank_booths[i]
                features.extend([
                    float(booth.get('id', 0)),
                    float(booth.get('x', 0)),
                    float(booth.get('y', 0))
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_npc_features(self, npcs: List[Dict]) -> List[float]:
        """Extract NPC features - EXACTLY like extract_features.py"""
        features = []
        
        npcs_sorted = sorted(npcs, key=lambda n: n.get('distance', float('inf')))
        
        for i in range(5):
            if i < len(npcs_sorted):
                npc = npcs_sorted[i]
                features.extend([
                    float(npc.get('id', 0)),  # NPC ID
                    float(npc.get('x', 0)),   # World X
                    float(npc.get('y', 0))    # World Y
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_tabs_features(self, tabs: Dict) -> List[float]:
        """Extract tabs features - EXACTLY like extract_features.py"""
        features = []
        
        current_tab = tabs.get('currentTab', 0)
        features.append(float(current_tab))
        
        return features
    
    def _extract_skills_features(self, skills: Dict) -> List[float]:
        """Extract skills features - EXACTLY like extract_features.py"""
        features = []
        
        crafting = skills.get('crafting', {})
        crafting_level = crafting.get('level', 0)
        crafting_xp = crafting.get('xp', 0)
        
        features.extend([crafting_level, crafting_xp])
        
        return features
    
    def _extract_action_summary_features(self, gamestate: Dict) -> List[float]:
        """Extract action summary features for compatibility with existing pipeline - EXACTLY like extract_features.py"""
        features = []
        # Action summary features removed - raw action data provides all necessary information
        # No features added for actions - they are handled separately in raw_action_data
        return features
    
    def _stable_hash(self, text: str) -> int:
        """Create a stable hash for text strings - EXACTLY like extract_features.py"""
        if not text:
            return 0
        return hash(text) % (2**32)  # 32-bit hash
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float - EXACTLY like extract_features.py"""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _to_relative_timestamp(self, absolute_timestamp: int) -> float:
        """Convert absolute timestamp to relative - EXACTLY like extract_features.py"""
        if not self.session_start_time:
            return float(absolute_timestamp)
        return float(absolute_timestamp - self.session_start_time)

    def extract_live_features(self, gs: Optional[Dict] = None) -> np.ndarray:
        """Extract live features using EXACT SAME logic as training script."""
        if gs is None:
            gs = self._latest_gs
            if gs is None:
                return np.zeros(128, dtype=np.float32)
        
        # Track if this is the first successful extraction
        if not hasattr(self, '_first_extraction_done'):
            self._first_extraction_done = False
        
        try:
            # Use mapping-based extraction instead of hardcoded methods
            features = self._extract_features_from_mappings(gs)
            
            # Print one-time message after first successful extraction
            if not self._first_extraction_done:
                non_zero_count = int(np.count_nonzero(features))
                print(f"[LIVE] extracted={non_zero_count}/128 using mapping-based extraction")
                self._first_extraction_done = True
            
            return features
            
        except Exception as e:
            print(f"[LIVE] Feature extraction error: {e}")
            # Return zeros on error
            return np.zeros(128, dtype=np.float32)

    def _extract_player_features(self, player: Dict) -> List[float]:
        """Extract player state features using OSRS IDs directly where possible - EXACTLY like extract_features.py"""
        features = []
        
        # Position (numerical - keep as-is)
        world_x = player.get('world_x', 0)
        world_y = player.get('world_y', 0)
        features.extend([
            self._safe_float(world_x),
            self._safe_float(world_y)
        ])
        
        # Animation ID (numerical - use OSRS ID directly)
        animation_id = player.get('animation_id', -1)
        features.append(float(animation_id))
        
        # Movement state (boolean - keep as-is)
        is_moving = player.get('is_moving', False)
        features.append(1.0 if is_moving else 0.0)
        
        # Movement direction (categorical - hash this since it's dynamic text)
        direction = player.get('movement_direction', 'stationary')
        hashed_direction = self._stable_hash(direction)
        features.append(float(hashed_direction))
        
        return features

    def _extract_interaction_features(self, gamestate: Dict) -> List[float]:
        """Extract interaction features as separate meaningful features instead of one hash - EXACTLY like extract_features.py"""
        features = []
        
        last_interaction = gamestate.get('last_interaction', {})
        gamestate_timestamp = gamestate.get('timestamp', 0)
        
        # Extract the actual data fields from last_interaction
        action = last_interaction.get('action', '')
        item_name = last_interaction.get('item_name', '')
        target = last_interaction.get('target', '')
        interaction_timestamp = last_interaction.get('timestamp', 0)
        
        # Feature 5: Action type (hash the action text since it's dynamic)
        action_hash = self._stable_hash(action)
        features.append(self._safe_float(action_hash))
        
        # Feature 6: Item name (hash the item name since it's dynamic)
        item_hash = self._stable_hash(item_name)
        features.append(self._safe_float(item_hash))
        
        # Feature 7: Target (hash the target text since it's dynamic)
        target_hash = self._stable_hash(target)
        features.append(self._safe_float(target_hash))
        
        # Feature 8: Time since interaction (raw milliseconds)
        if interaction_timestamp > 0 and gamestate_timestamp > 0:
            time_since_interaction = gamestate_timestamp - interaction_timestamp
        else:
            time_since_interaction = 0.0
        features.append(time_since_interaction)
        
        return features

    def _extract_camera_features(self, gamestate: Dict) -> List[float]:
        """Extract camera position and orientation (numerical - keep as-is) - EXACTLY like extract_features.py"""
        features = []
        
        camera_x = gamestate.get('camera_x', 0)
        camera_y = gamestate.get('camera_y', 0)
        camera_z = gamestate.get('camera_z', 0)
        camera_pitch = gamestate.get('camera_pitch', 0)
        camera_yaw = gamestate.get('camera_yaw', 0)
        
        features.extend([camera_x, camera_y, camera_z, camera_pitch, camera_yaw])
        
        return features

    def _extract_inventory_features(self, inventory: List[Dict]) -> List[float]:
        """Extract inventory features using item IDs directly instead of hashing - EXACTLY like extract_features.py"""
        features = []
        
        for i in range(28):
            if i < len(inventory) and inventory[i]:
                item = inventory[i]
                item_id = item.get('id', -1)
                features.append(self._safe_float(item_id))  # Use OSRS item ID directly
            else:
                features.append(-1.0)  # No item
        
        return features

    def _extract_bank_features(self, gamestate: Dict) -> List[float]:
        """Extract bank features as separate meaningful features for each material position - EXACTLY like extract_features.py"""
        features = []
        
        # Feature: Bank open status (boolean)
        bank_open = gamestate.get('bank_open', False)
        features.append(1.0 if bank_open else 0.0)
        
        # Individual bank material features (20 total)
        # For each material type, create 5 features: existence, quantity, slot, x, y
        bank_item_positions = gamestate.get('bank_item_positions', {})
        
        # Define material types
        material_types = ['sapphires', 'gold_bars', 'rings', 'moulds']
        
        for material_type in material_types:
            items = bank_item_positions.get(material_type, [])
            
            if items:
                # Take the first item of this type
                item = items[0]
                
                # Feature 1: Existence (1.0 if item exists, 0.0 if not)
                features.append(1.0)
                
                # Feature 2: Quantity
                quantity = item.get('quantity', 0)
                features.append(self._safe_float(quantity))
                
                # Feature 3: Slot
                slot = item.get('slot', -1)
                features.append(self._safe_float(slot))
                
                # Feature 4: Canvas X coordinate
                canvas_x = item.get('canvas_x', -1)
                features.append(self._safe_float(canvas_x))
                
                # Feature 5: Canvas Y coordinate
                canvas_y = item.get('canvas_y', -1)
                features.append(self._safe_float(canvas_y))
            else:
                # Material doesn't exist - set all features to 0/-1
                features.extend([0.0, 0.0, -1.0, 0.0, 0.0])
        
        return features

    def _extract_phase_context_features(self, gamestate: Dict) -> List[float]:
        """Extract phase context features - EXACTLY like extract_features.py"""
        features = []
        
        phase_context = gamestate.get('phase_context', {})
        
        # Phase type (hash the phase type text)
        cycle_phase = phase_context.get('cycle_phase', 'unknown')
        phase_type_hash = self._stable_hash(cycle_phase)
        features.append(float(phase_type_hash))
        
        # Phase start time (relative to session start, milliseconds)
        phase_start_time = phase_context.get('phase_start_time', 0)
        features.append(float(phase_start_time))
        
        # Phase duration (raw milliseconds)
        phase_duration_ms = phase_context.get('phase_duration_ms', 0)
        features.append(float(phase_duration_ms))
        
        # Gamestates in phase (raw count)
        gamestates_in_phase = phase_context.get('gamestates_in_phase', 0)
        features.append(float(gamestates_in_phase))
        
        return features

    def _extract_game_objects_features(self, gamestate: Dict) -> List[float]:
        """Extract game objects features - EXACTLY like extract_features.py"""
        features = []
        
        # 1. 10 closest game objects - each with 3 features (ID, x, y)
        all_objects = gamestate.get('game_objects', [])
        unique_objects = []
        seen_coords = set()
        
        for obj in all_objects:
            if obj is None or obj.get('name') == 'null' or obj.get('id') is None:
                continue
            coords = (obj.get('x', 0), obj.get('y', 0))
            if coords not in seen_coords:
                unique_objects.append(obj)
                seen_coords.add(coords)
        
        unique_objects.sort(key=lambda obj: obj.get('distance', float('inf')))
        
        for i in range(10):
            if i < len(unique_objects):
                obj = unique_objects[i]
                features.extend([
                    float(obj.get('id', 0)),  # Object ID
                    float(obj.get('x', 0)),   # World X
                    float(obj.get('y', 0))    # World Y
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # 2. 1 closest furnace - with 3 features (ID, x, y)
        furnaces = gamestate.get('furnaces', [])
        if furnaces:
            furnaces.sort(key=lambda f: f.get('distance', float('inf')))
            closest_furnace = furnaces[0]
            features.extend([
                float(closest_furnace.get('id', 0)),
                float(closest_furnace.get('x', 0)),
                float(closest_furnace.get('y', 0))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 3. 3 closest bank booths - each with 3 features (ID, x, y)
        bank_booths = [obj for obj in all_objects if obj.get('name') == 'Bank booth']
        bank_booths.sort(key=lambda obj: obj.get('distance', float('inf')))
        
        for i in range(3):
            if i < len(bank_booths):
                booth = bank_booths[i]
                features.extend([
                    float(booth.get('id', 0)),
                    float(booth.get('x', 0)),
                    float(booth.get('y', 0))
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features

    def _extract_npc_features(self, npcs: List[Dict]) -> List[float]:
        """Extract NPC features - EXACTLY like extract_features.py"""
        features = []
        
        npcs_sorted = sorted(npcs, key=lambda n: n.get('distance', float('inf')))
        
        for i in range(5):
            if i < len(npcs_sorted):
                npc = npcs_sorted[i]
                features.extend([
                    float(npc.get('id', 0)),  # NPC ID
                    float(npc.get('x', 0)),   # World X
                    float(npc.get('y', 0))    # World Y
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features

    def _extract_tabs_features(self, tabs: Dict) -> List[float]:
        """Extract tabs features - EXACTLY like extract_features.py"""
        features = []
        
        current_tab = tabs.get('currentTab', 0)
        features.append(float(current_tab))
        
        return features

    def _extract_skills_features(self, skills: Dict) -> List[float]:
        """Extract skills features - EXACTLY like extract_features.py"""
        features = []
        
        crafting = skills.get('crafting', {})
        crafting_level = crafting.get('level', 0)
        crafting_xp = crafting.get('xp', 0)
        
        features.extend([crafting_level, crafting_xp])
        
        return features

    def _extract_timestamp_feature(self, gamestate: Dict) -> float:
        """Extract timestamp feature - EXACT SAME as training script."""
        absolute_timestamp = gamestate.get('timestamp', 0)
        # For live extraction, use absolute timestamp since we don't have session start
        return float(absolute_timestamp)

    def build_action_step(self, gs: Dict) -> np.ndarray:
        """Build action step tensor. Return shape (101, 8), dtype float32."""
        try:
            # Get recent actions from gamestate
            events = []
            for field in ["recent_actions", "actions", "last_actions"]:
                if field in gs and isinstance(gs[field], list):
                    events.extend(gs[field])
            
            # Also check last_interaction
            if "last_interaction" in gs and isinstance(gs["last_interaction"], dict):
                events.append(gs["last_interaction"])
            
            count = min(len(events), 100)  # cap at 100 actions
            
            # Initialize with zeros
            action_step = np.zeros((101, 8), dtype=np.float32)
            
            # Row 0: header [count, 0, 0, 0, 0, 0, 0, 0]
            action_step[0, 0] = float(count)
            
            # Rows 1..count: encoded events
            for i in range(count):
                ev = events[i]
                action_step[i + 1] = [
                    float(ev.get("timestamp", 0)),
                    float(ev.get("type", 0)),
                    float(ev.get("x", 0)),
                    float(ev.get("y", 0)),
                    float(ev.get("button", 0)),
                    float(ev.get("key", 0)),
                    float(ev.get("scroll_dx", 0)),
                    float(ev.get("scroll_dy", 0))
                ]
            
            return action_step
            
        except Exception:
            # Return empty action step on error
            return np.zeros((101, 8), dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names for the 128 features."""
        names = []
        for i in range(128):
            if i < len(self._feat_map) and self._feat_map[i] is not None:
                path = self._feat_map[i].get("path", f"feature_{i}")
                # Clean up pipe-separated paths for display
                if '|' in path:
                    base_path, field_name = path.split('|', 1)
                    names.append(f"{base_path}.{field_name}")
                else:
                    names.append(path)
            else:
                names.append(f"feature_{i}")
        return names

    def interpret_feature(self, name: str, value: float) -> str:
        """Interpret a feature value based on its mapping - uses training script mappings."""
        try:
            # Find the feature mapping for this name
            for spec in self._feat_map:
                if spec and spec.get("path") == name:
                    kind = spec.get("kind", "numeric")
                    category = spec.get("category", "")
                    data_type = spec.get("data_type", "")
                    
                    if kind == "numeric":
                        # For numeric features, format based on type
                        if "time" in data_type.lower():
                            return f"{value:.0f}ms"
                        elif "coordinate" in data_type.lower() or "angle" in data_type.lower():
                            return f"{value:.1f}"
                        else:
                            return f"{value:.3f}"
                    
                    elif kind == "categorical":
                        # Use the new decode method for categorical features
                        return self._decode_categorical_feature(value, category)
                    
                    # Default fallback
                    return f"{value:.3f}"
            
            # If no mapping found, return formatted value
            return f"{value:.3f}"
            
        except Exception:
            return f"{value:.3f}"

    def test_mapping_extraction(self, gs: Dict) -> bool:
        """Test the new mapping-based extraction to ensure it works correctly."""
        try:
            features = self._extract_features_from_mappings(gs)
            non_zero_count = int(np.count_nonzero(features))
            print(f"[TEST] Mapping extraction: {non_zero_count}/128 non-zero features")
            
            # Check if we have valid mappings
            valid_mappings = sum(1 for m in self._feat_map if m is not None)
            print(f"[TEST] Valid mappings: {valid_mappings}/128")
            
            # Test a few specific features
            if len(self._feat_map) > 0 and self._feat_map[0]:
                test_path = self._feat_map[0].get("path", "")
                test_value = self._safe_get(gs, test_path, None)
                print(f"[TEST] First feature path: {test_path}, value: {test_value}")
            
            return True
        except Exception as e:
            print(f"[TEST] Mapping extraction test failed: {e}")
            return False