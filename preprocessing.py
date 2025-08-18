#!/usr/bin/env python3
"""
Data Preprocessing Module for OSRS Bot Training

This module provides the DataPreprocessor class for basic data preprocessing
during bot automation sessions.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any


class DataPreprocessor:
    """
    Basic data preprocessor for OSRS bot automation.
    Provides simple data loading and validation functionality.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.gamestate_file = os.path.join(data_dir, "runelite_gamestate.json")
        self.actions_file = os.path.join(data_dir, "actions.csv")
    
    def load_gamestate_data(self, file_path: str = None) -> Dict[str, Any]:
        """Load gamestate data from a JSON file."""
        if file_path is None:
            file_path = self.gamestate_file
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️  Gamestate file not found: {file_path}")
                return {}
        except Exception as e:
            print(f"❌ Error loading gamestate data: {e}")
            return {}
    
    def validate_gamestate(self, gamestate: Dict[str, Any]) -> bool:
        """Validate that a gamestate contains required fields."""
        if not gamestate:
            return False
        
        required_fields = ['player', 'inventory', 'game_objects']
        
        for field in required_fields:
            if field not in gamestate:
                print(f"⚠️  Missing required field: {field}")
                return False
        
        return True
    
    def get_player_position(self, gamestate: Dict[str, Any]) -> Optional[tuple]:
        """Extract player position from gamestate."""
        if not gamestate or 'player' not in gamestate:
            return None
        
        player = gamestate['player']
        x = player.get('x')
        y = player.get('y')
        
        if x is not None and y is not None:
            return (x, y)
        
        return None
    
    def get_inventory_status(self, gamestate: Dict[str, Any]) -> Dict[str, Any]:
        """Get inventory status from gamestate."""
        if not gamestate or 'inventory' not in gamestate:
            return {'is_full': False, 'item_count': 0, 'items': []}
        
        inventory = gamestate['inventory']
        valid_items = [item for item in inventory if item.get('id') != -1]
        
        return {
            'is_full': len(valid_items) >= 28,
            'item_count': len(valid_items),
            'items': valid_items
        }
    
    def find_objects_by_name(self, gamestate: Dict[str, Any], object_name: str) -> List[Dict]:
        """Find game objects by name."""
        if not gamestate or 'game_objects' not in gamestate:
            return []
        
        game_objects = gamestate['game_objects']
        matching_objects = []
        
        for obj in game_objects:
            if obj.get('name', '').lower() == object_name.lower():
                matching_objects.append(obj)
        
        return matching_objects
    
    def find_objects_by_id(self, gamestate: Dict[str, Any], object_id: int) -> List[Dict]:
        """Find game objects by ID."""
        if not gamestate or 'game_objects' not in gamestate:
            return []
        
        game_objects = gamestate['game_objects']
        matching_objects = []
        
        for obj in game_objects:
            if obj.get('id') == object_id:
                matching_objects.append(obj)
        
        return matching_objects
    
    def get_bank_status(self, gamestate: Dict[str, Any]) -> Dict[str, Any]:
        """Get bank status from gamestate."""
        if not gamestate or 'bank' not in gamestate:
            return {'is_open': False, 'items': []}
        
        bank = gamestate['bank']
        is_open = gamestate.get('bank_open', False)
        
        return {
            'is_open': is_open,
            'items': bank
        }
    
    def calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate Euclidean distance between two positions."""
        if not pos1 or not pos2:
            return float('inf')
        
        x1, y1 = pos1
        x2, y2 = pos2
        
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def find_nearest_object(self, gamestate: Dict[str, Any], player_pos: tuple, 
                           object_name: str = None, object_id: int = None) -> Optional[Dict]:
        """Find the nearest object to the player."""
        if not gamestate or 'game_objects' not in gamestate or not player_pos:
            return None
        
        game_objects = gamestate['game_objects']
        nearest_object = None
        nearest_distance = float('inf')
        
        for obj in game_objects:
            if 'x' in obj and 'y' in obj:
                obj_pos = (obj['x'], obj['y'])
                
                # Filter by name if specified
                if object_name and obj.get('name', '').lower() != object_name.lower():
                    continue
                
                # Filter by ID if specified
                if object_id and obj.get('id') != object_id:
                    continue
                
                distance = self.calculate_distance(player_pos, obj_pos)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_object = obj
        
        return nearest_object
