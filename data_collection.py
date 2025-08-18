#!/usr/bin/env python3
"""
Data Collection Module for OSRS Bot Training

This module provides the DataCollector class for basic data collection
during bot automation sessions.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional


class DataCollector:
    """
    Basic data collector for OSRS bot automation.
    Provides simple logging and data collection functionality.
    """
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.actions_file = os.path.join(output_dir, "actions.csv")
        self.gamestate_file = os.path.join(output_dir, "runelite_gamestate.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize actions CSV if it doesn't exist
        self._init_actions_csv()
    
    def _init_actions_csv(self):
        """Initialize the actions CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.actions_file):
            headers = [
                'timestamp', 'action_type', 'target_x', 'target_y', 
                'target_name', 'success', 'notes'
            ]
            with open(self.actions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_action(self, action_type: str, target_x: int = None, target_y: int = None,
                   target_name: str = None, success: bool = True, notes: str = None):
        """
        Log a simple action to the CSV file.
        
        Args:
            action_type: Type of action (click, move, etc.)
            target_x: X coordinate of target
            target_y: Y coordinate of target
            target_name: Name of target object
            success: Whether the action was successful
            notes: Additional notes about the action
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'target_x': target_x,
            'target_y': target_y,
            'target_name': target_name,
            'success': success,
            'notes': notes
        }
        
        # Write to CSV immediately
        self._write_action_to_csv(action)
    
    def _write_action_to_csv(self, action: Dict):
        """Write a single action to the CSV file."""
        try:
            with open(self.actions_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    action['timestamp'],
                    action['action_type'],
                    action['target_x'],
                    action['target_y'],
                    action['target_name'],
                    action['success'],
                    action['notes']
                ])
        except Exception as e:
            print(f"❌ Error writing action to CSV: {e}")
    
    def log_click(self, x: int, y: int, target_name: str = None, success: bool = True):
        """Log a mouse click action."""
        self.log_action(
            action_type="click",
            target_x=x,
            target_y=y,
            target_name=target_name,
            success=success
        )
    
    def log_move(self, x: int, y: int, target_name: str = None, success: bool = True):
        """Log a movement action."""
        self.log_action(
            action_type="move",
            target_x=x,
            target_y=y,
            target_name=target_name,
            success=success
        )
    
    def log_banking(self, action: str, success: bool = True, notes: str = None):
        """Log a banking action."""
        self.log_action(
            action_type=f"bank_{action}",
            success=success,
            notes=notes
        )
    
    def get_actions_summary(self) -> Dict:
        """Get a summary of logged actions."""
        try:
            if os.path.exists(self.actions_file):
                with open(self.actions_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    actions = list(reader)
                
                if not actions:
                    return {'total_actions': 0}
                
                action_types = {}
                for action in actions:
                    action_type = action.get('action_type', 'unknown')
                    action_types[action_type] = action_types.get(action_type, 0) + 1
                
                return {
                    'total_actions': len(actions),
                    'action_types': action_types,
                    'first_action': actions[0].get('timestamp'),
                    'last_action': actions[-1].get('timestamp')
                }
            else:
                return {'total_actions': 0}
        except Exception as e:
            print(f"❌ Error reading actions summary: {e}")
            return {'total_actions': 0, 'error': str(e)}
