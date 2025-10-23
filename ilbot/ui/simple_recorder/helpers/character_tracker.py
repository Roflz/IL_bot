#!/usr/bin/env python3
"""
Character Tracking System
========================

Tracks character data consistently and saves to CSV.
Integrates with the rules checking system to update data during plan execution.

Tracked Data:
- World number
- Username
- Skill levels (all skills)
- Session time (additive with previous sessions)
- Last updated timestamp
"""

import csv
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from .runtime_utils import ipc
from ..actions import player

sys.path.insert(0, str(Path(__file__).parent.parent))

class CharacterTracker:
    """Tracks character data and saves to CSV."""
    
    def __init__(self, username: str, data_dir: str = "character_data"):
        """
        Initialize character tracker.
        
        Args:
            username: Character username
            data_dir: Directory to save CSV files
        """
        self.username = username
        
        # Use strict absolute path for CSV file
        self.csv_file = Path("D:/repos/bot_runelite_IL/ilbot/ui/simple_recorder/character_data/character_stats.csv")
        self.csv_file.parent.mkdir(exist_ok=True)
        
        # Load existing session time
        self.session_start_time = time.time()
        self.total_session_time = self._load_total_session_time()
        
        logging.info(f"[CharacterTracker] Initialized for {username}")
        logging.info(f"[CharacterTracker] CSV file: {self.csv_file}")
        logging.info(f"[CharacterTracker] Total session time: {self.total_session_time:.2f} seconds")
    
    def _load_total_session_time(self) -> float:
        """Load total session time from CSV file for this specific character."""
        if not self.csv_file.exists():
            return 0.0
        
        try:
            with open(self.csv_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                if rows:
                    # Find the row for this specific character
                    for row in rows:
                        if row.get('username') == self.username:
                            return float(row.get('total_session_time', 0.0))
        except Exception as e:
            logging.warning(f"[CharacterTracker] Error loading session time: {e}")
        
        return 0.0
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current character data."""
        try:
            # Get world number
            world_number = 0
            try:
                world_resp = ipc.get_world()
                if world_resp and world_resp.get("ok"):
                    world_number = world_resp.get("world", 0)
            except Exception:
                pass
            
            # Get skill levels
            skills = player.get_skills()
            
            # Calculate current session time
            current_session_time = time.time() - self.session_start_time
            total_session_time = self.total_session_time + current_session_time
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Prepare data
            data = {
                'username': self.username,
                'world_number': world_number,
                'timestamp': timestamp,
                'session_time': current_session_time,
                'total_session_time': total_session_time
            }
            
            # Add all skill levels
            if skills:
                for skill_name, skill_data in skills.items():
                    if isinstance(skill_data, dict):
                        level = skill_data.get('level', 1)
                        xp = skill_data.get('xp', 0)
                        data[f'{skill_name.lower()}_level'] = level
                        data[f'{skill_name.lower()}_xp'] = xp
            
            return data
            
        except Exception as e:
            logging.error(f"[CharacterTracker] Error getting current data: {e}")
            return {
                'username': self.username,
                'world_number': 0,
                'timestamp': datetime.now().isoformat(),
                'session_time': 0.0,
                'total_session_time': self.total_session_time
            }
    
    def save_data(self, data: Dict[str, Any]) -> bool:
        """Save character data to CSV, updating existing row or adding new one."""
        try:
            # Read existing data
            existing_rows = []
            if self.csv_file.exists():
                try:
                    with open(self.csv_file, 'r', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        existing_rows = list(reader)
                except Exception as e:
                    logging.warning(f"[CharacterTracker] Error reading existing CSV: {e}")
                    existing_rows = []
            
            # Find if this character already exists
            character_found = False
            for i, row in enumerate(existing_rows):
                if row.get('username') == self.username:
                    # Update existing row
                    existing_rows[i] = data
                    character_found = True
                    break
            
            # If character not found, add new row
            if not character_found:
                existing_rows.append(data)
            
            # Write all data back to file
            if existing_rows:
                fieldnames = list(existing_rows[0].keys())
                with open(self.csv_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
            else:
                logging.warning(f"[CharacterTracker] No data to save for {self.username}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"[CharacterTracker] Error saving data: {e}")
            return False
    
    def update_tracking(self) -> bool:
        """Update tracking data and save to CSV."""
        try:
            data = self.get_current_data()
            success = self.save_data(data)
            
            if success:
                logging.debug(f"[CharacterTracker] Updated tracking for {self.username}")
                logging.debug(f"[CharacterTracker] Session time: {data['session_time']:.2f}s, Total: {data['total_session_time']:.2f}s")
            
            return success
            
        except Exception as e:
            logging.error(f"[CharacterTracker] Error updating tracking: {e}")
            return False
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest tracking data from CSV for this specific character."""
        if not self.csv_file.exists():
            return None
        
        try:
            with open(self.csv_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                if rows:
                    # Find the row for this specific character
                    for row in rows:
                        if row.get('username') == self.username:
                            return row
        except Exception as e:
            logging.error(f"[CharacterTracker] Error reading latest data: {e}")
        
        return None
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary data."""
        data = self.get_current_data()
        return {
            'username': data['username'],
            'current_session_time': data['session_time'],
            'total_session_time': data['total_session_time'],
            'world_number': data['world_number'],
            'timestamp': data['timestamp']
        }


# Global tracker instances (one per character)
_character_trackers = {}


def get_character_tracker(username: str) -> CharacterTracker:
    """Get or create character tracker for username."""
    if username not in _character_trackers:
        _character_trackers[username] = CharacterTracker(username)
    return _character_trackers[username]


def update_character_tracking(username: str) -> bool:
    """Update tracking for a specific character."""
    tracker = get_character_tracker(username)
    return tracker.update_tracking()


def get_character_summary(username: str) -> Optional[Dict[str, Any]]:
    """Get character summary data."""
    tracker = get_character_tracker(username)
    return tracker.get_session_summary()
