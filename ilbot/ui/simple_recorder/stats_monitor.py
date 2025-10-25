#!/usr/bin/env python3

import time
import threading
import logging
import random
from pathlib import Path
import csv
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .helpers.ipc import IPCClient


class StatsMonitor:
    """Background thread that monitors character stats and updates CSV file."""
    
    def __init__(self, port: int, username: str, csv_path: Optional[str] = None,
                 rule_params: Optional[Dict[str, Any]] = None, on_csv_update: Optional[Callable[[str], None]] = None,
                 on_username_changed: Optional[Callable[[str, str], None]] = None):
        self.port = port
        self.username = username
        # Use absolute path
        if csv_path is None:
            self.csv_path = Path(__file__).parent / "character_data" / "character_stats.csv"
        else:
            self.csv_path = Path(csv_path)
        self.running = False
        self.monitor_thread = None
        self.is_logged_in = False
        
        # Rule checking parameters (optional)
        # Can be passed directly or loaded from file
        self.rule_params = rule_params or {}
        self.triggered_rule = None
        # Use absolute path for rule check file
        self.rule_check_file = Path(__file__).parent / "character_data" / f"triggered_rule_{username}.txt"
        
        # Callback when CSV is updated (for GUI refresh)
        self.on_csv_update = on_csv_update
        # Callback when username changes (for GUI to update references)
        self.on_username_changed = on_username_changed
        
        # Create IPC client for this monitor (ONLY ONCE in __init__)
        self.ipc = IPCClient(host="127.0.0.1", port=self.port, timeout_s=2.0)
        
        # Stats tracking
        self.current_stats = {}
        self.last_bank_state = None
        self.last_inventory_state = None
        self.last_equipment_state = None
        self.last_skills_state = None
        self.last_world_number = None
        
        # Logged in time tracking
        self.total_logged_in_time = 0.0  # Cumulative logged-in time in seconds
        self.last_csv_timestamp = None  # Last timestamp from CSV
        self.is_first_update_after_login = True  # Flag to skip time on first update after login
        self._load_logged_in_time()
        
        # Tracked items for total amounts (bank + inventory)
        self.tracked_items = {
            'coins': 'Coins',
            'cowhides': 'Cowhides',
            'logs': 'Logs',
            'leather': 'Leather',
            'bow_string': 'Bow String',
            'iron_bar': 'Iron Bar',
            'steel_bar': 'Steel Bar',
            'coal': 'Coal',
            'iron_ore': 'Iron Ore',
            'raw_fish': 'Raw Fish',
            'cooked_fish': 'Cooked Fish',
            'bronze_bar': 'Bronze Bar',
            'silver_bar': 'Silver Bar',
            'gold_bar': 'Gold Bar',
            'mithril_bar': 'Mithril Bar',
            'adamant_bar': 'Adamant Bar',
            'rune_bar': 'Rune Bar',
            'bronze_ore': 'Bronze Ore',
            'silver_ore': 'Silver Ore',
            'gold_ore': 'Gold Ore',
            'mithril_ore': 'Mithril Ore',
            'adamant_ore': 'Adamant Ore',
            'rune_ore': 'Rune Ore',
            'willow_logs': 'Willow Logs',
            'oak_logs': 'Oak Logs',
            'maple_logs': 'Maple Logs',
            'yew_logs': 'Yew Logs',
            'magic_logs': 'Magic Logs',
            'redwood_logs': 'Redwood Logs'
        }
        
        # Skills to track
        self.skills = [
            'attack', 'defence', 'strength', 'hitpoints', 'ranged', 'prayer', 'magic',
            'cooking', 'woodcutting', 'fletching', 'fishing', 'firemaking', 'crafting',
            'smithing', 'mining', 'herblore', 'agility', 'thieving', 'slayer',
            'farming', 'runecraft', 'hunter', 'construction'
        ]
        
        # Try to load rule params from file if not provided
        self._load_rule_params_from_file()
        
        logging.info(f"[STATS_MONITOR] Initialized for {self.username} on port {self.port}")
        if self.rule_params:
            logging.info(f"[STATS_MONITOR] Rule params loaded: {self.rule_params}")
    
    def _load_rule_params_from_file(self):
        """Load rule parameters from JSON file if it exists."""
        rule_params_file = Path(__file__).parent / "character_data" / f"rule_params_{self.username}.json"
        if rule_params_file.exists():
            try:
                import json
                with open(rule_params_file, 'r', encoding='utf-8') as f:
                    new_params = json.load(f)
                # Only update if params actually changed
                if new_params != self.rule_params:
                    self.rule_params = new_params
                    logging.debug(f"[STATS_MONITOR] Loaded/updated rule parameters from {rule_params_file}")
            except Exception as e:
                logging.debug(f"[STATS_MONITOR] Error loading rule parameters: {e}")
        elif self.rule_params:
            # File was deleted, clear rule params
            self.rule_params = {}
            if self.triggered_rule:
                self.triggered_rule = None
                if self.rule_check_file.exists():
                    self.rule_check_file.unlink()
    
    def start(self):
        """Start the background monitoring thread."""
        if self.running:
            logging.debug(f"[STATS_MONITOR] Already running for {self.username}")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info(f"[STATS_MONITOR] Started monitoring for {self.username}")
        logging.debug(f"[STATS_MONITOR] Monitor thread started for {self.username}, running={self.running}")
    
    def stop(self):
        """Stop the background monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info(f"[STATS_MONITOR] Stopped monitoring for {self.username}")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background."""
        last_rule_file_check = 0
        rule_file_check_interval = 30  # Check for rule params file every 5 seconds
        cycle_counter = 0  # Track which checks to perform this cycle (0=inventory, 1=equipment, 2=bank)
        last_world_check = 0
        world_check_interval = 10  # Check world every 10 seconds
        
        while self.running:
            try:
                current_time = time.time()
                
                # Debug: Log every 10 seconds that we're still running
                if int(current_time) % 10 == 0:
                    logging.debug(f"[STATS_MONITOR] Monitor loop running for {self.username}, rule_params: {self.rule_params}")
                
                # Check for changes and update stats (with staggered checks based on cycle)
                check_world = (current_time - last_world_check) >= world_check_interval
                if check_world:
                    last_world_check = current_time
                
                self._check_and_update_stats(cycle_counter, check_world)
                cycle_counter = (cycle_counter + 1) % 3  # Cycle through 0, 1, 2 every 3 seconds
                
                # Periodically reload rule params from file (in case run_rj_loop.py updates it)
                if current_time - last_rule_file_check > rule_file_check_interval:
                    self._load_rule_params_from_file()
                    last_rule_file_check = current_time
                
                # Check rules if parameters are provided (reads from CSV only, no data collection)
                if self.rule_params:
                    logging.debug(f"[STATS_MONITOR] Checking rules for {self.username}, rule_params: {self.rule_params}")
                    self._check_rules()
                else:
                    logging.debug(f"[STATS_MONITOR] No rule_params for {self.username}, skipping rule check")
                
                # Sleep for 1 second between checks (we stagger what we check each cycle)
                time.sleep(1.0)
                
            except Exception as e:
                logging.error(f"[STATS_MONITOR] Error in monitor loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _check_and_update_stats(self, cycle_counter: int = 0, check_world: bool = False):
        """Check for changes and update stats if needed. Rotates checks based on cycle_counter."""
        try:
            # Always fetch player data (needed for connection check, skills, username)
            player_data = None
            try:
                player_data = self.ipc.get_player()
            except Exception as e:
                logging.debug(f"[STATS_MONITOR] Error fetching player data: {e}")
                self.is_logged_in = False
                self.is_first_update_after_login = True
                return
            
            # Check if logged in
            if not player_data or not player_data.get("ok"):
                self.is_logged_in = False
                self.is_first_update_after_login = True
                return
            
            # Extract player info once
            player_info = player_data.get("player", {})
            if not player_info.get("name"):
                self.is_logged_in = False
                self.is_first_update_after_login = True
                return
            
            self.is_logged_in = True
            stats_changed = False
            
            # Always check skills (using player_data we already fetched, no extra IPC call)
            # Update skills stats always (not just on change) so rules can check current levels
            current_skills = player_info.get("skills", {})
            if self.last_skills_state != current_skills:
                self.last_skills_state = current_skills
                stats_changed = True
            
            # Always update skills stats (even if unchanged) for rule checking
            for skill in self.skills:
                if skill in current_skills:
                    skill_data = current_skills[skill]
                    self.current_stats[f'{skill}_level'] = skill_data.get('level', 0)
                    self.current_stats[f'{skill}_xp'] = skill_data.get('xp', 0)
            
            # Rotate other checks based on cycle_counter (0=inventory, 1=equipment, 2=bank)
            # These cycle every 3 seconds (1 second each)
            # Always update stats even if unchanged, so rules can check current values
            if cycle_counter == 0:
                # Check inventory changes and update stats (reuse inventory_data)
                had_change, inventory_data = self._check_inventory_changes()
                if inventory_data:
                    self._update_inventory_stats(inventory_data)  # Pass data to avoid duplicate IPC call
                if had_change:
                    stats_changed = True
            elif cycle_counter == 1:
                # Check equipment changes and update stats (reuse equipment_data)
                had_change, equipment_data = self._check_equipment_changes()
                if equipment_data:
                    self._update_equipment_stats(equipment_data)  # Pass data to avoid duplicate IPC call
                if had_change:
                    stats_changed = True
            elif cycle_counter == 2:
                # Check bank changes (only if bank is open)
                try:
                    bank_data = self.ipc.get_bank()
                    if bank_data and bank_data.get("ok"):
                        # Bank is open, check if it changed
                        current_bank = bank_data.get("items")
                        had_change = (self.last_bank_state != current_bank)
                        if had_change:
                            self.last_bank_state = current_bank
                            stats_changed = True
                        self._update_bank_stats(bank_data)  # Always update for rule checking
                except Exception:
                    pass  # Bank check failed, skip this cycle
            
            # Check world separately (every 10 seconds)
            if check_world:
                try:
                    world_data = self.ipc.get_world()
                    if world_data and world_data.get("ok"):
                        current_world = world_data.get("world", 0)
                        if self.last_world_number != current_world:
                            self.current_stats['world_number'] = current_world
                            self.last_world_number = current_world
                            stats_changed = True
                except Exception:
                    pass  # World check is non-critical, skip if it fails
            
            # Always ensure current_stats has username for rule checking
            if player_data and player_data.get("ok"):
                player_info = player_data.get("player", {})
                self.current_stats['username'] = player_info.get("name", self.username)
            
            # Always update CSV if we have stats data (on any change)
            if stats_changed:
                # Check if we need to update username for unnamed characters (only if we have player_data)
                if player_data and player_data.get("ok"):
                    self._check_and_update_username()
                self._update_csv()
                
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error checking stats: {e}")
    
    def _is_game_connected(self) -> bool:
        """Check if we can connect to the game via IPC - DEPRECATED: now done inline."""
        # This method is deprecated - connection check is now done in _check_and_update_stats
        # to avoid duplicate IPC calls. Keeping for compatibility.
        return self.is_logged_in if hasattr(self, 'is_logged_in') else False
    
    def _check_bank_changes(self) -> bool:
        """Check if bank state has changed."""
        try:
            bank_data = self.ipc.get_bank()
            
            if not bank_data or not bank_data.get("ok"):
                return False
            
            current_bank = bank_data.get("items")
            
            # Compare with last known state
            if self.last_bank_state != current_bank:
                self.last_bank_state = current_bank
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error checking bank changes: {e}")
            return False
    
    def _check_inventory_changes(self) -> bool:
        """Check if inventory state has changed. Returns (changed, inventory_data) tuple."""
        try:
            inventory_data = self.ipc.get_inventory()
            
            if not inventory_data or not inventory_data.get("ok"):
                return False, None
            
            current_inventory = inventory_data.get("slots")
            
            # Compare with last known state
            if self.last_inventory_state != current_inventory:
                self.last_inventory_state = current_inventory
                return True, inventory_data
            
            return False, inventory_data
            
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error checking inventory changes: {e}")
            return False, None
    
    def _check_equipment_changes(self) -> tuple:
        """Check if equipment state has changed. Returns (changed, equipment_data) tuple."""
        try:
            equipment_data = self.ipc.get_equipment()
            
            if not equipment_data or not equipment_data.get("ok"):
                return False, None
            
            current_equipment = equipment_data.get("equipment")
            
            # Compare with last known state
            if self.last_equipment_state != current_equipment:
                self.last_equipment_state = current_equipment
                return True, equipment_data
            
            return False, equipment_data
            
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error checking equipment changes: {e}")
            return False, None
    
    def _check_and_update_username(self):
        """Check if username needs to be updated for unnamed characters and rename credentials file."""
        try:
            # Only check if current username contains 'unnamed_character'
            if 'unnamed_character' not in self.username or 'detected' not in self.username:
                return
            
            # Get actual username from IPC
            player_data = self.ipc.get_player()
            if not player_data or not player_data.get("ok"):
                return
            
            player_info = player_data.get("player", {})
            actual_username = player_info.get("name", "")
            
            if not actual_username or actual_username == self.username:
                return
            
            logging.info(f"[STATS_MONITOR] Detected username change: {self.username} -> {actual_username}")
            
            # Rename credentials file
            credentials_dir = Path("D:/repos/bot_runelite_IL/credentials")
            old_cred_file = credentials_dir / f"{self.username}.properties"
            new_cred_file = credentials_dir / f"{actual_username}.properties"
            
            if old_cred_file.exists() and not new_cred_file.exists():
                try:
                    old_cred_file.rename(new_cred_file)
                    logging.info(f"[STATS_MONITOR] Renamed credentials file: {old_cred_file.name} -> {new_cred_file.name}")
                except Exception as e:
                    logging.error(f"[STATS_MONITOR] Error renaming credentials file: {e}")
            
            # Update username
            old_username = self.username
            self.username = actual_username
            
            # Update CSV path with new username (CSV uses shared file, but update references)
            # Update rule check file path
            self.rule_check_file = Path(__file__).parent / "character_data" / f"triggered_rule_{self.username}.txt"
            
            # Notify GUI of username change
            if self.on_username_changed:
                try:
                    self.on_username_changed(old_username, self.username)
                except Exception as e:
                    logging.warning(f"[STATS_MONITOR] Error in username changed callback: {e}")
            
            logging.info(f"[STATS_MONITOR] Updated username from {old_username} to {self.username}")
            
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error checking/updating username: {e}")
    
    def _check_world_changes(self) -> bool:
        """Check if world number has changed - DEPRECATED: now done in _check_and_update_stats."""
        # This method is kept for compatibility but world checking is now done inline
        # to reduce IPC calls
        return False
    
    def _load_logged_in_time(self):
        """Load cumulative logged-in time and last timestamp from CSV file."""
        try:
            if not self.csv_path.exists():
                return
            
            import csv
            from datetime import datetime
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                # Find the most recent entry for this username
                for row in reader:
                    if row.get('username') == self.username:
                        logged_in_str = row.get('logged_in_time', '0')
                        try:
                            self.total_logged_in_time = float(logged_in_str) if logged_in_str else 0.0
                        except (ValueError, TypeError):
                            self.total_logged_in_time = 0.0
                        
                        # Get last timestamp
                        timestamp_str = row.get('timestamp', '')
                        if timestamp_str:
                            try:
                                self.last_csv_timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                            except (ValueError, TypeError):
                                self.last_csv_timestamp = None
        except Exception as e:
            logging.debug(f"[STATS_MONITOR] Error loading logged-in time: {e}")
            self.total_logged_in_time = 0.0
    
    def _check_skills_changes(self) -> bool:
        """Check if skills state has changed - DEPRECATED: now done in _check_and_update_stats."""
        # This method is kept for compatibility but skills checking is now done inline
        # to reduce IPC calls
        return False
    
    def _update_bank_stats(self, bank_data=None):
        """Update stats from bank data."""
        try:
            # Use provided bank_data if available, otherwise fetch it
            if bank_data is None:
                bank_data = self.ipc.get_bank()
            
            if not bank_data or not bank_data.get("ok"):
                return
            
            bank_items = bank_data.get("items", [])
            
            # Count total items in bank for each tracked item
            for item_key, display_name in self.tracked_items.items():
                bank_count = 0
                
                # Count in bank
                if bank_items:
                    for item in bank_items:
                        if not item:
                            continue
                        item_name = item.get('name', '').lower().replace(' ', '_')
                        if item_name == item_key:
                            bank_count += item.get('quantity', 0)
                
                # Store bank count (inventory count will be added in _update_inventory_stats)
                self.current_stats[f'{item_key}_bank'] = bank_count
                
            logging.debug(f"[STATS_MONITOR] Updated bank stats for {self.username}")
                
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error updating bank stats: {e}")
    
    def _update_inventory_stats(self, inventory_data=None):
        """Update stats from inventory data."""
        try:
            # Use provided inventory_data if available, otherwise fetch it
            if inventory_data is None:
                inventory_data = self.ipc.get_inventory()
            
            if not inventory_data or not inventory_data.get("ok"):
                return
            
            inventory_items = inventory_data.get("slots", [])
            
            # Clear all previous inventory stats (to remove items no longer in inventory)
            # Clear both tracked_items_inventory and generic inventory_ keys
            keys_to_remove = [key for key in self.current_stats.keys() if key.startswith('inventory_')]
            for key in keys_to_remove:
                del self.current_stats[key]
            
            # Track ALL items in inventory (not just tracked_items)
            # This creates keys like 'inventory_coins', 'inventory_sword', etc.
            # Accumulate quantities for items that appear in multiple slots
            inventory_counts = {}
            if inventory_items:
                for item in inventory_items:
                    if not item:
                        continue
                    
                    item_name = item.get('itemName', '')
                    if not item_name:
                        continue
                    
                    # Try multiple possible quantity field names
                    quantity = item.get('quantity', item.get('qty', item.get('count', 0)))
                    # Ensure it's an integer
                    try:
                        quantity = int(quantity)
                    except (ValueError, TypeError):
                        quantity = 0
                    
                    if quantity <= 0:
                        continue
                    
                    # Normalize item name for key
                    item_key = item_name.lower().replace(' ', '_')
                    inventory_key = f'inventory_{item_key}'
                    
                    # Accumulate quantity (in case same item appears in multiple slots)
                    inventory_counts[inventory_key] = inventory_counts.get(inventory_key, 0) + quantity
            
            # Store accumulated counts
            for inventory_key, total_quantity in inventory_counts.items():
                self.current_stats[inventory_key] = total_quantity
            
            # Calculate totals (bank + inventory) for tracked_items (for key items totals)
            for item_key, display_name in self.tracked_items.items():
                # Get inventory count from the generic inventory_ keys
                inventory_key = f'inventory_{item_key}'
                inventory_count = self.current_stats.get(inventory_key, 0)

                # Calculate total (bank + inventory) for tracked items
                bank_count = self.current_stats.get(f'{item_key}_bank', 0)
                total_count = bank_count + inventory_count
                self.current_stats[item_key] = total_count
                
            logging.debug(f"[STATS_MONITOR] Updated inventory stats for {self.username}")
                
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error updating inventory stats: {e}")
    
    def _update_equipment_stats(self, equipment_data=None):
        """Update stats from equipment data."""
        try:
            # Use provided equipment_data if available, otherwise fetch it
            if equipment_data is None:
                equipment_data = self.ipc.get_equipment()
            
            if not equipment_data or not equipment_data.get("ok"):
                return
            
            equipment_items = equipment_data.get("equipment")
            
            # Track equipped items
            if equipment_items:
                for slot, item in equipment_items.items():
                    if item and item.get('name'):
                        self.current_stats[f'equipped_{slot}'] = item['name']
                        
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error updating equipment stats: {e}")
    
    def _update_skills_stats(self):
        """Update stats from skills data - DEPRECATED: now done inline in _check_and_update_stats."""
        # This method is kept for compatibility but skills updating is now done inline
        # to avoid duplicate IPC calls
        pass
    
    def _update_csv(self):
        """Update the CSV file with current stats (overwrites row for this username)."""
        try:
            # Calculate logged-in time before writing CSV
            current_time = time.time()
            
            # Only accumulate time if logged in AND not first update after login
            if not self.is_first_update_after_login:
                if self.last_csv_timestamp is not None:
                    # Calculate time difference since last CSV update
                    time_diff = current_time - self.last_csv_timestamp
                    if time_diff > 0:  # Only add positive time
                        self.total_logged_in_time += time_diff
            
            # Reset first update flag after processing
            if self.is_first_update_after_login:
                self.is_first_update_after_login = False
            
            # Ensure CSV directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare row data
            row_data = {
                'username': self.username,
                'world_number': self.current_stats.get('world_number', ''),
                'timestamp': datetime.now().isoformat(),
                'logged_in_time': self.total_logged_in_time
            }
            
            # Add skills data
            for skill in self.skills:
                row_data[f'{skill}_level'] = self.current_stats.get(f'{skill}_level', 0)
                row_data[f'{skill}_xp'] = self.current_stats.get(f'{skill}_xp', 0)
            
            # Add tracked items data (for bank totals and key items)
            for item_key in self.tracked_items.keys():
                row_data[item_key] = self.current_stats.get(item_key, 0)
                row_data[f'{item_key}_bank'] = self.current_stats.get(f'{item_key}_bank', 0)
                row_data[f'{item_key}_inventory'] = self.current_stats.get(f'{item_key}_inventory', 0)
            
            # Add ALL inventory items (everything currently in inventory)
            for key, value in self.current_stats.items():
                if key.startswith('inventory_') and key not in [f'{item_key}_inventory' for item_key in self.tracked_items.keys()]:
                    row_data[key] = value
            
            # Add equipment data
            for slot in ['helmet', 'cape', 'amulet', 'weapon', 'body', 'shield', 'legs', 'gloves', 'boots', 'ring']:
                row_data[f'equipped_{slot}'] = self.current_stats.get(f'equipped_{slot}', '')
            
            # Read existing CSV and update row for this username
            # Note: Multiple instances may write simultaneously - retry on failure
            max_retries = 3
            retry_count = 0
            written = False
            
            while retry_count < max_retries and not written:
                try:
                    existing_rows = []
                    fieldnames = list(row_data.keys())
                    username_found = False
                    
                    if self.csv_path.exists():
                        try:
                            with open(self.csv_path, 'r', newline='', encoding='utf-8') as file:
                                reader = csv.DictReader(file)
                                # Get fieldnames from file (may have additional columns from other characters)
                                file_fieldnames = reader.fieldnames or []
                                # Merge fieldnames to include all possible columns
                                # Use dict.fromkeys to preserve order: file_fieldnames first (existing), then new ones
                                all_fieldnames = list(dict.fromkeys(file_fieldnames + fieldnames))
                                
                                for row in reader:
                                    if row.get('username') == self.username:
                                        # Update existing row for this username
                                        # Create a fresh row with only current data
                                        updated_row = {}
                                        # First, copy all non-inventory columns from existing row (preserve other data)
                                        for key, value in row.items():
                                            if not key.startswith('inventory_'):
                                                updated_row[key] = value
                                        # Then update with new row_data (which includes current inventory)
                                        updated_row.update(row_data)
                                        existing_rows.append(updated_row)
                                        username_found = True
                                    else:
                                        # Keep other users' rows as-is
                                        # DictWriter will fill missing columns with empty strings if needed
                                        existing_rows.append(row)
                        except Exception as e:
                            logging.warning(f"[STATS_MONITOR] Error reading CSV, creating new file: {e}")
                            all_fieldnames = fieldnames
                    else:
                        all_fieldnames = fieldnames
                    
                    # If username not found, add new row
                    if not username_found:
                        existing_rows.append(row_data)
                    
                    # Write all rows back to CSV
                    # Use unique temp file name per instance to avoid conflicts
                    # Include username and timestamp to make it unique
                    temp_path = self.csv_path.parent / f"character_stats_{self.username}_{int(time.time() * 1000000)}_{random.randint(0, 999999)}.tmp"
                    try:
                        with open(temp_path, 'x', newline='', encoding='utf-8') as file:
                            writer = csv.DictWriter(file, fieldnames=all_fieldnames)
                            writer.writeheader()
                            # Ensure all rows have all fieldnames (fill missing with empty string)
                            for row in existing_rows:
                                complete_row = {}
                                for fn in all_fieldnames:
                                    if fn.startswith('inventory_') and row.get('username') == self.username:
                                        # For inventory columns on our row, use row_data (current inventory)
                                        # If not in row_data, set to empty string (clears old items)
                                        complete_row[fn] = row_data.get(fn, '')
                                    else:
                                        # For all other columns, use existing value or empty
                                        complete_row[fn] = row.get(fn, '')
                                writer.writerow(complete_row)
                        
                        # Atomic rename (works on Windows too)
                        temp_path.replace(self.csv_path)
                        written = True
                    except FileExistsError:
                        # Temp file collision (shouldn't happen with unique names, but handle it)
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(0.1 * retry_count)  # Exponential backoff
                            continue
                        raise
                    finally:
                        # Clean up temp file if it still exists (shouldn't normally after successful rename)
                        if temp_path.exists() and temp_path != self.csv_path:
                            try:
                                temp_path.unlink()
                            except Exception:
                                pass  # Ignore cleanup errors
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"[STATS_MONITOR] Failed to update CSV after {max_retries} retries: {e}")
                        raise
                    time.sleep(0.1 * retry_count)  # Exponential backoff
            
            # Update last CSV timestamp after writing (for next time calculation)
            self.last_csv_timestamp = current_time
            
            logging.debug(f"[STATS_MONITOR] Updated CSV for {self.username}")
            
            # Notify GUI to update stats display
            if self.on_csv_update:
                try:
                    self.on_csv_update(self.username)
                except Exception as e:
                    logging.warning(f"[STATS_MONITOR] Error in CSV update callback: {e}")
            
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error updating CSV: {e}")
    
    def _check_rules(self):
        """Check rules by reading from CSV file only - simple check every loop, no data collection."""
        logging.debug(f"[STATS_MONITOR] _check_rules called for {self.username}")
        
        # Check if rule_params exists and has actual rules set
        if not self.rule_params:
            logging.debug(f"[STATS_MONITOR] No rule_params for {self.username}")
            return
        
        # Check if any rule parameters are actually set
        has_rule = (
            self.rule_params.get("max_minutes", 0) > 0 or
            (self.rule_params.get("skill_name") and self.rule_params.get("skill_level", 0) > 0) or
            self.rule_params.get("total_level", 0) > 0 or
            (self.rule_params.get("item_name") and self.rule_params.get("item_quantity", 0) > 0)
        )
        
        if not has_rule:
            logging.debug(f"[STATS_MONITOR] No actual rules set for {self.username}, rule_params: {self.rule_params}")
            return  # No rules actually set
        
        try:
            from .helpers.rules import check_rules_from_stats
            from datetime import datetime
            import csv
            
            # Read stats from CSV file for this username ONLY
            if not self.csv_path.exists():
                logging.debug(f"[STATS_MONITOR] CSV file not found: {self.csv_path}")
                return
            
            stats_from_csv = {}
            # Retry reading CSV in case another instance is writing to it
            max_retries = 3
            for retry in range(max_retries):
                try:
                    with open(self.csv_path, 'r', newline='', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if row.get('username') == self.username:
                                # Copy all values from CSV row to stats dict
                                for key, value in row.items():
                                    if key != 'username':  # Skip username field
                                        stats_from_csv[key] = value
                                break
                    break  # Success, exit retry loop
                except (IOError, OSError) as e:
                    if retry < max_retries - 1:
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                    else:
                        logging.warning(f"[STATS_MONITOR] Failed to read CSV after {max_retries} retries: {e}")
                        return
            
            if not stats_from_csv:
                logging.debug(f"[STATS_MONITOR] No stats found in CSV for username: {self.username}")
                return  # No stats found in CSV for this user
            
            logging.debug(f"[STATS_MONITOR] Found stats in CSV for {self.username}: {len(stats_from_csv)} fields")
            
            # Get rule parameters
            start_time = self.rule_params.get("start_time")
            if isinstance(start_time, str):
                # Parse ISO format string
                start_time = datetime.fromisoformat(start_time)
            elif not isinstance(start_time, datetime):
                # Fallback to current time if invalid
                start_time = datetime.now()
            
            max_minutes = self.rule_params.get("max_minutes", 0)
            skill_name = self.rule_params.get("skill_name", "")
            skill_level = self.rule_params.get("skill_level", 0)
            total_level = self.rule_params.get("total_level", 0)
            item_name = self.rule_params.get("item_name", "")
            item_quantity = self.rule_params.get("item_quantity", 0)
            
            # Check rules using CSV data ONLY
            logging.debug(f"[STATS_MONITOR] Calling check_rules_from_stats with: max_minutes={max_minutes}, skill_name={skill_name}, skill_level={skill_level}, total_level={total_level}, item_name={item_name}, item_quantity={item_quantity}")
            triggered = check_rules_from_stats(
                stats=stats_from_csv,
                start_time=start_time,
                max_minutes=max_minutes,
                skill_name=skill_name,
                skill_level=skill_level,
                total_level=total_level,
                item_name=item_name,
                item_quantity=item_quantity
            )
            logging.debug(f"[STATS_MONITOR] Rule check result: {triggered}")
            
            if triggered:
                # Only update if rule changed (avoid overwriting with same rule)
                if self.triggered_rule != triggered:
                    self.triggered_rule = triggered
                    # Write to file for run_rj_loop.py to read
                    self.rule_check_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.rule_check_file, 'w', encoding='utf-8') as f:
                        f.write(triggered)
                    logging.info(f"[STATS_MONITOR] Rule triggered: {triggered}")
            else:
                # Clear triggered rule if no longer triggered
                if self.triggered_rule is not None:
                    self.triggered_rule = None
                    if self.rule_check_file.exists():
                        self.rule_check_file.unlink()
                        
        except Exception as e:
            logging.error(f"[STATS_MONITOR] Error checking rules: {e}", exc_info=True)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current stats dictionary."""
        return self.current_stats.copy()
    
    def get_triggered_rule(self) -> Optional[str]:
        """Get the currently triggered rule, if any."""
        return self.triggered_rule
    
    def force_update(self):
        """Force an immediate stats update."""
        self._check_and_update_stats()