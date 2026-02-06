"""
Statistics Display Module (PySide6)
====================================

Loads and displays character statistics and skill icons.
"""

from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont
from typing import Dict, Optional, Callable
from pathlib import Path
import logging
import csv
import base64
from io import BytesIO

from helpers.ipc import IPCClient
from utils.stats_monitor import StatsMonitor
from gui.inventory_panel import InventoryPanel
from gui.equipment_panel import EquipmentPanel
from gui.inventory_panel import InventoryPanel
from gui.equipment_panel import EquipmentPanel


class StatisticsDisplay:
    """Displays character statistics and skill icons."""
    
    def __init__(self, root: QWidget, instance_tabs: Dict, instance_ports: Dict,
                 skill_icons: Dict, stats_monitors: Dict, log_callback: Optional[Callable] = None,
                 get_credential_name_callback: Optional[Callable] = None,
                 log_message_to_instance_callback: Optional[Callable] = None):
        """Initialize statistics display."""
        self.root = root
        self.instance_tabs = instance_tabs
        self.instance_ports = instance_ports
        self.skill_icons = skill_icons
        self.stats_monitors = stats_monitors
        self.log_callback = log_callback or (lambda msg, level='info': None)
        self.statistics_timers = {}
        self.get_credential_name_callback = get_credential_name_callback or (lambda name: name)
        self.log_message_to_instance_callback = log_message_to_instance_callback or (lambda name, msg, level='info': None)
        # Map credential names to instance names for CSV update callbacks
        self.credential_to_instance = {}
    
    def load_skill_icons(self) -> Dict:
        """Load skill icons from files."""
        if not hasattr(self, 'skill_icons') or not self.skill_icons:
            self.skill_icons = {}
        
        try:
            from PIL import Image
            
            # Skill name to icon filename mapping
            skill_icon_map = {
                'attack': 'attack.png',
                'strength': 'strength.png',
                'defence': 'defence.png',
                'hitpoints': 'hitpoints.png',
                'ranged': 'ranged.png',
                'prayer': 'prayer.png',
                'magic': 'magic.png',
                'cooking': 'cooking.png',
                'woodcutting': 'woodcutting.png',
                'fletching': 'fletching.png',
                'fishing': 'fishing.png',
                'firemaking': 'firemaking.png',
                'crafting': 'crafting.png',
                'smithing': 'smithing.png',
                'mining': 'mining.png',
                'herblore': 'herblore.png',
                'agility': 'agility.png',
                'thieving': 'thieving.png',
                'slayer': 'slayer.png',
                'farming': 'farming.png',
                'runecraft': 'runecrafting.png',
                'hunter': 'hunter.png',
                'construction': 'construction.png'
            }
            
            # Use absolute path relative to gui.py location
            icons_dir = Path(__file__).resolve().parent.parent / "skill_icons"
            logging.info(f"[GUI] Loading skill icons from: {icons_dir}")
            if icons_dir.exists():
                for skill_name, icon_file in skill_icon_map.items():
                    icon_path = icons_dir / icon_file
                    if icon_path.exists():
                        try:
                            img = Image.open(icon_path)
                            img = img.resize((20, 20), Image.Resampling.LANCZOS)
                            # Convert PIL Image to QPixmap
                            from PIL.ImageQt import ImageQt
                            qt_img = ImageQt(img)
                            pixmap = QPixmap.fromImage(qt_img)
                            self.skill_icons[skill_name] = pixmap
                            logging.debug(f"[GUI] Loaded icon: {icon_file}")
                        except Exception as e:
                            logging.warning(f"Could not load icon {icon_file}: {e}")
                            self.skill_icons[skill_name] = None
                    else:
                        logging.warning(f"Icon file not found: {icon_path}")
                        self.skill_icons[skill_name] = None
            else:
                logging.warning(f"Skill icons directory does not exist: {icons_dir}")
        except ImportError as e:
            logging.error(f"PIL/Pillow not available, skill icons will not be displayed. Error: {e}")
            self.skill_icons = {}
        except Exception as e:
            logging.error(f"Error loading skill icons: {e}")
            self.skill_icons = {}
        
        return self.skill_icons
    
    def decode_base64_icon(self, icon_base64: str) -> Optional[QPixmap]:
        """Decode base64 icon string to QPixmap, resized to 20x20."""
        try:
            from PIL import Image
            from PIL.ImageQt import ImageQt
            
            logging.debug(f"[GUI] Decoding icon: base64 length={len(icon_base64) if icon_base64 else 0}")
            # Decode base64
            image_data = base64.b64decode(icon_base64)
            logging.debug(f"[GUI] Decoded image data: {len(image_data)} bytes")
            # Load image from bytes
            img = Image.open(BytesIO(image_data))
            logging.debug(f"[GUI] Loaded image: size={img.size}, mode={img.mode}")
            # Resize to 20x20 (same as skill icons)
            img = img.resize((20, 20), Image.Resampling.LANCZOS)
            # Convert to QPixmap
            qt_img = ImageQt(img)
            pixmap = QPixmap.fromImage(qt_img)
            logging.debug(f"[GUI] Successfully created QPixmap: size={pixmap.size()}")
            return pixmap
        except Exception as e:
            logging.error(f"[GUI] Error decoding base64 icon: {e}", exc_info=True)
            return None
    
    def load_character_stats(self, username: str) -> Optional[Dict]:
        """Load character stats from CSV file."""
        try:
            csv_path = Path(__file__).resolve().parent.parent / "utils" / "character_data" / "character_stats.csv"
            if not csv_path.exists():
                logging.warning(f"[GUI] CSV file not found: {csv_path}")
                return None
            
            logging.info(f"[GUI] Loading stats from CSV for username: {username}")
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Find the most recent entry for this username
                latest_entry = None
                for row in reader:
                    csv_username = row.get('username', '').strip()
                    if csv_username.lower() == username.lower():
                        latest_entry = row
                        logging.debug(f"[GUI] Found matching entry for {username}")
                
                if latest_entry:
                    logging.info(f"[GUI] Loaded stats data with {len(latest_entry)} fields")
                else:
                    logging.warning(f"[GUI] No matching entry found for username: {username}")
                    # Log available usernames for debugging
                    file.seek(0)
                    reader = csv.DictReader(file)
                    available_usernames = [row.get('username', '').strip() for row in reader if row.get('username')]
                    logging.debug(f"[GUI] Available usernames in CSV: {set(available_usernames)}")
                
                return latest_entry
                    
        except Exception as e:
            logging.error(f"Error loading character stats: {e}")
            return None
    
    def update_stats_text(self, instance_name_or_credential: str, use_ipc: bool = True):
        """Update the stats display for an instance with icons.
        
        Args:
            instance_name_or_credential: Can be either instance_name (e.g., "detected_17000") 
                                        or credential name (e.g., "Batquinn"). If it's a credential
                                        name, we'll look up the instance_name from the mapping.
            use_ipc: If True, fetch inventory/equipment from IPC to get icons. Falls back to CSV if IPC fails.
        """
        logging.info(f"[GUI] update_stats_text called with: {instance_name_or_credential}")
        logging.info(f"[GUI] credential_to_instance mapping: {self.credential_to_instance}")
        logging.info(f"[GUI] Available instance_tabs keys: {list(self.instance_tabs.keys())}")
        
        # Check if this is a credential name (from CSV update callback)
        # If we have a mapping, use it to find the instance_name
        instance_name = self.credential_to_instance.get(instance_name_or_credential, instance_name_or_credential)
        logging.info(f"[GUI] After credential lookup, instance_name: {instance_name}")
        
        # If still not found, try to find instance_name by checking if credential name matches
        if instance_name == instance_name_or_credential:
            # Try to find instance_name by credential name
            for inst_name, cred_name in self.credential_to_instance.items():
                if cred_name == instance_name_or_credential:
                    instance_name = inst_name
                    logging.info(f"[GUI] Found instance_name by credential name: {instance_name}")
                    break
        
        # If still not found, assume instance_name_or_credential is the instance_name
        if instance_name not in self.instance_tabs:
            instance_name = instance_name_or_credential
            logging.info(f"[GUI] Using instance_name_or_credential as instance_name: {instance_name}")
        
        logging.info(f"[GUI] Final instance_name: {instance_name}")
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab:
            logging.warning(f"[GUI] Instance tab not found for {instance_name}")
            logging.warning(f"[GUI] Available instance_tabs: {list(self.instance_tabs.keys())}")
            return
        if not hasattr(instance_tab, 'plan_runner_tab'):
            logging.warning(f"[GUI] plan_runner_tab not found for {instance_name}")
            return
        
        plan_runner_tab = instance_tab.plan_runner_tab
        if not hasattr(plan_runner_tab, 'unified_stats_panel'):
            logging.warning(f"[GUI] unified_stats_panel not found for {instance_name}")
            return
        
        # Get credential name for CSV lookup (use credential name, not instance_name)
        credential_name = self.get_credential_name_callback(instance_name)
        logging.info(f"[GUI] Got credential name for {instance_name}: {credential_name}")
        if credential_name == "Detected" or credential_name == "Unknown":
            credential_name = instance_name
            logging.info(f"[GUI] Credential name was Detected/Unknown, using instance_name: {credential_name}")
        
        # Load current stats using credential name (CSV has usernames, not instance names)
        logging.info(f"[GUI] Loading stats for credential: {credential_name}")
        stats_data = self.load_character_stats(credential_name)
        logging.info(f"[GUI] Loaded stats data for {instance_name} (credential: {credential_name}): {stats_data is not None}")
        
        # Update logged-in time label
        if hasattr(plan_runner_tab, 'logged_in_time_label'):
            logged_in_time = stats_data.get('logged_in_time', 0) if stats_data else 0
            try:
                logged_in_seconds = float(logged_in_time)
                hours = int(logged_in_seconds // 3600)
                minutes = int((logged_in_seconds % 3600) // 60)
                seconds = int(logged_in_seconds % 60)
                time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            except (ValueError, TypeError):
                time_str = "0:00:00"
            plan_runner_tab.logged_in_time_label.setText(time_str)
        
        # Clear all sections
        # Get references from unified stats panel
        unified_panel = getattr(plan_runner_tab, 'unified_stats_panel', None)
        if not unified_panel:
            logging.warning(f"[GUI] unified_stats_panel not found for {instance_name}")
            return
        
        skills_layout = unified_panel.get_skills_layout()
        inventory_panel = unified_panel.get_inventory_panel()
        equipment_panel = unified_panel.get_equipment_panel()
        
        # Clear skills section
        while skills_layout.count():
            child = skills_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Inventory and Equipment panels manage their own content via set_items()
        # No need to manually clear them
        
        if not stats_data:
            no_data_label = QLabel("No stats data available")
            skills_layout.addWidget(no_data_label)
            skills_layout.addStretch()
            return
        
        # Skills section
        skill_names = [
            'attack', 'strength', 'defence', 'hitpoints', 'ranged', 'prayer', 'magic',
            'cooking', 'woodcutting', 'fletching', 'fishing', 'firemaking', 'crafting',
            'smithing', 'mining', 'herblore', 'agility', 'thieving', 'slayer',
            'farming', 'runecraft', 'hunter', 'construction'
        ]
        
        for skill_name in skill_names:
            skill_row = QWidget()
            skill_layout = QHBoxLayout(skill_row)
            skill_layout.setContentsMargins(2, 2, 2, 2)
            
            # Icon
            icon_label = QLabel()
            if skill_name in self.skill_icons and self.skill_icons[skill_name]:
                icon_label.setPixmap(self.skill_icons[skill_name])
            icon_label.setFixedSize(20, 20)
            skill_layout.addWidget(icon_label)
            
            # Skill name
            skill_display_name = skill_name.replace('_', ' ').title()
            name_label = QLabel(skill_display_name)
            name_label.setMinimumWidth(100)
            skill_layout.addWidget(name_label)
            
            # Level
            level_key = f"{skill_name}_level"
            level = stats_data.get(level_key, 0)
            try:
                level = int(level) if level else 0
            except (ValueError, TypeError):
                level = 0
            
            level_label = QLabel(f"Lvl: {level}")
            level_label.setMinimumWidth(60)
            skill_layout.addWidget(level_label)
            
            # XP
            xp_key = f"{skill_name}_xp"
            xp = stats_data.get(xp_key, 0)
            try:
                xp = int(float(xp)) if xp else 0
            except (ValueError, TypeError):
                xp = 0
            
            xp_label = QLabel(f"XP: {xp:,}")
            skill_layout.addWidget(xp_label)
            skill_layout.addStretch()
            
            skills_layout.addWidget(skill_row)
        
        skills_layout.addStretch()
        
        # Inventory section
        # Try to fetch from IPC first (to get icons), fallback to CSV
        inventory_items = []
        ipc_inventory_data = None
        
        logging.info(f"[GUI] Inventory fetch: use_ipc={use_ipc}, instance_name={instance_name}, instance_ports keys={list(self.instance_ports.keys())}")
        if use_ipc and instance_name in self.instance_ports:
            try:
                port = self.instance_ports[instance_name]
                logging.info(f"[GUI] Fetching inventory from IPC on port {port}")
                ipc_client = IPCClient(port=port)
                ipc_response = ipc_client.get_inventory()
                logging.info(f"[GUI] IPC response: ok={ipc_response.get('ok') if ipc_response else None}, has_slots={bool(ipc_response and ipc_response.get('slots'))}")
                if ipc_response and ipc_response.get('ok') and ipc_response.get('slots'):
                    ipc_inventory_data = ipc_response.get('slots', [])
                    logging.info(f"[GUI] Fetched {len(ipc_inventory_data)} inventory items from IPC")
                    # Log first item to check for iconBase64
                    if ipc_inventory_data:
                        first_item = ipc_inventory_data[0]
                        logging.info(f"[GUI] First inventory item: id={first_item.get('id')}, name={first_item.get('itemName')}, has_iconBase64={bool(first_item.get('iconBase64'))}")
                else:
                    logging.warning(f"[GUI] IPC inventory response invalid: {ipc_response}")
            except Exception as e:
                logging.error(f"[GUI] Failed to fetch inventory from IPC: {e}", exc_info=True)
        
        # Use the unified panel's inventory panel
        if ipc_inventory_data:
            # Use IPC data (includes icons and slot positions)
            inventory_panel.set_items(ipc_inventory_data)
            logging.info(f"[GUI] Set {len(ipc_inventory_data)} inventory items in panel")
        else:
            # Fallback to CSV parsing - convert to IPC format
            csv_items = []
            for key, value in stats_data.items():
                if key.startswith('inventory_') and value:
                    try:
                        # Skip columns that end with _inventory (these are for tracked items, duplicates)
                        if key.endswith('_inventory') and not key.startswith('inventory_inventory'):
                            continue
                        
                        quantity = int(float(value)) if value else 0
                        if quantity > 0:
                            # Convert inventory_itemname back to display name
                            item_key = key.replace('inventory_', '')
                            item_name = item_key.replace('_', ' ').title()
                            # CSV doesn't have slot info, so we can't display on grid
                            # Just create a placeholder entry
                            csv_items.append({
                                'slot': len(csv_items),  # Sequential slots
                                'itemName': item_name,
                                'quantity': quantity,
                                'id': -1  # No ID from CSV
                            })
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Error parsing inventory item {key}={value}: {e}")
                        continue
            
            if csv_items:
                inventory_panel.set_items(csv_items)
            else:
                # No items - panel will show empty
                inventory_panel.set_items([])
        
        # Equipment section
        # Try to fetch from IPC first (to get icons), fallback to CSV
        equipment = {}
        ipc_equipment_data = None
        
        logging.info(f"[GUI] Equipment fetch: use_ipc={use_ipc}, instance_name={instance_name}")
        if use_ipc and instance_name in self.instance_ports:
            try:
                port = self.instance_ports[instance_name]
                logging.info(f"[GUI] Fetching equipment from IPC on port {port}")
                ipc_client = IPCClient(port=port)
                ipc_response = ipc_client.get_equipment()
                logging.info(f"[GUI] IPC equipment response: ok={ipc_response.get('ok') if ipc_response else None}, has_slots={bool(ipc_response and ipc_response.get('slots'))}")
                if ipc_response and ipc_response.get('ok') and ipc_response.get('slots'):
                    ipc_equipment_data = ipc_response.get('slots', [])
                    logging.info(f"[GUI] Fetched {len(ipc_equipment_data)} equipment slots from IPC")
                    # Log first slot to check for iconBase64
                    if ipc_equipment_data:
                        first_slot = ipc_equipment_data[0]
                        logging.info(f"[GUI] First equipment slot: slot={first_slot.get('slot')}, id={first_slot.get('id')}, name={first_slot.get('name')}, has_iconBase64={bool(first_slot.get('iconBase64'))}")
                else:
                    logging.warning(f"[GUI] IPC equipment response invalid: {ipc_response}")
            except Exception as e:
                logging.error(f"[GUI] Failed to fetch equipment from IPC: {e}", exc_info=True)
        
        # Use the unified panel's equipment panel
        if ipc_equipment_data:
            # Use IPC data (includes icons and slot positions)
            equipment_panel.set_items(ipc_equipment_data)
            logging.info(f"[GUI] Set {len(ipc_equipment_data)} equipment slots in panel")
        else:
            # Fallback to CSV parsing - convert to IPC format
            csv_equipment = []
            slot_mapping = {
                'helmet': 'HEAD',
                'cape': 'CAPE',
                'amulet': 'AMULET',
                'weapon': 'WEAPON',
                'body': 'BODY',
                'shield': 'SHIELD',
                'legs': 'LEGS',
                'gloves': 'GLOVES',
                'boots': 'BOOTS',
                'ring': 'RING'
            }
            
            for csv_slot, ipc_slot in slot_mapping.items():
                equipped_key = f'equipped_{csv_slot}'
                item_name = stats_data.get(equipped_key, '').strip()
                if item_name:
                    csv_equipment.append({
                        'slot': ipc_slot,
                        'name': item_name,
                        'quantity': 1,
                        'id': -1  # No ID from CSV
                    })
            
            equipment_panel.set_items(csv_equipment)
    
    def start_stats_monitor(self, username: str, port: int, 
                           on_csv_update_callback=None, on_username_changed_callback=None,
                           instance_name: Optional[str] = None):
        """Start statistics monitoring for an instance.
        
        Args:
            username: Credential name (for CSV lookup)
            port: IPC port
            on_csv_update_callback: Optional callback when CSV is updated
            on_username_changed_callback: Optional callback when username changes
            instance_name: Instance name (e.g., "detected_17000") - used to map credential to instance
        """
        logging.info(f"[GUI] Starting stats monitor for {username} on port {port} (instance: {instance_name})")
        try:
            # Store mapping from credential name to instance name
            if instance_name:
                self.credential_to_instance[username] = instance_name
            
            # Check if monitor already exists
            if username in self.stats_monitors:
                existing_monitor = self.stats_monitors[username]
                if existing_monitor.running:
                    logging.debug(f"[GUI] Stats monitor already running for {username}, skipping")
                    return
                else:
                    del self.stats_monitors[username]
            
            # Create callback to update GUI when CSV is updated
            def on_csv_update(updated_username):
                # Get instance_name from credential mapping
                instance_name = self.credential_to_instance.get(updated_username, updated_username)
                
                # Log CSV update to instance output
                self.log_message_to_instance_callback(instance_name, f"Stats updated from CSV for {updated_username}", 'info')
                
                if on_csv_update_callback:
                    on_csv_update_callback(updated_username)
                else:
                    # Use QTimer to schedule on main thread
                    # Pass credential name (updated_username) - update_stats_text will map it to instance_name
                    QTimer.singleShot(0, lambda: self.update_stats_text(updated_username))
            
            # Create callback for username changes
            def on_username_changed(old_username, new_username):
                if on_username_changed_callback:
                    on_username_changed_callback(old_username, new_username)
            
            monitor = StatsMonitor(port, username, on_csv_update=on_csv_update, 
                                 on_username_changed=on_username_changed)
            logging.info(f"[GUI] Created StatsMonitor for {username}, starting...")
            monitor.start()
            
            self.stats_monitors[username] = monitor
            logging.info(f"[GUI] Started stats monitor for {username} on port {port}")
            
        except Exception as e:
            logging.error(f"[GUI] Error starting stats monitor for {username}: {e}")
    
    def stop_stats_monitor(self, username: str):
        """Stop the stats monitor for an instance."""
        if username in self.stats_monitors:
            self.stats_monitors[username].stop()
            del self.stats_monitors[username]
            logging.info(f"[GUI] Stopped stats monitor for {username}")
    
    def stop_statistics_timer(self, instance_name: str):
        """Stop statistics timer for an instance."""
        if instance_name in self.statistics_timers:
            timer = self.statistics_timers[instance_name]
            if timer.isActive():
                timer.stop()
            del self.statistics_timers[instance_name]
