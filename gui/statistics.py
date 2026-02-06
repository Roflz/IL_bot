"""
Statistics Display Module
=========================

Handles displaying character statistics, skills, inventory, and equipment.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional
import logging
from pathlib import Path

from helpers.ipc import IPCClient
from utils.stats_monitor import StatsMonitor


class StatisticsDisplay:
    """Manages statistics display for instances."""
    
    def __init__(self, root, instance_tabs: Dict, instance_ports: Dict, 
                 skill_icons: Dict, stats_monitors: Dict, log_callback=None):
        """
        Initialize statistics display.
        
        Args:
            root: Root tkinter window
            instance_tabs: Dictionary of instance tabs
            instance_ports: Dictionary mapping instance names to ports
            skill_icons: Dictionary of skill icon images
            stats_monitors: Dictionary of StatsMonitor instances
            log_callback: Optional callback for logging messages
        """
        self.root = root
        self.instance_tabs = instance_tabs
        self.instance_ports = instance_ports
        self.skill_icons = skill_icons
        self.stats_monitors = stats_monitors
        self.log_callback = log_callback
        self.statistics_timers = {}
    
    def load_skill_icons(self) -> Dict:
        """Load skill icon images from files."""
        if not hasattr(self, 'skill_icons') or not self.skill_icons:
            self.skill_icons = {}
        
        try:
            from PIL import Image, ImageTk
            
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
                            photo = ImageTk.PhotoImage(img)
                            self.skill_icons[skill_name] = photo
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
    
    def load_character_stats(self, username: str) -> Optional[Dict]:
        """Load character stats from CSV file."""
        try:
            csv_path = Path(__file__).resolve().parent.parent / "utils" / "character_data" / "character_stats.csv"
            if not csv_path.exists():
                return None
            
            import csv
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Find the most recent entry for this username
                latest_entry = None
                for row in reader:
                    if row.get('username') == username:
                        latest_entry = row
                
                return latest_entry
                    
        except Exception as e:
            logging.error(f"Error loading character stats: {e}")
            return None
    
    def update_stats_text(self, username: str):
        """Update the stats display for an instance with icons."""
        # This is a very large method - keeping it as a reference to be implemented
        # The full implementation is ~300 lines in the original gui.py
        # For now, we'll mark it as needing the full implementation
        instance_tab = self.instance_tabs.get(username)
        if not instance_tab or not hasattr(instance_tab, 'skills_scrollable_frame'):
            return
        
        # Load current stats
        stats_data = self.load_character_stats(username)
        
        # Update logged-in time label
        if hasattr(instance_tab, 'logged_in_time_label'):
            logged_in_time = stats_data.get('logged_in_time', 0) if stats_data else 0
            try:
                logged_in_seconds = float(logged_in_time)
                hours = int(logged_in_seconds // 3600)
                minutes = int((logged_in_seconds % 3600) // 60)
                seconds = int(logged_in_seconds % 60)
                time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            except (ValueError, TypeError):
                time_str = "0:00:00"
            instance_tab.logged_in_time_label.config(text=time_str)
        
        if not stats_data:
            # Clear all sections
            for widget in instance_tab.skills_scrollable_frame.winfo_children():
                widget.destroy()
            for widget in instance_tab.inventory_scrollable_frame.winfo_children():
                widget.destroy()
            for widget in instance_tab.equipment_scrollable_frame.winfo_children():
                widget.destroy()
            
            no_data_label = ttk.Label(instance_tab.skills_scrollable_frame, text="No stats data available")
            no_data_label.grid(row=0, column=0, padx=5, pady=5)
            return
        
        # NOTE: Full implementation of skills/inventory/equipment display
        # is ~250 lines. This should be extracted from gui.py lines 1332-1575
        # For brevity, marking as TODO for full extraction
        logging.warning(f"Full stats display update not yet implemented for {username}")
    
    def start_stats_monitor(self, username: str, port: int, 
                           on_csv_update_callback=None, on_username_changed_callback=None):
        """Start statistics monitoring for an instance."""
        logging.info(f"[GUI] Starting stats monitor for {username} on port {port}")
        try:
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
                if on_csv_update_callback:
                    on_csv_update_callback(updated_username)
                else:
                    self.root.after(0, lambda: self.update_stats_text(updated_username))
            
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
    
    def update_statistics_display(self, instance_name: str):
        """Update statistics display for an instance."""
        import time
        
        instance_tab = self.instance_tabs.get(instance_name)
        if not instance_tab:
            return
        
        # Update current plan
        if hasattr(instance_tab, 'current_plan_label'):
            current_plan = getattr(instance_tab, 'current_plan_name', 'None')
            instance_tab.current_plan_label.config(text=current_plan)
        
        # Update current phase
        if hasattr(instance_tab, 'current_phase_label'):
            current_phase = getattr(instance_tab, 'current_phase', 'None')
            instance_tab.current_phase_label.config(text=current_phase)
        
        # Update runtime
        if hasattr(instance_tab, 'runtime_label'):
            start_time = getattr(instance_tab, 'start_time', None)
            if start_time:
                elapsed = time.time() - start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                runtime_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                instance_tab.runtime_label.config(text=runtime_text)
            else:
                instance_tab.runtime_label.config(text="00:00:00")
        
        # Update rules display
        if hasattr(instance_tab, 'stats_rules_tree'):
            # Clear existing rules
            for item in instance_tab.stats_rules_tree.get_children():
                instance_tab.stats_rules_tree.delete(item)
            
            # Get current plan rules
            current_plan_index = getattr(instance_tab, 'current_plan_index', 0)
            if current_plan_index < len(instance_tab.plan_entries):
                plan_entry = instance_tab.plan_entries[current_plan_index]
                rules = plan_entry.get('rules', {})
                
                # Add rules to tree
                if rules.get('max_minutes'):
                    instance_tab.stats_rules_tree.insert('', 'end', text=f"Time Limit: {rules['max_minutes']} minutes")
                if rules.get('stop_skill'):
                    instance_tab.stats_rules_tree.insert('', 'end', text=f"Stop at Skill: {rules['stop_skill']} level {rules.get('stop_skill_level', 0)}")
                if rules.get('total_level'):
                    instance_tab.stats_rules_tree.insert('', 'end', text=f"Total Level: {rules['total_level']}")
                if rules.get('stop_items'):
                    items_node = instance_tab.stats_rules_tree.insert('', 'end', text="Stop with Items:")
                    for item in rules['stop_items']:
                        instance_tab.stats_rules_tree.insert(items_node, 'end', text=f"{item['name']} x{item['qty']}")
                
                if not any(rules.values()):
                    instance_tab.stats_rules_tree.insert('', 'end', text="No rules configured")
        
        # Also update stats text (skills, inventory, equipment)
        self.update_stats_text(instance_name)
    
    def start_statistics_timer(self, instance_name: str):
        """Start periodic statistics update timer."""
        def update_timer():
            if instance_name in self.instance_tabs:
                instance_tab = self.instance_tabs[instance_name]
                if getattr(instance_tab, 'is_running', False):
                    self.update_statistics_display(instance_name)
                    # Schedule next update in 1 second
                    self.statistics_timers[instance_name] = self.root.after(1000, update_timer)
        
        # Start the timer
        update_timer()
    
    def stop_statistics_timer(self, instance_name: str):
        """Stop the statistics update timer for an instance."""
        instance_tab = self.instance_tabs.get(instance_name)
        if instance_tab:
            instance_tab.is_running = False
            # Clear current tracking
            instance_tab.current_plan_name = "None"
            instance_tab.current_phase = "Stopped"
            self.update_statistics_display(instance_name)
        
        if instance_name in self.statistics_timers:
            self.root.after_cancel(self.statistics_timers[instance_name])
            del self.statistics_timers[instance_name]
    
    def estimate_item_value(self, item_key: str, quantity: int) -> int:
        """Estimate the value of an item (simplified pricing)."""
        prices = {
            'coins': 1, 'cowhides': 100, 'logs': 25, 'leather': 200,
            'bow_string': 50, 'iron_bar': 150, 'steel_bar': 300,
            'coal': 30, 'iron_ore': 50, 'raw_fish': 20, 'cooked_fish': 40,
            'bronze_bar': 100, 'silver_bar': 250, 'gold_bar': 500,
            'mithril_bar': 600, 'adamant_bar': 1200, 'rune_bar': 3000,
            'bronze_ore': 25, 'silver_ore': 100, 'gold_ore': 200,
            'mithril_ore': 300, 'adamant_ore': 600, 'rune_ore': 1500,
            'willow_logs': 50, 'oak_logs': 100, 'maple_logs': 200,
            'yew_logs': 500, 'magic_logs': 1000, 'redwood_logs': 2000
        }
        
        price_per_item = prices.get(item_key, 0)
        return price_per_item * quantity
