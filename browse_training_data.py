#!/usr/bin/env python3
"""
GUI to browse through all training data sequences
Shows complete raw data for every input sequence and target
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import numpy as np
import json
import csv
import pyperclip
import pandas as pd
from pathlib import Path

class ScrollableFrame:
    """A frame with both horizontal and vertical scrollbars that can contain any widget"""
    
    def __init__(self, parent, canvas_width=None, canvas_height=None, **kwargs):
        self.parent = parent
        
        # Create main frame
        self.main_frame = ttk.Frame(parent, **kwargs)
        
        # Create canvas
        self.canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        
        # Create scrollbars
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        # Configure canvas
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Create scrollable frame inside canvas
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bind events
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        
        # Pack scrollbars and canvas
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Set initial canvas size if specified
        if canvas_width:
            self.canvas.configure(width=canvas_width)
        if canvas_height:
            self.canvas.configure(height=canvas_height)
    
    def _on_frame_configure(self, event=None):
        """Update the scroll region when the frame size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Update the frame width when the canvas is resized"""
        if event.width > 1:  # Avoid setting width to 0
            self.canvas.itemconfig(self.canvas.find_withtag("all")[0], width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:  # Windows/Mac
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def pack(self, **kwargs):
        """Pack the main frame"""
        return self.main_frame.pack(**kwargs)
    
    def pack_forget(self):
        """Unpack the main frame"""
        return self.main_frame.pack_forget()
    
    def get_frame(self):
        """Get the scrollable frame for adding widgets"""
        return self.scrollable_frame
    
    def update_scroll_region(self):
        """Manually update the scroll region"""
        self._on_frame_configure()
    
    def scroll_to_top(self):
        """Scroll to the top of the frame"""
        self.canvas.yview_moveto(0)
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the frame"""
        self.canvas.yview_moveto(1)
    
    def scroll_to_left(self):
        """Scroll to the left of the frame"""
        self.canvas.xview_moveto(0)
    
    def scroll_to_right(self):
        """Scroll to the right of the frame"""
        self.canvas.xview_moveto(1)

class TrainingDataBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Data Browser")
        self.root.geometry("1400x800")
        
        # Configuration
        self.sequence_length = 10  # Number of gamestates for context
        
        # Load data
        self.load_data()
        
        # Create GUI
        self.create_widgets()
        
        # Start with first sequence
        self.current_sequence = 0
        self.update_display()
    
    def load_data(self):
        """Load all training data"""
        print("Loading training data...")
        
        # Load input sequences
        self.input_sequences = np.load('data/training_data/input_sequences.npy')
        print(f"Loaded input sequences: {self.input_sequences.shape}")
        
        # Load target sequences
        with open('data/training_data/target_sequences.json', 'r') as f:
            self.target_sequences = json.load(f)
        print(f"Loaded target sequences: {len(self.target_sequences)}")
        
        # Load feature names from feature mappings
        try:
            with open('data/features/feature_mappings.json', 'r') as f:
                self.feature_mappings = json.load(f)
            print(f"Loaded feature mappings for {len(self.feature_mappings)} features")
            
            # Extract feature names from mappings - the structure is a list of feature objects
            self.feature_names = {}
            for feature_data in self.feature_mappings:
                if isinstance(feature_data, dict):
                    feature_idx = feature_data.get('feature_index', -1)
                    feature_name = feature_data.get('feature_name', f'feature_{feature_idx}')
                    if feature_idx >= 0:
                        self.feature_names[str(feature_idx)] = {'feature_name': feature_name}
        except Exception as e:
            self.feature_mappings = {}
            self.feature_names = {}
            print(f"Warning: Could not load feature mappings: {e}")
        
        # Load ID mappings for better visualization
        try:
            with open('data/features/id_mappings.json', 'r') as f:
                self.id_mappings = json.load(f)
            print(f"Loaded ID mappings for items, NPCs, objects, and hashes")
        except Exception as e:
            self.id_mappings = {}
            print(f"Warning: Could not load ID mappings: {e}")
        
        # Load trimmed action data for Option 3 visualization (this is the "raw" data after trimming)
        try:
            with open('data/training_data/raw_action_data.json', 'r') as f:
                self.raw_action_data = json.load(f)
            print(f"Loaded trimmed action data for {len(self.raw_action_data)} gamestates")
        except Exception as e:
            # Fallback to features directory if training data not available
            try:
                with open('data/features/raw_action_data.json', 'r') as f:
                    self.raw_action_data = json.load(f)
                print(f"Loaded fallback action data for {len(self.raw_action_data)} gamestates")
            except Exception as e2:
                self.raw_action_data = []
                print(f"Warning: Could not load action data: {e2}")
        
        # Load features data for sequence alignment display (use processed data from training pipeline)
        try:
            self.features = np.load('data/training_data/state_features.npy')
            print(f"Loaded processed features: {self.features.shape}")
        except Exception as e:
            # Fallback to features directory if training data not available
            try:
                self.features = np.load('data/features/state_features.npy')
                print(f"Warning: Using fallback features from extraction (untrimmed): {self.features.shape}")
            except Exception as e2:
                self.features = None
                print(f"Warning: Could not load features: {e2}")
        
        # Data trimming is now done in phase1_data_preparation.py
        # No need to trim here since training data is already clean
        
        # Load pre-computed normalized data (no more on-the-fly normalization!)
        try:
            self.normalized_features = np.load('data/training_data/normalized_features.npy')
            print(f"Loaded pre-computed normalized features: {self.normalized_features.shape}")
        except Exception as e:
            print(f"Warning: Could not load normalized features: {e}")
            self.normalized_features = None
            
        try:
            self.normalized_input_sequences = np.load('data/training_data/normalized_input_sequences.npy')
            print(f"Loaded pre-computed normalized input sequences: {self.normalized_input_sequences.shape}")
        except Exception as e:
            print(f"Warning: Could not load normalized input sequences: {e}")
            self.normalized_input_sequences = None
            
        try:
            with open('data/training_data/normalized_action_data.json', 'r') as f:
                self.normalized_action_data = json.load(f)
            print(f"Loaded pre-computed normalized action data for {len(self.normalized_action_data)} gamestates")
        except Exception as e:
            print(f"Warning: Could not load normalized action data: {e}")
            self.normalized_action_data = None
        
        # Create reverse lookup for hash translation
        self.create_hash_reverse_lookup()
        
        # Print normalization status
        self.print_normalization_info()
        
        # Initialize feature analysis if the analysis tab exists
        if hasattr(self, 'feature_analysis_tree'):
            self.analyze_features()
        
        # Update action gamestate spinbox range if it exists
        if hasattr(self, 'action_gamestate_spinbox'):
            self.update_action_gamestate_range()
        
        # Update sequence range if it exists
        if hasattr(self, 'sequence_spinbox'):
            self.update_sequence_range()
    
    def create_hash_reverse_lookup(self):
        """Create reverse lookup from hash values to original values"""
        self.hash_reverse_lookup = {}
        
        # Use the new ID mappings structure
        if hasattr(self, 'id_mappings') and self.id_mappings:
            # Create lookup for global hash mappings
            if 'Global' in self.id_mappings and 'hash_mappings' in self.id_mappings['Global']:
                hash_mappings = self.id_mappings['Global']['hash_mappings']
            for hash_value, original_string in hash_mappings.items():
                # Convert hash value to int for consistent lookup
                try:
                    hash_key = int(float(hash_value))
                    # Store in a way that matches the feature structure
                    if 'hash_mappings' not in self.hash_reverse_lookup:
                        self.hash_reverse_lookup['hash_mappings'] = {}
                    self.hash_reverse_lookup['hash_mappings'][hash_key] = original_string
                except (ValueError, TypeError):
                    continue
            
            # Create lookups for feature-group-specific mappings
            for feature_group, group_mappings in self.id_mappings.items():
                if feature_group == 'Global':
                    continue  # Already handled above
                    
                for mapping_type, mappings in group_mappings.items():
                    if isinstance(mappings, dict):
                        for id_value, original_string in mappings.items():
                            try:
                                id_key = int(float(id_value))
                                if mapping_type not in self.hash_reverse_lookup:
                                    self.hash_reverse_lookup[mapping_type] = {}
                                self.hash_reverse_lookup[mapping_type][id_key] = original_string
                            except (ValueError, TypeError):
                                continue
        
    def translate_hash_value(self, feature_idx, hash_value):
        """Translate hash value to human-readable format using feature-specific mappings"""
        # First try to find translation in the new ID mappings structure
        if hasattr(self, 'id_mappings') and self.id_mappings:
            # Get feature info to determine the correct mapping category
            feature_name = None
            feature_group = None
            data_type = None
            
            for feature_data in self.feature_mappings:
                if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                    feature_name = feature_data.get('feature_name')
                    feature_group = feature_data.get('feature_group')
                    data_type = feature_data.get('data_type')
                    break
            
            # Handle boolean values automatically
            if data_type == 'boolean':
                if float(hash_value) == 1.0:
                    return f"{hash_value} → true"
                elif float(hash_value) == 0.0:
                    return f"{hash_value} → false"
            
            # Check feature-group-specific mappings
            if feature_group and feature_group in self.id_mappings:
                group_mappings = self.id_mappings[feature_group]
                
                # Check hash mappings first (for hashed strings)
                if 'hash_mappings' in group_mappings:
                    try:
                        hash_key = int(float(hash_value))
                        if str(hash_key) in group_mappings['hash_mappings']:
                            original_value = group_mappings['hash_mappings'][str(hash_key)]
                            return f"{hash_value} → {original_value}"
                    except (ValueError, TypeError):
                        pass
                
                # Check specific mapping types based on feature group
                if feature_group == "Player":
                    if 'player_animation_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['player_animation_ids']:
                                name = group_mappings['player_animation_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                    if 'player_movement_direction_hashes' in group_mappings:
                        try:
                            hash_key = int(float(hash_value))
                            if str(hash_key) in group_mappings['player_movement_direction_hashes']:
                                name = group_mappings['player_movement_direction_hashes'][str(hash_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                
                elif feature_group == "Interaction":
                    for mapping_type in ['action_type_hashes', 'item_name_hashes', 'target_hashes']:
                        if mapping_type in group_mappings:
                            try:
                                hash_key = int(float(hash_value))
                                if str(hash_key) in group_mappings[mapping_type]:
                                    name = group_mappings[mapping_type][str(hash_key)]
                                    return f"{hash_value} → {name}"
                            except (ValueError, TypeError):
                                pass
                
                elif feature_group == "Inventory":
                    if 'item_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['item_ids']:
                                name = group_mappings['item_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                    if 'empty_slot_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['empty_slot_ids']:
                                name = group_mappings['empty_slot_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                
                elif feature_group == "Bank":
                    if 'slot_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['slot_ids']:
                                name = group_mappings['slot_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                    # Only apply boolean mapping to features that are actually boolean
                    if data_type == 'boolean' and 'boolean_states' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['boolean_states']:
                                name = group_mappings['boolean_states'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                
                elif feature_group == "Game Objects":
                    if 'object_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['object_ids']:
                                name = group_mappings['object_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                
                elif feature_group == "NPCs":
                    if 'npc_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['npc_ids']:
                                name = group_mappings['npc_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                
                elif feature_group == "Tabs":
                    if 'tab_ids' in group_mappings:
                        try:
                            id_key = int(float(hash_value))
                            if str(id_key) in group_mappings['tab_ids']:
                                name = group_mappings['tab_ids'][str(id_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
                
                elif feature_group == "Phase Context":
                    if 'phase_type_hashes' in group_mappings:
                        try:
                            hash_key = int(float(hash_value))
                            if str(hash_key) in group_mappings['phase_type_hashes']:
                                name = group_mappings['phase_type_hashes'][str(hash_key)]
                                return f"{hash_value} → {name}"
                        except (ValueError, TypeError):
                            pass
            
            # Check global hash mappings as fallback
            if 'Global' in self.id_mappings and 'hash_mappings' in self.id_mappings['Global']:
                try:
                    hash_key = int(float(hash_value))
                    if str(hash_key) in self.id_mappings['Global']['hash_mappings']:
                        original_value = self.id_mappings['Global']['hash_mappings'][str(hash_key)]
                        return f"{hash_value} → {original_value}"
                except (ValueError, TypeError):
                    pass
        
        # Fallback: try the old feature-based lookup
        if feature_idx in self.hash_reverse_lookup:
            # Handle floating-point precision issues by converting to int
            try:
                # First try exact match
                if hash_value in self.hash_reverse_lookup[feature_idx]:
                    original_value = self.hash_reverse_lookup[feature_idx][hash_value]
                    return f"{hash_value} → {original_value}"
                
                # If no exact match, try converting to int for floating-point precision issues
                hash_key = int(round(float(hash_value)))
                if hash_key in self.hash_reverse_lookup[feature_idx]:
                    original_value = self.hash_reverse_lookup[feature_idx][hash_key]
                    return f"{hash_value} → {original_value}"
                    
            except (ValueError, TypeError):
                pass
        
        # No translation found
        return str(hash_value)
    
    def normalize_action_timestamp(self, timestamp: float) -> float:
        """Get normalized action timestamp from pre-computed data"""
        if hasattr(self, 'normalized_action_data') and self.normalized_action_data is not None:
            # Find the corresponding normalized action data for this gamestate
            if self.current_action_gamestate < len(self.normalized_action_data):
                action_data = self.normalized_action_data[self.current_action_gamestate]
                # Look for an action with this timestamp in the normalized data
                for action in action_data.get('actions', []):
                    if abs(action.get('timestamp', 0) - timestamp) < 0.001:  # Small tolerance
                        return action.get('timestamp', timestamp)
        return timestamp
    
    def normalize_action_coordinate(self, coord: float, coord_type: str) -> float:
        """Get normalized action coordinates from pre-computed data"""
        if hasattr(self, 'normalized_action_data') and self.normalized_action_data is not None:
            # Find the corresponding normalized action data for this gamestate
            if self.current_action_gamestate < len(self.normalized_action_data):
                action_data = self.normalized_action_data[self.current_action_gamestate]
                # Look for an action with this coordinate in the normalized data
                for action in action_data.get('actions', []):
                    if coord_type == 'screen_x' and 'x' in action:
                        if abs(action.get('x', 0) - coord) < 0.001:  # Small tolerance
                            return action.get('x', coord)
                    elif coord_type == 'screen_y' and 'y' in action:
                        if abs(action.get('y', 0) - coord) < 0.001:  # Small tolerance
                            return action.get('y', coord)
                    elif coord_type == 'scroll_delta' and 'dx' in action:
                        if abs(action.get('dx', 0) - coord) < 0.001:  # Small tolerance
                            return action.get('dx', coord)
        return coord
    
    def get_feature_category(self, feature_name):
        """Get feature category from feature mappings instead of hardcoded logic"""
        # Find the feature in our mappings
        for feature_data in self.feature_mappings:
            if isinstance(feature_data, dict) and feature_data.get('feature_name') == feature_name:
                return feature_data.get('feature_group', 'other')
        
        # Fallback to hardcoded logic if not found in mappings
        feature_name_lower = feature_name.lower()
        
        # Player state features (0-4)
        if "world_x" in feature_name_lower or "world_y" in feature_name_lower:
            return "Player"
        elif "animation" in feature_name_lower:
            return "Player"
        elif "moving" in feature_name_lower or "movement" in feature_name_lower:
            return "Player"
        
        # Interaction context features (5-8)
        elif "action_type" in feature_name_lower or "item_name" in feature_name_lower or "target" in feature_name_lower:
            return "Interaction"
        elif "time_since_interaction" in feature_name_lower:
            return "Phase Context"
        
        # Camera features (9-13)
        elif "camera" in feature_name_lower or "pitch" in feature_name_lower or "yaw" in feature_name_lower:
            return "Camera"
        
        # Inventory features (14-41)
        elif "inventory" in feature_name_lower or "slot" in feature_name_lower:
            return "Inventory"
        
        # Bank features (42-62)
        elif "bank" in feature_name_lower:
            return "Bank"
        
        # Phase context features (63-66)
        elif "phase" in feature_name_lower:
            return "Phase Context"
        
        # Game objects features (67-122)
        elif "game_object" in feature_name_lower or "furnace" in feature_name_lower or "bank_booth" in feature_name_lower:
            return "Game Objects"
        
        # NPC features (123-142)
        elif "npc" in feature_name_lower:
            return "NPCs"
        
        # Tab features (143)
        elif "tab" in feature_name_lower:
            return "Tabs"
        
        # Skills features (144-145)
        elif "level" in feature_name_lower or "xp" in feature_name_lower:
            return "Skills"
        

        
        # Timestamp feature (146)
        elif "timestamp" in feature_name_lower:
            return "Timestamp"
        
        else:
            return "other"
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Training Data Browser - 128 Features with Option 3 Action Implementation", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Updated with OSRS IDs directly, temporal context, clean game state features, and raw action data", font=("Arial", 10))
        subtitle_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Sequence navigation
        ttk.Label(nav_frame, text="Sequence:").pack(side=tk.LEFT)
        
        self.sequence_var = tk.StringVar()
        self.sequence_spinbox = ttk.Spinbox(
            nav_frame, 
            from_=0, 
            to=len(self.input_sequences)-1, 
            textvariable=self.sequence_var,
            width=10,
            command=self.on_sequence_change
        )
        self.sequence_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        
        # Navigation buttons
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_sequence).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next ▶", command=self.next_sequence).pack(side=tk.LEFT, padx=(0, 10))
        
        # Jump to specific sequence
        ttk.Label(nav_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(nav_frame, textvariable=self.jump_var, width=8)
        jump_entry.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(nav_frame, text="Go", command=self.jump_to_sequence).pack(side=tk.LEFT)
        
        # Info label
        self.info_label = ttk.Label(nav_frame, text="", font=("Arial", 10))
        self.info_label.pack(side=tk.RIGHT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind tab selection event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
        # Input sequences tab
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Input Sequences")
        
        # Target sequences tab
        self.target_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.target_frame, text="Target Sequences")
        
        # Feature Analysis tab
        self.feature_analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.feature_analysis_frame, text="Feature Analysis")
        
        # Final Training Data tab (NEW!)
        self.final_training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.final_training_frame, text="Final Training Data")
        
        # Action Tensors tab
        self.action_tensors_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.action_tensors_frame, text="Action Tensors")
        
        # Sequence Alignment tab
        self.sequence_alignment_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sequence_alignment_frame, text="Sequence Alignment")
        
        # Normalization Strategy tab
        self.normalization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.normalization_frame, text="Normalization Strategy")
        
        # Create input sequence display
        self.create_input_display()
        
        # Create target sequence display
        self.create_target_display()
        
        # Create feature analysis display
        self.create_feature_analysis_display()
        
        # Create final training data display
        self.create_final_training_display()
        
        # Create action tensors display
        self.create_action_tensors_display()
        
        # Create sequence alignment display
        self.create_sequence_alignment_display()
        
        # Create normalization strategy display
        self.create_normalization_strategy_display()
    
    def create_input_display(self):
        """Create the input sequence display"""
        # Info frame
        info_frame = ttk.Frame(self.input_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.input_info_label.pack(side=tk.LEFT)
        
        # Sequence info labels above table
        seq_info_frame = ttk.Frame(self.input_frame)
        seq_info_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.seq_num_label = ttk.Label(seq_info_frame, text="", font=("Arial", 10, "bold"))
        self.seq_num_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.seq_shape_label = ttk.Label(seq_info_frame, text="", font=("Arial", 10))
        self.seq_shape_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.seq_dtype_label = ttk.Label(seq_info_frame, text="", font=("Arial", 10))
        self.seq_dtype_label.pack(side=tk.LEFT)
        
        # Export and copy buttons
        export_frame = ttk.Frame(self.input_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Button(export_frame, text="📋 Copy Table to Clipboard", command=self.copy_table_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="💾 Export to CSV", command=self.export_table_to_csv).pack(side=tk.LEFT, padx=(0, 10))
        
        # View mode toggle
        self.show_translations = tk.BooleanVar(value=True)
        self.view_toggle = ttk.Checkbutton(
            export_frame, 
            text="🔍 Show Hash Translations", 
            variable=self.show_translations,
            command=self.toggle_view_mode
        )
        self.view_toggle.pack(side=tk.LEFT, padx=(0, 10))
        
        # Normalization toggle
        self.show_normalized_data = tk.BooleanVar(value=False)
        self.normalization_toggle = ttk.Checkbutton(
            export_frame, 
            text="📊 Show Normalized Data", 
            variable=self.show_normalized_data,
            command=self.toggle_normalization
        )
        self.normalization_toggle.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(export_frame, text="🔍 Search Features", command=self.show_search_dialog).pack(side=tk.LEFT, padx=(0, 10))
        
        # Feature group filter
        filter_frame = ttk.Frame(self.input_frame)
        filter_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Label(filter_frame, text="Filter by Feature Group:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        self.feature_group_filter = tk.StringVar(value="All")
        self.filter_combo = ttk.Combobox(
            filter_frame, 
            textvariable=self.feature_group_filter,
            values=["All", "Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"],
            state="readonly",
            width=20
        )
        self.filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.filter_combo.bind('<<ComboboxSelected>>', self.on_feature_group_filter_changed)
        
        ttk.Button(filter_frame, text="🔄 Refresh", command=self.refresh_input_display).pack(side=tk.LEFT, padx=(0, 10))
        
        # Feature summary frame
        summary_frame = ttk.LabelFrame(self.input_frame, text="Current Feature Summary", padding="5")
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.feature_summary_label = ttk.Label(summary_frame, text="", font=("Arial", 9), justify=tk.LEFT)
        self.feature_summary_label.pack(anchor=tk.W)
        
        # Color legend
        legend_frame = ttk.Frame(self.input_frame)
        legend_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        legend_label = ttk.Label(legend_frame, text="Color Legend:", font=("Arial", 9, "bold"))
        legend_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create legend items
        legend_items = [
            ("🔵", "Player", "Player State"),
            ("🟢", "Interaction", "Interaction Context"),
            ("🔵", "Camera", "Camera"),
            ("🟣", "Inventory", "Inventory Items"),
            ("🔴", "Bank", "Bank"),
            ("🟢", "Phase Context", "Phase Context"),
            ("🟠", "Game Objects", "Game Objects"),
            ("🟠", "NPCs", "NPCs"),
            ("🔴", "Tabs", "Tabs"),
            ("🟡", "Skills", "Skills"),
            ("⚪", "Timestamp", "Timestamp")
        ]
        
        for icon, category, description in legend_items:
            legend_item = ttk.Label(legend_frame, text=f"{icon} {description}", font=("Arial", 8))
            legend_item.pack(side=tk.LEFT, padx=(0, 15))
        
        # Create Treeview for table display
        self.create_input_table()
    
    def create_input_table(self):
        """Create the input sequence table"""
        # Frame for table
        table_frame = ttk.Frame(self.input_frame)
        table_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create Treeview with scrollbars
        self.input_tree = ttk.Treeview(table_frame, show="headings", height=20)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.input_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.input_tree.xview)
        self.input_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.input_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Configure columns
        self.input_tree["columns"] = ["Feature", "Index", "Timestep 0", "Timestep 1", "Timestep 2", "Timestep 3", "Timestep 4", 
                                     "Timestep 5", "Timestep 6", "Timestep 7", "Timestep 8", "Timestep 9"]
        
        # Set column headings
        for col in self.input_tree["columns"]:
            self.input_tree.heading(col, text=col)
            if col == "Feature":
                self.input_tree.column(col, width=200, minwidth=150)
            elif col == "Index":
                self.input_tree.column(col, width=50, minwidth=50)
            else:
                self.input_tree.column(col, width=100, minwidth=80)
        
        # Bind tooltip events
        self.input_tree.bind('<Motion>', self.on_table_motion)
        self.input_tree.bind('<Leave>', self.on_table_leave)
        
        # Tooltip variables
        self.tooltip = None
        self.tooltip_text = ""
    
    def on_table_motion(self, event):
        """Handle mouse motion over table for tooltips"""
        # Get the item and column under the cursor
        item = self.input_tree.identify_row(event.y)
        column = self.input_tree.identify_column(event.x)
        
        if item and column:
            # Get the value at this position
            values = self.input_tree.item(item)['values']
            col_idx = int(column[1]) - 1  # Convert column identifier to index
            
            if 0 <= col_idx < len(values):
                value = values[col_idx]
                
                # Get the original raw value for tooltip
                seq_idx = self.current_sequence
                sequence = self.input_sequences[seq_idx]
                
                if col_idx == 0:  # Feature name column
                    feature_name = value
                    category = self.get_feature_category(feature_name)
                    tooltip_text = f"Feature: {feature_name}\nCategory: {category}"
                elif col_idx == 1:  # Index column
                    tooltip_text = f"Feature Index: {value}"
                else:  # Timestep column
                    feature_idx = int(values[1])  # Get feature index from row
                    raw_value = sequence[col_idx - 2, feature_idx]  # Adjust for Feature and Index columns
                    feature_name = values[0]
                    category = self.get_feature_category(feature_name)
                    
                    # Enhanced tooltip with translation info
                    tooltip_text = f"Timestep {col_idx - 2}\n"
                    tooltip_text += f"Feature: {feature_name}\n"
                    tooltip_text += f"Category: {category}\n"
                    
                    # Show translation if available and enabled, otherwise show raw value
                    if self.show_translations.get():
                        # First try to find translation using the new ID mappings structure
                        translation_found = False
                        if hasattr(self, 'id_mappings') and self.id_mappings:
                            # Check hash mappings first
                            hash_mappings = self.id_mappings.get('hash_mappings', {})
                            try:
                                hash_key = int(float(raw_value))
                                if str(hash_key) in hash_mappings:
                                    original_value = hash_mappings[str(hash_key)]
                                    tooltip_text += f"Value: {original_value}"
                                    translation_found = True
                            except (ValueError, TypeError):
                                pass
                            
                            # Check feature-specific mappings based on feature index
                            if not translation_found and feature_idx is not None:
                                if 0 <= feature_idx <= 4:  # Player features (0-4)
                                    # Check player animation IDs
                                    if 'player_animation_ids' in self.id_mappings:
                                        try:
                                            id_key = int(float(raw_value))
                                            if str(id_key) in self.id_mappings['player_animation_ids']:
                                                name = self.id_mappings['player_animation_ids'][str(id_key)]
                                                tooltip_text += f"Value: {name}"
                                                translation_found = True
                                        except (ValueError, TypeError):
                                            pass
                                elif 14 <= feature_idx <= 41:  # Inventory features (14-41)
                                    # Check inventory slot IDs
                                    if 'inventory_slot_ids' in self.id_mappings:
                                        try:
                                            id_key = int(float(raw_value))
                                            if str(id_key) in self.id_mappings['inventory_slot_ids']:
                                                name = self.id_mappings['inventory_slot_ids'][str(id_key)]
                                                tooltip_text += f"Value: {name}"
                                                translation_found = True
                                        except (ValueError, TypeError):
                                            pass
                                    # Also check item IDs for actual items
                                    if not translation_found and 'item_ids' in self.id_mappings:
                                        try:
                                            id_key = int(float(raw_value))
                                            if str(id_key) in self.id_mappings['item_ids']:
                                                name = self.id_mappings['item_ids'][str(id_key)]
                                                tooltip_text += f"Value: {name}"
                                                translation_found = True
                                        except (ValueError, TypeError):
                                            pass
                            
                            # Check general mappings for other features
                            if not translation_found:
                                for mapping_type in ['item_ids', 'npc_ids', 'object_ids', 'movement_states']:
                                    if mapping_type in self.id_mappings:
                                        try:
                                            id_key = int(float(raw_value))
                                            if str(id_key) in self.id_mappings[mapping_type]:
                                                name = self.id_mappings[mapping_type][str(id_key)]
                                                tooltip_text += f"Value: {name}"
                                                translation_found = True
                                                break
                                        except (ValueError, TypeError):
                                            pass
                        
                        # Fallback: try the old feature-based lookup
                        if not translation_found and feature_idx in self.hash_reverse_lookup:
                            if raw_value in self.hash_reverse_lookup[feature_idx]:
                                original_value = self.hash_reverse_lookup[feature_idx][raw_value]
                                tooltip_text += f"Value: {original_value}"
                                translation_found = True
                        
                        # If no translation found, show raw value
                        if not translation_found:
                            tooltip_text += f"Raw Value: {raw_value}"
                    else:
                        tooltip_text += f"Raw Value: {raw_value}"
                    
                    # Add feature-specific information for new meaningful features
                    if "time_since_interaction" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nTime: {raw_value:.0f}ms since last interaction"
                    elif "phase_start_time" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nTime: {raw_value:.0f}ms since phase start"
                    elif "phase_duration" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nDuration: {raw_value:.0f}ms"
                    elif "mouse_movement_distance" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nDistance: {raw_value:.0f} pixels"
                    elif "mouse_movement_direction" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nDirection: {raw_value:.1f}°"
                    elif "click_x_coordinate" in feature_name or "click_y_coordinate" in feature_name:
                        if raw_value > 0:
                            coord_value = raw_value  # Already in pixels, no conversion needed
                            tooltip_text += f"\nScreen {'X' if 'x' in feature_name else 'Y'}: {coord_value:.0f} pixels"
                    elif "key_press_timing" in feature_name or "key_release_timing" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nTiming: {raw_value:.0f}ms in 600ms window"
                    elif "scroll_intensity" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nIntensity: {raw_value:.0f} scroll units"
                    elif "action_count" in feature_name:
                        if raw_value > 0:
                            tooltip_text += f"\nActions: {raw_value:.0f} in 600ms window"
                
                # Show tooltip if text changed or if tooltip doesn't exist
                if tooltip_text != self.tooltip_text or not self.tooltip:
                    self.tooltip_text = tooltip_text
                    self.show_tooltip(event.x_root, event.y_root, tooltip_text)
    
    def on_table_leave(self, event):
        """Handle mouse leave from table"""
        self.hide_tooltip()
    
    def show_tooltip(self, x, y, text):
        """Show tooltip at specified coordinates with text wrapping"""
        self.hide_tooltip()
        
        # Create tooltip window
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x+10}+{y+10}")
        
        # Wrap text to prevent very long tooltips
        wrapped_text = self.wrap_text(text, max_width=60)
        
        # Create tooltip label with wrapped text
        label = tk.Label(self.tooltip, text=wrapped_text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Tahoma", "8", "normal"), wraplength=400)
        label.pack(padx=5, pady=3)
    
    def wrap_text(self, text, max_width=60):
        """Wrap text to prevent very long lines in tooltips"""
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= max_width:
                wrapped_lines.append(line)
            else:
                # Split long lines at word boundaries
                words = line.split()
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= max_width:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        if current_line:
                            wrapped_lines.append(current_line)
                            current_line = word
                        else:
                            # If a single word is too long, just add it
                            wrapped_lines.append(word)
                
                if current_line:
                    wrapped_lines.append(current_line)
        
        return '\n'.join(wrapped_lines)
    
    def hide_tooltip(self):
        """Hide the tooltip"""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
    
    def create_target_display(self):
        """Create the target sequence display"""
        # Info frame
        info_frame = ttk.Frame(self.target_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.target_info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.target_info_label.pack(side=tk.LEFT)
        
        # Text widget for displaying target data
        self.target_text = scrolledtext.ScrolledText(
            self.target_frame, 
            wrap=tk.NONE, 
            font=("Consolas", 9),
            height=30
        )
        self.target_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def on_sequence_change(self):
        """Handle sequence change from spinbox"""
        try:
            new_seq = int(self.sequence_var.get())
            if 0 <= new_seq < len(self.input_sequences):
                self.current_sequence = new_seq
                self.update_display()
        except ValueError:
            pass
    
    def previous_sequence(self):
        """Go to previous sequence"""
        if self.current_sequence > 0:
            self.current_sequence -= 1
            self.sequence_var.set(str(self.current_sequence))
            self.update_display()
    
    def next_sequence(self):
        """Go to next sequence"""
        if self.current_sequence < len(self.input_sequences) - 1:
            self.current_sequence += 1
            self.sequence_var.set(str(self.current_sequence))
            self.update_display()
    
    def jump_to_sequence(self):
        """Jump to specific sequence number"""
        try:
            seq_num = int(self.jump_var.get())
            # Use the same range as the spinbox (0 to len-1)
            if 0 <= seq_num < len(self.input_sequences):
                self.current_sequence = seq_num
                self.sequence_var.set(str(seq_num))
                self.update_display()
                # Clear the jump entry after successful jump
                self.jump_var.set("")
            else:
                # Show error message and clear invalid input
                self.jump_var.set("")
                # You could add a temporary error message here if desired
        except ValueError:
            # Clear invalid input
            self.jump_var.set("")
    
    def update_display(self):
        """Update the display for current sequence"""
        self.update_input_sequences_display()  # Use the new filtered method
        self.update_target_display()
        self.update_info_labels()
        self.update_feature_summary()
        
        # Also update sequence alignment tab if it exists
        if hasattr(self, 'sequence_tree') and hasattr(self, 'features') and hasattr(self, 'raw_action_data'):
            self.display_sequence_alignment(self.current_sequence)
    
    def update_feature_summary(self, sequence=None):
        """Update the feature summary display"""
        if sequence is None:
            seq_idx = self.current_sequence
            sequence = self.input_sequences[seq_idx]
        else:
            seq_idx = self.current_sequence
        
        # Count features by category
        category_counts = {}
        for feature_idx in range(sequence.shape[1]):
            feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
            category = self.get_feature_category(feature_name)
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create summary text
        summary_parts = []
        for category, count in sorted(category_counts.items()):
            if count > 0:
                category_names = {
                    "Player": "Player State",
                    "Interaction": "Interaction Context", 
                    "Camera": "Camera",
                    "Inventory": "Inventory Items",
                    "Bank": "Bank",
                    "Phase Context": "Phase Context",
                    "Game Objects": "Game Objects",
                    "NPCs": "NPCs",
                    "Tabs": "Tabs",
                    "Skills": "Skills",
                    "Actions": "Input Actions",
                    "Timestamp": "Timestamp",
                    "other": "Other"
                }
                display_name = category_names.get(category, category.title())
                summary_parts.append(f"{display_name}: {count}")
        
        summary_text = f"Sequence {seq_idx} - Feature Distribution: {' | '.join(summary_parts)}"
        self.feature_summary_label.config(text=summary_text)
    
    def format_value_for_display(self, value, feature_idx=None, show_translation=True):
        """Format a value for display in the table with better readability and optional hash translation"""
        if isinstance(value, (int, float)):
            # Convert to float to handle numpy types
            value = float(value)
            
            # If showing translations, try to find a translation using the new ID mappings structure
            if show_translation and hasattr(self, 'id_mappings') and self.id_mappings:
                # Get feature info to determine the correct mapping category
                feature_name = None
                feature_group = None
                data_type = None
                
                if feature_idx is not None:
                    for feature_data in self.feature_mappings:
                        if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                            feature_name = feature_data.get('feature_name')
                            feature_group = feature_data.get('feature_group')
                            data_type = feature_data.get('data_type')
                            break
                
                # Handle boolean values automatically
                if data_type == 'boolean':
                    if value == 1.0:
                        return "true"
                    elif value == 0.0:
                        return "false"
                
                # Check feature-group-specific mappings
                if feature_group and feature_group in self.id_mappings:
                    group_mappings = self.id_mappings[feature_group]
                    
                    # Check hash mappings first (for hashed strings)
                    if 'hash_mappings' in group_mappings:
                        try:
                            hash_key = int(float(value))
                            if str(hash_key) in group_mappings['hash_mappings']:
                                original_value = group_mappings['hash_mappings'][str(hash_key)]
                                return str(original_value)
                        except (ValueError, TypeError):
                            pass
                    
                    # Check specific mapping types based on feature group
                    if feature_group == "Player":
                        if 'player_animation_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['player_animation_ids']:
                                    name = group_mappings['player_animation_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                        if 'player_movement_direction_hashes' in group_mappings:
                            try:
                                hash_key = int(float(value))
                                if str(hash_key) in group_mappings['player_movement_direction_hashes']:
                                    name = group_mappings['player_movement_direction_hashes'][str(hash_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Interaction":
                        for mapping_type in ['action_type_hashes', 'item_name_hashes', 'target_hashes']:
                            if mapping_type in group_mappings:
                                try:
                                    hash_key = int(float(value))
                                    if str(hash_key) in group_mappings[mapping_type]:
                                        name = group_mappings[mapping_type][str(hash_key)]
                                        return str(name)
                                except (ValueError, TypeError):
                                    pass
                    
                    elif feature_group == "Inventory":
                        if 'item_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['item_ids']:
                                    name = group_mappings['item_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                        if 'empty_slot_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['empty_slot_ids']:
                                    name = group_mappings['empty_slot_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Bank":
                        if 'slot_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['slot_ids']:
                                    name = group_mappings['slot_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                        # Only apply boolean mapping to features that are actually boolean
                        if data_type == 'boolean' and 'boolean_states' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['boolean_states']:
                                    name = group_mappings['boolean_states'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Game Objects":
                        if 'object_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['object_ids']:
                                    name = group_mappings['object_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "NPCs":
                        if 'npc_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['npc_ids']:
                                    name = group_mappings['npc_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Tabs":
                        if 'tab_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['tab_ids']:
                                    name = group_mappings['tab_ids'][str(id_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Phase Context":
                        if 'phase_type_hashes' in group_mappings:
                            try:
                                hash_key = int(float(value))
                                if str(hash_key) in group_mappings['phase_type_hashes']:
                                    name = group_mappings['phase_type_hashes'][str(hash_key)]
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                
                # Check global hash mappings as fallback
                if 'Global' in self.id_mappings and 'hash_mappings' in self.id_mappings['Global']:
                    try:
                        hash_key = int(float(value))
                        if str(hash_key) in self.id_mappings['Global']['hash_mappings']:
                            original_value = self.id_mappings['Global']['hash_mappings'][str(hash_key)]
                            return str(original_value)
                    except (ValueError, TypeError):
                        pass
                
                # Fallback: try the old feature-based lookup
                if show_translation and feature_idx is not None and feature_idx in self.hash_reverse_lookup:
                    if value in self.hash_reverse_lookup[feature_idx]:
                        original_value = self.hash_reverse_lookup[feature_idx][value]
                        return str(original_value)
            
            # Handle special cases for non-translated values (applies to all numeric values)
            if value == 0:
                result = "0"
            elif value == -1:
                result = "-1"
            else:
                # If it's a whole number, show as integer (no .0)
                if value == int(value):
                    result = f"{int(value)}"
                else:
                    # For decimal numbers, show without unnecessary trailing zeros
                    result = str(value)
            
            return result
        else:
            return str(value)
    
    def update_input_display(self):
        """Update input sequence display"""
        seq_idx = self.current_sequence
        sequence = self.input_sequences[seq_idx]
        
        # Clear table
        for item in self.input_tree.get_children():
            self.input_tree.delete(item)
        
        # Update sequence info labels above table
        self.seq_num_label.config(text=f"SEQUENCE {seq_idx}")
        self.seq_shape_label.config(text=f"Shape: {sequence.shape}")
        self.seq_dtype_label.config(text=f"Data type: {sequence.dtype}")
        
        # Populate table with feature data
        for feature_idx in range(sequence.shape[1]):
            # Get feature name if available
            feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
            
            # Get values for all timesteps
            values = sequence[:, feature_idx]
            
            # Format values nicely
            formatted_values = []
            for value in values:
                formatted_values.append(self.format_value_for_display(value, feature_idx, self.show_translations.get()))
            
            # Insert into table
            item = self.input_tree.insert("", "end", values=[
                feature_name,
                feature_idx,
                formatted_values[0],
                formatted_values[1],
                formatted_values[2],
                formatted_values[3],
                formatted_values[4],
                formatted_values[5],
                formatted_values[6],
                formatted_values[7],
                formatted_values[8],
                formatted_values[9]
            ])
            
            # Apply color coding based on feature category
            category = self.get_feature_category(feature_name)
            self.apply_row_coloring(item, category)
    
    def apply_row_coloring(self, item, category):
        """Apply color coding to table rows based on feature category"""
        colors = {
            "Player": "#e6f7ff",           # Light cyan for player state
            "Interaction": "#e6ffe6",      # Light green for interaction context
            "Camera": "#f0f0f0",           # Light gray for camera
            "Inventory": "#e6f3ff",        # Light blue for inventory items
            "Bank": "#ffe6e6",             # Light red for bank
            "Phase Context": "#e6ffe6",    # Light green for phase context
            "Game Objects": "#f0e6ff",     # Light purple for game objects
            "NPCs": "#fff2e6",             # Light orange for NPCs
            "Tabs": "#ffe6e6",             # Light red for tabs
            "Skills": "#fff7e6",           # Light yellow for skills
            "Timestamp": "#f5f5f5",        # Very light gray for timestamp
            "other": "#ffffff"             # White for other features
        }
        
        bg_color = colors.get(category, "#ffffff")
        self.input_tree.tag_configure(category, background=bg_color)
        self.input_tree.item(item, tags=(category,))
    
    def update_target_display(self):
        """Update target sequence display with comprehensive view"""
        seq_idx = self.current_sequence
        target = self.target_sequences[seq_idx]
        
        # Clear text widget
        self.target_text.delete(1.0, tk.END)
        
        # Header section
        self.target_text.insert(tk.END, f"🎯 TARGET SEQUENCE {seq_idx}\n", "header")
        self.target_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Summary section
        self.target_text.insert(tk.END, "📊 SEQUENCE SUMMARY\n", "section_header")
        self.target_text.insert(tk.END, "-" * 40 + "\n")
        
        if len(target) > 0:
            action_count = int(target[0])
            self.target_text.insert(tk.END, f"Total Actions: {action_count}\n")
            self.target_text.insert(tk.END, f"Data Length: {len(target)} values\n")
            
            if action_count > 0:
                # Calculate timing info
                first_timing = target[1] if len(target) > 1 else 0
                last_timing = target[1 + (action_count - 1) * 7] if action_count > 0 and len(target) > 1 + (action_count - 1) * 7 else 0
                total_duration = (last_timing - first_timing) * 600 if last_timing > first_timing else 0
                
                self.target_text.insert(tk.END, f"First Action: {first_timing:.6f} ticks ({first_timing * 600:.0f}ms)\n")
                self.target_text.insert(tk.END, f"Last Action: {last_timing:.6f} ticks ({last_timing * 600:.0f}ms)\n")
                self.target_text.insert(tk.END, f"Total Duration: {total_duration:.0f}ms\n")
        else:
            self.target_text.insert(tk.END, "No actions recorded\n")
        
        self.target_text.insert(tk.END, "\n")
        
        # Action breakdown section
        if len(target) > 1 and target[0] > 0:
            action_count = int(target[0])
            self.target_text.insert(tk.END, "🔄 ACTION BREAKDOWN\n", "section_header")
            self.target_text.insert(tk.END, "-" * 40 + "\n")
            
            # Action type statistics
            type_names = {0: 'move', 1: 'click', 2: 'key', 3: 'scroll'}
            type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for action_idx in range(action_count):
                base_idx = 1 + action_idx * 7
                if base_idx + 1 < len(target):
                    action_type = int(target[base_idx + 1])
                    if action_type in type_counts:
                        type_counts[action_type] += 1
            
            self.target_text.insert(tk.END, "Action Type Distribution:\n")
            for action_type, count in type_counts.items():
                if count > 0:
                    action_name = type_names.get(action_type, f"type_{action_type}")
                    percentage = (count / action_count) * 100
                    self.target_text.insert(tk.END, f"  {action_name.capitalize()}: {count} ({percentage:.1f}%)\n")
            
            self.target_text.insert(tk.END, "\n")
            
            # Detailed action list
            self.target_text.insert(tk.END, "📝 DETAILED ACTIONS\n", "section_header")
            self.target_text.insert(tk.END, "-" * 40 + "\n")
            
            for action_idx in range(action_count):
                base_idx = 1 + action_idx * 7
                if base_idx + 6 < len(target):
                    timing = target[base_idx]
                    action_type = target[base_idx + 1]
                    x = target[base_idx + 2]
                    y = target[base_idx + 3]
                    button = target[base_idx + 4]
                    key = target[base_idx + 5]
                    scroll = target[base_idx + 6]
                    
                    action_name = type_names.get(int(action_type), f"type_{action_type}")
                    timing_ms = timing * 600
                    
                    # Action header with timing
                    self.target_text.insert(tk.END, f"Action {action_idx + 1:2d} ", "action_header")
                    self.target_text.insert(tk.END, f"({action_name.upper()}) ", "action_type")
                    self.target_text.insert(tk.END, f"at {timing_ms:6.0f}ms\n", "timing")
                    
                    # Action details
                    self.target_text.insert(tk.END, f"  📍 Position: ({x:7.6f}, {y:7.6f})\n")
                    
                    if int(action_type) == 1:  # Click
                        button_names = {1: "Left", 2: "Right", 3: "Middle"}
                        button_name = button_names.get(int(button), f"Button {button}")
                        self.target_text.insert(tk.END, f"  🖱️  Button: {button_name}\n")
                    elif int(action_type) == 2:  # Key
                        if int(key) != 0:
                            self.target_text.insert(tk.END, f"  ⌨️  Key: {key}\n")
                    elif int(action_type) == 3:  # Scroll
                        if int(scroll) != 0:
                            scroll_dir = "Up" if scroll > 0 else "Down"
                            self.target_text.insert(tk.END, f"  📜 Scroll: {scroll_dir} ({abs(scroll)})\n")
                    
                    # Timing relative to previous action
                    if action_idx > 0:
                        prev_timing = target[1 + (action_idx - 1) * 7]
                        time_diff = (timing - prev_timing) * 600
                        self.target_text.insert(tk.END, f"  ⏱️  Since Previous: {time_diff:6.0f}ms\n")
                    
                    self.target_text.insert(tk.END, "\n")
        
        # Raw data section (collapsible)
        self.target_text.insert(tk.END, "🔍 RAW DATA\n", "section_header")
        self.target_text.insert(tk.END, "-" * 40 + "\n")
        self.target_text.insert(tk.END, "Raw target array values:\n")
        
        # Show all values in a compact format
        for i in range(0, len(target), 7):
            if i == 0:
                self.target_text.insert(tk.END, f"  [{i:2d}]: {target[i]} (action count)\n")
            else:
                action_num = (i - 1) // 7 + 1
                self.target_text.insert(tk.END, f"  [{i:2d}-{min(i+6, len(target)-1):2d}]: Action {action_num} data\n")
                for j in range(i, min(i+7, len(target))):
                    self.target_text.insert(tk.END, f"    [{j:2d}]: {target[j]}\n")
        
        # Configure text styling
        self.configure_target_text_styling()
        
        # Make text read-only
        self.target_text.config(state=tk.DISABLED)
    
    def configure_target_text_styling(self):
        """Configure text styling for the target display"""
        # Configure tags for different text styles
        self.target_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#2E86AB")
        self.target_text.tag_configure("section_header", font=("Arial", 10, "bold"), foreground="#A23B72")
        self.target_text.tag_configure("action_header", font=("Arial", 9, "bold"), foreground="#F18F01")
        self.target_text.tag_configure("action_type", font=("Arial", 9, "bold"), foreground="#C73E1D")
        self.target_text.tag_configure("timing", font=("Arial", 9), foreground="#6B5B95")
    
    def copy_table_to_clipboard(self):
        """Copy current table data to clipboard in a readable format"""
        try:
            # Get current sequence data (respects normalization toggle)
            seq_idx = self.current_sequence
            sequence = self.get_current_input_sequences()[seq_idx]
            
            # Create formatted text
            text_lines = []
            text_lines.append(f"SEQUENCE {seq_idx}")
            text_lines.append(f"Shape: {sequence.shape}")
            text_lines.append(f"Data type: {sequence.dtype}")
            text_lines.append("=" * 80)
            text_lines.append("")
            
            # Add header
            header = "Feature,Index," + ",".join([f"Timestep_{i}" for i in range(10)])
            text_lines.append(header)
            
            # Add data rows
            for feature_idx in range(sequence.shape[1]):
                feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
                
                # Apply filter
                if feature_idx in self.feature_mappings:
                    feature_info = self.feature_mappings[feature_idx]
                    feature_group = feature_info.get('feature_group', 'other')
                    current_filter = self.feature_group_filter.get()
                    if current_filter != "All" and feature_group != current_filter:
                        continue
                
                values = sequence[:, feature_idx]
                
                # Format values nicely using the same formatter
                formatted_values = []
                for value in values:
                    formatted_values.append(self.format_value_for_display(value, feature_idx, self.show_translations.get()))
                
                row = f"{feature_name},{feature_idx}," + ",".join(formatted_values)
                text_lines.append(row)
            
            # Copy to clipboard
            clipboard_text = "\n".join(text_lines)
            pyperclip.copy(clipboard_text)
            
            messagebox.showinfo("Success", "Table data copied to clipboard!\n\nYou can now paste it into Excel, Google Sheets, or any text editor.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
    
    def export_table_to_csv(self):
        """Export current table data to CSV file"""
        try:
            # Get current sequence data
            seq_idx = self.current_sequence
            sequence = self.input_sequences[seq_idx]
            
            # Ask user for save location
            filename = f"sequence_{seq_idx}_data.csv"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    header = ["Feature", "Index"] + [f"Timestep_{i}" for i in range(10)]
                    writer.writerow(header)
                    
                    # Write data rows
                    for feature_idx in range(sequence.shape[1]):
                        feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
                        
                        # Apply filter
                        if feature_idx in self.feature_mappings:
                            feature_info = self.feature_mappings[feature_idx]
                            feature_group = feature_info.get('feature_group', 'other')
                            current_filter = self.feature_group_filter.get()
                            if current_filter != "All" and feature_group != current_filter:
                                continue
                        
                        values = sequence[:, feature_idx]
                        
                        # Format values nicely using the same formatter
                        formatted_values = []
                        for value in values:
                            formatted_values.append(self.format_value_for_display(value, feature_idx, self.show_translations.get()))
                        
                        row = [feature_name, feature_idx] + formatted_values
                        writer.writerow(row)
                
                messagebox.showinfo("Success", f"Data exported to:\n{filepath}\n\nYou can now open this file in Excel, Google Sheets, or any spreadsheet application.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")
    
    def show_search_dialog(self):
        """Show dialog to search for specific features"""
        search_window = tk.Toplevel(self.root)
        search_window.title("Search Features")
        search_window.geometry("400x300")
        search_window.transient(self.root)
        search_window.grab_set()
        
        # Search frame
        search_frame = ttk.Frame(search_window, padding="10")
        search_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(search_frame, text="Search for features by name or index:").pack(pady=(0, 10))
        
        # Search entry
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(pady=(0, 10))
        search_entry.focus()
        
        # Results listbox
        results_frame = ttk.Frame(search_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_listbox = tk.Listbox(results_frame)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_listbox.yview)
        results_listbox.configure(yscrollcommand=results_scrollbar.set)
        
        results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def perform_search():
            """Perform the search"""
            query = search_var.get().lower()
            results_listbox.delete(0, tk.END)
            
            if not query:
                return
            
            # Search through features
            for feature_idx in range(self.input_sequences.shape[2]):
                feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
                
                # Search by feature name or index
                if query in feature_name.lower() or query in str(feature_idx):
                    results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name}")
                    continue
                
                # Search by original values (translations) - check new ID mappings first
                translation_found = False
                if hasattr(self, 'id_mappings') and self.id_mappings:
                    # Check hash mappings
                    hash_mappings = self.id_mappings.get('hash_mappings', {})
                    for hash_value, original_value in hash_mappings.items():
                        if query in str(original_value).lower():
                            results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {original_value}")
                            translation_found = True
                            break
                    
                    # Check feature-specific mappings based on feature index
                    if not translation_found:
                        if 0 <= feature_idx <= 4:  # Player features (0-4)
                            # Check player animation IDs
                            if 'player_animation_ids' in self.id_mappings:
                                for id_value, name in self.id_mappings['player_animation_ids'].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                        elif 14 <= feature_idx <= 41:  # Inventory features (14-41)
                            # Check inventory slot IDs
                            if 'inventory_slot_ids' in self.id_mappings:
                                for id_value, name in self.id_mappings['inventory_slot_ids'].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                            # Also check item IDs for actual items
                            if not translation_found and 'item_ids' in self.id_mappings:
                                for id_value, name in self.id_mappings['item_ids'].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                    
                    # Check general mappings for other features
                    if not translation_found:
                        for mapping_type in ['item_ids', 'npc_ids', 'object_ids', 'movement_states']:
                            if mapping_type in self.id_mappings:
                                for id_value, name in self.id_mappings[mapping_type].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                                if translation_found:
                                    break
                
                # Fallback: search by old feature-based lookup
                if not translation_found and feature_idx in self.hash_reverse_lookup:
                    for hash_value, original_value in self.hash_reverse_lookup[feature_idx].items():
                        if query in str(original_value).lower():
                            results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {original_value}")
                            break
        
        def on_search_enter(event):
            """Handle Enter key in search"""
            perform_search()
        
        def on_result_select(event):
            """Handle result selection"""
            selection = results_listbox.curselection()
            if selection:
                # Extract feature index from selection
                result_text = results_listbox.get(selection[0])
                feature_idx = int(result_text.split(':')[0])
                
                # Jump to this feature in the table
                self.jump_to_feature(feature_idx)
                search_window.destroy()
        
        # Bind events
        search_entry.bind('<Return>', on_search_enter)
        results_listbox.bind('<Double-Button-1>', on_result_select)
        
        # Search button
        ttk.Button(search_frame, text="Search", command=perform_search).pack(pady=(10, 0))
        
        # Initial search
        perform_search()
    
    def toggle_view_mode(self):
        """Toggle between showing hash values and translations"""
        print(f"Toggle view mode: show_translations = {self.show_translations.get()}")
        self.update_input_sequences_display()  # Use the new filtered method
    
    def toggle_normalization(self):
        """Toggle between showing raw and normalized data"""
        print(f"Toggle normalization: show_normalized_data = {self.show_normalized_data.get()}")
        self.refresh_input_display()
    
    def copy_table_to_clipboard(self):
        """Copy current table data to clipboard in a readable format"""
        try:
            # Get current sequence data (respects normalization toggle)
            seq_idx = self.current_sequence
            sequence = self.get_current_input_sequences()[seq_idx]
            
            # Create formatted text
            text_lines = []
            text_lines.append(f"SEQUENCE {seq_idx}")
            text_lines.append(f"Shape: {sequence.shape}")
            text_lines.append(f"Data type: {sequence.dtype}")
            text_lines.append("=" * 80)
            text_lines.append("")
            
            # Add header
            header = "Feature,Index," + ",".join([f"Timestep_{i}" for i in range(10)])
            text_lines.append(header)
            
            # Add data rows
            for feature_idx in range(sequence.shape[1]):
                feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
                
                # Apply filter
                if feature_idx in self.feature_mappings:
                    feature_info = self.feature_mappings[feature_idx]
                    feature_group = feature_info.get('feature_group', 'other')
                    current_filter = self.feature_group_filter.get()
                    if current_filter != "All" and feature_group != current_filter:
                        continue
                
                values = sequence[:, feature_idx]
                
                # Format values nicely using the same formatter
                formatted_values = []
                for value in values:
                    formatted_values.append(self.format_value_for_display(value, feature_idx, self.show_translations.get()))
                
                row = f"{feature_name},{feature_idx}," + ",".join(formatted_values)
                text_lines.append(row)
            
            # Copy to clipboard
            clipboard_text = "\n".join(text_lines)
            pyperclip.copy(clipboard_text)
            
            messagebox.showinfo("Success", "Table data copied to clipboard!\n\nYou can now paste it into Excel, Google Sheets, or any text editor.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
    
    def export_table_to_csv(self):
        """Export current table data to CSV file"""
        try:
            # Get current sequence data
            seq_idx = self.current_sequence
            sequence = self.input_sequences[seq_idx]
            
            # Ask user for save location
            filename = f"sequence_{seq_idx}_data.csv"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    header = ["Feature", "Index"] + [f"Timestep_{i}" for i in range(10)]
                    writer.writerow(header)
                    
                    # Write data rows
                    for feature_idx in range(sequence.shape[1]):
                        feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
                        
                        # Apply filter
                        if feature_idx in self.feature_mappings:
                            feature_info = self.feature_mappings[feature_idx]
                            feature_group = feature_info.get('feature_group', 'other')
                            current_filter = self.feature_group_filter.get()
                            if current_filter != "All" and feature_group != current_filter:
                                continue
                        
                        values = sequence[:, feature_idx]
                        
                        # Format values nicely using the same formatter
                        formatted_values = []
                        for value in values:
                            formatted_values.append(self.format_value_for_display(value, feature_idx, self.show_translations.get()))
                        
                        row = [feature_name, feature_idx] + formatted_values
                        writer.writerow(row)
                
                messagebox.showinfo("Success", f"Data exported to:\n{filepath}\n\nYou can now open this file in Excel, Google Sheets, or any spreadsheet application.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")
    
    def show_search_dialog(self):
        """Show dialog to search for specific features"""
        search_window = tk.Toplevel(self.root)
        search_window.title("Search Features")
        search_window.geometry("400x300")
        search_window.transient(self.root)
        search_window.grab_set()
        
        # Search frame
        search_frame = ttk.Frame(search_window, padding="10")
        search_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(search_frame, text="Search for features by name or index:").pack(pady=(0, 10))
        
        # Search entry
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(pady=(0, 10))
        search_entry.focus()
        
        # Results listbox
        results_frame = ttk.Frame(search_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_listbox = tk.Listbox(results_frame)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_listbox.yview)
        results_listbox.configure(yscrollcommand=results_scrollbar.set)
        
        results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def perform_search():
            """Perform the search"""
            query = search_var.get().lower()
            results_listbox.delete(0, tk.END)
            
            if not query:
                return
            
            # Search through features
            for feature_idx in range(self.input_sequences.shape[2]):
                feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
                
                # Search by feature name or index
                if query in feature_name.lower() or query in str(feature_idx):
                    results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name}")
                    continue
                
                # Search by original values (translations) - check new ID mappings first
                translation_found = False
                if hasattr(self, 'id_mappings') and self.id_mappings:
                    # Check hash mappings
                    hash_mappings = self.id_mappings.get('hash_mappings', {})
                    for hash_value, original_value in hash_mappings.items():
                        if query in str(original_value).lower():
                            results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {original_value}")
                            translation_found = True
                            break
                    
                    # Check feature-specific mappings based on feature index
                    if not translation_found:
                        if 0 <= feature_idx <= 4:  # Player features (0-4)
                            # Check player animation IDs
                            if 'player_animation_ids' in self.id_mappings:
                                for id_value, name in self.id_mappings['player_animation_ids'].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                        elif 14 <= feature_idx <= 41:  # Inventory features (14-41)
                            # Check inventory slot IDs
                            if 'inventory_slot_ids' in self.id_mappings:
                                for id_value, name in self.id_mappings['inventory_slot_ids'].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                            # Also check item IDs for actual items
                            if not translation_found and 'item_ids' in self.id_mappings:
                                for id_value, name in self.id_mappings['item_ids'].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                    
                    # Check general mappings for other features
                    if not translation_found:
                        for mapping_type in ['item_ids', 'npc_ids', 'object_ids', 'movement_states']:
                            if mapping_type in self.id_mappings:
                                for id_value, name in self.id_mappings[mapping_type].items():
                                    if query in str(name).lower():
                                        results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {name}")
                                        translation_found = True
                                        break
                                if translation_found:
                                    break
                
                # Fallback: search by old feature-based lookup
                if not translation_found and feature_idx in self.hash_reverse_lookup:
                    for hash_value, original_value in self.hash_reverse_lookup[feature_idx].items():
                        if query in str(original_value).lower():
                            results_listbox.insert(tk.END, f"{feature_idx:2d}: {feature_name} → {original_value}")
                            break
        
        def on_search_enter(event):
            """Handle Enter key in search"""
            perform_search()
        
        def on_result_select(event):
            """Handle result selection"""
            selection = results_listbox.curselection()
            if selection:
                # Extract feature index from selection
                result_text = results_listbox.get(selection[0])
                feature_idx = int(result_text.split(':')[0])
                
                # Jump to this feature in the table
                self.jump_to_feature(feature_idx)
                search_window.destroy()
        
        # Bind events
        search_entry.bind('<Return>', on_search_enter)
        results_listbox.bind('<Double-Button-1>', on_result_select)
        
        # Search button
        ttk.Button(search_frame, text="Search", command=perform_search).pack(pady=(10, 0))
        
        # Initial search
        perform_search()
    
    def jump_to_feature(self, feature_idx):
        """Jump to a specific feature in the table"""
        # Find the item in the treeview
        for item in self.input_tree.get_children():
            values = self.input_tree.item(item)['values']
            if values[1] == feature_idx:  # Index column
                # Select and scroll to this item
                self.input_tree.selection_set(item)
                self.input_tree.see(item)
                break
    
    def update_info_labels(self):
        """Update info labels"""
        seq_idx = self.current_sequence
        total_sequences = len(self.input_sequences)
        
        # Main info label
        self.info_label.config(text=f"Sequence {seq_idx + 1} of {total_sequences}")
        
        # Input info label
        sequence = self.input_sequences[seq_idx]
        self.input_info_label.config(text=f"Input: {sequence.shape[0]} timesteps × {sequence.shape[1]} features")
        
        # Target info label
        target = self.target_sequences[seq_idx]
        if len(target) > 0:
            action_count = target[0]
            self.target_info_label.config(text=f"Target: {action_count} actions, {len(target)} values")
        else:
            self.target_info_label.config(text=f"Target: 0 actions, {len(target)} values")

    def create_feature_analysis_display(self):
        """Create the feature analysis display with comprehensive feature information"""
        # Create a canvas and scrollbar for the entire feature analysis frame
        canvas = tk.Canvas(self.feature_analysis_frame)
        scrollbar = ttk.Scrollbar(self.feature_analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Info frame
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.feature_analysis_info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.feature_analysis_info_label.pack(side=tk.LEFT)
        
        # Controls frame
        controls_frame = ttk.Frame(scrollable_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Show translations checkbox
        self.show_analysis_translations = tk.BooleanVar(value=True)
        self.analysis_view_toggle = ttk.Checkbutton(
            controls_frame, 
            text="🔍 Show Hash Translations", 
            variable=self.show_analysis_translations,
            command=self.update_feature_analysis_display
        )
        self.analysis_view_toggle.pack(side=tk.LEFT, padx=(0, 20))
        
        # Normalization toggle for analysis
        self.show_analysis_normalized = tk.BooleanVar(value=False)
        self.analysis_normalization_toggle = ttk.Checkbutton(
            controls_frame, 
            text="📊 Show Normalized Data", 
            variable=self.show_analysis_normalized,
            command=self.update_feature_analysis_display
        )
        self.analysis_normalization_toggle.pack(side=tk.LEFT, padx=(0, 20))
        
        # Refresh button
        ttk.Button(controls_frame, text="🔄 Refresh Analysis", command=self.refresh_feature_analysis).pack(side=tk.LEFT, padx=(0, 20))
        
        # Feature group filter for analysis
        ttk.Label(controls_frame, text="Filter by Feature Group:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(20, 10))
        
        self.analysis_feature_group_filter = tk.StringVar(value="All")
        self.analysis_filter_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.analysis_feature_group_filter,
            values=["All", "Player", "Interaction", "Camera", "Inventory", "Bank", "Phase Context", "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"],
            state="readonly",
            width=20
        )
        self.analysis_filter_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.analysis_filter_combo.bind('<<ComboboxSelected>>', self.on_analysis_feature_group_filter_changed)
        
        # Export button
        ttk.Button(controls_frame, text="💾 Export Analysis", command=self.export_feature_analysis).pack(side=tk.LEFT)
        
        # Feature structure summary frame
        summary_frame = ttk.LabelFrame(scrollable_frame, text="Feature Structure Summary", padding="5")
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Create summary labels in a grid
        self.create_feature_summary_grid(summary_frame)
        
        # Create Treeview for feature analysis
        self.create_feature_analysis_table(scrollable_frame)
        
        # Feature details panel for selected features
        self.create_feature_details_panel(scrollable_frame)
        
        # Pack the canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize feature analysis data
        self.analyze_features()
    
    def create_final_training_display(self):
        """Create the final training data display tab"""
        # Info frame
        info_frame = ttk.Frame(self.final_training_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.final_training_info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.final_training_info_label.pack(side=tk.LEFT)
        
        # Navigation frame
        nav_frame = ttk.Frame(self.final_training_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_final_sequence).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next ▶", command=self.next_final_sequence).pack(side=tk.LEFT, padx=(0, 5))
        
        # Jump to sequence
        ttk.Label(nav_frame, text="Jump to:").pack(side=tk.LEFT, padx=(20, 5))
        self.final_sequence_jump_entry = ttk.Entry(nav_frame, width=10)
        self.final_sequence_jump_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Go", command=self.jump_to_final_sequence).pack(side=tk.LEFT)
        
        # Sequence info
        seq_info_frame = ttk.Frame(self.final_training_frame)
        seq_info_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.final_seq_num_label = ttk.Label(seq_info_frame, text="", font=("Arial", 10, "bold"))
        self.final_seq_num_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.final_seq_shape_label = ttk.Label(seq_info_frame, text="", font=("Arial", 10))
        self.final_seq_shape_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Export buttons
        export_frame = ttk.Frame(self.final_training_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Button(export_frame, text="📋 Copy to Clipboard", command=self.copy_final_training_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="💾 Export to JSON", command=self.export_final_training_to_json).pack(side=tk.LEFT, padx=(0, 10))
        
        # Main display frame
        display_frame = ttk.Frame(self.final_training_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create notebook for different views
        self.final_training_notebook = ttk.Notebook(display_frame)
        self.final_training_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Gamestate sequences tab
        self.final_gamestate_tab = ttk.Frame(self.final_training_notebook)
        self.final_training_notebook.add(self.final_gamestate_tab, text="Gamestate Sequences")
        self.create_final_gamestate_display()
        
        # Action input sequences tab
        self.final_action_input_tab = ttk.Frame(self.final_training_notebook)
        self.final_training_notebook.add(self.final_action_input_tab, text="Action Input Sequences")
        self.create_final_action_input_display()
        
        # Action targets tab
        self.final_action_targets_tab = ttk.Frame(self.final_training_notebook)
        self.final_training_notebook.add(self.final_action_targets_tab, text="Action Targets")
        self.create_final_action_targets_display()
        
        # Training pattern overview tab
        self.final_overview_tab = ttk.Frame(self.final_training_notebook)
        self.final_training_notebook.add(self.final_overview_tab, text="Training Pattern Overview")
        self.create_final_overview_display()
        
        # Initialize final training data
        self.current_final_sequence = 0
        self.load_final_training_data()
        self.update_final_training_display()
    
    def create_final_gamestate_display(self):
        """Create the gamestate sequences display"""
        # Create scrollable frame for the treeview
        self.gamestate_scrollable_frame = ScrollableFrame(self.final_gamestate_tab, canvas_height=400)
        self.gamestate_scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for gamestate sequences
        columns = ['Feature'] + [f'Timestep {i}' for i in range(10)]
        
        self.final_gamestate_tree = ttk.Treeview(self.gamestate_scrollable_frame.get_frame(), columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            self.final_gamestate_tree.heading(col, text=col)
            if col == 'Feature':
                self.final_gamestate_tree.column(col, width=200, minwidth=150)
            else:
                self.final_gamestate_tree.column(col, width=120, minwidth=100)
        
        # Pack the tree in the scrollable frame
        self.final_gamestate_tree.pack(fill=tk.BOTH, expand=True)
        
        # Update scroll region after tree is populated
        self.gamestate_scrollable_frame.update_scroll_region()
    
    def create_final_action_input_display(self):
        """Create the action input sequences display - same format as Action Tensors tab"""
        # Info frame
        info_frame = ttk.Frame(self.final_action_input_tab)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add description
        description_label = ttk.Label(
            info_frame,
            text="Action input sequences table: 10 timesteps of action history (t-9 to t-0) for sequence-to-sequence prediction",
            font=("Consolas", 9),
            foreground="blue"
        )
        description_label.pack(pady=(0, 5))
        
        # Create frame for controls
        controls_frame = ttk.Frame(self.final_action_input_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Collapse/Expand all buttons
        ttk.Button(controls_frame, text="🔽 Collapse All", command=self.collapse_all_action_timesteps).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls_frame, text="🔼 Expand All", command=self.expand_all_action_timesteps).pack(side=tk.LEFT, padx=(0, 10))
        
        # Help text
        help_label = ttk.Label(controls_frame, text="💡 Click on any timestep header to expand/collapse it", font=("Arial", 9))
        help_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Create scrollable frame for the treeview
        self.action_input_scrollable_frame = ScrollableFrame(self.final_action_input_tab, canvas_height=400)
        self.action_input_scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for action input sequences - same format as Action Tensors
        columns = ['Feature', 'Index'] + [f'Timestep {i}' for i in range(10)]
        
        self.final_action_input_tree = ttk.Treeview(self.action_input_scrollable_frame.get_frame(), columns=columns, show='headings', height=20)
        
        # Configure columns - make them consistent and smaller
        for col in columns:
            self.final_action_input_tree.heading(col, text=col)
            if col == 'Feature':
                self.final_action_input_tree.column(col, width=150, minwidth=120)
            elif col == 'Index':
                self.final_action_input_tree.column(col, width=50, minwidth=40)
            else:
                # Timestep columns - make them consistent and smaller
                self.final_action_input_tree.column(col, width=80, minwidth=70)
        
        # Pack the tree in the scrollable frame
        self.final_action_input_tree.pack(fill=tk.BOTH, expand=True)
        
        # Update scroll region after tree is populated
        self.action_input_scrollable_frame.update_scroll_region()
        
        # Bind click events for collapsing/expanding timesteps
        self.final_action_input_tree.bind('<Button-1>', self.on_action_input_timestep_click)
    
    def create_final_action_targets_display(self):
        """Create the action targets display - table format similar to Action Input Sequences"""
        # Info frame
        info_frame = ttk.Frame(self.final_action_targets_tab)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add description
        description_label = ttk.Label(
            info_frame,
            text="Action targets table: Action Count (first element) + 8 features per action for the target timestep",
            font=("Consolas", 9),
            foreground="blue"
        )
        description_label.pack(pady=(0, 5))
        
        # Create scrollable frame for the treeview
        self.action_targets_scrollable_frame = ScrollableFrame(self.final_action_targets_tab, canvas_height=400)
        self.action_targets_scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for action targets table
        # Columns: Feature, Index, Action 1, Action 2, Action 3, etc.
        columns = ['Feature', 'Index']
        
        self.final_action_targets_tree = ttk.Treeview(self.action_targets_scrollable_frame.get_frame(), columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            self.final_action_targets_tree.heading(col, text=col)
            if col == 'Feature':
                self.final_action_targets_tree.column(col, width=200, minwidth=150)
            elif col == 'Index':
                self.final_action_targets_tree.column(col, width=50, minwidth=50)
        
        # Pack the tree in the scrollable frame
        self.final_action_targets_tree.pack(fill=tk.BOTH, expand=True)
        
        # Update scroll region after tree is populated
        self.action_targets_scrollable_frame.update_scroll_region()
    
    def create_final_overview_display(self):
        """Create the training pattern overview display"""
        # Create scrollable frame for the text widget
        self.overview_scrollable_frame = ScrollableFrame(self.final_overview_tab, canvas_height=400)
        self.overview_scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for overview
        self.final_overview_text = tk.Text(self.overview_scrollable_frame.get_frame(), wrap=tk.WORD, height=20, font=("Consolas", 10))
        
        # Pack the text widget in the scrollable frame
        self.final_overview_text.pack(fill=tk.BOTH, expand=True)
        
        # Update scroll region after text is populated
        self.overview_scrollable_frame.update_scroll_region()
    
    def load_final_training_data(self):
        """Load the final training data from the separate folder"""
        try:
            # Load gamestate sequences
            gamestate_path = Path("data/final_training_data/gamestate_sequences.npy")
            if gamestate_path.exists():
                self.final_gamestate_sequences = np.load(gamestate_path)
                print(f"✓ Loaded final gamestate sequences: {self.final_gamestate_sequences.shape}")
            else:
                self.final_gamestate_sequences = None
                print("⚠ Final gamestate sequences not found")
            
            # Load action input sequences
            action_input_path = Path("data/final_training_data/action_input_sequences.json")
            if action_input_path.exists():
                with open(action_input_path, 'r') as f:
                    self.final_action_input_sequences = json.load(f)
                print(f"✓ Loaded final action input sequences: {len(self.final_action_input_sequences)} sequences")
            else:
                self.final_action_input_sequences = None
                print("⚠ Final action input sequences not found")
            
            # Load action targets
            action_targets_path = Path("data/final_training_data/action_targets.json")
            if action_targets_path.exists():
                with open(action_targets_path, 'r') as f:
                    self.final_action_targets = json.load(f)
                print(f"✓ Loaded final action targets: {len(self.final_action_targets)} targets")
            else:
                self.final_action_targets = None
                print("⚠ Final action targets not found")
            
            # Load metadata
            metadata_path = Path("data/final_training_data/metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.final_training_metadata = json.load(f)
                print(f"✓ Loaded final training metadata")
            else:
                self.final_training_metadata = None
                print("⚠ Final training metadata not found")
                
        except Exception as e:
            print(f"❌ Error loading final training data: {e}")
            self.final_gamestate_sequences = None
            self.final_action_input_sequences = None
            self.final_action_targets = None
            self.final_training_metadata = None
    
    def update_final_training_display(self):
        """Update the final training data display"""
        if not hasattr(self, 'final_gamestate_sequences') or self.final_gamestate_sequences is None:
            self.final_training_info_label.config(text="❌ Final training data not loaded")
            return
        
        # Update sequence info
        total_sequences = len(self.final_gamestate_sequences)
        self.final_seq_num_label.config(text=f"Sequence {self.current_final_sequence + 1} of {total_sequences}")
        self.final_seq_shape_label.config(text=f"Shape: {self.final_gamestate_sequences.shape}")
        
        # Update gamestate display
        self.update_final_gamestate_display()
        
        # Update action input display
        self.update_final_action_input_display()
        
        # Update action targets display
        self.update_final_action_targets_display()
        
        # Update overview display
        self.update_final_overview_display()
        
        # Update info label
        self.final_training_info_label.config(text=f"✅ Final training data loaded: {total_sequences} sequences")
    
    def update_final_gamestate_display(self):
        """Update the gamestate sequences display"""
        if not hasattr(self, 'final_gamestate_sequences') or self.final_gamestate_sequences is None:
            return
        
        # Clear existing items
        for item in self.final_gamestate_tree.get_children():
            self.final_gamestate_tree.delete(item)
        
        # Get current sequence data
        sequence_data = self.final_gamestate_sequences[self.current_final_sequence]
        
        # Add feature rows
        for feature_idx in range(sequence_data.shape[1]):
            feature_name = self.get_feature_name(feature_idx)
            row_values = [feature_name] + [f"{sequence_data[timestep, feature_idx]:.4f}" for timestep in range(10)]
            self.final_gamestate_tree.insert("", "end", values=row_values)
        
        # Update scroll region after populating data
        if hasattr(self, 'gamestate_scrollable_frame'):
            self.gamestate_scrollable_frame.update_scroll_region()
    
    def update_final_action_input_display(self):
        """Update the action input sequences display - same format as Action Tensors tab"""
        if not hasattr(self, 'final_action_input_sequences') or self.final_action_input_sequences is None:
            return
        
        # Clear existing items
        for item in self.final_action_input_tree.get_children():
            self.final_action_input_tree.delete(item)
        
        # Get current sequence data
        sequence_data = self.final_action_input_sequences[self.current_final_sequence]
        
        # Define feature names and indices - same as Action Tensors tab
        feature_rows = [
            ('Action Count', 0),
            ('Timestamp', 1),
            ('Action Type', 2),
            ('Mouse X', 3),
            ('Mouse Y', 4),
            ('Button', 5),
            ('Key', 6),
            ('Scroll DX', 7),
            ('Scroll DY', 8)
        ]
        
        # Add feature rows - same structure as Action Tensors tab
        for feature_name, feature_idx in feature_rows:
            row_values = [feature_name, feature_idx]
            
            # Add values for each timestep
            for timestep_idx in range(10):
                if timestep_idx < len(sequence_data):
                    timestep_data = sequence_data[timestep_idx]
                    if len(timestep_data) > feature_idx:
                        value = timestep_data[feature_idx]
                        # Format the value nicely
                        if isinstance(value, float):
                            row_values.append(f"{value:.4f}")
                        else:
                            row_values.append(str(value))
                    else:
                        row_values.append("N/A")
                else:
                    row_values.append("N/A")
            
            # Insert the row
            self.final_action_input_tree.insert("", "end", values=row_values)
        
        # Store the sequence data for click handling
        self.current_action_input_sequence_data = sequence_data
        
        # Debug: Print the structure of the first few timesteps
        print(f"DEBUG: Action input sequence structure:")
        for i, timestep_data in enumerate(sequence_data[:3]):  # First 3 timesteps
            print(f"  Timestep {i}: {len(timestep_data)} elements, first few: {timestep_data[:10]}")
        
        # Initialize timestep collapse state - start with all collapsed
        self.action_timestep_collapsed = {i: True for i in range(10)}
        
        # Update the display with the new structure
        self.update_action_input_timestep_display()
        
        # Update scroll region after populating data
        if hasattr(self, 'action_input_scrollable_frame'):
            self.action_input_scrollable_frame.update_scroll_region()
    
    def collapse_all_action_timesteps(self):
        """Collapse all action input timesteps"""
        for i in range(10):
            self.action_timestep_collapsed[i] = True
        self.update_action_input_timestep_display()
        
        # Update scroll region after collapsing
        if hasattr(self, 'action_input_scrollable_frame'):
            self.action_input_scrollable_frame.update_scroll_region()
    
    def expand_all_action_timesteps(self):
        """Expand all action input timesteps"""
        for i in range(10):
            self.action_timestep_collapsed[i] = False
        self.update_action_input_timestep_display()
        
        # Update scroll region after expanding
        if hasattr(self, 'action_input_scrollable_frame'):
            self.action_input_scrollable_frame.update_scroll_region()
    
    def on_action_input_timestep_click(self, event):
        """Handle clicks on action input timestep columns"""
        region = self.final_action_input_tree.identify("region", event.x, event.y)
        if region == "cell":
            column = self.final_action_input_tree.identify_column(event.x)
            if column.startswith('#') and int(column[1:]) > 2:  # Skip Feature and Index columns
                timestep_idx = int(column[1:]) - 3  # Convert column to timestep index
                if 0 <= timestep_idx < 10:
                    # Toggle collapse state
                    self.action_timestep_collapsed[timestep_idx] = not self.action_timestep_collapsed[timestep_idx]
                    self.update_action_input_timestep_display()
                    
                    # Update scroll region after toggling
                    if hasattr(self, 'action_input_scrollable_frame'):
                        self.action_input_scrollable_frame.update_scroll_region()
            
            # Also handle clicks on the timestep header rows
            item = self.final_action_input_tree.identify_row(event.y)
            if item:
                values = self.final_action_input_tree.item(item, 'values')
                if values and values[0].startswith("Timestep "):
                    # Extract timestep number from "Timestep X"
                    try:
                        timestep_num = int(values[0].split()[1])
                        if 0 <= timestep_num < 10:
                            # Toggle collapse state for this timestep
                            self.action_timestep_collapsed[timestep_num] = not self.action_timestep_collapsed[timestep_num]
                            self.update_action_input_timestep_display()
                            
                            # Update scroll region after toggling
                            if hasattr(self, 'action_input_scrollable_frame'):
                                self.action_input_scrollable_frame.update_scroll_region()
                    except (ValueError, IndexError):
                        pass
    
    def update_action_input_timestep_display(self):
        """Update the display based on timestep collapse state - show all 10 action tensors stacked"""
        if not hasattr(self, 'current_action_input_sequence_data'):
            return
        
        # Clear existing items
        for item in self.final_action_input_tree.get_children():
            self.final_action_input_tree.delete(item)
        
        sequence_data = self.current_action_input_sequence_data
        
        # Define feature names and indices
        feature_rows = [
            ('Action Count', 0),
            ('Timestamp', 1),
            ('Action Type', 2),
            ('Mouse X', 3),
            ('Mouse Y', 4),
            ('Button', 5),
            ('Key', 6),
            ('Scroll DX', 7),
            ('Scroll DY', 8)
        ]
        
        # Add all 10 action tensors stacked vertically
        for timestep_idx in range(10):
            if timestep_idx < len(sequence_data):
                timestep_data = sequence_data[timestep_idx]
                
                # Add timestep header row
                if self.action_timestep_collapsed[timestep_idx]:
                    # Collapsed view - just show timestep and action count
                    header_values = [f"Timestep {timestep_idx} [▼]", ""]
                    if len(timestep_data) > 0:
                        action_count = timestep_data[0]
                        header_values.extend([f"{action_count} actions"] + ["..."] * 9)  # 9 more columns
                    else:
                        header_values.extend(["0 actions"] + ["..."] * 9)
                    
                    # Insert header row with different styling
                    header_item = self.final_action_input_tree.insert("", "end", values=header_values, tags=('header_collapsed',))
                    self.final_action_input_tree.tag_configure('header_collapsed', background='lightblue')
                    
                else:
                    # Expanded view - show full action tensor
                    # Add timestep header
                    header_values = [f"Timestep {timestep_idx} [▲]", ""]
                    if len(timestep_data) > 0:
                        action_count = timestep_data[0]
                        header_values.extend([f"{action_count} actions"] + ["..."] * 9)
                    else:
                        header_values.extend(["0 actions"] + ["..."] * 9)
                    
                    header_item = self.final_action_input_tree.insert("", "end", values=header_values, tags=('header_expanded',))
                    self.final_action_input_tree.tag_configure('header_expanded', background='lightgreen')
                    
                    # Create a table for this timestep similar to Action Tensors tab
                    if len(timestep_data) > 0:
                        action_count = timestep_data[0]
                        if action_count > 0:
                            # Create feature rows showing values across all actions in this timestep
                            for feature_name, feature_idx in feature_rows:
                                if feature_idx == 0:  # Action Count
                                    # Action count is a single value, show it only in the first column
                                    values = [action_count] + ["N/A"] * (action_count - 1)
                                else:
                                    # For other features, extract values from the flattened tensor
                                    values = []
                                    for action_idx in range(action_count):
                                        # Calculate position in flattened tensor: 1 + action_idx * 8 + (feature_idx - 1)
                                        tensor_idx = 1 + action_idx * 8 + (feature_idx - 1)
                                        if tensor_idx < len(timestep_data):
                                            value = timestep_data[tensor_idx]
                                            if isinstance(value, float):
                                                values.append(f"{value:.4f}")
                                            else:
                                                values.append(str(value))
                                        else:
                                            values.append("N/A")
                                
                                # Create row for this feature
                                row_values = [f"  {feature_name}", feature_idx] + values
                                self.final_action_input_tree.insert("", "end", values=row_values, tags=('feature',))
                                self.final_action_input_tree.tag_configure('feature', background='white')
            else:
                # Timestep doesn't exist
                header_values = [f"Timestep {timestep_idx}", ""] + ["N/A"] * 10
                header_item = self.final_action_input_tree.insert("", "end", values=header_values, tags=('header',))
                self.final_action_input_tree.tag_configure('header', background='lightgray')
        
        # Update scroll region after populating data
        if hasattr(self, 'action_input_scrollable_frame'):
            self.action_input_scrollable_frame.update_scroll_region()
    
    def update_final_action_targets_display(self):
        """Update the action targets display - table format similar to Action Input Sequences"""
        if not hasattr(self, 'final_action_targets') or self.final_action_targets is None:
            return
        
        # Clear existing items
        for item in self.final_action_targets_tree.get_children():
            self.final_action_targets_tree.delete(item)
        
        # Get current sequence target
        target_data = self.final_action_targets[self.current_final_sequence]
        
        # Dynamically configure columns based on action count
        action_count = target_data[0] if len(target_data) > 0 else 0
        
        # Update treeview columns
        columns = ['Feature', 'Index']
        for action_idx in range(action_count):
            columns.append(f'Action {action_idx + 1}')
        
        self.final_action_targets_tree["columns"] = columns
        
        # Configure column headings and widths
        for col in columns:
            self.final_action_targets_tree.heading(col, text=col)
            if col == 'Feature':
                self.final_action_targets_tree.column(col, width=150, minwidth=120)
            elif col == 'Index':
                self.final_action_targets_tree.column(col, width=50, minwidth=40)
            else:
                # Action columns - make them consistent and smaller
                self.final_action_targets_tree.column(col, width=80, minwidth=70)
        
        # Update scroll region after column configuration
        if hasattr(self, 'action_targets_scrollable_frame'):
            self.action_targets_scrollable_frame.update_scroll_region()
        
        # Create feature rows for the table
        # Each action has 8 features: timestamp, type, x, y, button, key, scroll_dx, scroll_dy
        feature_names = ["Action Count", "Timestamp", "Action Type", "Mouse X", "Mouse Y", "Button", "Key", "Scroll DX", "Scroll DY"]
        
        # For each feature, create a row showing values across all actions
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_idx == 0:  # Action Count
                # Action count is a single value, show it only in the first column
                values = [action_count] + ["N/A"] * (action_count - 1)
            else:
                # For other features, we need to extract values from the flattened tensor
                values = []
                for action_idx in range(action_count):
                    # Calculate position in flattened tensor: 1 + action_idx * 8 + (feature_idx - 1)
                    tensor_idx = 1 + action_idx * 8 + (feature_idx - 1)
                    if tensor_idx < len(target_data):
                        value = target_data[tensor_idx]
                        if isinstance(value, float):
                            values.append(f"{value:.4f}")
                        else:
                            values.append(str(value))
                    else:
                        values.append("N/A")
            
            # Insert row into table
            row_values = [feature_name, feature_idx] + values
            self.final_action_targets_tree.insert("", "end", values=row_values)
        
        # Update scroll region after populating data
        if hasattr(self, 'action_targets_scrollable_frame'):
            self.action_targets_scrollable_frame.update_scroll_region()
    
    def update_final_overview_display(self):
        """Update the training pattern overview display"""
        if not hasattr(self, 'final_training_metadata') or self.final_training_metadata is None:
            return
        
        # Clear existing text
        self.final_overview_text.delete(1.0, tk.END)
        
        # Format metadata for display
        overview_text = "🎯 FINAL TRAINING DATA OVERVIEW\n"
        overview_text += "=" * 50 + "\n\n"
        
        # Training pattern
        if 'training_pattern' in self.final_training_metadata:
            pattern = self.final_training_metadata['training_pattern']
            overview_text += f"📚 TRAINING PATTERN:\n"
            overview_text += f"   Input: {pattern['input']}\n"
            overview_text += f"   Output: {pattern['output']}\n"
            overview_text += f"   Sequence Length: {pattern['sequence_length']}\n"
            overview_text += f"   Total Sequences: {pattern['total_sequences']}\n\n"
        
        # Data structure
        if 'data_structure' in self.final_training_metadata:
            structure = self.final_training_metadata['data_structure']
            overview_text += f"📊 DATA STRUCTURE:\n"
            overview_text += f"   Training Sequences: {structure['n_training_sequences']}\n"
            overview_text += f"   Gamestate Features: {structure['gamestate_features']}\n"
            overview_text += f"   Prediction Target: {structure['prediction_target']}\n\n"
        
        # Feature info
        if 'feature_info' in self.final_training_metadata:
            features = self.final_training_metadata['feature_info']
            overview_text += f"🔧 FEATURE INFO:\n"
            overview_text += f"   Gamestate Features: {features['gamestate_features']}\n"
            overview_text += f"   Action Features per Action: {features['action_features_per_action']}\n"
            overview_text += f"   Action Features: {', '.join(features['action_features'])}\n\n"
        
        # Normalization info
        if 'normalization' in self.final_training_metadata:
            norm = self.final_training_metadata['normalization']
            overview_text += f"📈 NORMALIZATION:\n"
            overview_text += f"   Gamestate: {norm['gamestate_features']}\n"
            overview_text += f"   Actions: {norm['action_features']}\n"
            overview_text += f"   Note: {norm['note']}\n\n"
        
        # Current sequence info
        if hasattr(self, 'final_gamestate_sequences') and self.final_gamestate_sequences is not None:
            sequence_data = self.final_gamestate_sequences[self.current_final_sequence]
            overview_text += f"📍 CURRENT SEQUENCE {self.current_final_sequence + 1}:\n"
            overview_text += f"   Shape: {sequence_data.shape}\n"
            overview_text += f"   Timesteps: 0-9 (history)\n"
            overview_text += f"   Target: Timestep 10 (prediction)\n\n"
        
        # Training flow
        overview_text += f"🔄 TRAINING FLOW:\n"
        overview_text += f"   1. Load sequence {self.current_final_sequence + 1}\n"
        overview_text += f"   2. Extract gamestates t-9 to t-0\n"
        overview_text += f"   3. Extract action history t-9 to t-0\n"
        overview_text += f"   4. Predict actions for timestep t+1\n"
        overview_text += f"   5. Compare with actual target actions\n"
        
        self.final_overview_text.insert(1.0, overview_text)
        
        # Update scroll region after populating data
        if hasattr(self, 'overview_scrollable_frame'):
            self.overview_scrollable_frame.update_scroll_region()
    
    def format_action_sequence(self, action_data):
        """Format action sequence data for display"""
        if not action_data:
            return "No actions"
        
        # Group into 8-feature actions
        actions = []
        for i in range(0, len(action_data), 8):
            if i + 7 < len(action_data):
                action = action_data[i:i+8]
                actions.append(f"[{action[0]:.2f},{action[1]},{action[2]:.2f},{action[3]:.2f},{action[4]},{action[5]:.2f},{action[6]:.2f},{action[7]:.2f}]")
        
        return " | ".join(actions) if actions else "No actions"
    
    def previous_final_sequence(self):
        """Navigate to previous final training sequence"""
        if hasattr(self, 'final_gamestate_sequences') and self.final_gamestate_sequences is not None:
            if self.current_final_sequence > 0:
                self.current_final_sequence -= 1
                self.update_final_training_display()
    
    def next_final_sequence(self):
        """Navigate to next final training sequence"""
        if hasattr(self, 'final_gamestate_sequences') and self.final_gamestate_sequences is not None:
            if self.current_final_sequence < len(self.final_gamestate_sequences) - 1:
                self.current_final_sequence += 1
                self.update_final_training_display()
    
    def jump_to_final_sequence(self):
        """Jump to a specific final training sequence"""
        try:
            sequence_num = int(self.final_sequence_jump_entry.get()) - 1
            if hasattr(self, 'final_gamestate_sequences') and self.final_gamestate_sequences is not None:
                if 0 <= sequence_num < len(self.final_gamestate_sequences):
                    self.current_final_sequence = sequence_num
                    self.update_final_training_display()
                else:
                    messagebox.showerror("Invalid Sequence", f"Sequence must be between 1 and {len(self.final_gamestate_sequences)}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid sequence number")
    
    def copy_final_training_to_clipboard(self):
        """Copy current final training sequence to clipboard"""
        try:
            if hasattr(self, 'final_gamestate_sequences') and self.final_gamestate_sequences is not None:
                sequence_data = {
                    'sequence_number': self.current_final_sequence + 1,
                    'gamestate_sequence': self.final_gamestate_sequences[self.current_final_sequence].tolist(),
                    'action_input_sequence': self.final_action_input_sequences[self.current_final_sequence] if hasattr(self, 'final_action_input_sequences') else None,
                    'action_target': self.final_action_targets[self.current_final_sequence] if hasattr(self, 'final_action_targets') else None
                }
                
                json_text = json.dumps(sequence_data, indent=2)
                pyperclip.copy(json_text)
                messagebox.showinfo("Copied", "Final training sequence copied to clipboard!")
            else:
                messagebox.showwarning("No Data", "No final training data available to copy")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy final training data: {e}")
    
    def export_final_training_to_json(self):
        """Export current final training sequence to JSON file"""
        try:
            if not hasattr(self, 'final_gamestate_sequences') or self.final_gamestate_sequences is None:
                messagebox.showwarning("No Data", "No final training data available to export")
                return
            
            filename = f"final_training_sequence_{self.current_final_sequence + 1}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename,
                title="Save Final Training Sequence"
            )
            
            if filepath:
                sequence_data = {
                    'sequence_number': self.current_final_sequence + 1,
                    'gamestate_sequence': self.final_gamestate_sequences[self.current_final_sequence].tolist(),
                    'action_input_sequence': self.final_action_input_sequences[self.current_final_sequence] if hasattr(self, 'final_action_input_sequences') else None,
                    'action_target': self.final_action_targets[self.current_final_sequence] if hasattr(self, 'final_action_targets') else None
                }
                
                with open(filepath, 'w') as f:
                    json.dump(sequence_data, f, indent=2)
                
                messagebox.showinfo("Exported", f"Final training sequence exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export final training data: {e}")
    
    def create_feature_summary_grid(self, parent_frame):
        """Create a grid layout for feature structure summary"""
        # Feature breakdown data
        feature_breakdown = [
            ("🔵 Player State", "5 features", "world_x, world_y, animation_id, is_moving, movement_direction"),
            ("🟢 Interaction Context", "4 features", "action_type, item_name, target, time_since_interaction"),
            ("🔵 Camera", "5 features", "x, y, z, pitch, yaw"),
            ("🟣 Inventory", "28 features", "using OSRS item IDs directly"),
            ("🔴 Bank", "21 features", "bank_open + 4 materials × 5 features each"),
            ("🟢 Phase Context", "4 features", "type, start_time, duration, gamestates_count"),
            ("🟠 Game Objects", "56 features", "10 objects × 4 + 1 furnace × 4 + 3 bank booths × 4"),
            ("🟠 NPCs", "20 features", "5 NPCs × 4 features each (ID, distance, x, y)"),
            ("🔴 Tabs", "1 feature", "current tab"),
            ("🟡 Skills", "2 features", "crafting level, xp"),

            ("⚪ Timestamp", "1 feature", "raw timestamp")
        ]
        
        # Update total count to reflect 147 features
        total_features = sum([
            5, 4, 5, 28, 21, 4, 56, 20, 1, 2, 1  # Player, Interaction, Camera, Inventory, Bank, Phase Context, Game Objects, NPCs, Tabs, Skills, Timestamp
        ])
        
        # Create grid layout
        for i, (category, count, description) in enumerate(feature_breakdown):
            row = i // 3
            col = (i % 3) * 3
            
            # Category label
            category_label = ttk.Label(parent_frame, text=category, font=("Arial", 9, "bold"))
            category_label.grid(row=row, column=col, sticky=tk.W, padx=(5, 2))
            
            # Count label
            count_label = ttk.Label(parent_frame, text=count, font=("Arial", 9))
            count_label.grid(row=row, column=col+1, sticky=tk.W, padx=(0, 5))
            
            # Description label
            desc_label = ttk.Label(parent_frame, text=description, font=("Arial", 8))
            desc_label.grid(row=row, column=col+2, sticky=tk.W, padx=(0, 15))
    
    def create_feature_analysis_table(self, parent_frame):
        """Create the feature analysis table"""
        # Frame for table
        table_frame = ttk.Frame(parent_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview with scrollbars
        self.feature_analysis_tree = ttk.Treeview(table_frame, show="headings", height=20)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.feature_analysis_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.feature_analysis_tree.xview)
        self.feature_analysis_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.feature_analysis_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Configure columns
        self.feature_analysis_tree["columns"] = [
            "Index", "Feature Name", "Category", "Data Type", "Unique Values", 
            "Top 5 Most Common", "Min Value", "Max Value", "Hashed", "Description"
        ]
        
        # Set column headings and widths
        column_widths = {
            "Index": 50,
            "Feature Name": 200,
            "Category": 120,
            "Data Type": 100,
            "Unique Values": 100,
            "Top 5 Most Common": 300,
            "Min Value": 100,
            "Max Value": 100,
            "Hashed": 80,
            "Description": 300
        }
        
        for col in self.feature_analysis_tree["columns"]:
            self.feature_analysis_tree.heading(col, text=col)
            self.feature_analysis_tree.column(col, width=column_widths.get(col, 100), minwidth=80)
        
        # Bind tooltip events
        self.feature_analysis_tree.bind('<Motion>', self.on_analysis_table_motion)
        self.feature_analysis_tree.bind('<Leave>', self.on_analysis_table_leave)
        
        # Bind feature selection event
        self.feature_analysis_tree.bind('<<TreeviewSelect>>', self.on_feature_selected)
        
        # Tooltip variables
        self.analysis_tooltip = None
        self.analysis_tooltip_text = ""
    
    def on_analysis_table_motion(self, event):
        """Handle mouse motion over analysis table for tooltips"""
        # Get the item and column under the cursor
        item = self.feature_analysis_tree.identify_row(event.y)
        column = self.feature_analysis_tree.identify_column(event.x)
        
        if item and column:
            # Get the value at this position
            values = self.feature_analysis_tree.item(item)['values']
            col_idx = int(column[1]) - 1  # Convert column identifier to index
            
            if 0 <= col_idx < len(values):
                value = values[col_idx]
                feature_idx = int(values[0]) if values[0] is not None else -1
                
                # Create tooltip text based on column
                tooltip_text = ""
                if col_idx == 0:  # Index
                    tooltip_text = f"Feature Index: {value}"
                elif col_idx == 1:  # Feature Name
                    tooltip_text = f"Feature: {value}"
                elif col_idx == 2:  # Category
                    tooltip_text = f"Category: {value}"
                elif col_idx == 3:  # Data Type
                    tooltip_text = f"Data Type: {value}"
                elif col_idx == 4:  # Unique Values
                    tooltip_text = f"Unique Values: {value}"
                elif col_idx == 5:  # Top 5 Most Common
                    tooltip_text = f"Top 5 Most Common Values: {value}"
                elif col_idx == 6:  # Min Value
                    tooltip_text = f"Minimum Value: {value}"
                elif col_idx == 7:  # Max Value
                    tooltip_text = f"Maximum Value: {value}"
                elif col_idx == 8:  # Hashed
                    tooltip_text = f"Hashed: {value}"
                elif col_idx == 9:  # Description
                    tooltip_text = f"Description: {value}"
                
                # Show tooltip if text changed or if tooltip doesn't exist
                if tooltip_text != self.analysis_tooltip_text or not self.analysis_tooltip:
                    self.analysis_tooltip_text = tooltip_text
                    self.show_analysis_tooltip(event.x_root, event.y_root, tooltip_text)
    
    def on_analysis_table_leave(self, event):
        """Handle mouse leave from analysis table"""
        self.hide_analysis_tooltip()
    
    def show_analysis_tooltip(self, x, y, text):
        """Show tooltip for analysis table"""
        self.hide_analysis_tooltip()
        
        # Create tooltip window
        self.analysis_tooltip = tk.Toplevel(self.root)
        self.analysis_tooltip.wm_overrideredirect(True)
        self.analysis_tooltip.wm_geometry(f"+{x+10}+{y+10}")
        
        # Wrap text to prevent very long tooltips
        wrapped_text = self.wrap_text(text, max_width=60)
        
        # Create tooltip label with wrapped text
        label = tk.Label(self.analysis_tooltip, text=wrapped_text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Tahoma", "8", "normal"), wraplength=400)
        label.pack(padx=5, pady=3)
    
    def hide_analysis_tooltip(self):
        """Hide the analysis tooltip"""
        if self.analysis_tooltip:
            self.analysis_tooltip.destroy()
            self.analysis_tooltip = None

    def analyze_features(self):
        """Analyze all features and populate the analysis table"""
        print("Analyzing features...")
        
        # Check if we have data to analyze
        if not hasattr(self, 'input_sequences') or self.input_sequences is None:
            print("No input sequences loaded yet")
            self.feature_analysis_info_label.config(text="No data loaded yet")
            return
        
        # Clear table
        for item in self.feature_analysis_tree.get_children():
            self.feature_analysis_tree.delete(item)
        
        # Analyze each feature
        for feature_idx in range(self.input_sequences.shape[2]):
            feature_name = self.feature_names.get(str(feature_idx), {}).get('feature_name', f'feature_{feature_idx}')
            category = self.get_feature_category(feature_name)
            
            # Apply filter
            current_filter = self.analysis_feature_group_filter.get()
            if current_filter != "All" and category != current_filter:
                continue
            
            # Get all values for this feature across all sequences
            all_values = self.input_sequences[:, :, feature_idx].flatten()  # Flatten to get all values
            
            # Calculate statistics
            unique_values = len(np.unique(all_values))
            min_value = float(np.min(all_values))
            max_value = float(np.max(all_values))
            
            # Determine data type
            data_type = self.determine_data_type(all_values, feature_name)
            
            # Check if hashed
            is_hashed = self.is_feature_hashed(feature_name, all_values)
            
            # Get most common value
            most_common = self.get_most_common_value(all_values, feature_idx)
            
            # Get description
            description = self.get_feature_description(feature_idx, feature_name, category)
            
            # Insert into table
            item = self.feature_analysis_tree.insert("", "end", values=[
                feature_idx,
                feature_name,
                category,
                data_type,
                unique_values,
                most_common,
                f"{min_value:.3f}" if isinstance(min_value, float) else str(min_value),
                f"{max_value:.3f}" if isinstance(max_value, float) else str(max_value),
                "Yes" if is_hashed else "No",
                description
            ])
            
            # Apply color coding
            self.apply_analysis_row_coloring(item, category)
        
        # Update info label
        total_features = self.input_sequences.shape[2]
        total_sequences = self.input_sequences.shape[0]
        current_filter = self.analysis_feature_group_filter.get()
        
        if current_filter != "All":
            # Count filtered features
            filtered_count = len([item for item in self.feature_analysis_tree.get_children()])
            self.feature_analysis_info_label.config(
                text=f"Showing {filtered_count} features from '{current_filter}' group (out of {total_features} total) across {total_sequences} sequences"
            )
        else:
            self.feature_analysis_info_label.config(
                text=f"Analyzed {total_features} features across {total_sequences} sequences"
            )
        
        print(f"Feature analysis completed: {total_features} features analyzed")
    
    def determine_data_type(self, values, feature_name):
        """Determine the data type of a feature from feature mappings instead of hardcoded logic"""
        # Find the feature in our mappings
        for feature_data in self.feature_mappings:
            if isinstance(feature_data, dict) and feature_data.get('feature_name') == feature_name:
                return feature_data.get('data_type', 'Numerical')
        
        # Fallback to hardcoded logic if not found in mappings
        # Check for boolean features
        if feature_name in ['player_is_moving', 'bank_open']:
            return "Boolean"
        
        # Check for categorical features (limited unique values)
        unique_count = len(np.unique(values))
        if unique_count <= 10:
            return "Categorical"
        
        # Check for continuous features
        if feature_name in ['world_x', 'world_y', 'camera_x', 'camera_y', 'camera_z', 'camera_pitch', 'camera_yaw']:
            return "Continuous"
        
        # Check for normalized features (0-1 range)
        if np.min(values) >= 0 and np.max(values) <= 1:
            return "Normalized (0-1)"
        
        # Check for ID features
        if 'id' in feature_name.lower() or 'slot' in feature_name.lower():
            return "ID"
        
        # Check for timestamp features
        if 'timestamp' in feature_name.lower():
            return "Timestamp"
        
        # Check for level/XP features
        if 'level' in feature_name.lower() or 'xp' in feature_name.lower():
            return "Skill"
        
        # Check for hash features (large numbers with high variance)
        if self.is_feature_hashed(feature_name, values):
            return "Hash"
        
        # Default to numerical
        return "Numerical"
    
    def is_feature_hashed(self, feature_name, values):
        """Determine if a feature is hashed"""
        # Features that are definitely hashed
        hashed_features = [
            'action_type', 'item_name', 'target', 'player_movement_direction',
            'phase_type', 'bank_materials_positions'
        ]
        
        if any(hf in feature_name for hf in hashed_features):
            return True
        
        # Check if values look like hashes (large numbers with high variance)
        if len(values) > 0:
            value_range = np.max(values) - np.min(values)
            if value_range > 10000 and np.std(values) > 1000:
                return True
        
        return False
    
    def get_most_common_value(self, values, feature_idx):
        """Get the top 5 most common values for a feature, with translation if available"""
        if len(values) == 0:
            return "N/A"
        
        # Get the top 5 most common values
        unique_values, counts = np.unique(values, return_counts=True)
        top_indices = np.argsort(counts)[-5:][::-1]  # Get top 5, reverse order
        
        result_parts = []
        for i, idx in enumerate(top_indices):
            value = unique_values[idx]
            count = counts[idx]
            
            # Try to get translation if available and translations are enabled
            if self.show_analysis_translations.get():
                # Get the translated value
                translated_value = self.translate_hash_value(feature_idx, value)
                # Extract just the mapped value part (after the arrow)
                if " → " in translated_value:
                    mapped_value = translated_value.split(" → ")[1]
                    formatted_value = self.format_value_for_analysis(mapped_value)
                else:
                    # If no translation found, use the original value
                    formatted_value = self.format_value_for_analysis(value)
                result_parts.append(f"{formatted_value} ({count})")
            else:
                # Format raw value without trailing zeros
                formatted_value = self.format_value_for_analysis(value)
                result_parts.append(f"{formatted_value} ({count})")
        
        return " | ".join(result_parts)
    
    def format_value_for_analysis(self, value):
        """Format a value for display in analysis table, removing trailing zeros"""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        elif isinstance(value, (float, np.floating)):
            # Remove trailing zeros and unnecessary decimal places
            if value == int(value):
                return str(int(value))
            else:
                # Format to remove trailing zeros
                formatted = f"{value:.6f}".rstrip('0').rstrip('.')
                return formatted
        else:
            return str(value)
    
    def get_feature_description(self, feature_idx, feature_name, category):
        """Get a description for a feature"""
        descriptions = {
            "Player": "Player state information including position, animation, and movement",
            "Interaction": "Information about the last player interaction",
            "Camera": "Camera position and orientation in 3D space",
            "Inventory": "Inventory item information using OSRS item IDs",
            "Bank": "Bank state and material information",
            "Phase Context": "Game phase and state context information",
            "Game Objects": "Game world objects using OSRS object IDs",
            "NPCs": "Non-player characters using OSRS NPC IDs",
            "Tabs": "UI tab information",
            "Skills": "Player skill levels and experience points",
            "Actions": "Player input actions including mouse, keyboard, and scroll",
            "Timestamp": "Raw timestamp value for temporal alignment"
        }
        
        return descriptions.get(category, f"Feature {feature_idx}: {feature_name}")
    
    def apply_analysis_row_coloring(self, item, category):
        """Apply color coding to analysis table rows"""
        colors = {
            "Player": "#e6f7ff",           # Light cyan for player state
            "Interaction": "#e6ffe6",      # Light green for interaction context
            "Camera": "#f0f0f0",           # Light gray for camera
            "Inventory": "#e6f3ff",        # Light blue for inventory items
            "Bank": "#ffe6e6",             # Light red for bank
            "Phase Context": "#e6ffe6",    # Light green for phase context
            "Game Objects": "#f0e6ff",     # Light purple for game objects
            "NPCs": "#fff2e6",             # Light orange for NPCs
            "Tabs": "#ffe6e6",             # Light red for tabs
            "Skills": "#fff7e6",           # Light yellow for skills
            "Timestamp": "#f5f5f5",        # Very light gray for timestamp
            "other": "#ffffff"             # White for other features
        }
        
        bg_color = colors.get(category, "#ffffff")
        self.feature_analysis_tree.tag_configure(category, background=bg_color)
        self.feature_analysis_tree.item(item, tags=(category,))
    
    def update_feature_analysis_display(self):
        """Update the feature analysis display when translations toggle changes"""
        self.analyze_features()
    
    def refresh_feature_analysis(self):
        """Refresh the feature analysis"""
        self.analyze_features()
    
    def export_feature_analysis(self):
        """Export feature analysis to CSV"""
        try:
            # Ask user for save location
            filename = "feature_analysis.csv"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filepath:
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    header = ["Index", "Feature Name", "Category", "Data Type", "Unique Values", 
                             "Top 5 Most Common", "Min Value", "Max Value", "Hashed", "Description"]
                    writer.writerow(header)
                    
                    # Write data rows
                    for item in self.feature_analysis_tree.get_children():
                        values = self.feature_analysis_tree.item(item)['values']
                        writer.writerow(values)
                
                messagebox.showinfo("Success", f"Feature analysis exported to:\n{filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")

    def on_tab_changed(self, event):
        """Handle tab selection change to update feature analysis display"""
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab == "Feature Analysis":
            self.update_feature_analysis_display()

    def on_feature_group_filter_changed(self, event):
        """Handle feature group filter change"""
        self.refresh_input_display()
    
    def refresh_input_display(self):
        """Refresh the input sequences display with current filter"""
        self.update_input_sequences_display()
    
    def update_input_sequences_display(self):
        """Update input sequences display with current sequence and filter"""
        if self.input_sequences is None:
            print("No input sequences loaded yet")
            return
        
        # Clear existing items
        for item in self.input_tree.get_children():
            self.input_tree.delete(item)
        
        seq_idx = self.current_sequence
        sequence = self.get_current_input_sequences()[seq_idx]
        
        # Get current filter
        current_filter = self.feature_group_filter.get()
        
        # Update sequence info labels
        self.seq_num_label.config(text=f"Sequence: {seq_idx}")
        self.seq_shape_label.config(text=f"Shape: {sequence.shape}")
        self.seq_dtype_label.config(text=f"Type: {sequence.dtype}")
        
        # Update feature summary
        self.update_feature_summary(sequence)
        
        # Count filtered features
        filtered_count = len([item for item in self.input_tree.get_children()])
        total_features = sequence.shape[1]
        current_filter = self.feature_group_filter.get()
        
        if current_filter != "All":
            self.input_info_label.config(text=f"Showing {filtered_count} features from '{current_filter}' group (out of {total_features} total)")
        else:
            self.input_info_label.config(text=f"Showing all {total_features} features")
        
        # Populate table with filtered features
        features_added = 0
        for feature_idx in range(sequence.shape[1]):
            # Find the feature info in feature_mappings list
            feature_info = None
            for feature_data in self.feature_mappings:
                if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                    feature_info = feature_data
                    break
            
            if feature_info:
                feature_name = feature_info.get('feature_name', f'Feature_{feature_idx}')
                feature_group = feature_info.get('feature_group', 'other')
                
                # Apply filter
                if current_filter != "All" and feature_group != current_filter:
                    continue
                
                values = sequence[:, feature_idx]
                
                # Format values nicely
                formatted_values = []
                for value in values:
                    formatted_values.append(self.format_value_for_display(value, feature_idx, self.show_translations.get()))
                
                # Insert into table
                item = self.input_tree.insert("", "end", values=[
                    feature_name,
                    feature_idx,
                    formatted_values[0],
                    formatted_values[1],
                    formatted_values[2],
                    formatted_values[3],
                    formatted_values[4],
                    formatted_values[5],
                    formatted_values[6],
                    formatted_values[7],
                    formatted_values[8],
                    formatted_values[9]
                ])
                
                # Apply color coding based on feature category
                self.apply_row_coloring(item, feature_group)
                features_added += 1
        
        # Update info label with filtered count
        if current_filter != "All":
            self.input_info_label.config(text=f"Showing {features_added} features from '{current_filter}' group (out of {total_features} total)")
        else:
            self.input_info_label.config(text=f"Showing all {features_added} features")

    def on_analysis_feature_group_filter_changed(self, event):
        """Handle feature group filter change for analysis"""
        self.refresh_feature_analysis()

    def create_feature_details_panel(self, parent_frame):
        """Create the feature details panel for showing selected feature information"""
        # Create a frame for the details panel
        details_frame = ttk.LabelFrame(parent_frame, text="Feature Details", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        
        # Feature info section
        info_frame = ttk.Frame(details_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.feature_name_label = ttk.Label(info_frame, text="Select a feature to view details", font=("Arial", 12, "bold"))
        self.feature_name_label.pack(side=tk.LEFT)
        
        self.feature_group_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.feature_group_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.feature_data_type_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.feature_data_type_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(details_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_all_values_btn = ttk.Button(
            buttons_frame, 
            text="📊 Show All Values", 
            command=self.show_feature_all_values,
            state="disabled"
        )
        self.show_all_values_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.show_timeline_btn = ttk.Button(
            buttons_frame, 
            text="📈 Show Timeline Graph", 
            command=self.show_feature_timeline,
            state="disabled"
        )
        self.show_timeline_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.show_distribution_btn = ttk.Button(
            buttons_frame, 
            text="📊 Show Distribution", 
            command=self.show_feature_distribution,
            state="disabled"
        )
        self.show_distribution_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create a horizontal frame to hold values and visualization side by side
        content_frame = ttk.Frame(details_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Values display area (left side)
        values_frame = ttk.LabelFrame(content_frame, text="Feature Values", padding="5")
        values_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create Treeview for values display
        self.values_tree = ttk.Treeview(values_frame, show="headings", height=8)
        
        # Create scrollbars for values
        vsb_values = ttk.Scrollbar(values_frame, orient="vertical", command=self.values_tree.yview)
        hsb_values = ttk.Scrollbar(values_frame, orient="horizontal", command=self.values_tree.xview)
        self.values_tree.configure(yscrollcommand=vsb_values.set, xscrollcommand=hsb_values.set)
        
        # Grid layout for values
        self.values_tree.grid(row=0, column=0, sticky="nsew")
        vsb_values.grid(row=0, column=1, sticky="ns")
        hsb_values.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        values_frame.columnconfigure(0, weight=1)
        values_frame.rowconfigure(0, weight=1)
        
        # Configure columns for values display
        self.values_tree["columns"] = ["Sequence", "Timestep", "Value", "Translated Value"]
        
        # Set column headings and widths
        for col in self.values_tree["columns"]:
            self.values_tree.heading(col, text=col)
            if col == "Sequence":
                self.values_tree.column(col, width=80, minwidth=80)
            elif col == "Timestep":
                self.values_tree.column(col, width=80, minwidth=80)
            elif col == "Value":
                self.values_tree.column(col, width=120, minwidth=120)
            else:
                self.values_tree.column(col, width=200, minwidth=150)
        
        # Mini visualization frame (right side)
        viz_frame = ttk.LabelFrame(content_frame, text="Mini Visualization", padding="5")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create a frame for the mini chart
        self.mini_chart_frame = ttk.Frame(viz_frame)
        self.mini_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize selected feature
        self.selected_feature_idx = None
        self.selected_feature_data = None
        
        # Initialize mini chart
        self.mini_chart_canvas = None
        self.mini_chart_figure = None
    
    def on_feature_selected(self, event):
        """Handle feature selection in the analysis table"""
        selection = self.feature_analysis_tree.selection()
        if not selection:
            return
        
        # Get the selected item
        item = selection[0]
        values = self.feature_analysis_tree.item(item)['values']
        
        print(f"Feature selected: {values}")  # Debug print
        
        if values and len(values) > 0:
            feature_idx = int(values[0])
            feature_name = values[1]
            feature_group = values[2]
            data_type = values[3]
            
            print(f"Processing feature {feature_idx}: {feature_name} ({feature_group})")  # Debug print
            
            # Store selected feature info
            self.selected_feature_idx = feature_idx
            self.selected_feature_data = {
                'name': feature_name,
                'group': feature_group,
                'data_type': data_type
            }
            
            # Update labels
            self.feature_name_label.config(text=f"Feature: {feature_name}")
            self.feature_group_label.config(text=f"Group: {feature_group}")
            self.feature_data_type_label.config(text=f"Type: {data_type}")
            
            # Enable buttons
            self.show_all_values_btn.config(state="normal")
            self.show_timeline_btn.config(state="normal")
            self.show_distribution_btn.config(state="normal")
            
            # Show initial values for this feature
            self.show_feature_all_values()
            
            # Create mini visualization
            self.create_mini_visualization()
            
            print(f"Feature details updated for feature {feature_idx}")  # Debug print
    
    def show_feature_all_values(self):
        """Show all values for the selected feature across all sequences and timesteps"""
        if self.selected_feature_idx is None:
            return
        
        # Clear existing values
        for item in self.values_tree.get_children():
            self.values_tree.delete(item)
        
        feature_idx = self.selected_feature_idx
        feature_name = self.selected_feature_data['name']
        
        # Get all values for this feature across all sequences and timesteps
        all_values = []
        for seq_idx in range(self.input_sequences.shape[0]):
            sequence = self.input_sequences[seq_idx]
            for timestep in range(sequence.shape[0]):
                value = sequence[timestep, feature_idx]
                all_values.append((seq_idx, timestep, value))
        
        # Sort by sequence, then by timestep
        all_values.sort(key=lambda x: (x[0], x[1]))
        
        # Add to the values tree
        for seq_idx, timestep, value in all_values:
            # Format the raw value
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.6f}".rstrip('0').rstrip('.')
            else:
                formatted_value = str(value)
            
            # Get translated value if translations are enabled
            if self.show_analysis_translations.get():
                translated_value = self.translate_hash_value(feature_idx, value)
                if " → " in translated_value:
                    translated_value = translated_value.split(" → ")[1]
            else:
                translated_value = formatted_value
            
            # Insert into values tree
            self.values_tree.insert("", "end", values=[
                f"Seq {seq_idx}",
                f"T{timestep}",
                formatted_value,
                translated_value
            ])
        
        # Update the values frame title to show count
        values_frame = self.values_tree.master
        values_frame.config(text=f"Feature Values ({len(all_values)} total values)")
        
        # Create mini visualization
        self.create_mini_visualization()
    
    def show_feature_timeline(self):
        """Show a timeline graph of the selected feature values over time"""
        if self.selected_feature_idx is None:
            return
        
        # Create a new window for the timeline
        timeline_window = tk.Toplevel(self.root)
        timeline_window.title(f"Timeline: {self.selected_feature_data['name']}")
        timeline_window.geometry("800x600")
        timeline_window.transient(self.root)
        
        # Create matplotlib figure
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import numpy as np
            
            # Create figure and canvas
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvasTkAgg(fig, timeline_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create subplot
            ax = fig.add_subplot(111)
            
            feature_idx = self.selected_feature_idx
            
            # Plot the feature values over time using the full features array
            if hasattr(self, 'features') and self.features is not None:
                # Use the full features array for better visualization
                all_values = self.features[:, feature_idx]
                timesteps = range(len(all_values))
                
                # Plot the full timeline
                ax.plot(timesteps, all_values, 'b-', linewidth=1, alpha=0.8, label='All Gamestates')
                
                # Add sequence markers (every 10th gamestate)
                seq_markers = timesteps[::10]
                seq_values = all_values[::10]
                ax.scatter(seq_markers, seq_values, c='red', s=30, alpha=0.7, label='Sequence Boundaries')
                
                # Add statistics lines
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                ax.axhline(y=mean_val, color='orange', linestyle='--', alpha=0.6, label=f'Mean: {mean_val:.3f}')
                ax.axhline(y=mean_val + std_val, color='green', linestyle=':', alpha=0.5, label=f'+1σ: {mean_val + std_val:.3f}')
                ax.axhline(y=mean_val - std_val, color='green', linestyle=':', alpha=0.5, label=f'-1σ: {mean_val - std_val:.3f}')
                
            else:
                # Fallback to input sequences if features not available
                for seq_idx in range(min(10, self.input_sequences.shape[0])):  # Limit to first 10 sequences for clarity
                    sequence = self.input_sequences[seq_idx]
                    timesteps = range(sequence.shape[0])
                    values = sequence[:, feature_idx]
                    
                    # Plot this sequence
                    ax.plot(timesteps, values, marker='o', markersize=4, linewidth=1, 
                           label=f'Sequence {seq_idx}', alpha=0.7)
            
            # Customize the plot
            if hasattr(self, 'features') and self.features is not None:
                ax.set_xlabel('Gamestate Index')
                ax.set_ylabel('Feature Value')
                ax.set_title(f'Feature: {self.selected_feature_data["name"]}\nGroup: {self.selected_feature_data["group"]}\nFull Timeline: {len(self.features)} gamestates')
            else:
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Feature Value')
                ax.set_title(f'Feature: {self.selected_feature_data["name"]}\nGroup: {self.selected_feature_data["group"]}')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add controls
            controls_frame = ttk.Frame(timeline_window)
            controls_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(controls_frame, text="Max Sequences to Show:").pack(side=tk.LEFT)
            seq_var = tk.IntVar(value=10)
            seq_spinbox = ttk.Spinbox(controls_frame, from_=1, to=50, textvariable=seq_var, width=10)
            seq_spinbox.pack(side=tk.LEFT, padx=(5, 10))
            
            def update_plot():
                # Clear the plot
                ax.clear()
                
                # Replot with new sequence limit
                if hasattr(self, 'features') and self.features is not None:
                    # Use the full features array
                    all_values = self.features[:, feature_idx]
                    timesteps = range(len(all_values))
                    
                    # Plot the full timeline
                    ax.plot(timesteps, all_values, 'b-', linewidth=1, alpha=0.8, label='All Gamestates')
                    
                    # Add sequence markers (every 10th gamestate)
                    seq_markers = timesteps[::10]
                    seq_values = all_values[::10]
                    ax.scatter(seq_markers, seq_values, c='red', s=30, alpha=0.7, label='Sequence Boundaries')
                    
                    # Add statistics lines
                    mean_val = np.mean(all_values)
                    std_val = np.std(all_values)
                    ax.axhline(y=mean_val, color='orange', linestyle='--', alpha=0.6, label=f'Mean: {mean_val:.3f}')
                    ax.axhline(y=mean_val + std_val, color='green', linestyle=':', alpha=0.5, label=f'+1σ: {mean_val + std_val:.3f}')
                    ax.axhline(y=mean_val - std_val, color='green', linestyle=':', alpha=0.5, label=f'-1σ: {mean_val - std_val:.3f}')
                    
                    ax.set_xlabel('Gamestate Index')
                    ax.set_ylabel('Feature Value')
                    ax.set_title(f'Feature: {self.selected_feature_data["name"]}\nGroup: {self.selected_feature_data["group"]}\nFull Timeline: {len(self.features)} gamestates')
                else:
                    # Fallback to input sequences
                    max_seqs = min(seq_var.get(), self.input_sequences.shape[0])
                    for seq_idx in range(max_seqs):
                        sequence = self.input_sequences[seq_idx]
                        timesteps = range(sequence.shape[0])
                        values = sequence[:, feature_idx]
                        
                        ax.plot(timesteps, values, marker='o', markersize=4, linewidth=1, 
                               label=f'Sequence {seq_idx}', alpha=0.7)
                    
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('Feature Value')
                    ax.set_title(f'Feature: {self.selected_feature_data["name"]}\nGroup: {self.selected_feature_data["group"]}')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                canvas.draw()
            
            ttk.Button(controls_frame, text="Update Plot", command=update_plot).pack(side=tk.LEFT)
            
        except ImportError:
            # Fallback if matplotlib is not available
            error_label = ttk.Label(timeline_window, text="Matplotlib is required for timeline visualization.\nPlease install it with: pip install matplotlib", 
                                  font=("Arial", 12), justify=tk.CENTER)
            error_label.pack(expand=True)
    
    def show_feature_distribution(self):
        """Show distribution graphs for the selected feature"""
        if self.selected_feature_idx is None:
            return
        
        # Create a new window for the distribution
        dist_window = tk.Toplevel(self.root)
        dist_window.title(f"Distribution: {self.selected_feature_data['name']}")
        dist_window.geometry("1000x700")
        dist_window.transient(self.root)
        
        # Create matplotlib figure
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import numpy as np
            
            # Create figure and canvas
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvasTkAgg(fig, dist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            feature_idx = self.selected_feature_idx
            
            # Get all values for this feature
            all_values = []
            for seq_idx in range(self.input_sequences.shape[0]):
                sequence = self.input_sequences[seq_idx]
                values = sequence[:, feature_idx]
                all_values.extend(values)
            
            all_values = np.array(all_values)
            
            # Create subplots
            ax1 = fig.add_subplot(221)  # Histogram
            ax2 = fig.add_subplot(222)  # Box plot
            ax3 = fig.add_subplot(223)  # Sequence-wise box plot
            ax4 = fig.add_subplot(224)  # Value range over time
            
            # 1. Histogram
            ax1.hist(all_values, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Feature Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Value Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 2. Box plot
            ax2.boxplot(all_values)
            ax2.set_ylabel('Feature Value')
            ax2.set_title('Value Statistics')
            ax2.grid(True, alpha=0.3)
            
            # 3. Sequence-wise box plot (first 10 sequences)
            seq_data = []
            seq_labels = []
            for seq_idx in range(min(10, self.input_sequences.shape[0])):
                sequence = self.input_sequences[seq_idx]
                values = sequence[:, feature_idx]
                seq_data.append(values)
                seq_labels.append(f'Seq {seq_idx}')
            
            if seq_data:
                ax3.boxplot(seq_data, labels=seq_labels)
                ax3.set_ylabel('Feature Value')
                ax3.set_title('Value Distribution by Sequence')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # 4. Value range over time
            timestep_data = []
            timestep_labels = []
            for timestep in range(self.input_sequences.shape[1]):
                values = self.input_sequences[:, timestep, feature_idx]
                timestep_data.append(values)
                timestep_labels.append(f'T{timestep}')
            
            if timestep_data:
                ax4.boxplot(timestep_data, labels=timestep_labels)
                ax4.set_ylabel('Feature Value')
                ax4.set_title('Value Distribution by Timestep')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
            
            # Add statistics text
            fig.suptitle(f'Feature: {self.selected_feature_data["name"]}\nGroup: {self.selected_feature_data["group"]}', fontsize=14)
            
            # Add controls
            controls_frame = ttk.Frame(dist_window)
            controls_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(controls_frame, text="Max Sequences for Sequence Plot:").pack(side=tk.LEFT)
            seq_var = tk.IntVar(value=10)
            seq_spinbox = ttk.Spinbox(controls_frame, from_=1, to=50, textvariable=seq_var, width=10)
            seq_spinbox.pack(side=tk.LEFT, padx=(5, 10))
            
            def update_distribution():
                # Clear the plot
                ax3.clear()
                
                # Replot sequence-wise box plot
                seq_data = []
                seq_labels = []
                max_seqs = min(seq_var.get(), self.input_sequences.shape[0])
                for seq_idx in range(max_seqs):
                    sequence = self.input_sequences[seq_idx]
                    values = sequence[:, feature_idx]
                    seq_data.append(values)
                    seq_labels.append(f'Seq {seq_idx}')
                
                if seq_data:
                    ax3.boxplot(seq_data, labels=seq_labels)
                    ax3.set_ylabel('Feature Value')
                    ax3.set_title('Value Distribution by Sequence')
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.grid(True, alpha=0.3)
                
                canvas.draw()
            
            ttk.Button(controls_frame, text="Update Plot", command=update_distribution).pack(side=tk.LEFT)
            
            # Add statistics summary
            stats_frame = ttk.LabelFrame(dist_window, text="Statistics Summary", padding="5")
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            stats_text = f"""
            Total Values: {len(all_values)}
            Mean: {np.mean(all_values):.6f}
            Median: {np.median(all_values):.6f}
            Std Dev: {np.std(all_values):.6f}
            Min: {np.min(all_values):.6f}
            Max: {np.max(all_values):.6f}
            Unique Values: {len(np.unique(all_values))}
            """
            
            stats_label = ttk.Label(stats_frame, text=stats_text, font=("Consolas", 9), justify=tk.LEFT)
            stats_label.pack(anchor=tk.W)
            
        except ImportError:
            # Fallback if matplotlib is not available
            error_label = ttk.Label(dist_window, text="Matplotlib is required for distribution visualization.\nPlease install it with: pip install matplotlib", 
                                  font=("Arial", 12), justify=tk.CENTER)
            error_label.pack(expand=True)

    def create_action_tensors_display(self):
        """Create the action tensors visualization display"""
        # Info frame
        info_frame = ttk.Frame(self.action_tensors_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.action_info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.action_info_label.pack(side=tk.LEFT)
        
        # Controls frame
        controls_frame = ttk.Frame(self.action_tensors_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Gamestate selector
        ttk.Label(controls_frame, text="Gamestate:").pack(side=tk.LEFT)
        self.action_gamestate_var = tk.StringVar()
        self.action_gamestate_spinbox = ttk.Spinbox(
            controls_frame,
            from_=0,
            to=len(self.raw_action_data)-1 if self.raw_action_data else 0,
            textvariable=self.action_gamestate_var,
            width=10,
            command=self.on_action_gamestate_change
        )
        self.action_gamestate_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        
        # Navigation buttons
        ttk.Button(controls_frame, text="◀ Previous", command=self.previous_action_gamestate).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Next ▶", command=self.next_action_gamestate).pack(side=tk.LEFT, padx=(0, 10))
        
        # Action type filter
        ttk.Label(controls_frame, text="Filter by Action Type:").pack(side=tk.LEFT)
        self.action_type_filter = tk.StringVar(value="All")
        self.action_filter_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.action_type_filter,
            values=["All", "mouse_movements", "clicks", "key_presses", "key_releases", "scrolls"],
            state="readonly",
            width=15
        )
        self.action_filter_combo.pack(side=tk.LEFT, padx=(5, 10))
        self.action_filter_combo.bind('<<ComboboxSelected>>', self.on_action_type_filter_changed)
        
        # Export button
        ttk.Button(controls_frame, text="📋 Copy to Clipboard", command=self.copy_action_data_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))
        
        # Action patterns button
        ttk.Button(controls_frame, text="📊 Show Patterns", command=self.show_action_patterns).pack(side=tk.LEFT, padx=(0, 10))
        
        # Export JSON button
        ttk.Button(controls_frame, text="💾 Export JSON", command=self.export_action_data_json).pack(side=tk.LEFT, padx=(0, 10))
        
        # Add help text for normalization
        help_label = ttk.Label(controls_frame, text="(Normalizes timestamps and coordinates using same scaling as main features)", 
                              font=("Arial", 8), foreground="gray")
        help_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create the action tensor display directly
        self.json_frame = ttk.LabelFrame(self.action_tensors_frame, text="Action Tensor Table", padding="5")
        self.json_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add simple description
        flow_description = ttk.Label(
            self.json_frame,
            text="Action tensor table: Action Count (first element only) + 8 features per action across all timesteps",
            font=("Consolas", 9),
            foreground="blue"
        )
        flow_description.pack(pady=(0, 5))
        
        # Create action tensor table (similar to input sequences table)
        self.create_action_tensor_table()
        
        # Initialize with first gamestate and populate table
        self.current_action_gamestate = 0
        self.update_action_json_display()
    
    def create_action_tensor_table(self):
        """Create the action tensor table similar to input sequences table"""
        # Info frame
        info_frame = ttk.Frame(self.json_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.action_tensor_info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.action_tensor_info_label.pack(side=tk.LEFT)
        
        # Export and copy buttons
        export_frame = ttk.Frame(self.json_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Button(export_frame, text="📋 Copy Table to Clipboard", command=self.copy_action_tensor_table_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="💾 Export to CSV", command=self.export_action_tensor_table_to_csv).pack(side=tk.LEFT, padx=(0, 10))
        
        # Normalization toggle
        self.show_action_normalized = tk.BooleanVar(value=False)
        self.action_normalization_toggle = ttk.Checkbutton(
            export_frame, 
            text="📊 Show Normalized Data", 
            variable=self.show_action_normalized,
            command=self.toggle_action_normalization
        )
        self.action_normalization_toggle.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create Treeview for table display
        self.create_action_tensor_tree()
    
    def create_action_tensor_tree(self):
        """Create the action tensor treeview table"""
        # Frame for table
        table_frame = ttk.Frame(self.json_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview with scrollbars
        self.action_tensor_tree = ttk.Treeview(table_frame, show="headings", height=20)
        
        # Create scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.action_tensor_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.action_tensor_tree.xview)
        self.action_tensor_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.action_tensor_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Initial columns - will be dynamically updated based on action count
        self.action_tensor_tree["columns"] = ["Feature", "Index"]
        
        # Set initial column headings
        for col in self.action_tensor_tree["columns"]:
            self.action_tensor_tree.heading(col, text=col)
            if col == "Feature":
                self.action_tensor_tree.column(col, width=200, minwidth=150)
            elif col == "Index":
                self.action_tensor_tree.column(col, width=50, minwidth=50)
        
        # Bind tooltip events
        self.action_tensor_tree.bind('<Motion>', self.on_action_tensor_table_motion)
        self.action_tensor_tree.bind('<Leave>', self.on_action_tensor_table_leave)
        
        # Tooltip variables
        self.action_tensor_tooltip = None
        self.action_tensor_tooltip_text = ""
    
    def update_action_tensor_table(self):
        """Update the action tensor table with current gamestate data"""
        print("DEBUG: update_action_tensor_table called!")
        try:
            # Clear existing items
            for item in self.action_tensor_tree.get_children():
                self.action_tensor_tree.delete(item)
            
            # Get current gamestate data
            gamestate_idx = self.current_action_gamestate
            
            # Load action tensors data based on normalization toggle
            if self.show_action_normalized.get():
                file_path = 'data/training_data/normalized_action_training_format.json'
                print(f"DEBUG: Loading normalized file: {file_path}")
                with open(file_path, 'r') as f:
                    action_tensors = json.load(f)
                print(f"DEBUG: Loaded {len(action_tensors)} gamestates from normalized file")
                print(f"DEBUG: First gamestate has {len(action_tensors[0])} elements: {action_tensors[0][:10]}")
                data_source = "normalized_action_training_format.json"
                normalization_status = "Normalized (timestamps in minutes, coordinates scaled)"
            else:
                file_path = 'data/training_data/raw_action_tensors.json'
                print(f"DEBUG: Loading raw file: {file_path}")
                with open(file_path, 'r') as f:
                    action_tensors = json.load(f)
                print(f"DEBUG: Loaded {len(action_tensors)} gamestates from raw file")
                print(f"DEBUG: First gamestate has {len(action_tensors[0])} elements: {action_tensors[0][:10]}")
                data_source = "raw_action_tensors.json"
                normalization_status = "Raw (timestamps in ms, coordinates as-is)"
            
            if gamestate_idx >= len(action_tensors):
                self.action_tensor_info_label.config(text="Gamestate index out of range")
                return
            
            # Get the action tensor for current gamestate
            action_tensor = action_tensors[gamestate_idx]
            action_count = int(action_tensor[0]) if action_tensor else 0
            
            # Update info label
            self.action_tensor_info_label.config(
                text=f"Gamestate {gamestate_idx} | {action_count} actions | {data_source} | {normalization_status} | Action Count: First element only"
            )
            
            # Dynamically configure columns for all timesteps
            columns = ["Feature", "Index"]
            for timestep in range(action_count):
                columns.append(f"Timestep {timestep}")
            
            # Update treeview columns
            self.action_tensor_tree["columns"] = columns
            
            # Configure column headings and widths
            for col in columns:
                self.action_tensor_tree.heading(col, text=col)
                if col == "Feature":
                    self.action_tensor_tree.column(col, width=200, minwidth=150)
                elif col == "Index":
                    self.action_tensor_tree.column(col, width=50, minwidth=50)
                else:
                    self.action_tensor_tree.column(col, width=100, minwidth=80)
            
            # Create feature rows for the table
            # Each action has 8 features: timestamp, type, x, y, button, key, scroll_dx, scroll_dy
            feature_names = ["Action Count", "Timestamp", "Action Type", "Mouse X", "Mouse Y", "Button", "Key", "Scroll DX", "Scroll DY"]
            
            # For each feature, create a row showing values across all timesteps
            for feature_idx, feature_name in enumerate(feature_names):
                if feature_idx == 0:  # Action Count
                    # Action count is a single value at the beginning of the tensor
                    # Show it only in the first column, with "N/A" for other timesteps
                    values = [action_count] + ["N/A"] * (action_count - 1)
                else:
                    # For other features, we need to extract values from the flattened tensor
                    values = []
                    for timestep in range(action_count):
                        # Calculate position in flattened tensor: 1 + timestep * 8 + (feature_idx - 1)
                        tensor_idx = 1 + timestep * 8 + (feature_idx - 1)
                        if tensor_idx < len(action_tensor):
                            value = action_tensor[tensor_idx]
                            values.append(value)
                            # Debug: print the first few values to verify indexing
                            if timestep < 3 and feature_idx == 1:  # Only for first few timestamps
                                print(f"DEBUG: Feature {feature_name}, Timestep {timestep}, Index {tensor_idx}, Value {value}")
                        else:
                            values.append("N/A")
                
                # Insert row into table
                row_values = [feature_name, feature_idx] + values
                self.action_tensor_tree.insert("", "end", values=row_values)
                
        except FileNotFoundError as e:
            self.action_tensor_info_label.config(text=f"File not found: {e}")
        except Exception as e:
            self.action_tensor_info_label.config(text=f"Error loading data: {e}")
    
    def toggle_action_normalization(self):
        """Toggle between raw and normalized action tensor data"""
        self.update_action_tensor_table()
    
    def copy_action_tensor_table_to_clipboard(self):
        """Copy action tensor table data to clipboard"""
        try:
            # Get all items from the table
            items = self.action_tensor_tree.get_children()
            if not items:
                messagebox.showwarning("No Data", "No data to copy!")
                return
            
            # Build CSV-like string with dynamic headers
            columns = self.action_tensor_tree["columns"]
            clipboard_text = ",".join(columns) + "\n"
            
            for item in items:
                values = self.action_tensor_tree.item(item)['values']
                row_text = ",".join(str(v) for v in values)
                clipboard_text += row_text + "\n"
            
            pyperclip.copy(clipboard_text)
            messagebox.showinfo("Copied", "Action tensor table data copied to clipboard!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy data: {e}")
    
    def export_action_tensor_table_to_csv(self):
        """Export action tensor table data to CSV file"""
        try:
            # Get all items from the table
            items = self.action_tensor_tree.get_children()
            if not items:
                messagebox.showwarning("No Data", "No data to export!")
                return
            
            # Get file path from user
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Action Tensor Table"
            )
            
            if file_path:
                with open(file_path, 'w', newline='') as csvfile:
                    # Write header with dynamic columns
                    columns = self.action_tensor_tree["columns"]
                    csvfile.write(",".join(columns) + "\n")
                    
                    # Write data rows
                    for item in items:
                        values = self.action_tensor_tree.item(item)['values']
                        row_text = ",".join(str(v) for v in values)
                        csvfile.write(row_text + "\n")
                
                messagebox.showinfo("Success", f"Action tensor table exported to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def on_action_tensor_table_motion(self, event):
        """Handle mouse motion over action tensor table for tooltips"""
        # Get the item and column under the cursor
        item = self.action_tensor_tree.identify_row(event.y)
        column = self.action_tensor_tree.identify_column(event.x)
        
        if item and column:
            # Get the value at this position
            values = self.action_tensor_tree.item(item)['values']
            col_idx = int(column[1]) - 1  # Convert column identifier to index
            
            if col_idx < len(values):
                value = values[col_idx]
                feature_name = values[0] if values else "Unknown"
                
                # Create tooltip text
                if col_idx == 0:  # Feature column
                    tooltip_text = f"Feature: {feature_name}"
                elif col_idx == 1:  # Index column
                    tooltip_text = f"Feature Index: {value}"
                elif col_idx < len(values):  # Timestep columns
                    timestep_num = col_idx - 2
                    tooltip_text = f"Timestep {timestep_num}: {value}"
                
                # Show tooltip
                self.show_action_tensor_tooltip(event, tooltip_text)
        else:
            self.hide_action_tensor_tooltip()
    
    def on_action_tensor_table_leave(self, event):
        """Handle mouse leave from action tensor table"""
        self.hide_action_tensor_tooltip()
    
    def show_action_tensor_tooltip(self, event, text):
        """Show tooltip for action tensor table"""
        if self.action_tensor_tooltip:
            self.action_tensor_tooltip.destroy()
        
        self.action_tensor_tooltip = tk.Toplevel()
        self.action_tensor_tooltip.wm_overrideredirect(True)
        self.action_tensor_tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = ttk.Label(self.action_tensor_tooltip, text=text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack()
        
        self.action_tensor_tooltip_text = text
    
    def hide_action_tensor_tooltip(self):
        """Hide action tensor tooltip"""
        if self.action_tensor_tooltip:
            self.action_tensor_tooltip.destroy()
            self.action_tensor_tooltip = None
    
    def on_action_gamestate_change(self):
        """Handle gamestate change in action tensors tab"""
        try:
            new_gamestate = int(self.action_gamestate_var.get())
            if 0 <= new_gamestate < len(self.raw_action_data):
                self.current_action_gamestate = new_gamestate
                self.update_action_display()
        except ValueError:
            pass
    
    def previous_action_gamestate(self):
        """Go to previous action gamestate"""
        if self.current_action_gamestate > 0:
            self.current_action_gamestate -= 1
            self.action_gamestate_var.set(str(self.current_action_gamestate))
            self.update_action_json_display()
    
    def next_action_gamestate(self):
        """Go to next action gamestate"""
        if self.current_action_gamestate < len(self.raw_action_data) - 1:
            self.current_action_gamestate += 1
            self.action_gamestate_var.set(str(self.current_action_gamestate))
            self.update_action_json_display()
    
    def on_action_type_filter_changed(self):
        """Handle action type filter change"""
        self.update_action_json_display()
        """Update the action tensors display"""
        if not self.raw_action_data or self.current_action_gamestate >= len(self.raw_action_data):
            return
        
        # Get current gamestate action data
        action_data = self.raw_action_data[self.current_action_gamestate]
        
        # Check if normalization is requested
        show_normalized = self.show_action_normalized.get()
        
        # This method is deprecated - redirect to the new method
        self.update_action_json_display()
    
    def copy_detailed_to_clipboard(self):
        """Copy the detailed action information to clipboard"""
        try:
            json_text = self.detailed_text.get(1.0, tk.END).strip()
            if json_text:
                pyperclip.copy(json_text)
                messagebox.showinfo("Copied", "Detailed action information copied to clipboard!")
            else:
                messagebox.showwarning("No Data", "No detailed data available to copy")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy detailed data: {e}")
    
    def save_detailed_to_file(self):
        """Save the detailed action information to a file"""
        try:
            json_text = self.detailed_text.get(1.0, tk.END).strip()
            if not json_text:
                messagebox.showwarning("No Data", "No detailed data available to save")
                return
            
            filename = f"detailed_actions_gamestate_{self.current_action_gamestate}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename,
                title="Save Detailed Action Information"
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                
                messagebox.showinfo("Success", f"Detailed action information saved to:\n{filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save detailed data: {e}")
    
    def copy_raw_training_to_clipboard(self):
        """Copy the raw training format data to clipboard"""
        try:
            json_text = self.raw_training_text.get(1.0, tk.END).strip()
            if json_text:
                pyperclip.copy(json_text)
                messagebox.showinfo("Copied", "Raw training format data copied to clipboard!")
            else:
                messagebox.showwarning("No Data", "No raw training data available to copy")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy raw training data: {e}")
    
    def save_raw_training_to_file(self):
        """Save the raw training format data to a file"""
        try:
            json_text = self.raw_training_text.get(1.0, tk.END).strip()
            if not json_text:
                messagebox.showwarning("No Data", "No raw training data available to save")
                return
            
            filename = f"raw_training_format_gamestate_{self.current_action_gamestate}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename,
                title="Save Raw Training Format Data"
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                
                messagebox.showinfo("Success", f"Raw training format data saved to:\n{filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save raw training data: {e}")
    
    def copy_normalized_training_to_clipboard(self):
        """Copy the normalized training format data to clipboard"""
        try:
            json_text = self.normalized_training_text.get(1.0, tk.END).strip()
            if json_text:
                pyperclip.copy(json_text)
                messagebox.showinfo("Copied", "Normalized training format data copied to clipboard!")
            else:
                messagebox.showwarning("No Data", "No normalized training data available to copy")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy normalized training data: {e}")
    
    def save_normalized_training_to_file(self):
        """Save the normalized training format data to a file"""
        try:
            json_text = self.normalized_training_text.get(1.0, tk.END).strip()
            if not json_text:
                messagebox.showwarning("No Data", "No normalized training data available to save")
                return
            
            filename = f"normalized_training_format_gamestate_{self.current_action_gamestate}.json"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename,
                title="Save Normalized Training Format Data"
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                
                messagebox.showinfo("Success", f"Normalized training format data saved to:\n{filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save normalized training data: {e}")
    
    def copy_action_data_to_clipboard(self):
        """Copy current action data to clipboard"""
        if not self.raw_action_data or self.current_action_gamestate >= len(self.raw_action_data):
            return
        
        action_data = self.raw_action_data[self.current_action_gamestate]
        
        # Create formatted text
        text_lines = [f"Gamestate {self.current_action_gamestate} Action Data:"]
        text_lines.append("=" * 50)
        
        for action_type, actions in action_data.items():
            if actions:
                text_lines.append(f"\n{action_type.upper()}:")
                for i, action in enumerate(actions):
                    if action_type == 'mouse_movements':
                        text_lines.append(f"  {i}: Move to ({action.get('x', 0)}, {action.get('y', 0)}) at {action.get('timestamp', 0):.0f}ms")
                    elif action_type == 'clicks':
                        text_lines.append(f"  {i}: {action.get('button', '')} click at ({action.get('x', 0)}, {action.get('y', 0)}) at {action.get('timestamp', 0):.0f}ms")
                    elif action_type == 'key_presses':
                        text_lines.append(f"  {i}: Key '{action.get('key', '')}' pressed at {action.get('timestamp', 0):.0f}ms")
                    elif action_type == 'key_releases':
                        text_lines.append(f"  {i}: Key '{action.get('key', '')}' released at {action.get('timestamp', 0):.0f}ms")
                    elif action_type == 'scrolls':
                        text_lines.append(f"  {i}: Scroll ({action.get('dx', 0)}, {action.get('dy', 0)}) at {action.get('timestamp', 0):.0f}ms")
        
        clipboard_text = "\n".join(text_lines)
        pyperclip.copy(clipboard_text)
        messagebox.showinfo("Copied", "Action data copied to clipboard!")
    
    def copy_action_sequences_to_clipboard(self):
        """Copy action sequences data to clipboard"""
        try:
            with open('data/training_data/action_sequences.json', 'r') as f:
                action_sequences = json.load(f)
            
            if self.current_action_gamestate < len(action_sequences):
                sequence_data = action_sequences[self.current_action_gamestate]
                clipboard_text = json.dumps(sequence_data, indent=2, ensure_ascii=False)
                pyperclip.copy(clipboard_text)
                messagebox.showinfo("Copied", "Action sequences data copied to clipboard!")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy action sequences: {e}")
    
    def save_action_sequences_to_file(self):
        """Save action sequences data to file"""
        try:
            with open('data/training_data/action_sequences.json', 'r') as f:
                action_sequences = json.load(f)
            
            if self.current_action_gamestate < len(action_sequences):
                sequence_data = action_sequences[self.current_action_gamestate]
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Save Action Sequences Data"
                )
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(sequence_data, f, indent=2)
                    messagebox.showinfo("Success", f"Action sequences data saved to:\n{file_path}")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save action sequences: {e}")
    
    def copy_action_targets_to_clipboard(self):
        """Copy action targets data to clipboard"""
        try:
            with open('data/training_data/action_targets.json', 'r') as f:
                action_targets = json.load(f)
            
            if self.current_action_gamestate < len(action_targets):
                target_data = action_targets[self.current_action_gamestate]
                clipboard_text = json.dumps(target_data, indent=2, ensure_ascii=False)
                pyperclip.copy(clipboard_text)
                messagebox.showinfo("Copied", "Action targets data copied to clipboard!")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy action targets: {e}")
    
    def save_action_targets_to_file(self):
        """Save action targets data to file"""
        try:
            with open('data/training_data/action_targets.json', 'r') as f:
                action_targets = json.load(f)
            
            if self.current_action_gamestate < len(action_targets):
                target_data = action_targets[self.current_action_gamestate]
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Save Action Targets Data"
                )
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(target_data, f, indent=2)
                    messagebox.showinfo("Success", f"Action targets data saved to:\n{file_path}")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save action targets: {e}")
    
    def copy_raw_tensors_to_clipboard(self):
        """Copy raw action tensors data to clipboard"""
        try:
            with open('data/training_data/raw_action_tensors.json', 'r') as f:
                raw_tensors = json.load(f)
            
            if self.current_action_gamestate < len(raw_tensors):
                tensor_data = raw_tensors[self.current_action_gamestate]
                clipboard_text = json.dumps(tensor_data, indent=2, ensure_ascii=False)
                pyperclip.copy(clipboard_text)
                messagebox.showinfo("Copied", "Raw action tensors data copied to clipboard!")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy raw tensors: {e}")
    
    def save_raw_tensors_to_file(self):
        """Save raw action tensors data to file"""
        try:
            with open('data/training_data/raw_action_tensors.json', 'r') as f:
                raw_tensors = json.load(f)
            
            if self.current_action_gamestate < len(raw_tensors):
                tensor_data = raw_tensors[self.current_action_gamestate]
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Save Raw Action Tensors Data"
                )
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(tensor_data, f, indent=2)
                    messagebox.showinfo("Success", f"Raw action tensors data saved to:\n{file_path}")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save raw tensors: {e}")
    
    def copy_normalized_tensors_to_clipboard(self):
        """Copy normalized action tensors data to clipboard"""
        try:
            with open('data/training_data/normalized_action_training_format.json', 'r') as f:
                normalized_tensors = json.load(f)
            
            if self.current_action_gamestate < len(normalized_tensors):
                tensor_data = normalized_tensors[self.current_action_gamestate]
                clipboard_text = json.dumps(tensor_data, indent=2, ensure_ascii=False)
                pyperclip.copy(clipboard_text)
                messagebox.showinfo("Copied", "Normalized action tensors data copied to clipboard!")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy normalized tensors: {e}")
    
    def save_normalized_tensors_to_file(self):
        """Save normalized action tensors data to file"""
        try:
            with open('data/training_data/normalized_action_training_format.json', 'r') as f:
                normalized_tensors = json.load(f)
            
            if self.current_action_gamestate < len(normalized_tensors):
                tensor_data = normalized_tensors[self.current_action_gamestate]
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Save Normalized Action Tensors Data"
                )
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(tensor_data, f, indent=2)
                    messagebox.showinfo("Success", f"Normalized action tensors data saved to:\n{file_path}")
            else:
                messagebox.showwarning("Warning", "Gamestate index out of range")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save normalized tensors: {e}")
    
    def export_action_data_json(self):
        """Export current action data to JSON file"""
        if not self.raw_action_data or self.current_action_gamestate >= len(self.raw_action_data):
            return
        
        # Get file path from user
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Action Data"
        )
        
        if file_path:
            try:
                # Export current gamestate data
                current_data = self.raw_action_data[self.current_action_gamestate]
                
                # Add metadata
                export_data = {
                    'metadata': {
                        'gamestate_index': self.current_action_gamestate,
                        'export_timestamp': str(pd.Timestamp.now()),
                        'total_actions': len(current_data.get('mouse_movements', [])) + len(current_data.get('clicks', [])) + 
                                        len(current_data.get('key_presses', [])) + len(current_data.get('key_releases', [])) + 
                                        len(current_data.get('scrolls', [])),
                        'action_breakdown': {
                            'mouse_movements': len(current_data.get('mouse_movements', [])),
                            'clicks': len(current_data.get('clicks', [])),
                            'key_presses': len(current_data.get('key_presses', [])),
                            'key_releases': len(current_data.get('key_releases', [])),
                            'scrolls': len(current_data.get('scrolls', []))
                        }
                    },
                    'action_data': current_data
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Action data exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export action data:\n{str(e)}")
    
    def show_action_patterns(self):
        """Show action patterns and statistics visualization"""
        if not self.raw_action_data:
            messagebox.showwarning("No Data", "No raw action data available!")
            return
        
        # Create new window
        patterns_window = tk.Toplevel(self.root)
        patterns_window.title("Action Patterns & Statistics")
        patterns_window.geometry("800x600")
        
        # Notebook for different visualizations
        notebook = ttk.Notebook(patterns_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Action counts tab
        counts_frame = ttk.Frame(notebook)
        notebook.add(counts_frame, text="Action Counts")
        
        # Action timing tab
        timing_frame = ttk.Frame(notebook)
        notebook.add(timing_frame, text="Action Timing")
        
        # Spatial patterns tab
        spatial_frame = ttk.Frame(notebook)
        notebook.add(spatial_frame, text="Spatial Patterns")
        
        # Create action counts visualization
        self.create_action_counts_visualization(counts_frame)
        
        # Create action timing visualization
        self.create_action_timing_visualization(timing_frame)
        
        # Create spatial patterns visualization
        self.create_spatial_patterns_visualization(spatial_frame)
    
    def create_action_counts_visualization(self, parent_frame):
        """Create action counts visualization"""
        # Summary frame
        summary_frame = ttk.LabelFrame(parent_frame, text="Action Counts Summary", padding="5")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Calculate total counts
        total_moves = sum(len(g.get('mouse_movements', [])) for g in self.raw_action_data)
        total_clicks = sum(len(g.get('clicks', [])) for g in self.raw_action_data)
        total_key_presses = sum(len(g.get('key_presses', [])) for g in self.raw_action_data)
        total_key_releases = sum(len(g.get('key_releases', [])) for g in self.raw_action_data)
        total_scrolls = sum(len(g.get('scrolls', [])) for g in self.raw_action_data)
        
        summary_text = f"""
        Total Actions Across All Gamestates:
        Mouse Movements: {total_moves:,}
        Clicks: {total_clicks:,}
        Key Presses: {total_key_presses:,}
        Key Releases: {total_key_releases:,}
        Scrolls: {total_scrolls:,}
        Grand Total: {total_moves + total_clicks + total_key_presses + total_key_releases + total_scrolls:,}
        """
        
        summary_label = ttk.Label(summary_frame, text=summary_text, font=("Consolas", 10), justify=tk.LEFT)
        summary_label.pack(anchor=tk.W)
        
        # Action distribution frame
        dist_frame = ttk.LabelFrame(parent_frame, text="Action Distribution by Gamestate", padding="5")
        dist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview for distribution
        columns = ("Gamestate", "Moves", "Clicks", "Keys", "Scrolls", "Total")
        dist_tree = ttk.Treeview(dist_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            dist_tree.heading(col, text=col)
            dist_tree.column(col, width=100)
        
        # Populate with data
        for i, gamestate_data in enumerate(self.raw_action_data):
            moves = len(gamestate_data.get('mouse_movements', []))
            clicks = len(gamestate_data.get('clicks', []))
            keys = len(gamestate_data.get('key_presses', [])) + len(gamestate_data.get('key_releases', []))
            scrolls = len(gamestate_data.get('scrolls', []))
            total = moves + clicks + keys + scrolls
            
            dist_tree.insert("", "end", values=(f"G{i}", moves, clicks, keys, scrolls, total))
        
        # Scrollbar
        dist_scrollbar = ttk.Scrollbar(dist_frame, orient=tk.VERTICAL, command=dist_tree.yview)
        dist_tree.configure(yscrollcommand=dist_scrollbar.set)
        
        # Pack widgets
        dist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dist_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_action_timing_visualization(self, parent_frame):
        """Create action timing visualization"""
        # Summary frame
        summary_frame = ttk.LabelFrame(parent_frame, text="Action Timing Analysis", padding="5")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Calculate timing statistics
        all_timestamps = []
        for gamestate_data in self.raw_action_data:
            for action_type in ['mouse_movements', 'clicks', 'key_presses', 'key_releases', 'scrolls']:
                for action in gamestate_data.get(action_type, []):
                    timestamp = action.get('timestamp', 0)
                    if timestamp > 0:
                        all_timestamps.append(timestamp)
        
        if all_timestamps:
            all_timestamps.sort()
            timing_text = f"""
            Timing Statistics:
            Total Actions: {len(all_timestamps):,}
            Time Range: {min(all_timestamps):.0f}ms to {max(all_timestamps):.0f}ms
            Duration: {max(all_timestamps) - min(all_timestamps):.0f}ms
            Average Interval: {np.mean(np.diff(all_timestamps)):.1f}ms
            Median Interval: {np.median(np.diff(all_timestamps)):.1f}ms
            """
        else:
            timing_text = "No timing data available"
        
        summary_label = ttk.Label(summary_frame, text=timing_text, font=("Consolas", 10), justify=tk.LEFT)
        summary_label.pack(anchor=tk.W)
        
        # Timing distribution frame
        timing_dist_frame = ttk.LabelFrame(parent_frame, text="Action Timing Distribution", padding="5")
        timing_dist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview for timing distribution
        columns = ("Time Window", "Actions", "Details")
        timing_tree = ttk.Treeview(timing_dist_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        timing_tree.heading("Time Window", text="Time Window (ms)")
        timing_tree.heading("Actions", text="Action Count")
        timing_tree.heading("Details", text="Action Types")
        timing_tree.column("Time Window", width=150)
        timing_tree.column("Actions", width=100)
        timing_tree.column("Details", width=300)
        
        # Group actions by time windows
        if all_timestamps:
            time_windows = {}
            window_size = 1000  # 1 second windows
            
            for timestamp in all_timestamps:
                window_start = (timestamp // window_size) * window_size
                window_end = window_start + window_size
                window_key = f"{window_start}-{window_end}"
                
                if window_key not in time_windows:
                    time_windows[window_key] = {'count': 0, 'types': set()}
                
                time_windows[window_key]['count'] += 1
                # Determine action type from timestamp (simplified)
                time_windows[window_key]['types'].add('action')
            
            # Populate tree
            for window_key, data in sorted(time_windows.items()):
                timing_tree.insert("", "end", values=(
                    window_key,
                    data['count'],
                    f"Types: {', '.join(sorted(data['types']))}"
                ))
        
        # Scrollbar
        timing_scrollbar = ttk.Scrollbar(timing_dist_frame, orient=tk.VERTICAL, command=timing_tree.yview)
        timing_tree.configure(yscrollcommand=timing_scrollbar.set)
        
        # Pack widgets
        timing_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        timing_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_spatial_patterns_visualization(self, parent_frame):
        """Create spatial patterns visualization"""
        # Summary frame
        summary_frame = ttk.LabelFrame(parent_frame, text="Spatial Patterns Summary", padding="5")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Calculate spatial statistics
        all_x_coords = []
        all_y_coords = []
        
        for gamestate_data in self.raw_action_data:
            # Mouse movements
            for move in gamestate_data.get('mouse_movements', []):
                x, y = move.get('x', 0), move.get('y', 0)
                if x > 0 and y > 0:
                    all_x_coords.append(x)
                    all_y_coords.append(y)
            
            # Clicks
            for click in gamestate_data.get('clicks', []):
                x, y = click.get('x', 0), click.get('y', 0)
                if x > 0 and y > 0:
                    all_x_coords.append(x)
                    all_y_coords.append(y)
        
        if all_x_coords and all_y_coords:
            spatial_text = f"""
            Spatial Statistics:
            Total Spatial Actions: {len(all_x_coords):,}
            X Range: {min(all_x_coords):.0f} to {max(all_x_coords):.0f}
            Y Range: {min(all_y_coords):.0f} to {max(all_y_coords):.0f}
            Average X: {np.mean(all_x_coords):.1f}
            Average Y: {np.mean(all_y_coords):.1f}
            """
        else:
            spatial_text = "No spatial data available"
        
        summary_label = ttk.Label(summary_frame, text=spatial_text, font=("Consolas", 10), justify=tk.LEFT)
        summary_label.pack(anchor=tk.W)
        
        # Spatial distribution frame
        spatial_dist_frame = ttk.LabelFrame(parent_frame, text="Spatial Distribution", padding="5")
        spatial_dist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview for spatial distribution
        columns = ("Region", "Actions", "Avg X", "Avg Y", "Details")
        spatial_tree = ttk.Treeview(spatial_dist_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        spatial_tree.heading("Region", text="Screen Region")
        spatial_tree.heading("Actions", text="Action Count")
        spatial_tree.heading("Avg X", text="Avg X Coord")
        spatial_tree.heading("Avg Y", text="Avg Y Coord")
        spatial_tree.heading("Details", text="Action Types")
        spatial_tree.column("Region", width=120)
        spatial_tree.column("Actions", width=100)
        spatial_tree.column("Avg X", width=100)
        spatial_tree.column("Avg Y", width=100)
        spatial_tree.column("Details", width=200)
        
        # Group actions by screen regions
        if all_x_coords and all_y_coords:
            regions = {
                'Top-Left': {'x': [], 'y': [], 'count': 0},
                'Top-Right': {'x': [], 'y': [], 'count': 0},
                'Bottom-Left': {'x': [], 'y': [], 'count': 0},
                'Bottom-Right': {'x': [], 'y': [], 'count': 0}
            }
            
            # Define region boundaries (assuming 800x600 screen)
            screen_width, screen_height = 800, 600
            mid_x, mid_y = screen_width // 2, screen_height // 2
            
            for i, (x, y) in enumerate(zip(all_x_coords, all_y_coords)):
                if x <= mid_x and y <= mid_y:
                    regions['Top-Left']['x'].append(x)
                    regions['Top-Left']['y'].append(y)
                    regions['Top-Left']['count'] += 1
                elif x > mid_x and y <= mid_y:
                    regions['Top-Right']['x'].append(x)
                    regions['Top-Right']['y'].append(y)
                    regions['Top-Right']['count'] += 1
                elif x <= mid_x and y > mid_y:
                    regions['Bottom-Left']['x'].append(x)
                    regions['Bottom-Left']['y'].append(y)
                    regions['Bottom-Left']['count'] += 1
                else:
                    regions['Bottom-Right']['x'].append(x)
                    regions['Bottom-Right']['y'].append(y)
                    regions['Bottom-Right']['count'] += 1
            
            # Populate tree
            for region_name, region_data in regions.items():
                if region_data['count'] > 0:
                    avg_x = np.mean(region_data['x']) if region_data['x'] else 0
                    avg_y = np.mean(region_data['y']) if region_data['y'] else 0
                    spatial_tree.insert("", "end", values=(
                        region_name,
                        region_data['count'],
                        f"{avg_x:.1f}",
                        f"{avg_y:.1f}",
                        f"Mouse moves: {len(region_data['x'])}"
                    ))
        
        # Scrollbar
        spatial_scrollbar = ttk.Scrollbar(spatial_dist_frame, orient=tk.VERTICAL, command=spatial_tree.yview)
        spatial_tree.configure(yscrollcommand=spatial_scrollbar.set)
        
        # Pack widgets
        spatial_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        spatial_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_action_json_display(self):
        """Update action tensor table with current gamestate data"""
        if not self.raw_action_data or self.current_action_gamestate >= len(self.raw_action_data):
            return
        
        # Update the frame title to show current gamestate
        self.json_frame.config(text=f"Action Tensors - Gamestate {self.current_action_gamestate}")
        
        # Update the action tensor table
        self.update_action_tensor_table()
    
    def update_detailed_tab(self, action_data, data_source):
        """Update the detailed action information tab"""
        # Create detailed view showing structured action data
        detailed_data = {
            "metadata": {
                "gamestate_index": self.current_action_gamestate,
                "data_source": data_source,
                "normalization_applied": "N/A - showing raw data",
                "processing_stage": "Step 2: After trimming, before tensor conversion",
                "description": "Raw action data with structured format (mouse movements, clicks, keys, scrolls)",
                "data_type": "Structured JSON with action categories",
                "normalization_status": "Raw data - no normalization applied yet",
                "note": "This tab shows the SAME timestep as all other tabs - just at this processing stage",
                "total_actions": len(action_data.get('mouse_movements', [])) + 
                                len(action_data.get('clicks', [])) + 
                                len(action_data.get('key_presses', [])) + 
                                len(action_data.get('key_releases', [])) + 
                                len(action_data.get('scrolls', [])),
                "action_breakdown": {
                    "mouse_movements": len(action_data.get('mouse_movements', [])),
                    "clicks": len(action_data.get('clicks', [])),
                    "key_presses": len(action_data.get('key_presses', [])),
                    "key_releases": len(action_data.get('key_releases', [])),
                    "scrolls": len(action_data.get('scrolls', []))
                }
            },
            "detailed_actions": {
                "mouse_movements": action_data.get('mouse_movements', []),
                "clicks": action_data.get('clicks', []),
                "key_presses": action_data.get('key_presses', []),
                "key_releases": action_data.get('key_releases', []),
                "scrolls": action_data.get('scrolls', [])
            }
        }
        
        # Format JSON with proper indentation
        try:
            import json as json_module
            formatted_json = json_module.dumps(detailed_data, indent=2, ensure_ascii=False)
        except Exception:
            formatted_json = str(detailed_data)
        
        # Update the detailed tab
        self.detailed_text.config(state=tk.NORMAL)
        self.detailed_text.delete(1.0, tk.END)
        self.detailed_text.insert(tk.END, formatted_json)
        self.detailed_text.config(state=tk.DISABLED)
    
    def update_raw_training_tab(self):
        """Update the raw training format tab"""
        try:
            # Load pre-processed raw training format data
            with open('data/training_data/raw_action_training_format.json', 'r') as f:
                raw_training_data = json.load(f)
            
            if self.current_action_gamestate < len(raw_training_data):
                training_sequence = raw_training_data[self.current_action_gamestate]
                
                # Create formatted display
                raw_data = {
                    "metadata": {
                        "gamestate_index": self.current_action_gamestate,
                        "data_source": "raw_action_training_format.json",
                        "processing_stage": "Step 10: Final output - raw training format",
                        "description": "Raw action data converted to flattened training format (not normalized)",
                        "data_type": "Flattened numerical array for training",
                        "normalization_status": "Raw data - no normalization applied",
                        "format": "[action_count, timing1, type1, x1, y1, button1, key1, timing2, type2, x2, y2, button2, key2, ...]",
                        "note": "This tab shows the SAME timestep as all other tabs - just at this processing stage",
                        "total_actions": training_sequence[0] if training_sequence else 0,
                        "sequence_length": len(training_sequence)
                    },
                    "training_sequence": training_sequence,
                    "action_breakdown": self.breakdown_training_sequence(training_sequence)
                }
                
                # Format JSON
                formatted_json = json.dumps(raw_data, indent=2, ensure_ascii=False)
                
                # Update the raw training tab
                self.raw_training_text.config(state=tk.NORMAL)
                self.raw_training_text.delete(1.0, tk.END)
                self.raw_training_text.insert(tk.END, formatted_json)
                self.raw_training_text.config(state=tk.DISABLED)
            else:
                self.raw_training_text.config(state=tk.NORMAL)
                self.raw_training_text.delete(1.0, tk.END)
                self.raw_training_text.insert(tk.END, "Gamestate index out of range")
                self.raw_training_text.config(state=tk.DISABLED)
                
        except FileNotFoundError:
            self.raw_training_text.config(state=tk.NORMAL)
            self.raw_training_text.delete(1.0, tk.END)
            self.raw_training_text.insert(tk.END, "raw_action_training_format.json not found.\nRun phase1_data_preparation.py first.")
            self.raw_training_text.config(state=tk.DISABLED)
        except Exception as e:
            self.raw_training_text.config(state=tk.NORMAL)
            self.raw_training_text.delete(1.0, tk.END)
            self.raw_training_text.insert(tk.END, f"Error loading raw training data: {e}")
            self.raw_training_text.config(state=tk.DISABLED)
    
    def update_normalized_training_tab(self):
        """Update the normalized training format tab"""
        try:
            # Load pre-processed normalized training format data
            with open('data/training_data/normalized_action_training_format.json', 'r') as f:
                normalized_training_data = json.load(f)
            
            if self.current_action_gamestate < len(normalized_training_data):
                training_sequence = normalized_training_data[self.current_action_gamestate]
                
                # Create formatted display
                norm_data = {
                    "metadata": {
                        "gamestate_index": self.current_action_gamestate,
                        "data_source": "normalized_action_training_format.json",
                        "processing_stage": "Step 10: Final output - normalized training format",
                        "description": "Normalized action data converted to flattened training format",
                        "data_type": "Flattened numerical array for training (normalized)",
                        "normalization_status": "Fully normalized - timestamps in minutes, coordinates scaled",
                        "format": "[action_count, timing1, type1, x1, y1, button1, key1, timing2, type2, x2, y2, button2, key2, ...]",
                        "normalization_details": "Timestamps: ms→minutes, Coordinates: RobustScaler, Keys: Hash normalized",
                        "note": "This tab shows the SAME timestep as all other tabs - just at this processing stage",
                        "total_actions": training_sequence[0] if training_sequence else 0,
                        "sequence_length": len(training_sequence)
                    },
                    "training_sequence": training_sequence,
                    "action_breakdown": self.breakdown_training_sequence(training_sequence)
                }
                
                # Format JSON
                formatted_json = json.dumps(norm_data, indent=2, ensure_ascii=False)
                
                # Update the normalized training tab
                self.normalized_training_text.config(state=tk.NORMAL)
                self.normalized_training_text.delete(1.0, tk.END)
                self.normalized_training_text.insert(tk.END, formatted_json)
                self.normalized_training_text.config(state=tk.DISABLED)
            else:
                self.normalized_training_text.config(state=tk.NORMAL)
                self.normalized_training_text.delete(1.0, tk.END)
                self.normalized_training_text.insert(tk.END, "Gamestate index out of range")
                self.normalized_training_text.config(state=tk.DISABLED)
                
        except FileNotFoundError:
            self.normalized_training_text.config(state=tk.NORMAL)
            self.normalized_training_text.delete(1.0, tk.END)
            self.normalized_training_text.insert(tk.END, "normalized_action_training_format.json not found.\nRun phase1_data_preparation.py first.")
            self.normalized_training_text.config(state=tk.DISABLED)
        except Exception as e:
            self.normalized_training_text.config(state=tk.NORMAL)
            self.normalized_training_text.delete(1.0, tk.END)
            self.normalized_training_text.insert(tk.END, f"Error loading normalized training data: {e}")
            self.normalized_training_text.config(state=tk.DISABLED)
    
    def breakdown_training_sequence(self, training_sequence):
        """Break down a training sequence into readable action descriptions"""
        if not training_sequence:
            return "Empty sequence"
        
        action_count = training_sequence[0]
        breakdown = []
        
        for action_idx in range(int(action_count)):
            base_idx = 1 + action_idx * 7
            if base_idx + 6 < len(training_sequence):
                timing = training_sequence[base_idx]
                action_type = training_sequence[base_idx + 1]
                x = training_sequence[base_idx + 2]
                y = training_sequence[base_idx + 3]
                button = training_sequence[base_idx + 4]
                key_or_scroll = training_sequence[base_idx + 5]
                
                action_type_names = {0: "Move", 1: "Click", 2: "Key", 3: "Scroll"}
                button_names = {0: "None", 1: "Left", 2: "Right", 3: "Middle"}
                
                action_desc = f"Action {action_idx + 1}: {action_type_names.get(int(action_type), 'Unknown')}"
                action_desc += f" at timing {timing:.3f}, position ({x:.3f}, {y:.3f})"
                
                if int(action_type) == 1:  # Click
                    action_desc += f", button: {button_names.get(int(button), 'Unknown')}"
                elif int(action_type) == 2:  # Key
                    action_desc += f", key hash: {key_or_scroll:.3f}"
                elif int(action_type) == 3:  # Scroll
                    action_desc += f", scroll delta: {key_or_scroll:.3f}"
                
                breakdown.append(action_desc)
        
        return breakdown
    
    def update_action_sequences_tab(self):
        """Update the action sequences tab"""
        try:
            # Load action sequences data
            with open('data/training_data/action_sequences.json', 'r') as f:
                action_sequences = json.load(f)
            
            if self.current_action_gamestate < len(action_sequences):
                sequence_data = action_sequences[self.current_action_gamestate]
                
                # Create formatted display
                seq_data = {
                    "metadata": {
                        "gamestate_index": self.current_action_gamestate,
                        "data_source": "action_sequences.json",
                        "processing_stage": "Step 4: After extracting action sequences",
                        "description": "Structured action sequences with relative timestamps (0-600ms window)",
                        "data_type": "Structured JSON with action metadata",
                        "normalization_status": "Raw data - no normalization applied yet",
                        "format": "Structured format with action categories and relative timing",
                        "timing_window": "600ms before each gamestate",
                        "note": "This tab shows the SAME timestep as all other tabs - just at this processing stage",
                        "action_count": sequence_data.get('action_count', 0),
                        "gamestate_timestamp": sequence_data.get('gamestate_timestamp', 0)
                    },
                    "sequence_data": sequence_data
                }
                
                # Format JSON
                formatted_json = json.dumps(seq_data, indent=2, ensure_ascii=False)
                
                # Update the action sequences tab
                self.action_sequences_text.config(state=tk.NORMAL)
                self.action_sequences_text.delete(1.0, tk.END)
                self.action_sequences_text.insert(tk.END, formatted_json)
                self.action_sequences_text.config(state=tk.DISABLED)
            else:
                self.action_sequences_text.config(state=tk.NORMAL)
                self.action_sequences_text.delete(1.0, tk.END)
                self.action_sequences_text.insert(tk.END, "Gamestate index out of range")
                self.action_sequences_text.config(state=tk.DISABLED)
                
        except FileNotFoundError:
            self.action_sequences_text.config(state=tk.NORMAL)
            self.action_sequences_text.delete(1.0, tk.END)
            self.action_sequences_text.insert(tk.END, "action_sequences.json not found.\nRun phase1_data_preparation.py first.")
            self.action_sequences_text.config(state=tk.DISABLED)
        except Exception as e:
            self.action_sequences_text.config(state=tk.NORMAL)
            self.action_sequences_text.delete(1.0, tk.END)
            self.action_sequences_text.insert(tk.END, f"Error loading action sequences: {e}")
            self.action_sequences_text.config(state=tk.DISABLED)
    
    def update_action_targets_tab(self):
        """Update the action targets tab"""
        try:
            # Load action targets data
            with open('data/training_data/action_targets.json', 'r') as f:
                action_targets = json.load(f)
            
            if self.current_action_gamestate < len(action_targets):
                target_data = action_targets[self.current_action_gamestate]
                
                # Create formatted display
                target_info = {
                    "metadata": {
                        "gamestate_index": self.current_action_gamestate,
                        "data_source": "action_targets.json",
                        "processing_stage": "Step 5: After creating variable-length targets",
                        "description": "Variable-length action targets for training (flattened format)",
                        "data_type": "Flattened numerical array for training targets",
                        "normalization_status": "Partially normalized - hardcoded scaling applied",
                        "format": "[action_count, timing1, type1, x1, y1, button1, key1, timing2, type2, x2, y2, button2, key2, ...]",
                        "normalization_details": "Timing: 0-1 (600ms window), Coordinates: 0-1 (800x600 screen), Keys: Hash normalized",
                        "note": "This tab shows the SAME timestep as all other tabs - just at this processing stage",
                        "target_length": len(target_data),
                        "action_count": target_data[0] if target_data else 0
                    },
                    "target_data": target_data,
                    "action_breakdown": self.breakdown_training_sequence(target_data)
                }
                
                # Format JSON
                formatted_json = json.dumps(target_info, indent=2, ensure_ascii=False)
                
                # Update the action targets tab
                self.action_targets_text.config(state=tk.NORMAL)
                self.action_targets_text.delete(1.0, tk.END)
                self.action_targets_text.insert(tk.END, formatted_json)
                self.action_targets_text.config(state=tk.DISABLED)
            else:
                self.action_targets_text.config(state=tk.NORMAL)
                self.action_targets_text.delete(1.0, tk.END)
                self.action_targets_text.insert(tk.END, "Gamestate index out of range")
                self.action_targets_text.config(state=tk.DISABLED)
                
        except FileNotFoundError:
            self.action_targets_text.config(state=tk.NORMAL)
            self.action_targets_text.delete(1.0, tk.END)
            self.action_targets_text.insert(tk.END, "action_targets.json not found.\nRun phase1_data_preparation.py first.")
            self.action_targets_text.config(state=tk.DISABLED)
        except Exception as e:
            self.action_targets_text.config(state=tk.NORMAL)
            self.action_targets_text.delete(1.0, tk.END)
            self.action_targets_text.insert(tk.END, f"Error loading action targets: {e}")
            self.action_targets_text.config(state=tk.DISABLED)
    
    def update_raw_tensors_tab(self):
        """Update the raw action tensors tab"""
        try:
            # Load raw action tensors data
            with open('data/training_data/raw_action_tensors.json', 'r') as f:
                raw_tensors = json.load(f)
            
            if self.current_action_gamestate < len(raw_tensors):
                tensor_data = raw_tensors[self.current_action_gamestate]
                
                # Create formatted display
                tensor_info = {
                    "metadata": {
                        "gamestate_index": self.current_action_gamestate,
                        "data_source": "raw_action_tensors.json",
                        "description": "Raw action tensors with 8 features per action",
                        "format": "[action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]",
                        "features_per_action": "timestamp, type, x, y, button, key, scroll_dx, scroll_dy",
                        "action_count": tensor_data[0] if tensor_data else 0,
                        "tensor_length": len(tensor_data)
                    },
                    "tensor_data": tensor_data,
                    "action_breakdown": self.breakdown_raw_tensor(tensor_data)
                }
                
                # Format JSON
                formatted_json = json.dumps(tensor_info, indent=2, ensure_ascii=False)
                
                # Update the raw tensors tab
                self.raw_tensors_text.config(state=tk.NORMAL)
                self.raw_tensors_text.delete(1.0, tk.END)
                self.raw_tensors_text.insert(tk.END, formatted_json)
                self.raw_tensors_text.config(state=tk.DISABLED)
            else:
                self.raw_tensors_text.config(state=tk.NORMAL)
                self.raw_tensors_text.delete(1.0, tk.END)
                self.raw_tensors_text.insert(tk.END, "Gamestate index out of range")
                self.action_targets_text.config(state=tk.DISABLED)
                
        except FileNotFoundError:
            self.raw_tensors_text.config(state=tk.NORMAL)
            self.raw_tensors_text.delete(1.0, tk.END)
            self.raw_tensors_text.insert(tk.END, "raw_action_tensors.json not found.\nRun phase1_data_preparation.py first.")
            self.raw_tensors_text.config(state=tk.DISABLED)
        except Exception as e:
            self.raw_tensors_text.config(state=tk.NORMAL)
            self.raw_tensors_text.delete(1.0, tk.END)
            self.raw_tensors_text.insert(tk.END, f"Error loading raw tensors: {e}")
            self.raw_tensors_text.config(state=tk.DISABLED)
    
    def update_normalized_tensors_tab(self):
        """Update the normalized action tensors tab"""
        try:
            # Load normalized action tensors data
            with open('data/training_data/normalized_action_training_format.json', 'r') as f:
                normalized_tensors = json.load(f)
            
            if self.current_action_gamestate < len(normalized_tensors):
                tensor_data = normalized_tensors[self.current_action_gamestate]
                
                # Create formatted display
                tensor_info = {
                    "metadata": {
                        "gamestate_index": self.current_action_gamestate,
                        "data_source": "normalized_action_training_format.json",
                        "description": "Normalized action tensors with 8 features per action",
                        "format": "[action_count, timestamp1, type1, x1, y1, button1, key1, scroll_dx1, scroll_dy1, ...]",
                        "features_per_action": "timestamp, type, x, y, button, key, scroll_dx, scroll_dy",
                        "normalization": "Timestamps: ms→minutes, Coordinates: RobustScaler",
                        "action_count": tensor_data[0] if tensor_data else 0,
                        "tensor_length": len(tensor_data)
                    },
                    "tensor_data": tensor_data,
                    "action_breakdown": self.breakdown_raw_tensor(tensor_data)
                }
                
                # Format JSON
                formatted_json = json.dumps(tensor_info, indent=2, ensure_ascii=False)
                
                # Update the normalized tensors tab
                self.normalized_tensors_text.config(state=tk.NORMAL)
                self.normalized_tensors_text.delete(1.0, tk.END)
                self.normalized_tensors_text.insert(tk.END, formatted_json)
                self.normalized_tensors_text.config(state=tk.DISABLED)
            else:
                self.normalized_tensors_text.config(state=tk.NORMAL)
                self.normalized_tensors_text.delete(1.0, tk.END)
                self.normalized_tensors_text.insert(tk.END, "Gamestate index out of range")
                
        except FileNotFoundError:
            self.normalized_tensors_text.config(state=tk.NORMAL)
            self.normalized_tensors_text.delete(1.0, tk.END)
            self.normalized_tensors_text.insert(tk.END, "normalized_action_training_format.json not found.\nRun phase1_data_preparation.py first.")
            self.normalized_tensors_text.config(state=tk.DISABLED)
        except Exception as e:
            self.normalized_tensors_text.config(state=tk.NORMAL)
            self.normalized_tensors_text.delete(1.0, tk.END)
            self.normalized_tensors_text.insert(tk.END, f"Error loading normalized tensors: {e}")
            self.normalized_tensors_text.config(state=tk.DISABLED)
    
    def breakdown_raw_tensor(self, tensor_data):
        """Break down a raw action tensor into readable action descriptions"""
        if not tensor_data:
            return "Empty tensor"
        
        action_count = tensor_data[0]
        breakdown = []
        
        for action_idx in range(int(action_count)):
            base_idx = 1 + action_idx * 8
            if base_idx + 7 < len(tensor_data):
                timestamp = tensor_data[base_idx]
                action_type = tensor_data[base_idx + 1]
                x = tensor_data[base_idx + 2]
                y = tensor_data[base_idx + 3]
                button = tensor_data[base_idx + 4]
                key = tensor_data[base_idx + 5]
                scroll_dx = tensor_data[base_idx + 6]
                scroll_dy = tensor_data[base_idx + 7]
                
                action_type_names = {0: "Move", 1: "Click", 2: "Key", 3: "Scroll"}
                button_names = {0: "None", 1: "Left", 2: "Right", 3: "Middle"}
                
                action_desc = f"Action {action_idx + 1}: {action_type_names.get(int(action_type), 'Unknown')}"
                action_desc += f" at timestamp {timestamp:.3f}, position ({x:.3f}, {y:.3f})"
                
                if int(action_type) == 1:  # Click
                    action_desc += f", button: {button_names.get(int(button), 'Unknown')}"
                elif int(action_type) == 2: # Key
                    action_desc += f", key hash: {key:.3f}"
                elif int(action_type) == 3: # Scroll
                    action_desc += f", scroll delta: ({scroll_dx:.3f}, {scroll_dy:.3f})"
                
                breakdown.append(action_desc)
        
        return breakdown
    
    # The convert_actions_to_training_format method has been removed
    # All data processing is now done in phase1_data_preparation.py
    # The browser only loads and displays pre-processed data
    
    def update_action_gamestate_range(self):
        """Update the action gamestate spinbox range"""
        if hasattr(self, 'action_gamestate_spinbox') and self.raw_action_data:
            max_gamestate = len(self.raw_action_data) - 1
            self.action_gamestate_spinbox.configure(to=max_gamestate)
            # Update current gamestate if it's out of range
            if self.current_action_gamestate > max_gamestate:
                self.current_action_gamestate = max_gamestate
                self.action_gamestate_var.set(str(max_gamestate))
    
    def create_sequence_alignment_display(self):
        """Create the sequence alignment display to show gamestate-action relationships"""
        # Main info frame
        info_frame = ttk.LabelFrame(self.sequence_alignment_frame, text="Sequence Alignment Info", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = """
        This tab shows how gamestates and their action sequences are aligned for training:
        
        • Each gamestate contains the last 600ms of actions that happened BEFORE it
        • Training sequence: Use gamestates 0-9 (with their embedded action history) to predict actions for gamestate 10
        • This creates rich temporal context for the model to learn from
        
        The alignment ensures the model sees both the game state AND the actions that led to it.
        """
        
        info_label = ttk.Label(info_frame, text=info_text, font=("Arial", 10), justify=tk.LEFT)
        info_label.pack(anchor=tk.W)
        
        # Sequence navigation frame
        nav_frame = ttk.LabelFrame(self.sequence_alignment_frame, text="Sequence Navigation", padding="10")
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Sequence selector
        seq_frame = ttk.Frame(nav_frame)
        seq_frame.pack(fill=tk.X)
        
        ttk.Label(seq_frame, text="Training Sequence:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        self.sequence_var = tk.StringVar(value="0")
        self.sequence_spinbox = ttk.Spinbox(
            seq_frame, 
            from_=0, 
            to=100,  # Will be updated based on available data
            textvariable=self.sequence_var,
            width=10,
            command=self.on_sequence_change
        )
        self.sequence_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(seq_frame, text="◀ Previous", command=self.previous_sequence).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(seq_frame, text="Next ▶", command=self.next_sequence).pack(side=tk.LEFT, padx=(0, 10))
        
        # Sequence info
        self.sequence_info_label = ttk.Label(seq_frame, text="", font=("Arial", 10))
        self.sequence_info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Sequence visualization frame
        viz_frame = ttk.LabelFrame(self.sequence_alignment_frame, text="Sequence Visualization", padding="10")
        viz_frame.pack(fill=tk.X, padx=10, pady=5)  # Changed from BOTH to X, removed expand=True
        
        # Create Treeview for sequence display
        columns = ("Gamestate", "Timestamp", "Actions (600ms before)", "Action Count", "Key Actions")
        self.sequence_tree = ttk.Treeview(viz_frame, columns=columns, show="headings", height=8)  # Reduced height from 15 to 8
        
        # Configure columns
        self.sequence_tree.heading("Gamestate", text="Gamestate Index")
        self.sequence_tree.heading("Timestamp", text="Timestamp")
        self.sequence_tree.heading("Actions (600ms before)", text="Actions (600ms before)")
        self.sequence_tree.heading("Action Count", text="Action Count")
        self.sequence_tree.heading("Key Actions", text="Key Actions")
        self.sequence_tree.column("Gamestate", width=100)
        self.sequence_tree.column("Timestamp", width=150)
        self.sequence_tree.column("Actions (600ms before)", width=200)
        self.sequence_tree.column("Action Count", width=100)
        self.sequence_tree.column("Key Actions", width=300)
        
        # Scrollbar
        sequence_scrollbar = ttk.Scrollbar(viz_frame, orient=tk.VERTICAL, command=self.sequence_tree.yview)
        self.sequence_tree.configure(yscrollcommand=sequence_scrollbar.set)
        
        # Pack widgets
        self.sequence_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sequence_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Target info frame with scrollbar
        target_frame = ttk.LabelFrame(self.sequence_alignment_frame, text="Target Actions (Next 600ms)", padding="10")
        target_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)  # Changed to BOTH and expand=True
        
        # Create a frame for the target info with scrollbar
        target_scroll_frame = ttk.Frame(target_frame)
        target_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas and scrollbar for the target info
        target_canvas = tk.Canvas(target_scroll_frame)
        target_scrollbar = ttk.Scrollbar(target_scroll_frame, orient=tk.VERTICAL, command=target_canvas.yview)
        target_scrollable_frame = ttk.Frame(target_canvas)
        
        target_scrollable_frame.bind(
            "<Configure>",
            lambda e: target_canvas.configure(scrollregion=target_canvas.bbox("all"))
        )
        
        target_canvas.create_window((0, 0), window=target_scrollable_frame, anchor="nw")
        target_canvas.configure(yscrollcommand=target_scrollbar.set)
        
        # Target info label inside the scrollable frame
        self.target_info_label = ttk.Label(target_scrollable_frame, text="", font=("Arial", 10), justify=tk.LEFT, wraplength=800)
        self.target_info_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Pack the canvas and scrollbar
        target_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        target_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize sequence alignment display for current sequence
        self.display_sequence_alignment(0)
    
    def display_sequence_alignment(self, sequence_idx):
        """Display the alignment for a specific training sequence"""
        if not hasattr(self, 'features') or not hasattr(self, 'raw_action_data'):
            return
        
        # Clear existing items
        for item in self.sequence_tree.get_children():
            self.sequence_tree.delete(item)
        
        # Convert training sequence index to gamestate indices
        # Training sequences are 0-179, but we need to map to the trimmed gamestates (0-191)
        # Since we trimmed 5 timesteps from the start, we need to add 5 to get the actual gamestate indices
        trimmed_start_offset = 5  # This matches the trimming in phase1_data_preparation.py
        
        start_idx = sequence_idx + trimmed_start_offset
        end_idx = start_idx + 9  # 10 gamestates (0-9)
        target_idx = end_idx + 1  # Target is gamestate 10
        
        if target_idx >= len(self.features):
            self.sequence_info_label.config(text="Sequence extends beyond available data")
            return
        
        # Update sequence info
        self.sequence_info_label.config(text=f"Sequence {sequence_idx}: Gamestates {start_idx}-{end_idx} → Target {target_idx}")
        
        # Display input gamestates (0-9)
        for i in range(start_idx, end_idx + 1):
            if i < len(self.raw_action_data):
                action_data = self.raw_action_data[i]
                timestamp = self.features[i, -1] if self.features is not None else 0
                
                # Get action summary
                action_count = len(action_data.get('mouse_movements', [])) + len(action_data.get('clicks', [])) + \
                              len(action_data.get('key_presses', [])) + len(action_data.get('key_releases', [])) + \
                              len(action_data.get('scrolls', []))
                
                # Get key actions (first few actions for display)
                key_actions = []
                all_actions = []
                
                for move in action_data.get('mouse_movements', [])[:3]:
                    all_actions.append(f"Move({move.get('x', 0)},{move.get('y', 0)})")
                for click in action_data.get('clicks', [])[:3]:
                    all_actions.append(f"Click({click.get('button', 'left')})")
                for key in action_data.get('key_presses', [])[:3]:
                    all_actions.append(f"Key({key.get('key', '')})")
                for scroll in action_data.get('scrolls', [])[:3]:
                    all_actions.append(f"Scroll({scroll.get('dx', 0)},{scroll.get('dy', 0)})")
                
                key_actions_str = ", ".join(all_actions[:5])  # Show first 5 actions
                if len(all_actions) > 5:
                    key_actions_str += f" (+{len(all_actions) - 5} more)"
                
                self.sequence_tree.insert("", "end", values=(
                    f"Gamestate {i}",
                    f"{timestamp:.2f}",
                    f"600ms before gamestate {i}",
                    action_count,
                    key_actions_str
                ))
        
        # Display target info
        if target_idx < len(self.raw_action_data):
            target_action_data = self.raw_action_data[target_idx]
            target_timestamp = self.features[target_idx, -1] if self.features is not None else 0
            
            target_action_count = len(target_action_data.get('mouse_movements', [])) + len(target_action_data.get('clicks', [])) + \
                                 len(target_action_data.get('key_presses', [])) + len(target_action_data.get('key_releases', [])) + \
                                 len(target_action_data.get('scrolls', []))
            
            # Get target action summary
            target_actions = []
            for move in target_action_data.get('mouse_movements', [])[:3]:
                target_actions.append(f"Move({move.get('x', 0)},{move.get('y', 0)})")
            for click in target_action_data.get('clicks', [])[:3]:
                target_actions.append(f"Click({click.get('button', 'left')})")
            for key in target_action_data.get('key_presses', [])[:3]:
                target_actions.append(f"Key({key.get('key', '')})")
            for scroll in target_action_data.get('scrolls', [])[:3]:
                target_actions.append(f"Scroll({scroll.get('dx', 0)},{scroll.get('dy', 0)})")
            
            target_actions_str = ", ".join(target_actions[:5])
            if len(target_actions) > 5:
                target_actions_str += f" (+{len(target_actions) - 5} more)"
            
            self.target_info_label.config(text=f"""
            Target Gamestate {target_idx} (Timestamp: {target_timestamp:.2f}):
            • Actions in 600ms AFTER gamestate {end_idx}: {target_action_count} actions
            • Key actions: {target_actions_str}
            • This is what the model learns to predict from the input sequence
            """)
        else:
            self.target_info_label.config(text="Target gamestate not available")
    
    def create_mini_visualization(self):
        """Create a mini visualization of the selected feature values over time"""
        print(f"Creating mini visualization for feature {self.selected_feature_idx}")  # Debug print
        
        if self.selected_feature_idx is None or not hasattr(self, 'input_sequences'):
            print("Cannot create visualization: selected_feature_idx is None or no input_sequences")  # Debug print
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import numpy as np
            
            # Clear previous chart
            for widget in self.mini_chart_frame.winfo_children():
                widget.destroy()
            
            # Create figure for mini chart
            fig = Figure(figsize=(8, 3), dpi=80)
            canvas = FigureCanvasTkAgg(fig, self.mini_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create subplot
            ax = fig.add_subplot(111)
            
            feature_idx = self.selected_feature_idx
            
            # Get data for visualization
            if hasattr(self, 'features') and self.features is not None:
                # Use the full features array for better visualization
                all_values = self.features[:, feature_idx]
                timesteps = range(len(all_values))
                
                # Plot the feature values over time
                ax.plot(timesteps, all_values, 'b-', linewidth=1, alpha=0.7)
                ax.scatter(timesteps[::10], all_values[::10], c='red', s=20, alpha=0.6)  # Sample every 10th point
                
                ax.set_xlabel('Gamestate Index')
                ax.set_ylabel('Feature Value')
                ax.set_title(f'Feature {feature_idx}: {self.selected_feature_data.get("name", "Unknown")}')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.3f}')
                ax.axhline(y=mean_val + std_val, color='g', linestyle=':', alpha=0.5, label=f'+1σ: {mean_val + std_val:.3f}')
                ax.axhline(y=mean_val - std_val, color='g', linestyle=':', alpha=0.5, label=f'-1σ: {mean_val - std_val:.3f}')
                ax.legend(fontsize=8)
                
            else:
                # Fallback to input sequences if features not available
                max_seqs = min(5, self.input_sequences.shape[0])  # Show first 5 sequences
                for seq_idx in range(max_seqs):
                    sequence = self.input_sequences[seq_idx]
                    timesteps = range(sequence.shape[0])
                    values = sequence[:, feature_idx]
                    
                    ax.plot(timesteps, values, marker='o', markersize=3, linewidth=1, 
                           label=f'Seq {seq_idx}', alpha=0.7)
                
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Feature Value')
                ax.set_title(f'Feature {feature_idx}: {self.selected_feature_data.get("name", "Unknown")}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Store references
            self.mini_chart_canvas = canvas
            self.mini_chart_figure = fig
            
        except ImportError:
            # Fallback if matplotlib is not available
            error_label = ttk.Label(self.mini_chart_frame, 
                                  text="Matplotlib is required for visualization.\nPlease install it with: pip install matplotlib", 
                                  font=("Arial", 10), justify=tk.CENTER)
            error_label.pack(expand=True)
        except Exception as e:
            # Fallback for any other errors
            error_label = ttk.Label(self.mini_chart_frame, 
                                  text=f"Visualization error: {str(e)}", 
                                  font=("Arial", 10), justify=tk.CENTER)
            error_label.pack(expand=True)
    
    def print_normalization_info(self):
        """Print information about the loaded normalized data"""
        print("\n" + "="*60)
        print("NORMALIZED DATA STATUS")
        print("="*60)
        
        if self.normalized_features is not None:
            print(f"✓ Normalized features loaded: {self.normalized_features.shape}")
        else:
            print("✗ Normalized features not available")
            
        if self.normalized_input_sequences is not None:
            print(f"✓ Normalized input sequences loaded: {self.normalized_input_sequences.shape}")
        else:
            print("✗ Normalized input sequences not available")
            
        if hasattr(self, 'normalized_action_data') and self.normalized_action_data is not None:
            print(f"✓ Normalized action data loaded: {len(self.normalized_action_data)} gamestates")
        else:
            print("✗ Normalized action data not available")
            
        print("\nNote: All normalization is now pre-computed in phase1_data_preparation.py")
        print("No more on-the-fly normalization - world coordinates are preserved!")
        print("="*60)
    
    def get_current_features(self):
        """Get the current features (raw or normalized) based on normalization toggle"""
        if hasattr(self, 'show_normalized_data') and self.show_normalized_data.get():
            if self.normalized_features is not None:
                print(f"DEBUG: Returning normalized features: {self.normalized_features.shape}")
                # Debug: Show some sample values
                if self.normalized_features.shape[1] > 110:  # Check if we have NPC features
                    print(f"DEBUG: Sample NPC coordinates from normalized data:")
                    print(f"  npc_1_x (feature 110): {self.normalized_features[:3, 110]}")
                    print(f"  npc_1_y (feature 111): {self.normalized_features[:3, 111]}")
                    print(f"  npc_2_x (feature 113): {self.normalized_features[:3, 113]}")
                    print(f"  npc_2_y (feature 114): {self.normalized_features[:3, 114]}")
                return self.normalized_features
            else:
                print("Warning: Normalized features not available, showing raw data")
                return self.features
        else:
            print(f"DEBUG: Returning raw features: {self.features.shape}")
            return self.features
    
    def get_current_input_sequences(self):
        """Get the current input sequences (raw or normalized) based on normalization toggle"""
        if hasattr(self, 'show_normalized_data') and self.show_normalized_data.get():
            if self.normalized_input_sequences is not None:
                return self.normalized_input_sequences
            else:
                print("Warning: Normalized input sequences not available, showing raw data")
                return self.input_sequences
        else:
            return self.input_sequences
    
    def get_normalization_info(self, feature_idx):
        """Get normalization information for a specific feature"""
        # Get feature info from feature mappings
        feature_info = None
        for feature_data in self.feature_mappings:
            if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                feature_info = feature_data
                break
        
        if feature_info:
            data_type = feature_info.get('data_type', 'unknown')
            feature_name = feature_info.get('feature_name', 'unknown')
            
            if data_type == 'world_coordinate':
                return f"World Coordinate: No normalization - {feature_name} (spatial position preserved)"
            elif data_type in ['item_id', 'animation_id', 'hashed_string', 'boolean', 'tab_id', 'object_id', 'npc_id', 'count', 'slot_id', 'skill_level', 'skill_xp']:
                return f"Categorical: No normalization - {feature_name} ({data_type} values kept as-is)"
            elif data_type in ['camera_coordinate', 'angle_degrees', 'time_ms', 'screen_coordinate']:
                return f"Normalized: {feature_name} (robust scaling applied)"
            else:
                return f"Feature: {feature_name} (Data Type: {data_type})"
        else:
            return "Unknown feature"
    
    # Data trimming is now handled in phase1_data_preparation.py
    # This ensures training data is clean from the start
    pass
    
    # Trimming info is now handled in phase1_data_preparation.py
    pass
    
    def print_normalization_stats(self):
        """Print normalization information for pre-computed data"""
        self.print_normalization_info()
    
    def create_normalization_strategy_display(self):
        """Create the normalization strategy management interface"""
        # Main container
        main_container = ttk.Frame(self.normalization_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and description
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="Normalization Strategy Manager", font=("Arial", 14, "bold")).pack()
        ttk.Label(title_frame, text="Configure how features are normalized and grouped for consistent scaling", font=("Arial", 10)).pack()
        
        # Split into left and right panels
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
        # Left panel: Feature Groups
        left_panel = ttk.LabelFrame(content_frame, text="Feature Groups & Normalization", padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Feature groups treeview
        groups_frame = ttk.Frame(left_panel)
        groups_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(groups_frame, text="Feature Groups:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Create groups treeview
        self.groups_tree = ttk.Treeview(groups_frame, columns=("Group", "Method", "Features", "Description"), show="headings", height=15)
        self.groups_tree.heading("Group", text="Group Name")
        self.groups_tree.heading("Method", text="Normalization Method")
        self.groups_tree.heading("Features", text="Feature Count")
        self.groups_tree.heading("Description", text="Description")
        
        # Set column widths
        self.groups_tree.column("Group", width=120)
        self.groups_tree.column("Method", width=120)
        self.groups_tree.column("Features", width=80)
        self.groups_tree.column("Description", width=200)
        
        # Bind double-click to view group details
        self.groups_tree.bind('<Double-1>', self.view_group_details)
        
        # Bind right-click for context menu
        self.groups_tree.bind('<Button-3>', self.show_groups_context_menu)
        
        # Scrollbar for groups
        groups_scrollbar = ttk.Scrollbar(groups_frame, orient="vertical", command=self.groups_tree.yview)
        self.groups_tree.configure(yscrollcommand=groups_scrollbar.set)
        
        self.groups_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        groups_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Group management buttons
        group_buttons_frame = ttk.Frame(left_panel)
        group_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(group_buttons_frame, text="➕ Add Group", command=self.add_normalization_group).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(group_buttons_frame, text="✏️ Edit Group", command=self.edit_normalization_group).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(group_buttons_frame, text="🗑️ Delete Group", command=self.delete_normalization_group).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(group_buttons_frame, text="💾 Save Strategy", command=self.save_normalization_strategy).pack(side=tk.RIGHT)
        
        # Right panel: Feature Assignment
        right_panel = ttk.LabelFrame(content_frame, text="Feature Assignment", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Feature assignment controls
        assignment_frame = ttk.Frame(right_panel)
        assignment_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(assignment_frame, text="Available Features:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Feature filter
        filter_frame = ttk.Frame(assignment_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.feature_filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.feature_filter_var, 
                                   values=["all", "gamestate", "action", "unassigned"], width=15)
        filter_combo.pack(side=tk.LEFT, padx=(5, 0))
        filter_combo.bind('<<ComboboxSelected>>', self.filter_features)
        
        # Features treeview
        features_frame = ttk.Frame(assignment_frame)
        features_frame.pack(fill=tk.BOTH, expand=True)
        
        self.features_tree = ttk.Treeview(features_frame, columns=("Index", "Name", "Type", "Group", "Current Value"), show="headings", height=15)
        self.features_tree.heading("Index", text="Index")
        self.features_tree.heading("Name", text="Feature Name")
        self.features_tree.heading("Type", text="Data Type")
        self.features_tree.heading("Group", text="Assigned Group")
        self.features_tree.heading("Current Value", text="Sample Value")
        
        # Set column widths
        self.features_tree.column("Index", width=50)
        self.features_tree.column("Name", width=150)
        self.features_tree.column("Type", width=100)
        self.features_tree.column("Group", width=100)
        self.features_tree.column("Current Value", width=100)
        
        # Scrollbar for features
        features_scrollbar = ttk.Scrollbar(features_frame, orient="vertical", command=self.features_tree.yview)
        self.features_tree.configure(yscrollcommand=features_scrollbar.set)
        
        self.features_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        features_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature assignment buttons
        assign_buttons_frame = ttk.Frame(right_panel)
        assign_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(assign_buttons_frame, text="🔗 Assign to Group", command=self.assign_feature_to_group).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(assign_buttons_frame, text="🔓 Remove Assignment", command=self.remove_feature_assignment).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(assign_buttons_frame, text="🔄 Refresh", command=self.refresh_feature_display).pack(side=tk.RIGHT)
        
        # Bottom panel: Strategy Preview
        bottom_panel = ttk.LabelFrame(main_container, text="Strategy Preview & Export", padding="10")
        bottom_panel.pack(fill=tk.X, pady=(10, 0))
        
        # Strategy info
        strategy_info_frame = ttk.Frame(bottom_panel)
        strategy_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.strategy_info_label = ttk.Label(strategy_info_frame, text="No normalization strategy configured", font=("Arial", 10))
        self.strategy_info_label.pack(anchor=tk.W)
        
        # Export buttons
        export_frame = ttk.Frame(bottom_panel)
        export_frame.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="📋 Copy Strategy to Clipboard", command=self.copy_strategy_to_clipboard).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="💾 Export Strategy to JSON", command=self.export_strategy_to_json).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="📊 Apply Strategy to Data", command=self.apply_strategy_to_data).pack(side=tk.RIGHT)
        
        # Strategy Creation Panel
        strategy_creation_panel = ttk.LabelFrame(main_container, text="Normalization Strategy Creation", padding="10")
        strategy_creation_panel.pack(fill=tk.X, pady=(10, 0))
        
        # Strategy creation controls
        creation_frame = ttk.Frame(strategy_creation_panel)
        creation_frame.pack(fill=tk.X)
        
        # Left side: Strategy templates
        templates_frame = ttk.LabelFrame(creation_frame, text="Strategy Templates", padding="5")
        templates_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(templates_frame, text="Quick setup templates:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        template_buttons_frame = ttk.Frame(templates_frame)
        template_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(template_buttons_frame, text="🎯 Conservative", 
                  command=lambda: self.apply_strategy_template("conservative")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(template_buttons_frame, text="⚡ Aggressive", 
                  command=lambda: self.apply_strategy_template("aggressive")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(template_buttons_frame, text="🔄 Balanced", 
                  command=lambda: self.apply_strategy_template("balanced")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(template_buttons_frame, text="🧹 Reset to Defaults", 
                  command=self.reset_to_default_groups).pack(side=tk.LEFT, padx=(0, 5))
        
        # Right side: Custom strategy creation
        custom_frame = ttk.LabelFrame(creation_frame, text="Custom Strategy", padding="5")
        custom_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(custom_frame, text="Create custom normalization rules:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        custom_buttons_frame = ttk.Frame(custom_frame)
        custom_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(custom_buttons_frame, text="🔧 Create Custom Rule", 
                  command=self.create_custom_normalization_rule).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(custom_buttons_frame, text="📊 Batch Assign by Type", 
                  command=self.batch_assign_by_type).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(custom_buttons_frame, text="🎨 Smart Auto-Assign", 
                  command=self.smart_auto_assign).pack(side=tk.LEFT, padx=(0, 5))
        
        # Initialize with default groups
        self.initialize_default_groups()
        self.populate_features_tree()
    
    def initialize_default_groups(self):
        """Initialize with sensible default normalization groups"""
        self.normalization_groups = {
            "world_coordinates": {
                "method": "none",
                "description": "World coordinates - no normalization (preserve spatial relationships)",
                "features": []
            },
            "screen_coordinates": {
                "method": "robust_scaler",
                "description": "Screen coordinates - robust scaling (preserve UI relationships)",
                "features": []
            },
            "timestamps": {
                "method": "ms_to_minutes",
                "description": "Timestamps - convert milliseconds to minutes (consistent units)",
                "features": []
            },
            "camera_angles": {
                "method": "robust_scaler",
                "description": "Camera angles - robust scaling (preserve angular relationships)",
                "features": []
            },
            "categorical": {
                "method": "none",
                "description": "Categorical data - no normalization (preserve IDs and values)",
                "features": []
            },
            "continuous_scaled": {
                "method": "robust_scaler",
                "description": "Other continuous features - robust scaling",
                "features": []
            }
        }
        
        # Automatically assign features to groups based on their types
        self.auto_assign_features_to_groups()
        
        self.update_groups_tree()
    
    def auto_assign_features_to_groups(self):
        """Automatically assign features to groups based on their data types"""
        print("Auto-assigning features to normalization groups...")
        
        for feature_idx in range(128):  # 128 gamestate features
            feature_type = self.get_feature_type(feature_idx)
            
            # Assign based on data type
            if feature_type == "world_coordinate":
                self.normalization_groups["world_coordinates"]["features"].append(feature_idx)
            elif feature_type == "screen_coordinate":
                self.normalization_groups["screen_coordinates"]["features"].append(feature_idx)
            elif feature_type == "time_ms":
                self.normalization_groups["timestamps"]["features"].append(feature_idx)
            elif feature_type in ["camera_coordinate", "angle_degrees"]:
                self.normalization_groups["camera_angles"]["features"].append(feature_idx)
            elif feature_type in ["item_id", "animation_id", "hashed_string", "boolean", "tab_id", "object_id", "npc_id", "count", "slot_id", "skill_level", "skill_xp"]:
                self.normalization_groups["categorical"]["features"].append(feature_idx)
            else:
                # Default to continuous scaled for unknown types
                self.normalization_groups["continuous_scaled"]["features"].append(feature_idx)
        
        # Print assignment summary
        for group_name, group_info in self.normalization_groups.items():
            print(f"  {group_name}: {len(group_info['features'])} features")
    
    def update_groups_tree(self):
        """Update the groups treeview display"""
        # Clear existing items
        for item in self.groups_tree.get_children():
            self.groups_tree.delete(item)
        
        # Add each group
        for group_name, group_info in self.normalization_groups.items():
            feature_count = len(group_info["features"])
            self.groups_tree.insert("", "end", values=(
                group_name,
                group_info["method"],
                feature_count,
                group_info["description"]
            ))
    
    def populate_features_tree(self):
        """Populate the features treeview with available features"""
        # Clear existing items
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        
        # Add gamestate features
        for feature_idx in range(128):  # 128 gamestate features
            feature_name = self.get_feature_name(feature_idx)
            feature_type = self.get_feature_type(feature_idx)
            assigned_group = self.get_assigned_group(feature_idx)
            sample_value = self.get_sample_value(feature_idx)
            
            self.features_tree.insert("", "end", values=(
                feature_idx,
                feature_name,
                feature_type,
                assigned_group,
                sample_value
            ))
        
        # Add action features (if available)
        if hasattr(self, 'raw_action_data') and self.raw_action_data:
            # Action features are dynamic, so we'll add them as a special category
            self.features_tree.insert("", "end", values=(
                "A*",
                "Action Features (Dynamic)",
                "action_tensor",
                "action_normalization",
                f"{len(self.raw_action_data)} gamestates"
            ))
    
    def get_feature_name(self, feature_idx):
        """Get the name of a feature by index"""
        if str(feature_idx) in self.feature_names:
            return self.feature_names[str(feature_idx)].get('feature_name', f'feature_{feature_idx}')
        return f'feature_{feature_idx}'
    
    def get_feature_type(self, feature_idx):
        """Get the data type of a feature by index"""
        for feature_data in self.feature_mappings:
            if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                return feature_data.get('data_type', 'unknown')
        return 'unknown'
    
    def get_assigned_group(self, feature_idx):
        """Get the assigned normalization group for a feature"""
        for group_name, group_info in self.normalization_groups.items():
            if feature_idx in group_info["features"]:
                return group_name
        return "unassigned"
    
    def get_sample_value(self, feature_idx):
        """Get a sample value for a feature"""
        try:
            if hasattr(self, 'features') and self.features is not None:
                if feature_idx < self.features.shape[1]:
                    sample_val = self.features[0, feature_idx]
                    if isinstance(sample_val, (int, float)):
                        return f"{sample_val:.3f}"
                    return str(sample_val)
        except:
            pass
        return "N/A"
    
    def filter_features(self, event=None):
        """Filter features based on selection"""
        filter_value = self.feature_filter_var.get()
        
        # Clear existing items
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        
        # Add gamestate features based on filter
        for feature_idx in range(128):  # 128 gamestate features
            feature_name = self.get_feature_name(feature_idx)
            feature_type = self.get_feature_type(feature_idx)
            assigned_group = self.get_assigned_group(feature_idx)
            sample_value = self.get_sample_value(feature_idx)
            
            # Apply filter
            if filter_value == "all":
                show_feature = True
            elif filter_value == "gamestate":
                show_feature = True  # All gamestate features
            elif filter_value == "action":
                show_feature = False  # No action features in gamestate
            elif filter_value == "unassigned":
                show_feature = assigned_group == "unassigned"
            else:
                show_feature = True
            
            if show_feature:
                self.features_tree.insert("", "end", values=(
                    feature_idx,
                    feature_name,
                    feature_type,
                    assigned_group,
                    sample_value
                ))
        
        # Add action features if showing all or action features
        if filter_value in ["all", "action"] and hasattr(self, 'raw_action_data') and self.raw_action_data:
            self.features_tree.insert("", "end", values=(
                "A*",
                "Action Features (Dynamic)",
                "action_tensor",
                "action_normalization",
                f"{len(self.raw_action_data)} gamestates"
            ))
    
    def add_normalization_group(self):
        """Add a new normalization group"""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Normalization Group")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Group name
        ttk.Label(dialog, text="Group Name:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=40)
        name_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Normalization method
        ttk.Label(dialog, text="Normalization Method:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        method_var = tk.StringVar(value="robust_scaler")
        method_combo = ttk.Combobox(dialog, textvariable=method_var, 
                                   values=["none", "robust_scaler", "standard_scaler", "minmax_scaler", "ms_to_minutes", "custom"], 
                                   width=37)
        method_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Description
        ttk.Label(dialog, text="Description:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        desc_var = tk.StringVar()
        desc_entry = ttk.Entry(dialog, textvariable=desc_var, width=40)
        desc_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        def save_group():
            group_name = name_var.get().strip()
            method = method_var.get()
            description = desc_var.get().strip()
            
            if not group_name:
                messagebox.showerror("Error", "Group name is required!")
                return
            
            if group_name in self.normalization_groups:
                messagebox.showerror("Error", "Group name already exists!")
                return
            
            # Add the new group
            self.normalization_groups[group_name] = {
                "method": method,
                "description": description,
                "features": []
            }
            
            self.update_groups_tree()
            self.update_strategy_info()
            dialog.destroy()
            messagebox.showinfo("Success", f"Group '{group_name}' created successfully!")
        
        ttk.Button(button_frame, text="Save", command=save_group).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        
        # Focus on name entry
        name_entry.focus()
    
    def edit_normalization_group(self):
        """Edit an existing normalization group"""
        # Get selected group
        selection = self.groups_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a group to edit!")
            return
        
        group_name = self.groups_tree.item(selection[0])['values'][0]
        group_info = self.normalization_groups.get(group_name)
        
        if not group_info:
            messagebox.showerror("Error", "Selected group not found!")
            return
        
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit Group: {group_name}")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Group name (read-only)
        ttk.Label(dialog, text="Group Name:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        name_label = ttk.Label(dialog, text=group_name, font=("Arial", 10))
        name_label.pack(anchor=tk.W, padx=10, pady=(0, 10))
        
        # Normalization method
        ttk.Label(dialog, text="Normalization Method:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        method_var = tk.StringVar(value=group_info["method"])
        method_combo = ttk.Combobox(dialog, textvariable=method_var, 
                                   values=["none", "robust_scaler", "standard_scaler", "minmax_scaler", "ms_to_minutes", "custom"], 
                                   width=37)
        method_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Description
        ttk.Label(dialog, text="Description:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        desc_var = tk.StringVar(value=group_info["description"])
        desc_entry = ttk.Entry(dialog, textvariable=desc_var, width=40)
        desc_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        def save_changes():
            method = method_var.get()
            description = desc_var.get().strip()
            
            # Update the group
            self.normalization_groups[group_name]["method"] = method
            self.normalization_groups[group_name]["description"] = description
            
            self.update_groups_tree()
            self.update_strategy_info()
            dialog.destroy()
            messagebox.showinfo("Success", f"Group '{group_name}' updated successfully!")
        
        ttk.Button(button_frame, text="Save Changes", command=save_changes).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def delete_normalization_group(self):
        """Delete a normalization group"""
        # Get selected group
        selection = self.groups_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a group to delete!")
            return
        
        group_name = self.groups_tree.item(selection[0])['values'][0]
        group_info = self.normalization_groups.get(group_name)
        
        if not group_info:
            messagebox.showerror("Error", "Selected group not found!")
            return
        
        # Check if group has features
        if group_info["features"]:
            result = messagebox.askyesno("Confirm Delete", 
                                       f"Group '{group_name}' has {len(group_info['features'])} assigned features.\n"
                                       f"Deleting will unassign all features. Continue?")
            if not result:
                return
        
        # Confirm deletion
        result = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete group '{group_name}'?")
        if not result:
            return
        
        # Remove group and unassign features
        del self.normalization_groups[group_name]
        
        # Update display
        self.update_groups_tree()
        self.populate_features_tree()
        self.update_strategy_info()
        
        messagebox.showinfo("Success", f"Group '{group_name}' deleted successfully!")
    
    def assign_feature_to_group(self):
        """Assign a feature to a normalization group"""
        # Get selected feature
        selection = self.features_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a feature to assign!")
            return
        
        feature_item = self.features_tree.item(selection[0])
        feature_values = feature_item['values']
        
        # Check if it's an action feature (special case)
        if feature_values[0] == "A*":
            messagebox.showinfo("Info", "Action features are automatically assigned to the 'action_normalization' group.")
            return
        
        feature_idx = int(feature_values[0])
        feature_name = feature_values[1]
        
        # Create assignment dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Assign Feature: {feature_name}")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Feature info
        ttk.Label(dialog, text=f"Feature: {feature_name} (Index {feature_idx})", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        ttk.Label(dialog, text=f"Type: {feature_values[2]}", font=("Arial", 9)).pack(anchor=tk.W, padx=10, pady=(0, 10))
        
        # Current assignment
        current_group = feature_values[3]
        if current_group != "unassigned":
            ttk.Label(dialog, text=f"Currently assigned to: {current_group}", font=("Arial", 9)).pack(anchor=tk.W, padx=10, pady=(0, 10))
        
        # Group selection
        ttk.Label(dialog, text="Assign to group:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        group_var = tk.StringVar(value="unassigned")
        group_combo = ttk.Combobox(dialog, textvariable=group_var, 
                                  values=["unassigned"] + list(self.normalization_groups.keys()), 
                                  width=37)
        group_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        def save_assignment():
            selected_group = group_var.get()
            
            # Remove from current group
            for group_name, group_info in self.normalization_groups.items():
                if feature_idx in group_info["features"]:
                    group_info["features"].remove(feature_idx)
            
            # Add to new group
            if selected_group != "unassigned":
                self.normalization_groups[selected_group]["features"].append(feature_idx)
            
            # Update displays
            self.update_groups_tree()
            self.populate_features_tree()
            self.update_strategy_info()
            
            dialog.destroy()
            messagebox.showinfo("Success", f"Feature '{feature_name}' assigned to '{selected_group}'!")
        
        ttk.Button(button_frame, text="Save Assignment", command=save_assignment).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def remove_feature_assignment(self):
        """Remove a feature's group assignment"""
        # Get selected feature
        selection = self.features_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a feature to unassign!")
            return
        
        feature_item = self.features_tree.item(selection[0])
        feature_values = feature_item['values']
        
        # Check if it's an action feature
        if feature_values[0] == "A*":
            messagebox.showinfo("Info", "Action features cannot be unassigned from their group.")
            return
        
        feature_idx = int(feature_values[0])
        feature_name = feature_values[1]
        current_group = feature_values[3]
        
        if current_group == "unassigned":
            messagebox.showinfo("Info", "Feature is not assigned to any group.")
            return
        
        # Confirm removal
        result = messagebox.askyesno("Confirm Unassign", f"Remove feature '{feature_name}' from group '{current_group}'?")
        if not result:
            return
        
        # Remove from group
        self.normalization_groups[current_group]["features"].remove(feature_idx)
        
        # Update displays
        self.update_groups_tree()
        self.populate_features_tree()
        self.update_strategy_info()
        
        messagebox.showinfo("Success", f"Feature '{feature_name}' unassigned from '{current_group}'!")
    
    def refresh_feature_display(self):
        """Refresh the feature display"""
        self.populate_features_tree()
        self.update_strategy_info()
    
    def update_strategy_info(self):
        """Update the strategy info display"""
        total_features = 128  # Gamestate features
        assigned_features = sum(len(group["features"]) for group in self.normalization_groups.values())
        unassigned_features = total_features - assigned_features
        
        group_summary = []
        for group_name, group_info in self.normalization_groups.items():
            if group_info["features"]:
                group_summary.append(f"{group_name}: {len(group_info['features'])} features")
        
        strategy_text = f"Strategy Status: {assigned_features}/{total_features} features assigned to {len(group_summary)} groups"
        if unassigned_features > 0:
            strategy_text += f" ({unassigned_features} unassigned)"
        
        self.strategy_info_label.config(text=strategy_text)
    
    def save_normalization_strategy(self):
        """Save the current normalization strategy"""
        # Get file path from user
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Normalization Strategy"
        )
        
        if file_path:
            try:
                # Prepare strategy data
                strategy_data = {
                    "metadata": {
                        "created": str(pd.Timestamp.now()),
                        "description": "Normalization strategy for OSRS training data",
                        "total_features": 128,
                        "groups_count": len(self.normalization_groups)
                    },
                    "normalization_groups": self.normalization_groups,
                    "feature_assignments": {}
                }
                
                # Add feature assignments
                for group_name, group_info in self.normalization_groups.items():
                    for feature_idx in group_info["features"]:
                        strategy_data["feature_assignments"][str(feature_idx)] = {
                            "group": group_name,
                            "method": group_info["method"],
                            "feature_name": self.get_feature_name(feature_idx),
                            "feature_type": self.get_feature_type(feature_idx)
                        }
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(strategy_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Normalization strategy saved to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save strategy: {e}")
    
    def copy_strategy_to_clipboard(self):
        """Copy the normalization strategy to clipboard"""
        try:
            # Prepare strategy summary
            strategy_summary = "Normalization Strategy Summary\n"
            strategy_summary += "=" * 40 + "\n\n"
            
            for group_name, group_info in self.normalization_groups.items():
                strategy_summary += f"Group: {group_name}\n"
                strategy_summary += f"Method: {group_info['method']}\n"
                strategy_summary += f"Description: {group_info['description']}\n"
                strategy_summary += f"Features: {len(group_info['features'])}\n"
                
                if group_info['features']:
                    feature_names = [self.get_feature_name(idx) for idx in group_info['features']]
                    strategy_summary += f"Feature Names: {', '.join(feature_names[:5])}"
                    if len(feature_names) > 5:
                        strategy_summary += f" ... and {len(feature_names) - 5} more"
                    strategy_summary += "\n"
                
                strategy_summary += "\n"
            
            pyperclip.copy(strategy_summary)
            messagebox.showinfo("Success", "Normalization strategy copied to clipboard!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy strategy: {e}")
    
    def export_strategy_to_json(self):
        """Export the normalization strategy to JSON"""
        self.save_normalization_strategy()
    
    def apply_strategy_to_data(self):
        """Apply the normalization strategy to the data"""
        messagebox.showinfo("Info", "This feature will integrate with phase1_data_preparation.py to apply the normalization strategy to your data.\n\nFor now, you can save the strategy and manually integrate it into your data processing pipeline.")
    
    def view_group_details(self, event=None):
        """View details of a normalization group including its features"""
        # Get selected group
        selection = self.groups_tree.selection()
        if not selection:
            return
        
        group_name = self.groups_tree.item(selection[0])['values'][0]
        group_info = self.normalization_groups.get(group_name)
        
        if not group_info:
            return
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Group Details: {group_name}")
        details_window.geometry("600x500")
        details_window.transient(self.root)
        details_window.grab_set()
        
        # Center the window
        details_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        # Group info
        info_frame = ttk.LabelFrame(details_window, text="Group Information", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(info_frame, text=f"Group Name: {group_name}", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Normalization Method: {group_info['method']}", font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Description: {group_info['description']}", font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Feature Count: {len(group_info['features'])}", font=("Arial", 10)).pack(anchor=tk.W)
        
        # Features list
        if group_info['features']:
            features_frame = ttk.LabelFrame(details_window, text="Assigned Features", padding="10")
            features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            # Create features treeview
            features_tree = ttk.Treeview(features_frame, columns=("Index", "Name", "Type", "Sample Value"), show="headings", height=15)
            features_tree.heading("Index", text="Index")
            features_tree.heading("Name", text="Feature Name")
            features_tree.heading("Type", text="Data Type")
            features_tree.heading("Sample Value", text="Sample Value")
            
            # Set column widths
            features_tree.column("Index", width=60)
            features_tree.column("Name", width=200)
            features_tree.column("Type", width=120)
            features_tree.column("Sample Value", width=120)
            
            # Add features
            for feature_idx in sorted(group_info['features']):
                feature_name = self.get_feature_name(feature_idx)
                feature_type = self.get_feature_type(feature_idx)
                sample_value = self.get_sample_value(feature_idx)
                
                features_tree.insert("", "end", values=(
                    feature_idx,
                    feature_name,
                    feature_type,
                    sample_value
                ))
            
            # Scrollbar
            features_scrollbar = ttk.Scrollbar(features_frame, orient="vertical", command=features_tree.yview)
            features_tree.configure(yscrollcommand=features_scrollbar.set)
            
            features_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            features_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Export features button
            ttk.Button(features_frame, text="📋 Copy Features to Clipboard", 
                      command=lambda: self.copy_features_to_clipboard(group_info['features'])).pack(pady=(10, 0))
        else:
            # No features assigned
            no_features_frame = ttk.Frame(details_window)
            no_features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            ttk.Label(no_features_frame, text="No features assigned to this group yet.", 
                     font=("Arial", 10), foreground="gray").pack(expand=True)
        
        # Close button
        ttk.Button(details_window, text="Close", command=details_window.destroy).pack(pady=10)
    
    def copy_features_to_clipboard(self, feature_indices):
        """Copy feature information to clipboard"""
        try:
            feature_info = []
            for feature_idx in sorted(feature_indices):
                feature_name = self.get_feature_name(feature_idx)
                feature_type = self.get_feature_type(feature_idx)
                feature_info.append(f"{feature_idx}: {feature_name} ({feature_type})")
            
            clipboard_text = f"Features in Group ({len(feature_indices)} total):\n" + "\n".join(feature_info)
            pyperclip.copy(clipboard_text)
            messagebox.showinfo("Success", "Feature information copied to clipboard!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy features: {e}")
    
    def show_groups_context_menu(self, event):
        """Show context menu for groups tree"""
        # Get clicked item
        item = self.groups_tree.identify_row(event.y)
        if not item:
            return
        
        # Select the clicked item
        self.groups_tree.selection_set(item)
        
        # Create context menu
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="📋 View Group Details", command=self.view_group_details)
        context_menu.add_command(label="✏️ Edit Group", command=self.edit_normalization_group)
        context_menu.add_command(label="🗑️ Delete Group", command=self.delete_normalization_group)
        context_menu.add_separator()
        context_menu.add_command(label="📋 Copy Group Info to Clipboard", command=self.copy_group_info_to_clipboard)
        
        # Show context menu at cursor position
        context_menu.tk_popup(event.x_root, event.y_root)
    
    def copy_group_info_to_clipboard(self):
        """Copy group information to clipboard"""
        selection = self.groups_tree.selection()
        if not selection:
            return
        
        group_name = self.groups_tree.item(selection[0])['values'][0]
        group_info = self.normalization_groups.get(group_name)
        
        if not group_info:
            return
        
        try:
            # Prepare group info
            group_summary = f"Group: {group_name}\n"
            group_summary += f"Method: {group_info['method']}\n"
            group_summary += f"Description: {group_info['description']}\n"
            group_summary += f"Features: {len(group_info['features'])}\n\n"
            
            if group_info['features']:
                group_summary += "Assigned Features:\n"
                for feature_idx in sorted(group_info['features']):
                    feature_name = self.get_feature_name(feature_idx)
                    feature_type = self.get_feature_type(feature_idx)
                    group_summary += f"  {feature_idx}: {feature_name} ({feature_type})\n"
            
            pyperclip.copy(group_summary)
            messagebox.showinfo("Success", f"Group '{group_name}' information copied to clipboard!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy group info: {e}")
    
    def apply_strategy_template(self, template_name):
        """Apply a predefined normalization strategy template"""
        if template_name == "conservative":
            # Conservative: Minimal normalization, preserve most relationships
            self.normalization_groups = {
                "world_coordinates": {"method": "none", "description": "World coordinates - no normalization", "features": []},
                "screen_coordinates": {"method": "none", "description": "Screen coordinates - no normalization", "features": []},
                "timestamps": {"method": "ms_to_minutes", "description": "Timestamps - convert to minutes", "features": []},
                "camera_angles": {"method": "none", "description": "Camera angles - no normalization", "features": []},
                "categorical": {"method": "none", "description": "Categorical data - no normalization", "features": []},
                "continuous_scaled": {"method": "robust_scaler", "description": "Other continuous - robust scaling", "features": []}
            }
        elif template_name == "aggressive":
            # Aggressive: Maximum normalization for model training
            self.normalization_groups = {
                "world_coordinates": {"method": "robust_scaler", "description": "World coordinates - robust scaling", "features": []},
                "screen_coordinates": {"method": "robust_scaler", "description": "Screen coordinates - robust scaling", "features": []},
                "timestamps": {"method": "ms_to_minutes", "description": "Timestamps - convert to minutes", "features": []},
                "camera_angles": {"method": "robust_scaler", "description": "Camera angles - robust scaling", "features": []},
                "categorical": {"method": "none", "description": "Categorical data - no normalization", "features": []},
                "continuous_scaled": {"method": "robust_scaler", "description": "Other continuous - robust scaling", "features": []}
            }
        elif template_name == "balanced":
            # Balanced: Smart normalization preserving important relationships
            self.normalization_groups = {
                "world_coordinates": {"method": "none", "description": "World coordinates - no normalization", "features": []},
                "screen_coordinates": {"method": "robust_scaler", "description": "Screen coordinates - robust scaling", "features": []},
                "timestamps": {"method": "ms_to_minutes", "description": "Timestamps - convert to minutes", "features": []},
                "camera_angles": {"method": "robust_scaler", "description": "Camera angles - robust scaling", "features": []},
                "categorical": {"method": "none", "description": "Categorical data - no normalization", "features": []},
                "continuous_scaled": {"method": "robust_scaler", "description": "Other continuous - robust scaling", "features": []}
            }
        
        # Reassign features to the new groups
        self.auto_assign_features_to_groups()
        self.update_groups_tree()
        self.populate_features_tree()
        self.update_strategy_info()
        
        messagebox.showinfo("Success", f"Applied {template_name} strategy template!")
    
    def reset_to_default_groups(self):
        """Reset to the original default groups"""
        result = messagebox.askyesno("Confirm Reset", "Reset to default normalization groups? This will clear all custom assignments.")
        if result:
            self.initialize_default_groups()
            self.populate_features_tree()
            self.update_strategy_info()
            messagebox.showinfo("Success", "Reset to default groups!")
    
    def create_custom_normalization_rule(self):
        """Create a custom normalization rule"""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Custom Normalization Rule")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Rule name
        ttk.Label(dialog, text="Rule Name:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        rule_name_var = tk.StringVar()
        rule_name_entry = ttk.Entry(dialog, textvariable=rule_name_var, width=40)
        rule_name_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Normalization method
        ttk.Label(dialog, text="Normalization Method:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        method_var = tk.StringVar(value="robust_scaler")
        method_combo = ttk.Combobox(dialog, textvariable=method_var, 
                                   values=["none", "robust_scaler", "standard_scaler", "minmax_scaler", "ms_to_minutes", "custom"], 
                                   width=37)
        method_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Feature selection criteria
        ttk.Label(dialog, text="Apply to features matching:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Data type filter
        type_frame = ttk.Frame(dialog)
        type_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ttk.Label(type_frame, text="Data Type:").pack(side=tk.LEFT)
        type_var = tk.StringVar(value="all")
        type_combo = ttk.Combobox(type_frame, textvariable=type_var, 
                                 values=["all", "world_coordinate", "screen_coordinate", "time_ms", "camera_coordinate", "angle_degrees", "item_id", "animation_id", "hashed_string", "boolean", "tab_id", "object_id", "npc_id", "count", "slot_id", "skill_level", "skill_xp"], 
                                 width=25)
        type_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Feature index range
        range_frame = ttk.Frame(dialog)
        range_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ttk.Label(range_frame, text="Feature Index Range:").pack(side=tk.LEFT)
        start_var = tk.StringVar(value="0")
        end_var = tk.StringVar(value="127")
        ttk.Entry(range_frame, textvariable=start_var, width=8).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(range_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(range_frame, textvariable=end_var, width=8).pack(side=tk.LEFT, padx=(2, 0))
        
        # Description
        ttk.Label(dialog, text="Description:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(0, 5))
        desc_var = tk.StringVar()
        desc_entry = ttk.Entry(dialog, textvariable=desc_var, width=40)
        desc_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        def create_rule():
            rule_name = rule_name_var.get().strip()
            method = method_var.get()
            data_type = type_var.get()
            start_idx = int(start_var.get())
            end_idx = int(end_var.get())
            description = desc_var.get().strip()
            
            if not rule_name:
                messagebox.showerror("Error", "Rule name is required!")
                return
            
            if rule_name in self.normalization_groups:
                messagebox.showerror("Error", "Rule name already exists!")
                return
            
            # Create the new group
            self.normalization_groups[rule_name] = {
                "method": method,
                "description": description,
                "features": []
            }
            
            # Assign features based on criteria
            for feature_idx in range(start_idx, min(end_idx + 1, 128)):
                if data_type == "all" or self.get_feature_type(feature_idx) == data_type:
                    # Remove from other groups first
                    for group_name, group_info in self.normalization_groups.items():
                        if group_name != rule_name and feature_idx in group_info["features"]:
                            group_info["features"].remove(feature_idx)
                    
                    # Add to new group
                    self.normalization_groups[rule_name]["features"].append(feature_idx)
            
            self.update_groups_tree()
            self.populate_features_tree()
            self.update_strategy_info()
            dialog.destroy()
            messagebox.showinfo("Success", f"Custom rule '{rule_name}' created with {len(self.normalization_groups[rule_name]['features'])} features!")
        
        ttk.Button(button_frame, text="Create Rule", command=create_rule).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        
        # Focus on name entry
        rule_name_entry.focus()
    
    def batch_assign_by_type(self):
        """Batch assign features by their data type"""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Assign by Data Type")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        ttk.Label(dialog, text="Select data types to batch assign:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Data type checkboxes
        type_vars = {}
        types = ["world_coordinate", "screen_coordinate", "time_ms", "camera_coordinate", "angle_degrees", "item_id", "animation_id", "hashed_string", "boolean", "tab_id", "object_id", "npc_id", "count", "slot_id", "skill_level", "skill_xp"]
        
        for data_type in types:
            var = tk.BooleanVar()
            type_vars[data_type] = var
            ttk.Checkbutton(dialog, text=data_type, variable=var).pack(anchor=tk.W, padx=20)
        
        # Target group
        ttk.Label(dialog, text="Assign to group:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
        group_var = tk.StringVar(value="continuous_scaled")
        group_combo = ttk.Combobox(dialog, textvariable=group_var, 
                                  values=list(self.normalization_groups.keys()), 
                                  width=37)
        group_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        def apply_batch():
            selected_types = [data_type for data_type, var in type_vars.items() if var.get()]
            target_group = group_var.get()
            
            if not selected_types:
                messagebox.showwarning("Warning", "Please select at least one data type!")
                return
            
            # Count features that will be assigned
            feature_count = 0
            for feature_idx in range(128):
                if self.get_feature_type(feature_idx) in selected_types:
                    feature_count += 1
            
            result = messagebox.askyesno("Confirm Batch Assignment", 
                                       f"Assign {feature_count} features of types {', '.join(selected_types)} to group '{target_group}'?")
            if not result:
                return
            
            # Perform batch assignment
            for feature_idx in range(128):
                if self.get_feature_type(feature_idx) in selected_types:
                    # Remove from other groups
                    for group_name, group_info in self.normalization_groups.items():
                        if group_name != target_group and feature_idx in group_info["features"]:
                            group_info["features"].remove(feature_idx)
                    
                    # Add to target group
                    if feature_idx not in self.normalization_groups[target_group]["features"]:
                        self.normalization_groups[target_group]["features"].append(feature_idx)
            
            self.update_groups_tree()
            self.populate_features_tree()
            self.update_strategy_info()
            dialog.destroy()
            messagebox.showinfo("Success", f"Batch assigned {feature_count} features to '{target_group}'!")
        
        ttk.Button(button_frame, text="Apply Batch Assignment", command=apply_batch).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def smart_auto_assign(self):
        """Smart auto-assignment based on feature analysis"""
        result = messagebox.askyesno("Smart Auto-Assign", 
                                   "This will analyze your features and intelligently assign them to normalization groups.\n\nContinue?")
        if not result:
            return
        
        # Clear all assignments
        for group_info in self.normalization_groups.values():
            group_info["features"].clear()
        
        # Smart assignment based on feature analysis
        for feature_idx in range(128):
            feature_type = self.get_feature_type(feature_idx)
            feature_name = self.get_feature_name(feature_idx)
            
            # Analyze feature name and type for smart assignment
            if "world" in feature_name.lower() or feature_type == "world_coordinate":
                self.normalization_groups["world_coordinates"]["features"].append(feature_idx)
            elif "screen" in feature_name.lower() or feature_type == "screen_coordinate":
                self.normalization_groups["screen_coordinates"]["features"].append(feature_idx)
            elif "time" in feature_name.lower() or feature_type == "time_ms":
                self.normalization_groups["timestamps"]["features"].append(feature_idx)
            elif "camera" in feature_name.lower() or feature_type in ["camera_coordinate", "angle_degrees"]:
                self.normalization_groups["camera_angles"]["features"].append(feature_idx)
            elif feature_type in ["item_id", "animation_id", "hashed_string", "boolean", "tab_id", "object_id", "npc_id", "count", "slot_id", "skill_level", "skill_xp"]:
                self.normalization_groups["categorical"]["features"].append(feature_idx)
            else:
                self.normalization_groups["continuous_scaled"]["features"].append(feature_idx)
        
        self.update_groups_tree()
        self.populate_features_tree()
        self.update_strategy_info()
        messagebox.showinfo("Success", "Smart auto-assignment completed!")

def main():
    """Main function"""
    root = tk.Tk()
    app = TrainingDataBrowser(root)
    root.mainloop()

if __name__ == "__main__":
    main()
