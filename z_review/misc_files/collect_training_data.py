#!/usr/bin/env python3
"""
Training Data Collection Script for Sapphire Ring Crafting Bot

This script helps you collect training data by:
1. Recording your manual crafting sessions
2. Aligning gamestates with your actions
3. Creating properly formatted training data

Usage:
    python collect_training_data.py --session_name "session_1" --duration 300
"""

import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue
import numpy as np

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from feature_extraction_rings import extract_ring_crafting_features, extract_action_features

class TrainingDataCollector:
    """
    Collects training data for the sapphire ring crafting bot.
    Records gamestates and actions simultaneously.
    """
    
    def __init__(self, session_name: str, output_dir: str = "training_data"):
        self.session_name = session_name
        self.output_dir = output_dir
        self.session_dir = os.path.join(output_dir, session_name)
        
        # Create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "gamestates"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "screenshots"), exist_ok=True)
        
        # Data collection
        self.gamestates = []
        self.actions = []
        self.timestamps = []
        
        # Threading
        self.running = False
        self.gamestate_queue = queue.Queue()
        self.action_queue = queue.Queue()
        
        # Files
        self.gamestate_file = os.path.join(self.session_dir, "gamestates.json")
        self.actions_file = os.path.join(self.session_dir, "actions.csv")
        self.metadata_file = os.path.join(self.session_dir, "metadata.json")
        
        print(f"📁 Training data will be saved to: {self.session_dir}")
    
    def start_collection(self, duration: int = 300):
        """Start collecting training data."""
        print(f"🚀 Starting training data collection for session: {self.session_name}")
        print(f"⏱️  Duration: {duration} seconds")
        
        if duration == 0:
            print("📋 Copying existing data...")
            self._copy_existing_data()
            self._save_data()
            print(f"💾 Data saved to: {self.session_dir}")
            return
        
        print("📝 Instructions:")
        print("   1. Make sure RuneLite is running with the StateExporterPlugin")
        print("   2. Start crafting sapphire rings manually")
        print("   3. The script will collect gamestates and actions")
        print("   4. Press Ctrl+C to stop early")
        
        self.running = True
        
        # Start gamestate collection thread
        gamestate_thread = threading.Thread(target=self._collect_gamestates, args=(duration,))
        gamestate_thread.daemon = True
        gamestate_thread.start()
        
        # Start action collection thread
        action_thread = threading.Thread(target=self._collect_actions, args=(duration,))
        action_thread.daemon = True
        action_thread.start()
        
        # Wait for completion or interruption
        try:
            start_time = time.time()
            while time.time() - start_time < duration and self.running:
                time.sleep(1)
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                print(f"\r⏱️  Collecting... {elapsed}s / {duration}s (remaining: {remaining}s)", end="", flush=True)
            
            print("\n✅ Collection time completed!")
            
        except KeyboardInterrupt:
            print("\n⏹️  Collection stopped by user")
        
        finally:
            self.running = False
            self._save_data()
            print(f"💾 Data saved to: {self.session_dir}")
    
    def _copy_existing_data(self):
        """Copy existing data when duration=0."""
        # Copy existing gamestates
        gamestates_dir = "data/gamestates"
        if os.path.exists(gamestates_dir):
            files = [f for f in os.listdir(gamestates_dir) if f.endswith('.json')]
            files.sort()
            
            for file in files:
                filepath = os.path.join(gamestates_dir, file)
                try:
                    with open(filepath, 'r') as f:
                        gamestate = json.load(f)
                    
                    # Add timestamp
                    timestamp = time.time()
                    gamestate['collection_timestamp'] = timestamp
                    gamestate['session_name'] = self.session_name
                    
                    # Store gamestate
                    self.gamestates.append(gamestate)
                    self.timestamps.append(timestamp)
                    
                    # Copy to session directory
                    session_gamestate_path = os.path.join(
                        self.session_dir, "gamestates", file
                    )
                    with open(session_gamestate_path, 'w') as f:
                        json.dump(gamestate, f, indent=2)
                    
                    print(f"📊 Copied gamestate: {file}")
                    
                except Exception as e:
                    print(f"❌ Error copying gamestate {file}: {e}")
        
        # Copy existing actions
        actions_file = "data/actions.csv"
        if os.path.exists(actions_file):
            try:
                with open(actions_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    actions = list(reader)
                
                for action in actions:
                    # Add timestamp and session info
                    action['collection_timestamp'] = time.time()
                    action['session_name'] = self.session_name
                    self.actions.append(action)
                
                # Copy to session directory
                session_actions_path = os.path.join(self.session_dir, "actions.csv")
                with open(session_actions_path, 'w', newline='') as f:
                    if actions:
                        writer = csv.DictWriter(f, fieldnames=actions[0].keys())
                        writer.writeheader()
                        writer.writerows(actions)
                
                print(f"📝 Copied {len(actions)} actions")
                
            except Exception as e:
                print(f"❌ Error copying actions: {e}")
        
        # Copy existing screenshots
        screenshots_dir = "data/runelite_screenshots"
        if os.path.exists(screenshots_dir):
            session_screenshots_dir = os.path.join(self.session_dir, "screenshots")
            os.makedirs(session_screenshots_dir, exist_ok=True)
            
            screenshot_files = [f for f in os.listdir(screenshots_dir) if f.endswith('.png')]
            for file in screenshot_files:
                src = os.path.join(screenshots_dir, file)
                dst = os.path.join(session_screenshots_dir, file)
                try:
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"📸 Copied screenshot: {file}")
                except Exception as e:
                    print(f"❌ Error copying screenshot {file}: {e}")
    
    def _collect_gamestates(self, duration: int):
        """Collect gamestates from the RuneLite plugin output."""
        start_time = time.time()
        
        while time.time() - start_time < duration and self.running:
            try:
                # Check for new gamestate files
                gamestates_dir = "data/gamestates"
                if os.path.exists(gamestates_dir):
                    files = [f for f in os.listdir(gamestates_dir) if f.endswith('.json')]
                    files.sort()
                    
                    for file in files:
                        filepath = os.path.join(gamestates_dir, file)
                        
                        # Check if we've already processed this file
                        if file not in [os.path.basename(f) for f in self.gamestates]:
                            try:
                                with open(filepath, 'r') as f:
                                    gamestate = json.load(f)
                                
                                # Add timestamp
                                timestamp = time.time()
                                gamestate['collection_timestamp'] = timestamp
                                gamestate['session_name'] = self.session_name
                                
                                # Store gamestate
                                self.gamestates.append(gamestate)
                                self.timestamps.append(timestamp)
                                
                                # Copy to session directory
                                session_gamestate_path = os.path.join(
                                    self.session_dir, "gamestates", file
                                )
                                with open(session_gamestate_path, 'w') as f:
                                    json.dump(gamestate, f, indent=2)
                                
                                print(f"\n📊 Collected gamestate: {file}")
                                
                            except Exception as e:
                                print(f"\n❌ Error reading gamestate {file}: {e}")
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"\n❌ Error in gamestate collection: {e}")
                time.sleep(1)
    
    def _collect_actions(self, duration: int):
        """Collect actions from the main app recording."""
        start_time = time.time()
        
        while time.time() - start_time < duration and self.running:
            try:
                # Check for actions.csv updates
                actions_file = "data/actions.csv"
                if os.path.exists(actions_file):
                    # Read the actions file
                    with open(actions_file, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        actions = list(reader)
                    
                    # Check if we have new actions
                    if len(actions) > len(self.actions):
                        new_actions = actions[len(self.actions):]
                        
                        for action in new_actions:
                            # Add timestamp and session info
                            action['collection_timestamp'] = time.time()
                            action['session_name'] = self.session_name
                            
                            # Store action
                            self.actions.append(action)
                            
                            print(f"\n🎯 Collected action: {action.get('event_type', 'unknown')}")
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"\n❌ Error in action collection: {e}")
                time.sleep(1)
    
    def _save_data(self):
        """Save collected data to files."""
        print("\n💾 Saving collected data...")
        
        # Save gamestates
        if self.gamestates:
            with open(self.gamestate_file, 'w') as f:
                json.dump(self.gamestates, f, indent=2)
            print(f"📊 Saved {len(self.gamestates)} gamestates")
        
        # Save actions
        if self.actions:
            with open(self.actions_file, 'w', newline='') as f:
                if self.actions:
                    fieldnames = self.actions[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.actions)
            print(f"🎯 Saved {len(self.actions)} actions")
        
        # Save metadata
        metadata = {
            'session_name': self.session_name,
            'collection_start': min(self.timestamps) if self.timestamps else None,
            'collection_end': max(self.timestamps) if self.timestamps else None,
            'total_gamestates': len(self.gamestates),
            'total_actions': len(self.actions),
            'duration_seconds': max(self.timestamps) - min(self.timestamps) if len(self.timestamps) > 1 else 0
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Saved metadata")
    
    def create_training_features(self):
        """Create training features from collected data."""
        print("\n🔧 Creating training features...")
        
        if not self.gamestates or not self.actions:
            print("❌ No data to process!")
            return
        
        # Extract features
        state_features = []
        action_features = []
        
        for gamestate in self.gamestates:
            try:
                state_feat = extract_ring_crafting_features(gamestate)
                state_features.append(state_feat)
            except Exception as e:
                print(f"❌ Error extracting state features: {e}")
                continue
        
        for action in self.actions:
            try:
                action_feat = extract_action_features(action)
                action_features.append(action_feat)
            except Exception as e:
                print(f"❌ Error extracting action features: {e}")
                continue
        
        if state_features and action_features:
            # Save features
            features_dir = os.path.join(self.session_dir, "features")
            os.makedirs(features_dir, exist_ok=True)
            
            np_state_features = np.array(state_features)
            np_action_features = np.array(action_features)
            
            np.save(os.path.join(features_dir, "state_features.npy"), np_state_features)
            np.save(os.path.join(features_dir, "action_features.npy"), np_action_features)
            
            print(f"✅ Features saved:")
            print(f"   State features: {np_state_features.shape}")
            print(f"   Action features: {np_action_features.shape}")
        else:
            print("❌ No features could be extracted!")

def main():
    parser = argparse.ArgumentParser(description="Collect training data for sapphire ring crafting bot")
    parser.add_argument("--session_name", required=True, help="Name for this training session")
    parser.add_argument("--duration", type=int, default=300, help="Collection duration in seconds (default: 300)")
    parser.add_argument("--output_dir", default="training_data", help="Output directory (default: training_data)")
    
    args = parser.parse_args()
    
    # Create collector
    collector = TrainingDataCollector(args.session_name, args.output_dir)
    
    try:
        # Start collection
        collector.start_collection(args.duration)
        
        # Create training features
        collector.create_training_features()
        
        print(f"\n🎉 Training data collection complete!")
        print(f"📁 Data saved to: {collector.session_dir}")
        print(f"\n📋 Next steps:")
        print(f"1. Review the collected data")
        print(f"2. Run the training script: python model/train_ring_bot.py")
        print(f"3. Collect more sessions for better training data")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
