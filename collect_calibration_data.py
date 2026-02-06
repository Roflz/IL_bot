#!/usr/bin/env python3
"""
Collect calibration data for camera Jacobian model.

This script moves the camera around and records:
- Player position
- Camera state (yaw, pitch, zoom)
- Visible objects with their world and screen positions

Usage:
    python collect_calibration_data.py [--output FILE] [--port PORT] [--samples N] [--interval SECONDS]
    
Examples:
    # Collect 1000 samples with 0.1s interval
    python collect_calibration_data.py --samples 1000 --interval 0.1
    
    # Collect data continuously until stopped (Ctrl+C)
    python collect_calibration_data.py --samples 0
    
    # Custom output file
    python collect_calibration_data.py --output my_calibration.jsonl
"""

import sys
import argparse
import json
import time
import random
import signal
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from helpers.ipc import IPCClient
from helpers.runtime_utils import set_ipc
from actions import player
from services.camera_integration import _ensure_camera_thread
from helpers.camera import move_camera_random as camera_move_random

# ANSI color codes
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_CYAN = '\033[96m'
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_RED = '\033[91m'
COLOR_MAGENTA = '\033[95m'

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    print(f"\n{COLOR_YELLOW}[INFO]{COLOR_RESET} Shutdown requested, finishing current sample...")
    _shutdown_requested = True


def get_camera_state(ipc):
    """Get current camera state."""
    camera_data = ipc.get_camera()
    if not camera_data:
        return None
    
    return {
        "yaw": camera_data.get("yaw", 0),
        "pitch": camera_data.get("pitch", 256),
        "zoom": camera_data.get("scale", 512)
    }


def get_objects_data(ipc):
    """Get visible objects with world and screen positions."""
    objects = ipc.get_objects()
    if not objects:
        return []
    
    # Handle different response formats
    if isinstance(objects, dict):
        objects = objects.get("objects", [])
    elif not isinstance(objects, list):
        return []
    
    objects_data = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        
        world = obj.get("world")
        screen = obj.get("screen")
        
        # Only include objects with valid world and screen positions
        if world and screen:
            world_x = world.get("x")
            world_y = world.get("y")
            screen_x = screen.get("x")
            screen_y = screen.get("y")
            
            if (world_x is not None and world_y is not None and
                screen_x is not None and screen_y is not None):
                objects_data.append({
                    "name": obj.get("name", "Unknown"),
                    "world": {"x": int(world_x), "y": int(world_y)},
                    "screen": {"x": float(screen_x), "y": float(screen_y)}
                })
    
    return objects_data


def record_sample(ipc, output_file):
    """Record a single calibration sample."""
    try:
        # Get player position
        player_x = player.get_x()
        player_y = player.get_y()
        if player_x is None or player_y is None:
            return False
        
        # Get camera state
        camera_state = get_camera_state(ipc)
        if not camera_state:
            return False
        
        # Get objects
        objects_data = get_objects_data(ipc)
        
        # Only record if we have at least some objects
        if len(objects_data) == 0:
            return False
        
        # Create record
        record = {
            "timestamp": time.time(),
            "player": {"x": int(player_x), "y": int(player_y)},
            "camera": camera_state,
            "objects": objects_data
        }
        
        # Write to file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        return True
    except Exception as e:
        print(f"{COLOR_RED}[ERROR]{COLOR_RESET} Failed to record sample: {e}")
        return False


def move_camera_random(ipc):
    """Move camera to a random position using IPC key presses."""
    try:
        from services.camera_integration import _camera_movement_queue
        
        camera_data = ipc.get_camera()
        if not camera_data:
            return
        
        current_yaw = camera_data.get("yaw", 0)
        current_pitch = camera_data.get("pitch", 256)
        current_zoom = camera_data.get("scale", 512)
        
        # Random yaw change (-100 to +100)
        yaw_delta = random.randint(-100, 100)
        if abs(yaw_delta) > 2:
            yaw_key = "RIGHT" if yaw_delta > 0 else "LEFT"
            yaw_duration = min(abs(yaw_delta) * 0.5, 500)  # Max 500ms
            movement = {
                "type": "key_hold",
                "key": yaw_key,
                "duration_ms": int(yaw_duration),
                "cancel_opposite": "LEFT" if yaw_key == "RIGHT" else "RIGHT"
            }
            _camera_movement_queue.put(movement)
        
        # Random pitch change (-20 to +20)
        pitch_delta = random.randint(-20, 20)
        if abs(pitch_delta) > 2:
            pitch_key = "UP" if pitch_delta > 0 else "DOWN"
            pitch_duration = min(abs(pitch_delta) * 2.0, 400)  # Max 400ms
            movement = {
                "type": "key_hold",
                "key": pitch_key,
                "duration_ms": int(pitch_duration),
                "cancel_opposite": "DOWN" if pitch_key == "UP" else "UP"
            }
            _camera_movement_queue.put(movement)
        
        # Random zoom change (small scrolls)
        zoom_delta = random.randint(-3, 3)
        if abs(zoom_delta) > 0:
            scroll_amount = 1 if zoom_delta > 0 else -1
            movement = {
                "type": "scroll",
                "amount": scroll_amount,
                "count": abs(zoom_delta)
            }
            _camera_movement_queue.put(movement)
        
        # Wait for movement to complete
        time.sleep(0.3)
        
    except Exception as e:
        pass  # Ignore errors in random movement


def collect_calibration_data(
    output_file: str,
    port: int = 17000,
    num_samples: int = 1000,
    interval: float = 0.1,
    move_camera: bool = True
):
    """
    Collect calibration data.
    
    Args:
        output_file: Path to output JSONL file
        port: IPC port number
        num_samples: Number of samples to collect (0 = infinite until Ctrl+C)
        interval: Time between samples in seconds
        move_camera: Whether to randomly move camera between samples
    """
    global _shutdown_requested
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize IPC
    try:
        ipc = IPCClient(port=port)
        set_ipc(ipc)
        print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Connected to IPC on port {port}")
    except Exception as e:
        print(f"{COLOR_RED}[ERROR]{COLOR_RESET} Could not connect to IPC on port {port}: {e}")
        return False
    
    # Ensure camera thread is running
    _ensure_camera_thread()
    
    # Get player position
    player_x = player.get_x()
    player_y = player.get_y()
    if player_x is None or player_y is None:
        print(f"{COLOR_RED}[ERROR]{COLOR_RESET} Could not get player position")
        return False
    
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Player position: ({player_x}, {player_y})")
    
    # Create output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    if output_path.exists():
        response = input(f"{COLOR_YELLOW}[WARNING]{COLOR_RESET} File {output_file} already exists. Append? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
    
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Output file: {output_file}")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Samples: {num_samples if num_samples > 0 else 'infinite (Ctrl+C to stop)'}")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Interval: {interval:.2f}s")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Move camera: {move_camera}")
    print(f"\n{COLOR_CYAN}{COLOR_BOLD}{'='*70}")
    print(f"  COLLECTING CALIBRATION DATA")
    print(f"{'='*70}{COLOR_RESET}\n")
    
    # Collect samples
    samples_collected = 0
    samples_skipped = 0
    start_time = time.time()
    
    try:
        while True:
            if _shutdown_requested:
                break
            
            if num_samples > 0 and samples_collected >= num_samples:
                break
            
            # Record sample
            success = record_sample(ipc, output_file)
            
            if success:
                samples_collected += 1
                if samples_collected % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = samples_collected / elapsed if elapsed > 0 else 0
                    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Collected {samples_collected} samples "
                          f"(skipped {samples_skipped}, rate: {rate:.1f} samples/s)")
            else:
                samples_skipped += 1
            
            # Move camera if requested
            if move_camera and not _shutdown_requested:
                # Use the existing camera movement function for better coverage
                try:
                    camera_move_random()
                    time.sleep(0.2)  # Wait for camera to settle
                except Exception as e:
                    # Fallback to simple random movement if helper function fails
                    move_camera_random(ipc)
                    time.sleep(0.2)
            
            # Wait for next sample
            if not _shutdown_requested:
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n{COLOR_YELLOW}[INFO]{COLOR_RESET} Interrupted by user")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{COLOR_CYAN}{COLOR_BOLD}{'='*70}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*70}{COLOR_RESET}\n")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Samples collected: {samples_collected}")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Samples skipped: {samples_skipped}")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Total time: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Average rate: {samples_collected / elapsed:.1f} samples/s")
    print(f"{COLOR_GREEN}[INFO]{COLOR_RESET} Output file: {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect calibration data for camera Jacobian model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 1000 samples with 0.1s interval
  python collect_calibration_data.py --samples 1000 --interval 0.1
  
  # Collect continuously until Ctrl+C
  python collect_calibration_data.py --samples 0
  
  # Custom output file and port
  python collect_calibration_data.py --output my_data.jsonl --port 17000
  
  # Don't move camera (record at current positions only)
  python collect_calibration_data.py --no-move-camera
        """
    )
    
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSONL file path (default: screen_projection_calibration_TIMESTAMP.jsonl)")
    parser.add_argument("--port", type=int, default=17000,
                       help="IPC port number (default: 17000)")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to collect (0 = infinite until Ctrl+C, default: 1000)")
    parser.add_argument("--interval", type=float, default=0.1,
                       help="Time between samples in seconds (default: 0.1)")
    parser.add_argument("--move-camera", action="store_true", default=False,
                       help="Move camera randomly between samples (default: False)")
    parser.add_argument("--no-move-camera", dest="move_camera", action="store_false",
                       help="Don't move camera (record at current positions only)")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"screen_projection_calibration_{timestamp}.jsonl"
    
    # Run collection
    success = collect_calibration_data(
        output_file=args.output,
        port=args.port,
        num_samples=args.samples,
        interval=args.interval,
        move_camera=args.move_camera
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
