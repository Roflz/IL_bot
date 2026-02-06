"""
Analyze Screen Projection Calibration Data

This script loads and analyzes the JSONL calibration data collected from
calibrate_screen_projection() to:
1. Understand data distribution
2. Determine zoom-to-focal length relationship
3. Calculate initial error metrics
4. Visualize patterns in the data
"""
import json
import os
import glob
from typing import List, Dict, Any, Tuple
import statistics
from collections import defaultdict

# Import the projection model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.camera_projection import (
    predict_screen_position,
    calculate_reprojection_error,
    focal_length_from_zoom
)


def load_calibration_files(pattern: str = "screen_projection_calibration_*.jsonl") -> List[Dict[str, Any]]:
    """
    Load all calibration JSONL files matching the pattern.
    
    Args:
        pattern: Glob pattern to match calibration files
    
    Returns:
        List of data entries (one per line in all files)
    """
    files = glob.glob(pattern)
    if not files:
        print(f"[WARNING] No files found matching pattern: {pattern}")
        return []
    
    print(f"[INFO] Found {len(files)} calibration file(s)")
    
    all_data = []
    for filepath in sorted(files):
        print(f"[INFO] Loading: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        all_data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Failed to parse line {line_num} in {filepath}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
    
    print(f"[INFO] Loaded {len(all_data)} data entries total")
    return all_data


def analyze_data_distribution(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the distribution of camera states and object positions in the data.
    
    Returns:
        Dictionary with statistics about the data
    """
    stats = {
        "total_entries": len(data),
        "total_objects": 0,
        "camera_states": {
            "yaw": {"min": None, "max": None, "values": []},
            "pitch": {"min": None, "max": None, "values": []},
            "zoom": {"min": None, "max": None, "values": []}
        },
        "object_positions": {
            "world_x": {"min": None, "max": None, "values": []},
            "world_y": {"min": None, "max": None, "values": []},
            "screen_x": {"min": None, "max": None, "values": []},
            "screen_y": {"min": None, "max": None, "values": []}
        }
    }
    
    for entry in data:
        camera = entry.get("camera", {})
        objects = entry.get("objects", [])
        
        # Track camera states
        yaw = camera.get("yaw")
        pitch = camera.get("pitch")
        zoom = camera.get("zoom")
        
        if yaw is not None:
            stats["camera_states"]["yaw"]["values"].append(yaw)
            if stats["camera_states"]["yaw"]["min"] is None or yaw < stats["camera_states"]["yaw"]["min"]:
                stats["camera_states"]["yaw"]["min"] = yaw
            if stats["camera_states"]["yaw"]["max"] is None or yaw > stats["camera_states"]["yaw"]["max"]:
                stats["camera_states"]["yaw"]["max"] = yaw
        
        if pitch is not None:
            stats["camera_states"]["pitch"]["values"].append(pitch)
            if stats["camera_states"]["pitch"]["min"] is None or pitch < stats["camera_states"]["pitch"]["min"]:
                stats["camera_states"]["pitch"]["min"] = pitch
            if stats["camera_states"]["pitch"]["max"] is None or pitch > stats["camera_states"]["pitch"]["max"]:
                stats["camera_states"]["pitch"]["max"] = pitch
        
        if zoom is not None:
            stats["camera_states"]["zoom"]["values"].append(zoom)
            if stats["camera_states"]["zoom"]["min"] is None or zoom < stats["camera_states"]["zoom"]["min"]:
                stats["camera_states"]["zoom"]["min"] = zoom
            if stats["camera_states"]["zoom"]["max"] is None or zoom > stats["camera_states"]["zoom"]["max"]:
                stats["camera_states"]["zoom"]["max"] = zoom
        
        # Track object positions
        stats["total_objects"] += len(objects)
        for obj in objects:
            world = obj.get("world", {})
            screen = obj.get("screen", {})
            
            world_x = world.get("x")
            world_y = world.get("y")
            screen_x = screen.get("x")
            screen_y = screen.get("y")
            
            if world_x is not None:
                stats["object_positions"]["world_x"]["values"].append(world_x)
                if stats["object_positions"]["world_x"]["min"] is None or world_x < stats["object_positions"]["world_x"]["min"]:
                    stats["object_positions"]["world_x"]["min"] = world_x
                if stats["object_positions"]["world_x"]["max"] is None or world_x > stats["object_positions"]["world_x"]["max"]:
                    stats["object_positions"]["world_x"]["max"] = world_x
            
            if world_y is not None:
                stats["object_positions"]["world_y"]["values"].append(world_y)
                if stats["object_positions"]["world_y"]["min"] is None or world_y < stats["object_positions"]["world_y"]["min"]:
                    stats["object_positions"]["world_y"]["min"] = world_y
                if stats["object_positions"]["world_y"]["max"] is None or world_y > stats["object_positions"]["world_y"]["max"]:
                    stats["object_positions"]["world_y"]["max"] = world_y
            
            if screen_x is not None:
                stats["object_positions"]["screen_x"]["values"].append(screen_x)
                if stats["object_positions"]["screen_x"]["min"] is None or screen_x < stats["object_positions"]["screen_x"]["min"]:
                    stats["object_positions"]["screen_x"]["min"] = screen_x
                if stats["object_positions"]["screen_x"]["max"] is None or screen_x > stats["object_positions"]["screen_x"]["max"]:
                    stats["object_positions"]["screen_x"]["max"] = screen_x
            
            if screen_y is not None:
                stats["object_positions"]["screen_y"]["values"].append(screen_y)
                if stats["object_positions"]["screen_y"]["min"] is None or screen_y < stats["object_positions"]["screen_y"]["min"]:
                    stats["object_positions"]["screen_y"]["min"] = screen_y
                if stats["object_positions"]["screen_y"]["max"] is None or screen_y > stats["object_positions"]["screen_y"]["max"]:
                    stats["object_positions"]["screen_y"]["max"] = screen_y
    
    # Calculate statistics
    for key in ["yaw", "pitch", "zoom"]:
        values = stats["camera_states"][key]["values"]
        if values:
            stats["camera_states"][key]["mean"] = statistics.mean(values)
            stats["camera_states"][key]["median"] = statistics.median(values)
            stats["camera_states"][key]["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    
    for key in ["world_x", "world_y", "screen_x", "screen_y"]:
        values = stats["object_positions"][key]["values"]
        if values:
            stats["object_positions"][key]["mean"] = statistics.mean(values)
            stats["object_positions"][key]["median"] = statistics.median(values)
            stats["object_positions"][key]["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    
    return stats


def analyze_zoom_to_focal_relationship(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze how zoom affects screen positions to determine the zoom-to-focal relationship.
    
    Strategy:
    1. Find objects at the same world position with same yaw/pitch but different zoom
    2. Measure how screen position changes with zoom
    3. Determine if relationship is inverse, direct, or other
    
    Returns:
        Dictionary with analysis results
    """
    print("\n[ANALYSIS] Analyzing zoom-to-focal length relationship...")
    
    # Group data by (player_x, player_y, obj_x, obj_y, yaw, pitch) to find same object at different zooms
    groups = defaultdict(list)
    
    for entry in data:
        player = entry.get("player", {})
        camera = entry.get("camera", {})
        objects = entry.get("objects", [])
        
        player_x = player.get("x")
        player_y = player.get("y")
        yaw = camera.get("yaw")
        pitch = camera.get("pitch")
        zoom = camera.get("zoom")
        
        if None in [player_x, player_y, yaw, pitch, zoom]:
            continue
        
        for obj in objects:
            world = obj.get("world", {})
            screen = obj.get("screen", {})
            
            obj_x = world.get("x")
            obj_y = world.get("y")
            screen_x = screen.get("x")
            screen_y = screen.get("y")
            
            if None in [obj_x, obj_y, screen_x, screen_y]:
                continue
            
            # Create a key for grouping (same object, same camera yaw/pitch, different zoom)
            key = (player_x, player_y, obj_x, obj_y, yaw, pitch)
            groups[key].append({
                "zoom": zoom,
                "screen_x": screen_x,
                "screen_y": screen_y,
                "distance_from_center": ((screen_x - 823.5) ** 2 + (screen_y - 508.5) ** 2) ** 0.5
            })
    
    # Filter groups that have multiple zoom levels
    multi_zoom_groups = {k: v for k, v in groups.items() if len(v) > 1}
    
    print(f"[INFO] Found {len(multi_zoom_groups)} object positions with multiple zoom levels")
    
    if not multi_zoom_groups:
        print("[WARNING] No data with varying zoom found. Need more calibration data.")
        return {"status": "insufficient_data"}
    
    # Analyze patterns
    zoom_increases = []  # When zoom increases, does distance from center increase or decrease?
    
    for key, zoom_data in multi_zoom_groups.items():
        # Sort by zoom
        zoom_data_sorted = sorted(zoom_data, key=lambda x: x["zoom"])
        
        for i in range(len(zoom_data_sorted) - 1):
            current = zoom_data_sorted[i]
            next_item = zoom_data_sorted[i + 1]
            
            zoom_delta = next_item["zoom"] - current["zoom"]
            distance_delta = next_item["distance_from_center"] - current["distance_from_center"]
            
            if zoom_delta > 0:  # Zoom increased
                zoom_increases.append({
                    "zoom_delta": zoom_delta,
                    "distance_delta": distance_delta,
                    "distance_ratio": distance_delta / zoom_delta if zoom_delta > 0 else 0
                })
    
    if not zoom_increases:
        return {"status": "insufficient_data"}
    
    # Determine relationship
    avg_distance_delta = statistics.mean([x["distance_delta"] for x in zoom_increases])
    
    # If distance_from_center decreases as zoom increases → inverse relationship (f = k/zoom)
    # If distance_from_center increases as zoom increases → direct relationship (f = k*zoom)
    
    likely_formula = "inverse" if avg_distance_delta < 0 else "direct"
    
    analysis = {
        "status": "success",
        "num_groups": len(multi_zoom_groups),
        "num_zoom_transitions": len(zoom_increases),
        "avg_distance_delta_per_zoom": avg_distance_delta,
        "likely_formula": likely_formula,
        "explanation": (
            "Objects move TOWARD center as zoom increases → inverse relationship (f = k/zoom)"
            if avg_distance_delta < 0 else
            "Objects move AWAY from center as zoom increases → direct relationship (f = k*zoom)"
        ),
        "sample_data": list(multi_zoom_groups.items())[:5]  # First 5 groups as examples
    }
    
    return analysis


def calculate_initial_errors(
    data: List[Dict[str, Any]],
    focal_formula: str = "inverse",
    focal_k: float = 100000.0,
    focal_offset: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate reprojection errors using the current model parameters.
    
    This gives us a baseline to see how well the model performs before fitting.
    
    Args:
        data: Calibration data entries
        focal_formula: Formula type to test
        focal_k: Constant parameter to test
        focal_offset: Offset parameter to test
    
    Returns:
        Dictionary with error statistics
    """
    print(f"\n[ANALYSIS] Calculating initial errors (formula={focal_formula}, k={focal_k})...")
    
    errors = []
    skipped = 0
    
    for entry in data:
        player = entry.get("player", {})
        camera = entry.get("camera", {})
        objects = entry.get("objects", [])
        
        player_x = player.get("x")
        player_y = player.get("y")
        yaw = camera.get("yaw")
        pitch = camera.get("pitch")
        zoom = camera.get("zoom")
        
        if None in [player_x, player_y, yaw, pitch, zoom]:
            skipped += 1
            continue
        
        player_world = {"x": player_x, "y": player_y}
        
        for obj in objects:
            world = obj.get("world", {})
            screen = obj.get("screen", {})
            
            obj_x = world.get("x")
            obj_y = world.get("y")
            screen_x = screen.get("x")
            screen_y = screen.get("y")
            
            if None in [obj_x, obj_y, screen_x, screen_y]:
                continue
            
            obj_world = {"x": obj_x, "y": obj_y, "p": world.get("p", 0)}
            observed_screen = {"x": screen_x, "y": screen_y}
            
            # Predict screen position
            predicted_screen = predict_screen_position(
                obj_world,
                player_world,
                yaw,
                pitch,
                zoom,
                focal_formula,
                focal_k,
                focal_offset
            )
            
            if predicted_screen is None:
                skipped += 1
                continue
            
            # Calculate error
            error = calculate_reprojection_error(predicted_screen, observed_screen)
            errors.append(error)
    
    if not errors:
        return {"status": "no_valid_predictions", "skipped": skipped}
    
    error_distances = [e["error_distance"] for e in errors]
    error_x = [e["error_x"] for e in errors]
    error_y = [e["error_y"] for e in errors]
    
    stats = {
        "status": "success",
        "num_predictions": len(errors),
        "num_skipped": skipped,
        "error_distance": {
            "mean": statistics.mean(error_distances),
            "median": statistics.median(error_distances),
            "stdev": statistics.stdev(error_distances) if len(error_distances) > 1 else 0.0,
            "min": min(error_distances),
            "max": max(error_distances),
            "p95": sorted(error_distances)[int(len(error_distances) * 0.95)] if error_distances else 0.0
        },
        "error_x": {
            "mean": statistics.mean(error_x),
            "median": statistics.median(error_x),
            "stdev": statistics.stdev(error_x) if len(error_x) > 1 else 0.0
        },
        "error_y": {
            "mean": statistics.mean(error_y),
            "median": statistics.median(error_y),
            "stdev": statistics.stdev(error_y) if len(error_y) > 1 else 0.0
        }
    }
    
    return stats


def print_analysis_report(stats: Dict[str, Any], zoom_analysis: Dict[str, Any], error_stats: Dict[str, Any]):
    """Print a formatted analysis report."""
    print("\n" + "=" * 80)
    print("SCREEN PROJECTION CALIBRATION DATA ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n--- DATA DISTRIBUTION ---")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total objects: {stats['total_objects']}")
    
    print("\nCamera State Ranges:")
    for key in ["yaw", "pitch", "zoom"]:
        cam = stats["camera_states"][key]
        print(f"  {key.upper()}:")
        print(f"    Range: {cam['min']} - {cam['max']}")
        if "mean" in cam:
            print(f"    Mean: {cam['mean']:.2f}, Median: {cam['median']:.2f}, StdDev: {cam['stdev']:.2f}")
    
    print("\nObject Position Ranges:")
    for key in ["world_x", "world_y", "screen_x", "screen_y"]:
        pos = stats["object_positions"][key]
        print(f"  {key}:")
        print(f"    Range: {pos['min']:.2f} - {pos['max']:.2f}")
        if "mean" in pos:
            print(f"    Mean: {pos['mean']:.2f}, Median: {pos['median']:.2f}, StdDev: {pos['stdev']:.2f}")
    
    print("\n--- ZOOM-TO-FOCAL LENGTH ANALYSIS ---")
    if zoom_analysis.get("status") == "success":
        print(f"Found {zoom_analysis['num_groups']} object positions with multiple zoom levels")
        print(f"Analyzed {zoom_analysis['num_zoom_transitions']} zoom transitions")
        print(f"Average distance change per zoom unit: {zoom_analysis['avg_distance_delta_per_zoom']:.4f}")
        print(f"Likely formula: {zoom_analysis['likely_formula']}")
        print(f"Explanation: {zoom_analysis['explanation']}")
    else:
        print("Insufficient data to determine zoom-to-focal relationship")
        print("Need more calibration data with varying zoom levels")
    
    print("\n--- INITIAL ERROR STATISTICS ---")
    if error_stats.get("status") == "success":
        ed = error_stats["error_distance"]
        print(f"Valid predictions: {error_stats['num_predictions']}")
        print(f"Skipped: {error_stats['num_skipped']}")
        print(f"\nError Distance (pixels):")
        print(f"  Mean: {ed['mean']:.2f}")
        print(f"  Median: {ed['median']:.2f}")
        print(f"  StdDev: {ed['stdev']:.2f}")
        print(f"  Min: {ed['min']:.2f}")
        print(f"  Max: {ed['max']:.2f}")
        print(f"  95th percentile: {ed['p95']:.2f}")
        
        ex = error_stats["error_x"]
        ey = error_stats["error_y"]
        print(f"\nError X (pixels): Mean={ex['mean']:.2f}, StdDev={ex['stdev']:.2f}")
        print(f"Error Y (pixels): Mean={ey['mean']:.2f}, StdDev={ey['stdev']:.2f}")
    else:
        print(f"Status: {error_stats.get('status', 'unknown')}")
        print(f"Skipped: {error_stats.get('skipped', 0)}")
    
    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    print("[INFO] Starting screen projection data analysis...")
    
    # Load data
    data = load_calibration_files()
    
    if not data:
        print("[ERROR] No calibration data found. Run calibrate_screen_projection() first.")
        return
    
    # Analyze data distribution
    print("\n[ANALYSIS] Analyzing data distribution...")
    stats = analyze_data_distribution(data)
    
    # Analyze zoom-to-focal relationship
    zoom_analysis = analyze_zoom_to_focal_relationship(data)
    
    # Calculate initial errors (test with inverse formula as default)
    error_stats = calculate_initial_errors(data, focal_formula="inverse", focal_k=100000.0)
    
    # Print report
    print_analysis_report(stats, zoom_analysis, error_stats)
    
    print("\n[INFO] Analysis complete!")


if __name__ == "__main__":
    main()


