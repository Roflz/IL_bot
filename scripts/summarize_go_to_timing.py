#!/usr/bin/env python3
"""
Summarize go_to() timing data from JSONL logs.

This script reads all travel.go_to.timing.jsonl files in the repository
and generates a summary table showing median, p90, p95 for each segment
separately for LONG_DISTANCE and SHORT_DISTANCE branches.
"""

import json
import glob
import statistics
from collections import defaultdict
from typing import Dict, List, Any, Optional

def load_timing_data() -> List[Dict[str, Any]]:
    """Load all timing data from JSONL files."""
    timing_files = glob.glob("**/travel.go_to.timing.jsonl", recursive=True)
    all_data = []
    
    for file_path in timing_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GO_TO_TIMING_JSON:"):
                        json_str = line[18:]  # Remove prefix
                        try:
                            data = json.loads(json_str)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON from {file_path}: {e}")
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
    
    return all_data

def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate median, p90, p95 for a list of values."""
    if not values:
        return {"median": 0.0, "p90": 0.0, "p95": 0.0}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        "median": sorted_values[n // 2],
        "p90": sorted_values[int(n * 0.9)],
        "p95": sorted_values[int(n * 0.95)]
    }

def analyze_timing_data(data: List[Dict[str, Any]]) -> None:
    """Analyze timing data and print summary."""
    if not data:
        print("No timing data found.")
        return
    
    # Separate by branch
    long_distance_data = [d for d in data if d.get("branch") == "LONG_DISTANCE"]
    short_distance_data = [d for d in data if d.get("branch") == "SHORT_DISTANCE"]
    
    print(f"Total go_to() calls: {len(data)}")
    print(f"LONG_DISTANCE calls: {len(long_distance_data)}")
    print(f"SHORT_DISTANCE calls: {len(short_distance_data)}")
    print()
    
    # Define segments to analyze
    segments = [
        "rect_lookup", "player_pos", "ldp_select", 
        "short_path_gen", "short_project_merge", "short_door_path_gen", "short_door_open",
        "ldp_wp_path_gen", "ldp_wp_project_merge", "ldp_door_path_gen", "ldp_door_open",
        "click_ground", "total"
    ]
    
    def analyze_branch(branch_data: List[Dict[str, Any]], branch_name: str) -> None:
        """Analyze timing data for a specific branch."""
        if not branch_data:
            print(f"No data for {branch_name} branch")
            return
        
        print(f"=== {branch_name} BRANCH ANALYSIS ===")
        print(f"Calls: {len(branch_data)}")
        
        # Calculate statistics for each segment
        segment_stats = {}
        for segment in segments:
            values = []
            for call in branch_data:
                dur_ms = call.get("dur_ms", {})
                if segment in dur_ms and dur_ms[segment] is not None:
                    values.append(dur_ms[segment])
            
            if values:
                stats = calculate_percentiles(values)
                segment_stats[segment] = stats
            else:
                segment_stats[segment] = {"median": 0.0, "p90": 0.0, "p95": 0.0}
        
        # Print table
        print(f"{'Segment':<20} {'Median':<8} {'P90':<8} {'P95':<8} {'Samples':<8}")
        print("-" * 60)
        
        for segment in segments:
            stats = segment_stats[segment]
            sample_count = len([d for d in branch_data if d.get("dur_ms", {}).get(segment) is not None])
            print(f"{segment:<20} {stats['median']:<8.1f} {stats['p90']:<8.1f} {stats['p95']:<8.1f} {sample_count:<8}")
        
        print()
        
        # Find top 10 slowest calls
        slowest_calls = sorted(branch_data, key=lambda x: x.get("dur_ms", {}).get("total", 0), reverse=True)[:10]
        
        print("Top 10 slowest calls:")
        print(f"{'Rank':<4} {'Total (ms)':<12} {'Dominant Segment':<20} {'Door Attempted':<15} {'Error':<20}")
        print("-" * 80)
        
        for i, call in enumerate(slowest_calls, 1):
            total_ms = call.get("dur_ms", {}).get("total", 0)
            error = call.get("error", "")
            
            # Find dominant segment (excluding total)
            dur_ms = call.get("dur_ms", {})
            max_segment = "unknown"
            max_duration = 0
            for segment in segments:
                if segment != "total" and segment in dur_ms and dur_ms[segment] is not None:
                    if dur_ms[segment] > max_duration:
                        max_duration = dur_ms[segment]
                        max_segment = segment
            
            door_attempted = call.get("context", {}).get("door_attempted", False)
            print(f"{i:<4} {total_ms:<12.1f} {max_segment:<20} {str(door_attempted):<15} {error[:20]:<20}")
        
        print()
    
    # Analyze both branches
    analyze_branch(long_distance_data, "LONG_DISTANCE")
    analyze_branch(short_distance_data, "SHORT_DISTANCE")
    
    # Overall analysis
    print("=== OVERALL ANALYSIS ===")
    total_times = [d.get("dur_ms", {}).get("total", 0) for d in data if d.get("dur_ms", {}).get("total") is not None]
    if total_times:
        overall_stats = calculate_percentiles(total_times)
        print(f"Overall total time - Median: {overall_stats['median']:.1f}ms, P90: {overall_stats['p90']:.1f}ms, P95: {overall_stats['p95']:.1f}ms")
    
    # Error analysis
    error_calls = [d for d in data if not d.get("ok", True)]
    if error_calls:
        print(f"Error rate: {len(error_calls)}/{len(data)} ({len(error_calls)/len(data)*100:.1f}%)")
        error_types = defaultdict(int)
        for call in error_calls:
            error = call.get("error", "Unknown")
            error_types[error] += 1
        print("Error types:")
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count}")

def main():
    """Main function."""
    print("Go_to() Timing Analysis")
    print("=" * 50)
    
    data = load_timing_data()
    analyze_timing_data(data)

if __name__ == "__main__":
    main()

