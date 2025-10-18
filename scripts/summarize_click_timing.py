#!/usr/bin/env python3
"""
Click timing analysis script.
Reads all *.timing.jsonl files and produces summary statistics.
"""

import json
import glob
import os
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict

def read_timing_files() -> List[Dict[str, Any]]:
    """Read all timing JSONL files in the repository."""
    timing_files = []
    
    # Find all .timing.jsonl files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.timing.jsonl'):
                filepath = os.path.join(root, file)
                timing_files.append(filepath)
    
    print(f"Found {len(timing_files)} timing files:")
    for f in timing_files:
        print(f"  - {f}")
    
    all_entries = []
    for filepath in timing_files:
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entry['_source_file'] = filepath
                        entry['_line_num'] = line_num
                        all_entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in {filepath}:{line_num}: {e}")
        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return all_entries

def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 90, 95]) -> Dict[int, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {p: 0.0 for p in percentiles}
    
    sorted_values = sorted(values)
    result = {}
    for p in percentiles:
        if p == 50:
            result[p] = statistics.median(sorted_values)
        else:
            idx = int((p / 100.0) * (len(sorted_values) - 1))
            result[p] = sorted_values[idx]
    return result

def analyze_click_timing(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze click timing data."""
    # Filter for click timing entries
    click_entries = [e for e in entries if e.get('phase') == 'click_timing']
    
    if not click_entries:
        print("No click timing entries found!")
        return {}
    
    print(f"Found {len(click_entries)} click timing entries")
    
    # Group by click type
    by_type = defaultdict(list)
    for entry in click_entries:
        click_type = entry.get('who', 'unknown')
        by_type[click_type].append(entry)
    
    results = {}
    
    for click_type, type_entries in by_type.items():
        print(f"\n=== {click_type.upper()} CLICKS ({len(type_entries)} entries) ===")
        
        # Extract timing segments
        segments = ['resolve', 'cam', 'resample', 'hover', 'menu_open', 'menu_verify', 'dispatch', 'post_ack', 'total']
        segment_data = {seg: [] for seg in segments}
        
        for entry in type_entries:
            dur_ms = entry.get('dur_ms', {})
            for seg in segments:
                if seg in dur_ms and dur_ms[seg] is not None:
                    segment_data[seg].append(dur_ms[seg])
        
        # Calculate statistics for each segment
        segment_stats = {}
        for seg, values in segment_data.items():
            if values:
                stats = calculate_percentiles(values)
                segment_stats[seg] = {
                    'count': len(values),
                    'median': stats[50],
                    'p90': stats[90],
                    'p95': stats[95],
                    'min': min(values),
                    'max': max(values)
                }
            else:
                segment_stats[seg] = {
                    'count': 0,
                    'median': 0.0,
                    'p90': 0.0,
                    'p95': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        
        # Print table
        print(f"{'Segment':<12} {'Count':<6} {'Median':<8} {'P90':<8} {'P95':<8} {'Min':<8} {'Max':<8}")
        print("-" * 70)
        for seg in segments:
            stats = segment_stats[seg]
            print(f"{seg:<12} {stats['count']:<6} {stats['median']:<8.1f} {stats['p90']:<8.1f} {stats['p95']:<8.1f} {stats['min']:<8.1f} {stats['max']:<8.1f}")
        
        # Find slowest clicks
        slow_entries = []
        for entry in type_entries:
            total_time = entry.get('dur_ms', {}).get('total', 0)
            if total_time > 0:
                slow_entries.append((entry, total_time))
        
        slow_entries.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 slowest {click_type} clicks:")
        print(f"{'Rank':<4} {'Total (ms)':<10} {'Action':<15} {'Error':<20} {'Dominant Segment':<20}")
        print("-" * 80)
        
        for i, (entry, total_time) in enumerate(slow_entries[:10]):
            action = entry.get('action', 'Unknown')[:14]
            error = entry.get('error', '')[:19] if entry.get('error') else 'None'
            
            # Find dominant segment
            dur_ms = entry.get('dur_ms', {})
            dominant_seg = 'unknown'
            max_dur = 0
            for seg in segments[:-1]:  # Exclude 'total'
                if seg in dur_ms and dur_ms[seg] and dur_ms[seg] > max_dur:
                    max_dur = dur_ms[seg]
                    dominant_seg = seg
            
            print(f"{i+1:<4} {total_time:<10.1f} {action:<15} {error:<20} {dominant_seg:<20}")
        
        results[click_type] = {
            'segment_stats': segment_stats,
            'slow_entries': slow_entries[:10]
        }
    
    return results

def analyze_camera_timing(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze camera timing data."""
    camera_entries = [e for e in entries if e.get('phase') == 'camera_timing']
    
    if not camera_entries:
        print("\nNo camera timing entries found!")
        return {}
    
    print(f"\n=== CAMERA TIMING ({len(camera_entries)} entries) ===")
    
    # Extract timing segments
    segments = ['first_projection', 'aim_total']
    segment_data = {seg: [] for seg in segments}
    
    for entry in camera_entries:
        dur_ms = entry.get('dur_ms', {})
        for seg in segments:
            if seg in dur_ms and dur_ms[seg] is not None:
                segment_data[seg].append(dur_ms[seg])
    
    # Calculate statistics
    segment_stats = {}
    for seg, values in segment_data.items():
        if values:
            stats = calculate_percentiles(values)
            segment_stats[seg] = {
                'count': len(values),
                'median': stats[50],
                'p90': stats[90],
                'p95': stats[95],
                'min': min(values),
                'max': max(values)
            }
        else:
            segment_stats[seg] = {
                'count': 0,
                'median': 0.0,
                'p90': 0.0,
                'p95': 0.0,
                'min': 0.0,
                'max': 0.0
            }
    
    # Print table
    print(f"{'Segment':<15} {'Count':<6} {'Median':<8} {'P90':<8} {'P95':<8} {'Min':<8} {'Max':<8}")
    print("-" * 70)
    for seg in segments:
        stats = segment_stats[seg]
        print(f"{seg:<15} {stats['count']:<6} {stats['median']:<8.1f} {stats['p90']:<8.1f} {stats['p95']:<8.1f} {stats['min']:<8.1f} {stats['max']:<8.1f}")
    
    return {'segment_stats': segment_stats}

def main():
    """Main analysis function."""
    print("Click Timing Analysis")
    print("===================")
    
    # Read all timing files
    entries = read_timing_files()
    
    if not entries:
        print("No timing entries found!")
        return
    
    print(f"\nTotal entries: {len(entries)}")
    
    # Analyze click timing
    click_results = analyze_click_timing(entries)
    
    # Analyze camera timing
    camera_results = analyze_camera_timing(entries)
    
    # Summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    if click_results:
        print("\nClick Type Analysis:")
        for click_type, data in click_results.items():
            segment_stats = data['segment_stats']
            total_stats = segment_stats.get('total', {})
            print(f"  {click_type}: {total_stats.get('count', 0)} clicks, "
                  f"median {total_stats.get('median', 0):.1f}ms, "
                  f"p95 {total_stats.get('p95', 0):.1f}ms")
            
            # Find dominant segment
            max_median = 0
            dominant_seg = 'unknown'
            for seg, stats in segment_stats.items():
                if seg != 'total' and stats['median'] > max_median:
                    max_median = stats['median']
                    dominant_seg = seg
            
            print(f"    Dominant segment: {dominant_seg} ({max_median:.1f}ms median)")
    
    if camera_results:
        print(f"\nCamera Analysis:")
        aim_stats = camera_results['segment_stats'].get('aim_total', {})
        print(f"  Camera aiming: {aim_stats.get('count', 0)} operations, "
              f"median {aim_stats.get('median', 0):.1f}ms, "
              f"p95 {aim_stats.get('p95', 0):.1f}ms")

if __name__ == "__main__":
    main()
