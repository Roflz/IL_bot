#!/usr/bin/env python3
"""
Analyze camera Jacobian accuracy test results.
"""

import json
import statistics
from pathlib import Path

def analyze_results(file_path):
    """Analyze camera accuracy test results."""
    results = []
    
    # Load all results
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not results:
        print("No results found!")
        return
    
    print(f"\n{'='*80}")
    print(f"CAMERA JACOBIAN ACCURACY ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total tests: {len(results)}")
    
    # Success rate
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    # Error analysis (only for successful tests with error data)
    errors = []
    for r in successful:
        err = r.get("error", {})
        if err and err.get("distance") is not None:
            errors.append(err["distance"])
    
    if errors:
        print(f"\n{'='*80}")
        print(f"ERROR ANALYSIS (Successful tests only)")
        print(f"{'='*80}\n")
        print(f"Total with error data: {len(errors)}")
        print(f"Mean error: {statistics.mean(errors):.2f} pixels")
        print(f"Median error: {statistics.median(errors):.2f} pixels")
        print(f"Min error: {min(errors):.2f} pixels")
        print(f"Max error: {max(errors):.2f} pixels")
        print(f"Std deviation: {statistics.stdev(errors) if len(errors) > 1 else 0:.2f} pixels")
        
        # Error distribution
        excellent = [e for e in errors if e < 50]
        good = [e for e in errors if 50 <= e < 150]
        moderate = [e for e in errors if 150 <= e < 300]
        poor = [e for e in errors if e >= 300]
        
        print(f"\nError Distribution:")
        print(f"  Excellent (<50px):   {len(excellent)} ({len(excellent)/len(errors)*100:.1f}%)")
        print(f"  Good (50-150px):     {len(good)} ({len(good)/len(errors)*100:.1f}%)")
        print(f"  Moderate (150-300px): {len(moderate)} ({len(moderate)/len(errors)*100:.1f}%)")
        print(f"  Poor (>300px):       {len(poor)} ({len(poor)/len(errors)*100:.1f}%)")
        
        # Worst cases
        if poor:
            print(f"\nWorst cases (>300px error):")
            worst = sorted([(r, r.get("error", {}).get("distance", 0)) for r in successful if r.get("error", {}).get("distance", 0) >= 300], 
                          key=lambda x: x[1], reverse=True)[:10]
            for r, err_dist in worst:
                obj = r.get("object", {})
                dist = r.get("distance", 0)
                print(f"  {obj.get('name', 'unknown')} at ({obj.get('world', {}).get('x')}, {obj.get('world', {}).get('y')}) - "
                      f"Distance: {dist} tiles, Error: {err_dist:.1f}px")
    
    # Movement accuracy analysis
    print(f"\n{'='*80}")
    print(f"MOVEMENT ACCURACY ANALYSIS")
    print(f"{'='*80}\n")
    
    yaw_diffs = []
    pitch_diffs = []
    
    for r in successful:
        calc = r.get("calculated_movement", {})
        actual = r.get("actual_movement", {})
        
        calc_yaw = calc.get("delta_yaw")
        actual_yaw = actual.get("delta_yaw")
        if calc_yaw is not None and actual_yaw is not None:
            # Handle yaw wrapping
            diff = abs(actual_yaw - calc_yaw)
            if diff > 1024:
                diff = 2048 - diff
            yaw_diffs.append(diff)
        
        calc_pitch = calc.get("delta_pitch")
        actual_pitch = actual.get("delta_pitch")
        if calc_pitch is not None and actual_pitch is not None:
            pitch_diffs.append(abs(actual_pitch - calc_pitch))
    
    if yaw_diffs:
        print(f"Yaw Movement Accuracy:")
        print(f"  Mean difference: {statistics.mean(yaw_diffs):.2f} units")
        print(f"  Median difference: {statistics.median(yaw_diffs):.2f} units")
        print(f"  Max difference: {max(yaw_diffs):.2f} units")
        print(f"  Tests with >100 unit difference: {len([d for d in yaw_diffs if d > 100])}")
    
    if pitch_diffs:
        print(f"\nPitch Movement Accuracy:")
        print(f"  Mean difference: {statistics.mean(pitch_diffs):.2f} units")
        print(f"  Median difference: {statistics.median(pitch_diffs):.2f} units")
        print(f"  Max difference: {max(pitch_diffs):.2f} units")
        print(f"  Tests with >50 unit difference: {len([d for d in pitch_diffs if d > 50])}")
    
    # Distance vs error correlation
    print(f"\n{'='*80}")
    print(f"DISTANCE VS ERROR CORRELATION")
    print(f"{'='*80}\n")
    
    distance_error_pairs = []
    for r in successful:
        dist = r.get("distance")
        err = r.get("error", {}).get("distance")
        if dist is not None and err is not None:
            distance_error_pairs.append((dist, err))
    
    if distance_error_pairs:
        # Group by distance ranges
        ranges = [
            (0, 5, "0-5 tiles"),
            (5, 10, "5-10 tiles"),
            (10, 15, "10-15 tiles"),
            (15, 20, "15-20 tiles"),
            (20, 25, "20-25 tiles")
        ]
        
        for min_dist, max_dist, label in ranges:
            group = [err for dist, err in distance_error_pairs if min_dist <= dist < max_dist]
            if group:
                print(f"{label:15s}: {len(group):3d} tests, Mean error: {statistics.mean(group):.1f}px, "
                      f"Median: {statistics.median(group):.1f}px")
    
    # Sign error analysis (yaw direction issues)
    print(f"\n{'='*80}")
    print(f"YAW DIRECTION ANALYSIS")
    print(f"{'='*80}\n")
    
    sign_errors = []
    for r in successful:
        calc = r.get("calculated_movement", {})
        actual = r.get("actual_movement", {})
        calc_yaw = calc.get("delta_yaw")
        actual_yaw = actual.get("delta_yaw")
        
        if calc_yaw is not None and actual_yaw is not None and abs(calc_yaw) > 10:
            # Check if signs are opposite
            if (calc_yaw > 0 and actual_yaw < 0) or (calc_yaw < 0 and actual_yaw > 0):
                # Check if magnitudes are similar (within 200 units)
                if abs(abs(calc_yaw) - abs(actual_yaw)) < 200:
                    sign_errors.append(r)
    
    if sign_errors:
        print(f"Potential sign errors (opposite direction, similar magnitude): {len(sign_errors)}")
        for r in sign_errors[:5]:
            obj = r.get("object", {})
            calc = r.get("calculated_movement", {})
            actual = r.get("actual_movement", {})
            print(f"  {obj.get('name')}: calc={calc.get('delta_yaw'):+.1f}, actual={actual.get('delta_yaw'):+.1f}")
    else:
        print("No obvious sign errors detected")
    
    # Large error cases analysis
    print(f"\n{'='*80}")
    print(f"LARGE ERROR CASES (>500px)")
    print(f"{'='*80}\n")
    
    large_errors = [r for r in successful if r.get("error", {}).get("distance", 0) > 500]
    if large_errors:
        print(f"Found {len(large_errors)} cases with >500px error:")
        for r in large_errors[:10]:
            obj = r.get("object", {})
            err = r.get("error", {})
            calc = r.get("calculated_movement", {})
            screen_before = r.get("screen_before", {})
            screen_after = r.get("screen_after", {})
            
            print(f"\n  {obj.get('name')} at ({obj.get('world', {}).get('x')}, {obj.get('world', {}).get('y')}):")
            print(f"    Distance: {r.get('distance')} tiles")
            print(f"    Error: {err.get('distance', 0):.1f}px (x={err.get('x', 0):.1f}, y={err.get('y', 0):.1f})")
            print(f"    Screen before: ({screen_before.get('x')}, {screen_before.get('y')})")
            print(f"    Screen after: ({screen_after.get('x')}, {screen_after.get('y')})")
            print(f"    Target: ({r.get('target_screen', {}).get('x')}, {r.get('target_screen', {}).get('y')})")
            print(f"    Calculated delta_yaw: {calc.get('delta_yaw', 0):+.1f}")
            print(f"    Calculated error_pixels: {calc.get('error_pixels', 0):.1f}px")
    else:
        print("No cases with >500px error")
    
    # Failed tests analysis
    if failed:
        print(f"\n{'='*80}")
        print(f"FAILED TESTS ANALYSIS")
        print(f"{'='*80}\n")
        
        for r in failed:
            obj = r.get("object", {})
            err_msg = r.get("error", "Unknown")
            print(f"  {obj.get('name')} at ({obj.get('world', {}).get('x')}, {obj.get('world', {}).get('y')}): {err_msg}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Find most recent results file
        results_dir = Path("test_results")
        if results_dir.exists():
            files = list(results_dir.glob("camera_jacobian_accuracy_*.jsonl"))
            if files:
                file_path = max(files, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent file: {file_path}")
            else:
                print("No results files found!")
                sys.exit(1)
        else:
            print("test_results directory not found!")
            sys.exit(1)
    
    analyze_results(file_path)

