#!/usr/bin/env python3
"""
Quick script to check JSON training data structure
"""

import json
from pathlib import Path

def check_json_structure(file_path):
    """Check the structure of JSON training data"""
    print(f"\n{'='*60}")
    print(f"CHECKING JSON STRUCTURE: {file_path}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Length: {len(data)}")
        
        if isinstance(data, list):
            print(f"\nFirst sequence structure:")
            first_seq = data[0]
            print(f"  Type: {type(first_seq)}")
            print(f"  Length: {len(first_seq)}")
            
            if isinstance(first_seq, list):
                print(f"  First timestep type: {type(first_seq[0])}")
                if len(first_seq) > 0:
                    print(f"  First timestep length: {len(first_seq[0])}")
                    if len(first_seq[0]) > 0:
                        print(f"  First timestep first element: {first_seq[0][0]}")
                        print(f"  First timestep first 8 elements: {first_seq[0][:8]}")
            
            print(f"\nLast sequence structure:")
            last_seq = data[-1]
            print(f"  Type: {type(last_seq)}")
            print(f"  Length: {len(last_seq)}")
            
            if isinstance(last_seq, list) and len(last_seq) > 0:
                print(f"  Last timestep length: {len(last_seq[-1])}")
                if len(last_seq[-1]) > 0:
                    print(f"  Last timestep first 8 elements: {last_seq[-1][:8]}")
        
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python check_json_structure.py <json_filepath>")
        print("Example: python check_json_structure.py data/06_final_training_data/action_input_sequences.json")
        return
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    if not file_path.suffix == '.json':
        print(f"❌ Not a JSON file: {file_path}")
        return
    
    check_json_structure(file_path)

if __name__ == "__main__":
    main()
