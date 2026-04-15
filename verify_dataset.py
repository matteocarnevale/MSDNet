#!/usr/bin/env python3
"""
Verify VoD dataset consistency before training.

Usage: python verify_dataset.py --data_root /path/to/dataset --fix_splits
"""

import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def verify_file_pair(data_root, frame_id):
    """Check if both LiDAR and radar files exist."""
    lidar_file = Path(data_root) / 'lidar' / f'{frame_id}.bin'
    radar_file = Path(data_root) / 'radar' / f'{frame_id}.bin'
    return lidar_file.exists() and radar_file.exists()


def verify_dataset(data_root, fix_splits=False):
    """Verify dataset and optionally fix splits."""
    data_path = Path(data_root)
    
    print(f"Verifying dataset: {data_root}")
    
    # Load splits
    splits = {}
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = data_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                splits[split_name] = [line.strip() for line in f if line.strip()]
    
    # Check each split
    corrected_splits = {}
    for split_name, frame_ids in splits.items():
        print(f"\nChecking {split_name}: {len(frame_ids)} frames")
        
        valid_frames = []
        for frame_id in tqdm(frame_ids):
            if verify_file_pair(data_root, frame_id):
                valid_frames.append(frame_id)
        
        corrected_splits[split_name] = valid_frames
        valid_count = len(valid_frames)
        total_count = len(frame_ids) 
        print(f"Valid: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    # Fix splits if requested
    if fix_splits:
        print("\nFixing splits...")
        for split_name, valid_frames in corrected_splits.items():
            split_file = data_path / 'split' / split_name
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_frames))
            print(f"Updated {split_name}: {len(valid_frames)} frames")
    
    return corrected_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--fix_splits', action='store_true')
    args = parser.parse_args()
    
    verify_dataset(args.data_root, args.fix_splits)


if __name__ == '__main__':
    main()
