#!/usr/bin/env python3
"""
Debug exactly why radar files are lost during conversion.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def debug_specific_files(radar_dir, frame_ids_lost, max_debug=20):
    """Debug specific files that were lost."""
    velodyne_dir = radar_dir / 'training' / 'velodyne'
    
    print(f"Debugging {min(max_debug, len(frame_ids_lost))} lost radar files:")
    
    for i, frame_id in enumerate(frame_ids_lost[:max_debug]):
        radar_file = velodyne_dir / f'{frame_id}.bin'
        print(f"\n{i+1}. File: {frame_id}.bin")
        
        if not radar_file.exists():
            print(f"   Status: FILE NOT FOUND")
            continue
            
        try:
            data = np.fromfile(radar_file, dtype=np.float32)
            print(f"   Size: {data.size} elements")
            
            # Check all possible formats
            possible_formats = []
            for n_feat in range(1, 11):
                if data.size % n_feat == 0:
                    n_points = data.size // n_feat
                    possible_formats.append((n_feat, n_points))
            
            print(f"   Possible formats: {possible_formats}")
            
            # Show raw data sample
            if data.size >= 10:
                print(f"   First 10 elements: {data[:10]}")
                
            # Try to process with current logic
            if data.size % 7 == 0:
                points = data.reshape(-1, 7)[:, :5]
                print(f"   Would process as 7->5 features: {points.shape}")
            elif data.size % 5 == 0:
                points = data.reshape(-1, 5)
                print(f"   Would process as 5 features: {points.shape}")
            elif data.size % 4 == 0:
                points = data.reshape(-1, 4)
                print(f"   Would process as 4+velocity: {points.shape} -> {(points.shape[0], 5)}")
            elif data.size % 3 == 0:
                points = data.reshape(-1, 3)
                print(f"   Would process as 3+extras: {points.shape} -> {(points.shape[0], 5)}")
            else:
                print(f"   ISSUE: No standard format fits")
                
        except Exception as e:
            print(f"   ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--radar_type', default='radar_5frames')
    parser.add_argument('--debug_count', type=int, default=20)
    args = parser.parse_args()
    
    vod_path = Path(args.vod_root)
    lidar_dir = vod_path / 'lidar'
    radar_dir = vod_path / args.radar_type
    
    # Get frame IDs from splits
    imagesets = lidar_dir / 'ImageSets'
    all_frame_ids = set()
    
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = imagesets / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
                all_frame_ids.update(frame_ids)
    
    all_frame_ids = sorted(all_frame_ids)
    print(f"Total frame IDs in splits: {len(all_frame_ids)}")
    
    # Find which radar files exist
    radar_velodyne = radar_dir / 'training' / 'velodyne'
    existing_radar = {f.stem for f in radar_velodyne.glob('*.bin')} if radar_velodyne.exists() else set()
    
    missing_radar = [fid for fid in all_frame_ids if fid not in existing_radar]
    
    print(f"Missing radar files: {len(missing_radar)}")
    if len(missing_radar) > 0:
        print(f"Examples: {missing_radar[:10]}")
    
    # Debug processing failures
    existing_frame_ids = [fid for fid in all_frame_ids if fid in existing_radar]
    print(f"Existing radar files: {len(existing_frame_ids)}")
    
    if len(existing_frame_ids) > 0:
        debug_specific_files(radar_dir, existing_frame_ids[:args.debug_count])


if __name__ == '__main__':
    main()
