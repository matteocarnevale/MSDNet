#!/usr/bin/env python3
"""
Permissive VoD converter that preserves maximum files.
Applies minimal filtering to avoid losing files during conversion.

Usage: python convert_vod_permissive.py --vod_root /path/to/view_of_delft_PUBLIC --output_dir data/vod
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_vod_permissive(vod_root, output_dir, radar_type="radar_5frames"):
    """Convert with minimal filtering to preserve maximum files."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    lidar_dir = vod_path / 'lidar'
    radar_dir = vod_path / radar_type
    
    print(f"Converting VoD dataset (PERMISSIVE MODE)")
    print(f"Source: {vod_path}")
    print(f"Radar: {radar_type}")
    print(f"Output: {output_dir}")
    
    # Create directories
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Copy splits
    imagesets = lidar_dir / 'ImageSets'
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        src = imagesets / split_name
        dst = out_path / 'split' / split_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied: {split_name}")
    
    # Find all existing files
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    radar_velodyne = radar_dir / 'training' / 'velodyne'
    
    lidar_files = {f.stem for f in lidar_velodyne.glob('*.bin')} if lidar_velodyne.exists() else set()
    radar_files = {f.stem for f in radar_velodyne.glob('*.bin')} if radar_velodyne.exists() else set()
    
    valid_ids = sorted(lidar_files & radar_files)
    
    print(f"LiDAR files found: {len(lidar_files)}")
    print(f"Radar files found: {len(radar_files)}")
    print(f"Valid pairs: {len(valid_ids)}")
    
    # Process LiDAR with minimal filtering
    print("Processing LiDAR files...")
    lidar_success = 0
    
    for frame_id in tqdm(valid_ids, desc="LiDAR"):
        try:
            src_file = lidar_velodyne / f'{frame_id}.bin'
            data = np.fromfile(src_file, dtype=np.float32)
            
            # Handle formats flexibly
            if data.size % 4 == 0:
                points = data.reshape(-1, 4)
            elif data.size % 3 == 0:
                points = data.reshape(-1, 3)
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, intensity])
            else:
                # For other formats, try to interpret as x,y,z,intensity
                if data.size >= 4:
                    n_points = data.size // 4
                    points = data[:n_points*4].reshape(-1, 4)
                else:
                    continue
            
            # Minimal preprocessing - only remove extreme outliers
            if points.shape[0] > 0:
                # Remove only extreme outliers (very permissive)
                valid_mask = (
                    (points[:, 0] > -100) & (points[:, 0] < 100) &  # x range
                    (points[:, 1] > -100) & (points[:, 1] < 100) &  # y range  
                    (points[:, 2] > -10) & (points[:, 2] < 20)      # z range
                )
                points = points[valid_mask]
            
            if points.shape[0] > 0:
                dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                lidar_success += 1
                
        except Exception as e:
            print(f"Error with LiDAR {frame_id}: {e}")
    
    # Process radar with minimal filtering
    print("Processing radar files...")
    radar_success = 0
    
    for frame_id in tqdm(valid_ids, desc="Radar"):
        try:
            src_file = radar_velodyne / f'{frame_id}.bin'
            data = np.fromfile(src_file, dtype=np.float32)
            
            # Handle formats flexibly
            if data.size % 5 == 0:
                points = data.reshape(-1, 5)
            elif data.size % 4 == 0:
                points = data.reshape(-1, 4)
                velocity = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, velocity])
            elif data.size % 6 == 0:
                points = data.reshape(-1, 6)[:, :5]  # Take first 5 features
            elif data.size % 7 == 0:
                points = data.reshape(-1, 7)[:, :5]  # Take first 5 features
            else:
                # Try to interpret as 5-feature points
                if data.size >= 5:
                    n_points = data.size // 5
                    points = data[:n_points*5].reshape(-1, 5)
                else:
                    continue
            
            # Minimal preprocessing - only remove extreme outliers
            if points.shape[0] > 0:
                valid_mask = (
                    (points[:, 0] > -100) & (points[:, 0] < 100) &  # x range
                    (points[:, 1] > -100) & (points[:, 1] < 100) &  # y range
                    (points[:, 2] > -10) & (points[:, 2] < 20)      # z range  
                )
                points = points[valid_mask]
            
            if points.shape[0] > 0:
                dst_file = out_path / 'radar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                radar_success += 1
                
        except Exception as e:
            print(f"Error with radar {frame_id}: {e}")
    
    # Update splits to include only successfully processed files
    final_valid_ids = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        radar_file = out_path / 'radar' / f'{frame_id}.bin'
        if radar_file.exists():
            final_valid_ids.add(frame_id)
    
    print(f"\nFinal processing results:")
    print(f"LiDAR processed: {lidar_success}")
    print(f"Radar processed: {radar_success}")
    print(f"Final valid pairs: {len(final_valid_ids)}")
    
    # Update all split files
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                original_ids = [line.strip() for line in f if line.strip()]
            
            valid_ids_for_split = [fid for fid in original_ids if fid in final_valid_ids]
            
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_ids_for_split))
            
            print(f"Updated {split_name}: {len(original_ids)} -> {len(valid_ids_for_split)} frames")
    
    success_rate = len(final_valid_ids) / len(valid_ids) * 100
    print(f"\nConversion complete!")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Dataset ready for training")


def main():
    parser = argparse.ArgumentParser(description="Permissive VoD converter")
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', default='data/vod')
    parser.add_argument('--radar_type', default='radar_5frames',
                       choices=['radar', 'radar_3frames', 'radar_5frames'])
    args = parser.parse_args()
    
    convert_vod_permissive(args.vod_root, args.output_dir, args.radar_type)


if __name__ == '__main__':
    main()
