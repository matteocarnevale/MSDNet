#!/usr/bin/env python3
"""
Convert View of Delft dataset to MSDNet format.

Handles 3 radar variants:
- radar/: Single-frame (sparse, ~50-200 points)
- radar_3frames/: 3-frame accumulation (medium, ~150-600 points)  
- radar_5frames/: 5-frame accumulation (dense, ~250-1000 points) [RECOMMENDED]

Usage:
    python convert_vod_real.py --vod_root /path/to/view_of_delft_PUBLIC --output_dir data/vod --radar_type radar_5frames
"""

import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def crop_points_to_range(points, point_cloud_range, min_points=10):
    """Crop points to the specified range, but keep files with minimum points."""
    pc_min = np.array(point_cloud_range[:3])
    pc_max = np.array(point_cloud_range[3:])
    mask = ((points[:, :3] >= pc_min) & (points[:, :3] < pc_max)).all(axis=1)
    cropped = points[mask]
    
    # If too few points after cropping, expand range slightly
    if cropped.shape[0] < min_points and points.shape[0] > min_points:
        # Try with expanded range (+20%)
        pc_range_expanded = pc_max - pc_min
        pc_min_exp = pc_min - 0.2 * pc_range_expanded
        pc_max_exp = pc_max + 0.2 * pc_range_expanded
        mask_exp = ((points[:, :3] >= pc_min_exp) & (points[:, :3] <= pc_max_exp)).all(axis=1)
        expanded = points[mask_exp]
        if expanded.shape[0] >= min_points:
            return expanded
    
    return cropped


def remove_ground_points(lidar_points, ground_height=-2.0):
    """Remove ground points from LiDAR data - more permissive threshold."""
    return lidar_points[lidar_points[:, 2] > ground_height]


def apply_fov_filter(points, fov_degrees=140.0):
    """Apply horizontal field of view filter - more permissive FOV."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    half_fov = np.deg2rad(fov_degrees / 2)
    mask = np.abs(angles) <= half_fov
    return points[mask]


def convert_vod_to_msdnet(vod_root, output_dir, radar_type="radar_5frames"):
    """Convert VoD dataset to MSDNet format."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    # Verify structure exists
    if not (vod_path / 'lidar').exists():
        raise FileNotFoundError(f"LiDAR directory not found: {vod_path / 'lidar'}")
    if not (vod_path / radar_type).exists():
        raise FileNotFoundError(f"Radar directory not found: {vod_path / radar_type}")
    
    print(f"Source: {vod_path}")
    print(f"Radar type: {radar_type}")
    print(f"Output: {output_dir}")
    
    # Create output directories
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Processing parameters from MSDNet paper
    point_cloud_range = [0, -16, -2, 32, 16, 4]
    
    # Copy dataset splits
    imagesets = vod_path / 'lidar' / 'ImageSets'
    for split_file in ['train.txt', 'test.txt', 'val.txt']:
        src = imagesets / split_file
        dst = out_path / 'split' / split_file
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied split: {split_file}")
    
    # Load frame IDs
    train_ids = []
    test_ids = []
    
    if (out_path / 'split' / 'train.txt').exists():
        with open(out_path / 'split' / 'train.txt') as f:
            train_ids = [line.strip() for line in f if line.strip()]
    
    if (out_path / 'split' / 'test.txt').exists():
        with open(out_path / 'split' / 'test.txt') as f:
            test_ids = [line.strip() for line in f if line.strip()]
    
    all_ids = train_ids + test_ids
    print(f"Train frames: {len(train_ids)}, Test frames: {len(test_ids)}")
    
    # Process LiDAR data
    print("Processing LiDAR data...")
    lidar_processed = 0
    
    for frame_id in tqdm(all_ids, desc="LiDAR"):
        lidar_file = vod_path / 'lidar' / 'training' / 'velodyne' / f'{frame_id}.bin'
        if not lidar_file.exists():
            continue
            
        try:
            data = np.frombuffer(lidar_file.read_bytes(), dtype=np.float32)
            if data.size % 4 != 0:
                continue
            points = data.reshape(-1, 4)
            
            # Apply MSDNet preprocessing
            points = remove_ground_points(points)
            points = apply_fov_filter(points)
            points = crop_points_to_range(points, point_cloud_range)
            
            if points.shape[0] > 0:
                output_file = out_path / 'lidar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(output_file)
                lidar_processed += 1
                
        except Exception as e:
            print(f"Error processing LiDAR {frame_id}: {e}")
    
    # Process radar data
    print(f"Processing {radar_type} data...")
    radar_processed = 0
    
    for frame_id in tqdm(all_ids, desc="Radar"):
        radar_file = vod_path / radar_type / 'training' / 'velodyne' / f'{frame_id}.bin'
        if not radar_file.exists():
            continue
            
        try:
            data = np.frombuffer(radar_file.read_bytes(), dtype=np.float32)
            if data.size % 5 == 0:
                points = data.reshape(-1, 5)
            elif data.size % 4 == 0:
                pts = data.reshape(-1, 4)
                velocity = np.zeros((pts.shape[0], 1))
                points = np.hstack([pts, velocity])
            else:
                continue
            
            # Apply range cropping only
            points = crop_points_to_range(points, point_cloud_range)
            
            if points.shape[0] > 0:
                output_file = out_path / 'radar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(output_file)
                radar_processed += 1
                
        except Exception as e:
            print(f"Error processing radar {frame_id}: {e}")
    
    print(f"Conversion completed")
    print(f"LiDAR files: {lidar_processed}")
    print(f"Radar files: {radar_processed}")
    print(f"Ready for training")


def main():
    parser = argparse.ArgumentParser(description="Convert VoD to MSDNet format")
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', default='data/vod')
    parser.add_argument('--radar_type', default='radar_5frames',
                       choices=['radar', 'radar_3frames', 'radar_5frames'])
    
    args = parser.parse_args()
    
    print("VoD Dataset Converter for MSDNet")
    print(f"Radar variants available:")
    print(f"  radar: Single-frame (sparse, ~50-200 points)")
    print(f"  radar_3frames: 3-frame accumulation (medium, ~150-600 points)")
    print(f"  radar_5frames: 5-frame accumulation (dense, ~250-1000 points) [RECOMMENDED]")
    print()
    
    convert_vod_to_msdnet(args.vod_root, args.output_dir, args.radar_type)


if __name__ == '__main__':
    main()
