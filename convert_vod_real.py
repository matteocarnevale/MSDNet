#!/usr/bin/env python3
"""
Convert actual View of Delft dataset to MSDNet format.

This script processes the real VoD dataset structure as shown in the provided image.

Usage:
    python convert_vod_real.py --vod_root /path/to/view_of_delft_PUBLIC --output_dir data/vod
"""

import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def read_calibration(calib_file):
    """Read VoD calibration file and extract transformations."""
    calib = {}
    if not calib_file.exists():
        return calib
        
    with open(calib_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
    return calib


def crop_points_to_range(points, point_cloud_range):
    """Crop points to the specified range."""
    pc_min = np.array(point_cloud_range[:3])
    pc_max = np.array(point_cloud_range[3:])
    
    mask = ((points[:, :3] >= pc_min) & (points[:, :3] < pc_max)).all(axis=1)
    return points[mask]


def remove_ground_points(lidar_points, ground_height=-1.5):
    """Remove ground points from LiDAR data."""
    return lidar_points[lidar_points[:, 2] > ground_height]


def apply_fov_filter(points, fov_degrees=120.0):
    """Apply horizontal field of view filter (radar FoV)."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    half_fov = np.deg2rad(fov_degrees / 2)
    mask = np.abs(angles) <= half_fov
    return points[mask]


def convert_vod_to_msdnet(vod_root, output_dir, radar_type="radar_5frames"):
    """
    Convert VoD dataset to MSDNet format.
    
    Args:
        vod_root: Path to view_of_delft_PUBLIC directory
        output_dir: Output directory for MSDNet format
        radar_type: "radar", "radar_3frames", or "radar_5frames"
    """
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    # Verify VoD structure
    required_dirs = [
        vod_path / 'lidar',
        vod_path / radar_type,
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    print(f"✅ VoD dataset found at: {vod_path}")
    print(f"📡 Using radar type: {radar_type}")
    
    # Create output directories
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Point cloud processing parameters (from MSDNet paper)
    point_cloud_range = [0, -16, -2, 32, 16, 4]  # meters
    
    # Copy splits from LiDAR ImageSets
    lidar_imagesets = vod_path / 'lidar' / 'ImageSets'
    split_files_copied = []
    
    for split_file in ['train.txt', 'test.txt', 'val.txt', 'full.txt']:
        src_file = lidar_imagesets / split_file
        dst_file = out_path / 'split' / split_file
        
        if src_file.exists():
            shutil.copy(src_file, dst_file)
            split_files_copied.append(split_file)
            print(f"✅ Copied split: {split_file}")
    
    if not split_files_copied:
        raise FileNotFoundError(f"No split files found in {lidar_imagesets}")
    
    # Read frame IDs from available splits
    all_frame_ids = set()
    train_ids = []
    test_ids = []
    
    # Read train split
    train_file = out_path / 'split' / 'train.txt'
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_ids = [line.strip() for line in f if line.strip()]
            all_frame_ids.update(train_ids)
    
    # Read test split
    test_file = out_path / 'split' / 'test.txt'
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_ids = [line.strip() for line in f if line.strip()]
            all_frame_ids.update(test_ids)
    
    # If no train/test, use full.txt
    if not all_frame_ids:
        full_file = out_path / 'split' / 'full.txt'
        if full_file.exists():
            with open(full_file, 'r') as f:
                all_frame_ids = set(line.strip() for line in f if line.strip())
                # Split full into train/test (80/20)
                frame_list = sorted(list(all_frame_ids))
                split_idx = int(len(frame_list) * 0.8)
                train_ids = frame_list[:split_idx]
                test_ids = frame_list[split_idx:]
                
                # Write new splits
                with open(out_path / 'split' / 'train.txt', 'w') as f:
                    f.write('\n'.join(train_ids))
                with open(out_path / 'split' / 'test.txt', 'w') as f:
                    f.write('\n'.join(test_ids))
    
    print(f"📊 Found {len(train_ids)} training and {len(test_ids)} test frames")
    print(f"📊 Total frames to process: {len(all_frame_ids)}")
    
    # Process LiDAR data
    print("\n🔄 Processing LiDAR data...")
    lidar_dir = vod_path / 'lidar' / 'training'
    processed_lidar = 0
    
    for frame_id in tqdm(all_frame_ids, desc="LiDAR Processing"):
        try:
            # Load LiDAR points
            lidar_file = lidar_dir / 'velodyne' / f'{frame_id}.bin'
            if not lidar_file.exists():
                # Try without .bin extension
                alt_file = lidar_dir / 'velodyne' / frame_id
                if alt_file.exists():
                    lidar_file = alt_file
                else:
                    continue
                
            lidar_points = np.frombuffer(lidar_file.read_bytes(), dtype=np.float32)
            
            # Handle different point cloud formats
            if lidar_points.size % 4 == 0:
                lidar_points = lidar_points.reshape(-1, 4)  # [x, y, z, intensity]
            elif lidar_points.size % 3 == 0:
                lidar_points = lidar_points.reshape(-1, 3)  # [x, y, z] - add intensity
                intensity = np.ones((lidar_points.shape[0], 1))
                lidar_points = np.hstack([lidar_points, intensity])
            else:
                print(f"Warning: Unexpected LiDAR format for {frame_id}")
                continue
            
            # Apply preprocessing (following MSDNet paper Section IV-B)
            original_count = lidar_points.shape[0]
            lidar_points = remove_ground_points(lidar_points)
            lidar_points = apply_fov_filter(lidar_points)
            lidar_points = crop_points_to_range(lidar_points, point_cloud_range)
            
            if lidar_points.shape[0] == 0:
                print(f"Warning: No LiDAR points remaining after filtering for {frame_id}")
                continue
            
            # Save processed LiDAR
            output_file = out_path / 'lidar' / f'{frame_id}.bin'
            lidar_points.astype(np.float32).tofile(output_file)
            processed_lidar += 1
            
        except Exception as e:
            print(f"Error processing LiDAR {frame_id}: {e}")
    
    print(f"✅ Processed {processed_lidar} LiDAR files")
    
    # Process 4D Radar data
    print(f"\n📡 Processing {radar_type} data...")
    radar_dir = vod_path / radar_type / 'training'
    processed_radar = 0
    
    for frame_id in tqdm(all_frame_ids, desc="Radar Processing"):
        try:
            # Load radar points
            radar_file = radar_dir / 'velodyne' / f'{frame_id}.bin'
            if not radar_file.exists():
                # Try without .bin extension
                alt_file = radar_dir / 'velodyne' / frame_id
                if alt_file.exists():
                    radar_file = alt_file
                else:
                    continue
            
            radar_points = np.frombuffer(radar_file.read_bytes(), dtype=np.float32)
            
            # Handle different radar formats
            if radar_points.size % 5 == 0:
                radar_points = radar_points.reshape(-1, 5)  # [x, y, z, intensity, velocity]
            elif radar_points.size % 4 == 0:
                radar_points = radar_points.reshape(-1, 4)  # [x, y, z, intensity] - add velocity
                velocity = np.zeros((radar_points.shape[0], 1))
                radar_points = np.hstack([radar_points, velocity])
            else:
                print(f"Warning: Unexpected radar format for {frame_id}")
                continue
            
            # Apply coordinate transformation if calibration available
            calib_file = radar_dir / 'calib' / f'{frame_id}.txt'
            if calib_file.exists():
                calib = read_calibration(calib_file)
                # Apply transformation if needed (implementation depends on calibration format)
            
            # Apply range cropping (no ground removal or FoV filter for radar)
            original_count = radar_points.shape[0]
            radar_points = crop_points_to_range(radar_points, point_cloud_range)
            
            if radar_points.shape[0] == 0:
                print(f"Warning: No radar points remaining after filtering for {frame_id}")
                continue
            
            # Save processed radar
            output_file = out_path / 'radar' / f'{frame_id}.bin'
            radar_points.astype(np.float32).tofile(output_file)
            processed_radar += 1
            
        except Exception as e:
            print(f"Error processing radar {frame_id}: {e}")
    
    print(f"✅ Processed {processed_radar} radar files")
    
    # Final verification
    print(f"\n🎯 Conversion Summary:")
    print(f"   Input: {vod_root}")
    print(f"   Output: {output_dir}")
    print(f"   LiDAR files: {processed_lidar}")
    print(f"   Radar files: {processed_radar}")
    print(f"   Training IDs: {len(train_ids)}")
    print(f"   Test IDs: {len(test_ids)}")
    
    # Create dataset info file
    info_file = out_path / 'dataset_info.txt'
    with open(info_file, 'w') as f:
        f.write(f"VoD Dataset Conversion Info\n")
        f.write(f"===========================\n")
        f.write(f"Source: {vod_root}\n")
        f.write(f"Radar type: {radar_type}\n")
        f.write(f"Point cloud range: {point_cloud_range}\n")
        f.write(f"LiDAR files processed: {processed_lidar}\n")
        f.write(f"Radar files processed: {processed_radar}\n")
        f.write(f"Training samples: {len(train_ids)}\n")
        f.write(f"Test samples: {len(test_ids)}\n")
    
    print(f"✅ Dataset info saved to: {info_file}")
    print(f"🚀 Ready for MSDNet training!")


def main():
    parser = argparse.ArgumentParser(description="Convert VoD dataset to MSDNet format")
    parser.add_argument('--vod_root', required=True, 
                       help='Path to view_of_delft_PUBLIC directory')
    parser.add_argument('--output_dir', default='data/vod',
                       help='Output directory for MSDNet format')
    parser.add_argument('--radar_type', default='radar_5frames',
                       choices=['radar', 'radar_3frames', 'radar_5frames'],
                       help='Type of radar data to use (radar_5frames recommended)')
    args = parser.parse_args()
    
    print(f"🔄 Converting VoD dataset...")
    print(f"📁 Source: {args.vod_root}")
    print(f"📡 Radar type: {args.radar_type}")
    print(f"🎯 Output: {args.output_dir}")
    print(f"━" * 50)
    
    convert_vod_to_msdnet(args.vod_root, args.output_dir, args.radar_type)


if __name__ == '__main__':
    main()
