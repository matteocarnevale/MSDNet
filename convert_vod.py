#!/usr/bin/env python3
"""
VoD Dataset Converter for MSDNet

Converts View of Delft dataset to MSDNet format with paper-compliant preprocessing.
Supports all 3 radar variants and includes verification.

Usage: python convert_vod.py --vod_root /path/to/view_of_delft_PUBLIC --output_dir data/vod
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def remove_ground_elevation_map(points, grid_size=0.5, height_threshold=0.3, ground_height=-1.5):
    """Standard elevation map ground removal."""
    if points.shape[0] == 0:
        return points
        
    xyz = points[:, :3]
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    
    if x_max - x_min < 0.1 or y_max - y_min < 0.1:
        return points[xyz[:, 2] > ground_height]
    
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    ground_heights = np.full((x_bins, y_bins), np.inf)
    x_indices = np.clip(((xyz[:, 0] - x_min) / grid_size).astype(int), 0, x_bins - 1)
    y_indices = np.clip(((xyz[:, 1] - y_min) / grid_size).astype(int), 0, y_bins - 1)
    
    for i in range(len(xyz)):
        x_idx, y_idx = x_indices[i], y_indices[i]
        ground_heights[x_idx, y_idx] = min(ground_heights[x_idx, y_idx], xyz[i, 2])
    
    non_ground_mask = np.zeros(len(xyz), dtype=bool)
    for i in range(len(xyz)):
        x_idx, y_idx = x_indices[i], y_indices[i]
        ground_h = ground_heights[x_idx, y_idx]
        
        if ground_h == np.inf:
            non_ground_mask[i] = xyz[i, 2] > ground_height
        else:
            non_ground_mask[i] = (xyz[i, 2] - ground_h) > height_threshold
    
    return points[non_ground_mask]


def crop_to_radar_fov(points, fov_degrees=120.0):
    """Crop LiDAR to radar field of view."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    half_fov = np.deg2rad(fov_degrees / 2)
    mask = np.abs(angles) <= half_fov
    return points[mask]


def convert_vod_final(vod_root, output_dir, radar_type="radar_5frames"):
    """Final VoD converter with paper-compliant preprocessing."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    # Verify input structure
    lidar_dir = vod_path / 'lidar'
    radar_dir = vod_path / radar_type
    
    if not lidar_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError(f"Required directories not found: {lidar_dir}, {radar_dir}")
    
    print(f"VoD to MSDNet Converter")
    print(f"Source: {vod_path}")
    print(f"Radar: {radar_type}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Create output structure
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Copy dataset splits
    imagesets = lidar_dir / 'ImageSets'
    split_info = {}
    
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        src = imagesets / split_name
        dst = out_path / 'split' / split_name
        if src.exists():
            shutil.copy(src, dst)
            with open(dst, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
            split_info[split_name] = frame_ids
            print(f"Split {split_name}: {len(frame_ids)} frames")
    
    # Find all existing files
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    radar_velodyne = radar_dir / 'training' / 'velodyne'
    
    lidar_files = {f.stem for f in lidar_velodyne.glob('*.bin')} if lidar_velodyne.exists() else set()
    radar_files = {f.stem for f in radar_velodyne.glob('*.bin')} if radar_velodyne.exists() else set()
    
    valid_pairs = sorted(lidar_files & radar_files)
    print(f"File pairs available: {len(valid_pairs)}")
    
    # Process LiDAR (paper preprocessing)
    print("Processing LiDAR (ground removal + FOV crop)...")
    lidar_success = 0
    
    for frame_id in tqdm(valid_pairs, desc="LiDAR"):
        try:
            src_file = lidar_velodyne / f'{frame_id}.bin'
            data = np.fromfile(src_file, dtype=np.float32)
            
            if data.size % 4 == 0:
                points = data.reshape(-1, 4)
            elif data.size % 3 == 0:
                points = data.reshape(-1, 3)
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, intensity])
            else:
                continue
            
            if points.shape[0] < 10:
                continue
            
            # Paper preprocessing: ground removal + FOV crop
            points = remove_ground_elevation_map(points)
            points = crop_to_radar_fov(points)
            
            if points.shape[0] > 10:
                dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                lidar_success += 1
                
        except Exception as e:
            print(f"Error LiDAR {frame_id}: {e}")
    
    # Process radar (no preprocessing)
    print("Processing radar (no preprocessing)...")
    radar_success = 0
    
    for frame_id in tqdm(valid_pairs, desc="Radar"):
        try:
            src_file = radar_velodyne / f'{frame_id}.bin'
            data = np.fromfile(src_file, dtype=np.float32)
            
            if data.size % 5 == 0:
                points = data.reshape(-1, 5)
            elif data.size % 4 == 0:
                points = data.reshape(-1, 4)
                velocity = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, velocity])
            else:
                continue
            
            if points.shape[0] > 5:
                dst_file = out_path / 'radar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                radar_success += 1
                
        except Exception as e:
            print(f"Error radar {frame_id}: {e}")
    
    # Update splits with processed files
    final_pairs = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        if (out_path / 'radar' / f'{frame_id}.bin').exists():
            final_pairs.add(frame_id)
    
    print(f"\nProcessing results:")
    print(f"LiDAR processed: {lidar_success}")
    print(f"Radar processed: {radar_success}")
    print(f"Final pairs: {len(final_pairs)}")
    
    # Update splits
    for split_name, original_ids in split_info.items():
        valid_ids = [fid for fid in original_ids if fid in final_pairs]
        
        split_file = out_path / 'split' / split_name
        with open(split_file, 'w') as f:
            f.write('\n'.join(valid_ids))
        
        print(f"Updated {split_name}: {len(original_ids)} -> {len(valid_ids)} frames")
    
    return len(final_pairs)


def main():
    parser = argparse.ArgumentParser(description="Convert VoD dataset to MSDNet format")
    parser.add_argument('--vod_root', required=True, help='Path to view_of_delft_PUBLIC')
    parser.add_argument('--output_dir', default='data/vod', help='Output directory')
    parser.add_argument('--radar_type', default='radar_5frames',
                       choices=['radar', 'radar_3frames', 'radar_5frames'],
                       help='Radar variant (radar_5frames recommended)')
    
    args = parser.parse_args()
    
    print("VoD Dataset Converter for MSDNet")
    print("Radar variants:")
    print("  radar: Single-frame (sparse, ~50-200 points)")
    print("  radar_3frames: 3-frame accumulation (~150-600 points)")
    print("  radar_5frames: 5-frame accumulation (~250-1000 points) [RECOMMENDED]")
    print()
    
    success_count = convert_vod_final(args.vod_root, args.output_dir, args.radar_type)
    
    if success_count > 0:
        print(f"\nConversion complete: {success_count} valid pairs")
        print("Ready for MSDNet training!")
    else:
        print("\nERROR: No valid pairs found")


if __name__ == '__main__':
    main()
