#!/usr/bin/env python3
"""
Paper-compliant VoD converter following exact MSDNet preprocessing (Section IV-B).

"Our data preprocessing follows R2LDM [15], which includes removing ground points 
from the LiDAR data and cropping the LiDAR point cloud to match the Field of View 
(FOV) of the 4D radar."

Usage: python convert_vod_paper_compliant.py --vod_root /path/to/view_of_delft_PUBLIC --output_dir data/vod
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def remove_ground_points_lidar(points, ground_height=-1.5):
    """Remove ground points from LiDAR (paper preprocessing)."""
    return points[points[:, 2] > ground_height]


def crop_lidar_to_radar_fov(points, fov_degrees=120.0):
    """Crop LiDAR to match radar FOV (paper preprocessing)."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    half_fov = np.deg2rad(fov_degrees / 2)
    mask = np.abs(angles) <= half_fov
    return points[mask]


def convert_vod_paper_compliant(vod_root, output_dir, radar_type="radar_5frames"):
    """Convert following exact paper preprocessing."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    lidar_dir = vod_path / 'lidar'
    radar_dir = vod_path / radar_type
    
    print(f"VoD Paper-Compliant Converter")
    print(f"Source: {vod_path}")
    print(f"Radar type: {radar_type}")
    print(f"Output: {output_dir}")
    print(f"Preprocessing: LiDAR only (ground removal + FOV crop)")
    
    # Create directories
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Copy splits as-is
    imagesets = lidar_dir / 'ImageSets'
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        src = imagesets / split_name
        dst = out_path / 'split' / split_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied split: {split_name}")
    
    # Find all files that exist
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    radar_velodyne = radar_dir / 'training' / 'velodyne'
    
    lidar_files = {f.stem for f in lidar_velodyne.glob('*.bin')} if lidar_velodyne.exists() else set()
    radar_files = {f.stem for f in radar_velodyne.glob('*.bin')} if radar_velodyne.exists() else set()
    
    valid_pairs = sorted(lidar_files & radar_files)
    print(f"Valid file pairs found: {len(valid_pairs)}")
    
    # Process LiDAR with paper preprocessing
    print("Processing LiDAR (ground removal + FOV crop)...")
    lidar_processed = 0
    
    for frame_id in tqdm(valid_pairs, desc="LiDAR"):
        try:
            src_file = lidar_velodyne / f'{frame_id}.bin'
            data = np.fromfile(src_file, dtype=np.float32)
            
            # Standard LiDAR format: (N, 4)
            if data.size % 4 == 0:
                points = data.reshape(-1, 4)
            else:
                print(f"Skipping {frame_id}: unexpected LiDAR format")
                continue
            
            # Apply paper preprocessing
            original_count = points.shape[0]
            points = remove_ground_points_lidar(points)  # Paper: remove ground
            points = crop_lidar_to_radar_fov(points)     # Paper: crop to radar FOV
            
            # Keep all points that pass basic filters (no range cropping yet)
            if points.shape[0] > 10:  # Keep files with reasonable point count
                dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                lidar_processed += 1
            
        except Exception as e:
            print(f"Error processing LiDAR {frame_id}: {e}")
    
    # Process radar with NO preprocessing (paper doesn't mention radar preprocessing)
    print("Processing radar (no preprocessing)...")
    radar_processed = 0
    
    for frame_id in tqdm(valid_pairs, desc="Radar"):
        try:
            src_file = radar_velodyne / f'{frame_id}.bin'
            data = np.fromfile(src_file, dtype=np.float32)
            
            # Handle radar formats flexibly
            if data.size % 5 == 0:
                points = data.reshape(-1, 5)
            elif data.size % 4 == 0:
                points = data.reshape(-1, 4)
                velocity = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, velocity])
            else:
                print(f"Skipping {frame_id}: unexpected radar format")
                continue
            
            # NO preprocessing for radar - keep all points as-is
            if points.shape[0] > 5:  # Minimal sanity check
                dst_file = out_path / 'radar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                radar_processed += 1
            
        except Exception as e:
            print(f"Error processing radar {frame_id}: {e}")
    
    # Update splits with actually processed files
    final_pairs = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        radar_file = out_path / 'radar' / f'{frame_id}.bin'
        if radar_file.exists():
            final_pairs.add(frame_id)
    
    print(f"Final valid pairs: {len(final_pairs)}")
    
    # Update split files
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                original_ids = [line.strip() for line in f if line.strip()]
            
            valid_ids = [fid for fid in original_ids if fid in final_pairs]
            
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_ids))
            
            print(f"Updated {split_name}: {len(original_ids)} -> {len(valid_ids)} frames")
    
    print(f"Conversion complete - paper-compliant preprocessing applied")
    return len(final_pairs)


def main():
    parser = argparse.ArgumentParser(description="Paper-compliant VoD converter")
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', default='data/vod')
    parser.add_argument('--radar_type', default='radar_5frames',
                       choices=['radar', 'radar_3frames', 'radar_5frames'])
    args = parser.parse_args()
    
    convert_vod_paper_compliant(args.vod_root, args.output_dir, args.radar_type)


if __name__ == '__main__':
    main()
