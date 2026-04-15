#!/usr/bin/env python3
"""
ULTRA-PERMISSIVE VoD converter for MSDNet.
Preserves maximum radar files with flexible format handling.
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_lidar_file(src_file, ground_height=-1.5, fov_degrees=120.0):
    """Process LiDAR with paper preprocessing."""
    data = np.fromfile(src_file, dtype=np.float32)
    
    if data.size % 4 == 0:
        points = data.reshape(-1, 4)
    elif data.size % 3 == 0:
        points = data.reshape(-1, 3)
        intensity = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points, intensity])
    else:
        return None
    
    if points.shape[0] < 10:
        return None
    
    # Ground removal (elevation map method)
    xyz = points[:, :3]
    if xyz[:, 2].max() - xyz[:, 2].min() > 1.0:  # Has height variation
        # Simple but effective: remove bottom 10% of points by height
        height_threshold = np.percentile(xyz[:, 2], 10)
        points = points[xyz[:, 2] > height_threshold]
    else:
        # Fallback: simple threshold
        points = points[xyz[:, 2] > ground_height]
    
    # FOV crop
    angles = np.arctan2(points[:, 1], points[:, 0])
    half_fov = np.deg2rad(fov_degrees / 2)
    mask = np.abs(angles) <= half_fov
    points = points[mask]
    
    return points if points.shape[0] > 5 else None


def process_radar_file(src_file):
    """Process radar with ULTRA-FLEXIBLE format handling."""
    data = np.fromfile(src_file, dtype=np.float32)
    
    if data.size == 0:
        return None
    
    # Try EVERY possible format to maximize file recovery
    points = None
    
    # Standard formats first
    for n_features in [5, 4, 6, 7, 8, 9, 10]:
        if data.size % n_features == 0:
            temp_points = data.reshape(-1, n_features)
            
            if temp_points.shape[0] > 0:
                if n_features >= 5:
                    points = temp_points[:, :5]  # x,y,z,intensity,velocity
                elif n_features == 4:
                    # Add velocity=0
                    velocity = np.zeros((temp_points.shape[0], 1))
                    points = np.hstack([temp_points, velocity])
                elif n_features == 3:
                    # Add intensity=1, velocity=0
                    extras = np.ones((temp_points.shape[0], 2))
                    extras[:, 1] = 0  # velocity
                    points = np.hstack([temp_points, extras])
                break
    
    # Emergency format recovery for weird formats
    if points is None:
        # Try to interpret as consecutive x,y,z,... values
        if data.size >= 3:
            n_points = data.size // 3
            if n_points > 0:
                xyz = data[:n_points*3].reshape(-1, 3)
                # Add intensity=1, velocity=0
                intensity = np.ones((n_points, 1))
                velocity = np.zeros((n_points, 1))
                points = np.hstack([xyz, intensity, velocity])
    
    # Even more emergency: try interpreting as raw coordinates
    if points is None and data.size >= 15:  # At least 5 points with 3 coords
        # Assume every 3 values are x,y,z
        try:
            n_coords = (data.size // 3) * 3
            xyz = data[:n_coords].reshape(-1, 3)
            intensity = np.ones((xyz.shape[0], 1))
            velocity = np.zeros((xyz.shape[0], 1))
            points = np.hstack([xyz, intensity, velocity])
        except:
            pass
    
    return points if points is not None and points.shape[0] > 0 else None


def convert_vod_ultra_permissive(vod_root, output_dir, radar_type="radar_5frames"):
    """Ultra-permissive conversion to maximize file recovery."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    lidar_dir = vod_path / 'lidar'
    radar_dir = vod_path / radar_type
    
    print(f"Ultra-Permissive VoD Converter")
    print(f"Source: {vod_path}")
    print(f"Radar: {radar_type}")
    print(f"Goal: Maximum file preservation")
    print("-" * 60)
    
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
    
    # Get ALL possible frame IDs from splits
    all_frame_ids = set()
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
                all_frame_ids.update(frame_ids)
    
    all_frame_ids = sorted(all_frame_ids)
    print(f"Total frame IDs from splits: {len(all_frame_ids)}")
    
    # Process ALL frame IDs (don't pre-filter by file existence)
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    radar_velodyne = radar_dir / 'training' / 'velodyne'
    
    # Process LiDAR
    print("Processing ALL LiDAR files...")
    lidar_success = 0
    lidar_errors = 0
    
    for frame_id in tqdm(all_frame_ids, desc="LiDAR"):
        src_file = lidar_velodyne / f'{frame_id}.bin'
        if not src_file.exists():
            lidar_errors += 1
            continue
            
        try:
            processed_points = process_lidar_file(src_file)
            if processed_points is not None:
                dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                processed_points.astype(np.float32).tofile(dst_file)
                lidar_success += 1
            else:
                lidar_errors += 1
        except Exception as e:
            lidar_errors += 1
    
    # Process ALL radar files with ultra-flexible handling
    print("Processing ALL radar files (ultra-permissive)...")
    radar_success = 0
    radar_errors = 0
    radar_format_stats = {}
    
    for frame_id in tqdm(all_frame_ids, desc="Radar"):
        src_file = radar_velodyne / f'{frame_id}.bin'
        if not src_file.exists():
            radar_errors += 1
            continue
            
        try:
            data = np.fromfile(src_file, dtype=np.float32)
            
            # Track format statistics
            data_size = data.size
            if data_size not in radar_format_stats:
                radar_format_stats[data_size] = 0
            radar_format_stats[data_size] += 1
            
            processed_points = process_radar_file(src_file)
            
            if processed_points is not None:
                dst_file = out_path / 'radar' / f'{frame_id}.bin'
                processed_points.astype(np.float32).tofile(dst_file)
                radar_success += 1
            else:
                radar_errors += 1
                
        except Exception as e:
            radar_errors += 1
    
    print(f"\nRadar format statistics:")
    for size, count in sorted(radar_format_stats.items()):
        features = "unknown"
        if size % 5 == 0:
            features = f"{size//5} points × 5 features"
        elif size % 4 == 0:
            features = f"{size//4} points × 4 features"
        print(f"  Size {size}: {count} files ({features})")
    
    # Find final valid pairs
    final_pairs = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        if (out_path / 'radar' / f'{frame_id}.bin').exists():
            final_pairs.add(frame_id)
    
    print(f"\nProcessing summary:")
    print(f"LiDAR: {lidar_success}/{len(all_frame_ids)} ({lidar_success/len(all_frame_ids)*100:.1f}%)")
    print(f"Radar: {radar_success}/{len(all_frame_ids)} ({radar_success/len(all_frame_ids)*100:.1f}%)")
    print(f"Final pairs: {len(final_pairs)}")
    print(f"Radar errors: {radar_errors}")
    
    # Update splits
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                original_ids = [line.strip() for line in f if line.strip()]
            
            valid_ids = [fid for fid in original_ids if fid in final_pairs]
            
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_ids))
            
            retention = len(valid_ids) / len(original_ids) * 100
            print(f"Updated {split_name}: {len(original_ids)} -> {len(valid_ids)} ({retention:.1f}%)")
    
    return len(final_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', default='data/vod')
    parser.add_argument('--radar_type', default='radar_5frames',
                       choices=['radar', 'radar_3frames', 'radar_5frames'])
    
    args = parser.parse_args()
    
    print("ULTRA-PERMISSIVE VoD Converter")
    print("Goal: Maximum file preservation")
    print()
    
    success_count = convert_vod_ultra_permissive(args.vod_root, args.output_dir, args.radar_type)
    
    print(f"\nFinal result: {success_count} valid pairs")
    if success_count > 5000:
        print("EXCELLENT: High file retention achieved!")
    elif success_count > 3000:
        print("GOOD: Reasonable file retention")
    else:
        print("WARNING: Low file retention - check radar data format")


if __name__ == '__main__':
    main()
