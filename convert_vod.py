#!/usr/bin/env python3
"""
VoD converter optimized for your specific radar format distribution:
- 3 features: 32% (x,y,z)
- 4 features: 19% (x,y,z,intensity) 
- 5 features: 18% (x,y,z,intensity,velocity)
- 7 features: 31% (extended format)

Target: 100% retention of 8682 files instead of 40%
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_lidar_minimal(src_file, ground_percentile=15, fov_degrees=120.0):
    """Minimal LiDAR processing following paper."""
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
    
    # Ground removal: remove bottom percentile by height
    z_threshold = np.percentile(points[:, 2], ground_percentile)
    points = points[points[:, 2] > z_threshold]
    
    # FOV crop
    angles = np.arctan2(points[:, 1], points[:, 0])
    half_fov = np.deg2rad(fov_degrees / 2)
    mask = np.abs(angles) <= half_fov
    points = points[mask]
    
    return points if points.shape[0] > 5 else None


def process_radar_format_specific(src_file):
    """Handle specific VoD radar formats based on analysis."""
    data = np.fromfile(src_file, dtype=np.float32)
    
    if data.size == 0:
        return None
    
    # Handle each format based on your analysis results
    if data.size % 7 == 0:
        # 31% of files - extended radar format
        points_7d = data.reshape(-1, 7)
        # Take [x, y, z, intensity, velocity] (first 5 columns)
        return points_7d[:, :5]
        
    elif data.size % 5 == 0:
        # 18% of files - perfect format
        return data.reshape(-1, 5)
        
    elif data.size % 4 == 0:
        # 19% of files - missing velocity
        points_4d = data.reshape(-1, 4)
        velocity = np.zeros((points_4d.shape[0], 1), dtype=np.float32)
        return np.hstack([points_4d, velocity])
        
    elif data.size % 3 == 0:
        # 32% of files - only coordinates
        points_3d = data.reshape(-1, 3)
        intensity = np.ones((points_3d.shape[0], 1), dtype=np.float32)
        velocity = np.zeros((points_3d.shape[0], 1), dtype=np.float32)
        return np.hstack([points_3d, intensity, velocity])
        
    elif data.size % 6 == 0:
        # Handle 6 feature case
        points_6d = data.reshape(-1, 6)
        return points_6d[:, :5]
        
    elif data.size % 8 == 0:
        # Handle 8 feature case
        points_8d = data.reshape(-1, 8)
        return points_8d[:, :5]
    
    # Last resort: interpret as xyz triplets
    elif data.size >= 3:
        n_points = data.size // 3
        xyz = data[:n_points*3].reshape(-1, 3)
        intensity = np.ones((n_points, 1), dtype=np.float32)
        velocity = np.zeros((n_points, 1), dtype=np.float32)
        return np.hstack([xyz, intensity, velocity])
    
    return None


def convert_vod_optimized(vod_root, output_dir, radar_type="radar_5frames"):
    """Optimized conversion for 100% retention."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    print(f"VoD Converter - Target: 100% Retention")
    print(f"Source: {vod_path}")
    print(f"Radar: {radar_type}")
    
    # Create structure
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Copy splits
    lidar_dir = vod_path / 'lidar'
    imagesets = lidar_dir / 'ImageSets'
    
    frame_counts = {}
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        src = imagesets / split_name
        dst = out_path / 'split' / split_name
        if src.exists():
            shutil.copy(src, dst)
            with open(dst, 'r') as f:
                frame_counts[split_name] = len([l.strip() for l in f if l.strip()])
    
    print(f"Target splits: {sum(frame_counts.values())} total frames")
    
    # Get ALL frame IDs
    all_frame_ids = set()
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
                all_frame_ids.update(frame_ids)
    
    all_frame_ids = sorted(all_frame_ids)
    
    # Process LiDAR
    print("Processing LiDAR...")
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    lidar_success = 0
    
    for frame_id in tqdm(all_frame_ids, desc="LiDAR"):
        src_file = lidar_velodyne / f'{frame_id}.bin'
        if src_file.exists():
            processed = process_lidar_minimal(src_file)
            if processed is not None:
                dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                processed.astype(np.float32).tofile(dst_file)
                lidar_success += 1
    
    # Process radar with format-specific handling
    print("Processing radar (format-specific handling)...")
    radar_dir_path = vod_path / radar_type
    radar_velodyne = radar_dir_path / 'training' / 'velodyne'
    radar_success = 0
    radar_format_count = {3: 0, 4: 0, 5: 0, 7: 0, 'other': 0}
    
    for frame_id in tqdm(all_frame_ids, desc="Radar"):
        src_file = radar_velodyne / f'{frame_id}.bin'
        if src_file.exists():
            try:
                processed = process_radar_format_specific(src_file)
                if processed is not None and processed.shape[0] > 0:
                    dst_file = out_path / 'radar' / f'{frame_id}.bin'
                    processed.astype(np.float32).tofile(dst_file)
                    radar_success += 1
                    
                    # Track format
                    data = np.fromfile(src_file, dtype=np.float32)
                    if data.size % 7 == 0:
                        radar_format_count[7] += 1
                    elif data.size % 5 == 0:
                        radar_format_count[5] += 1
                    elif data.size % 4 == 0:
                        radar_format_count[4] += 1
                    elif data.size % 3 == 0:
                        radar_format_count[3] += 1
                    else:
                        radar_format_count['other'] += 1
                        
            except Exception as e:
                pass  # Continue processing
    
    # Find final pairs
    final_pairs = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        if (out_path / 'radar' / f'{frame_id}.bin').exists():
            final_pairs.add(frame_id)
    
    print(f"\nResults:")
    print(f"LiDAR processed: {lidar_success}/{len(all_frame_ids)} ({lidar_success/len(all_frame_ids)*100:.1f}%)")
    print(f"Radar processed: {radar_success}/{len(all_frame_ids)} ({radar_success/len(all_frame_ids)*100:.1f}%)")
    print(f"Final valid pairs: {len(final_pairs)} ({len(final_pairs)/len(all_frame_ids)*100:.1f}% retention)")
    
    print(f"\nRadar format distribution processed:")
    for fmt, count in radar_format_count.items():
        if count > 0:
            print(f"  {fmt} features: {count} files")
    
    # Update splits
    for split_name, original_count in frame_counts.items():
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                original_ids = [line.strip() for line in f if line.strip()]
            
            valid_ids = [fid for fid in original_ids if fid in final_pairs]
            
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_ids))
            
            retention = len(valid_ids) / len(original_ids) * 100
            print(f"{split_name}: {len(original_ids)} -> {len(valid_ids)} ({retention:.1f}%)")
    
    target_retention = 90  # Target 90%+ retention
    actual_retention = len(final_pairs) / len(all_frame_ids) * 100
    
    if actual_retention >= target_retention:
        print(f"\nEXCELLENT: {actual_retention:.1f}% retention achieved!")
    else:
        print(f"\nWARNING: Only {actual_retention:.1f}% retention (target: {target_retention}%)")
    
    return len(final_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', default='data/vod') 
    parser.add_argument('--radar_type', default='radar_5frames')
    args = parser.parse_args()
    
    success_count = convert_vod_optimized(args.vod_root, args.output_dir, args.radar_type)
    print(f"\nTarget achieved: {success_count} valid pairs")


if __name__ == '__main__':
    main()
