#!/usr/bin/env python3
"""
SIMPLE converter based on exact user format analysis.
Handles only the 4 formats found: 3feat(32%), 4feat(19%), 5feat(18%), 7feat(31%)
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_vod_simple(vod_root, output_dir, radar_type="radar_5frames"):
    """Simple converter handling your exact format distribution."""
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    print(f"Simple VoD Converter")
    print(f"Target formats: 3(32%), 4(19%), 5(18%), 7(31%)")
    print(f"Source: {vod_path}")
    print(f"Output: {output_dir}")
    
    # Create directories
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)  
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Copy splits exactly
    lidar_imagesets = vod_path / 'lidar' / 'ImageSets'
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        src = lidar_imagesets / split_name
        dst = out_path / 'split' / split_name
        if src.exists():
            shutil.copy(src, dst)
    
    # Get frame IDs
    all_frame_ids = set()
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
                all_frame_ids.update(frame_ids)
    
    all_frame_ids = sorted(all_frame_ids)
    print(f"Processing {len(all_frame_ids)} frame IDs")
    
    # Process LiDAR (minimal)
    print("Processing LiDAR...")
    lidar_dir = vod_path / 'lidar' / 'training' / 'velodyne'
    lidar_success = 0
    
    for frame_id in tqdm(all_frame_ids, desc="LiDAR"):
        src_file = lidar_dir / f'{frame_id}.bin'
        if not src_file.exists():
            continue
            
        try:
            data = np.fromfile(src_file, dtype=np.float32)
            if data.size % 4 == 0:
                points = data.reshape(-1, 4)
                
                # Minimal preprocessing: remove extreme outliers only
                valid_mask = (
                    (points[:, 0] > -50) & (points[:, 0] < 100) &
                    (points[:, 1] > -50) & (points[:, 1] < 50) &
                    (points[:, 2] > -5) & (points[:, 2] < 15)
                )
                points = points[valid_mask]
                
                if points.shape[0] > 10:
                    dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                    points.astype(np.float32).tofile(dst_file)
                    lidar_success += 1
        except:
            pass
    
    # Process radar (format-specific)
    print("Processing radar...")
    radar_dir = vod_path / radar_type / 'training' / 'velodyne'
    radar_success = 0
    format_counts = {3: 0, 4: 0, 5: 0, 7: 0, 'other': 0}
    
    for frame_id in tqdm(all_frame_ids, desc="Radar"):
        src_file = radar_dir / f'{frame_id}.bin'
        if not src_file.exists():
            continue
            
        try:
            data = np.fromfile(src_file, dtype=np.float32)
            points = None
            
            # Process based on exact format analysis
            if data.size % 7 == 0:
                # 31% of files: Take first 5 features
                temp = data.reshape(-1, 7)
                points = temp[:, :5]  # [x, y, z, intensity, velocity]
                format_counts[7] += 1
                
            elif data.size % 5 == 0:
                # 18% of files: Perfect format
                points = data.reshape(-1, 5)
                format_counts[5] += 1
                
            elif data.size % 4 == 0:
                # 19% of files: Add velocity=0
                temp = data.reshape(-1, 4)
                velocity = np.zeros((temp.shape[0], 1), dtype=np.float32)
                points = np.hstack([temp, velocity])
                format_counts[4] += 1
                
            elif data.size % 3 == 0:
                # 32% of files: Add intensity=1, velocity=0
                temp = data.reshape(-1, 3)
                intensity = np.ones((temp.shape[0], 1), dtype=np.float32)
                velocity = np.zeros((temp.shape[0], 1), dtype=np.float32)
                points = np.hstack([temp, intensity, velocity])
                format_counts[3] += 1
            else:
                format_counts['other'] += 1
                continue  # Skip unknown formats
            
            # Save if processed successfully
            if points is not None and points.shape[0] > 0:
                dst_file = out_path / 'radar' / f'{frame_id}.bin'
                points.astype(np.float32).tofile(dst_file)
                radar_success += 1
                
        except Exception as e:
            format_counts['other'] += 1
    
    # Find final valid pairs
    final_pairs = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        if (out_path / 'radar' / f'{frame_id}.bin').exists():
            final_pairs.add(frame_id)
    
    print(f"\nProcessing Results:")
    print(f"LiDAR: {lidar_success}/{len(all_frame_ids)} ({lidar_success/len(all_frame_ids)*100:.1f}%)")
    print(f"Radar: {radar_success}/{len(all_frame_ids)} ({radar_success/len(all_frame_ids)*100:.1f}%)")
    print(f"Final pairs: {len(final_pairs)} ({len(final_pairs)/len(all_frame_ids)*100:.1f}%)")
    
    print(f"\nRadar format processing:")
    for fmt, count in format_counts.items():
        print(f"  {fmt} features: {count} files")
    
    # Update splits with valid pairs
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                original_ids = [line.strip() for line in f if line.strip()]
            
            valid_ids = [fid for fid in original_ids if fid in final_pairs]
            
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_ids))
            
            print(f"{split_name}: {len(original_ids)} -> {len(valid_ids)} ({len(valid_ids)/len(original_ids)*100:.1f}%)")
    
    return len(final_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', default='data/vod')
    parser.add_argument('--radar_type', default='radar_5frames')
    args = parser.parse_args()
    
    result = convert_vod_simple(args.vod_root, args.output_dir, args.radar_type)
    
    if result >= 8000:
        print(f"\nEXCELLENT: {result} pairs (>95% retention)")
    elif result >= 7000:
        print(f"\nGOOD: {result} pairs (>80% retention)")  
    else:
        print(f"\nWARNING: {result} pairs - check for processing errors")


if __name__ == '__main__':
    main()
