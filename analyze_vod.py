#!/usr/bin/env python3
"""
Analyze VoD dataset structure to understand file formats and distributions.

Usage: python analyze_vod.py --vod_root /path/to/view_of_delft_PUBLIC --radar_type radar_5frames
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def analyze_radar_formats(radar_dir, sample_size=100):
    """Analyze radar file formats to understand data structure."""
    velodyne_dir = radar_dir / 'training' / 'velodyne'
    if not velodyne_dir.exists():
        print(f"Radar velodyne directory not found: {velodyne_dir}")
        return
    
    radar_files = list(velodyne_dir.glob('*.bin'))
    print(f"Found {len(radar_files)} radar files")
    
    # Sample files for analysis
    sample_files = radar_files[:min(sample_size, len(radar_files))]
    
    format_stats = defaultdict(int)
    size_distribution = defaultdict(int)
    point_counts = []
    
    print(f"Analyzing {len(sample_files)} sample files...")
    
    for radar_file in tqdm(sample_files):
        try:
            data = np.fromfile(radar_file, dtype=np.float32)
            data_size = data.size
            size_distribution[data_size] += 1
            
            # Determine likely format
            possible_formats = []
            for n_features in [3, 4, 5, 6, 7, 8]:
                if data_size % n_features == 0:
                    n_points = data_size // n_features
                    possible_formats.append((n_features, n_points))
            
            if possible_formats:
                # Choose most likely format (prefer 5 features for radar)
                if any(fmt[0] == 5 for fmt in possible_formats):
                    best_format = next(fmt for fmt in possible_formats if fmt[0] == 5)
                else:
                    best_format = max(possible_formats, key=lambda x: x[1])  # Most points
                
                format_stats[best_format[0]] += 1
                point_counts.append(best_format[1])
            else:
                format_stats['unknown'] += 1
                
        except Exception as e:
            format_stats['error'] += 1
    
    print("\nRadar Format Analysis:")
    print("-" * 40)
    for n_features, count in sorted(format_stats.items()):
        percentage = count / len(sample_files) * 100
        if isinstance(n_features, int):
            print(f"{n_features} features: {count} files ({percentage:.1f}%)")
        else:
            print(f"{n_features}: {count} files ({percentage:.1f}%)")
    
    if point_counts:
        print(f"\nPoint count statistics:")
        print(f"  Mean: {np.mean(point_counts):.1f} points")
        print(f"  Std: {np.std(point_counts):.1f}")
        print(f"  Min: {np.min(point_counts)} points")
        print(f"  Max: {np.max(point_counts)} points")
    
    # Show some file size examples
    print(f"\nFile size distribution (top 10):")
    top_sizes = sorted(size_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    for size, count in top_sizes:
        print(f"  {size} elements: {count} files")


def analyze_missing_files(vod_root, radar_type):
    """Analyze which files exist vs which are in splits."""
    vod_path = Path(vod_root)
    lidar_dir = vod_path / 'lidar' 
    radar_dir = vod_path / radar_type
    
    # Get frame IDs from splits
    imagesets = lidar_dir / 'ImageSets'
    split_frame_ids = set()
    
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = imagesets / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
                split_frame_ids.update(frame_ids)
    
    print(f"Frame IDs in splits: {len(split_frame_ids)}")
    
    # Check which files actually exist
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    radar_velodyne = radar_dir / 'training' / 'velodyne'
    
    existing_lidar = {f.stem for f in lidar_velodyne.glob('*.bin')} if lidar_velodyne.exists() else set()
    existing_radar = {f.stem for f in radar_velodyne.glob('*.bin')} if radar_velodyne.exists() else set()
    
    print(f"Existing LiDAR files: {len(existing_lidar)}")
    print(f"Existing radar files: {len(existing_radar)}")
    
    # Find missing files
    missing_lidar = split_frame_ids - existing_lidar
    missing_radar = split_frame_ids - existing_radar
    
    print(f"Missing LiDAR files: {len(missing_lidar)}")
    print(f"Missing radar files: {len(missing_radar)}")
    
    if len(missing_radar) > 0:
        print(f"Examples of missing radar files:")
        for frame_id in list(missing_radar)[:10]:
            print(f"  {frame_id}.bin")
    
    # Find valid pairs
    valid_pairs = existing_lidar & existing_radar
    print(f"Valid file pairs available: {len(valid_pairs)}")
    print(f"Potential retention rate: {len(valid_pairs)/len(split_frame_ids)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze VoD dataset structure")
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--radar_type', default='radar_5frames')
    args = parser.parse_args()
    
    print("VoD Dataset Structure Analysis")
    print("=" * 60)
    
    vod_path = Path(args.vod_root)
    radar_dir = vod_path / args.radar_type
    
    analyze_missing_files(args.vod_root, args.radar_type)
    print()
    analyze_radar_formats(radar_dir)


if __name__ == '__main__':
    main()
