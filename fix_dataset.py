#!/usr/bin/env python3
"""
Fix VoD dataset by creating splits based on actually existing files.

Usage: python fix_dataset.py --data_root /path/to/dataset
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm


def fix_dataset_splits(data_root):
    """Create new splits based on existing file pairs."""
    data_path = Path(data_root)
    
    print(f"Scanning for valid file pairs in: {data_root}")
    
    # Find all existing LiDAR files
    lidar_dir = data_path / 'lidar'
    radar_dir = data_path / 'radar'
    
    if not lidar_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError("LiDAR or radar directory missing")
    
    # Get all LiDAR file IDs
    lidar_files = list(lidar_dir.glob('*.bin'))
    lidar_ids = {f.stem for f in lidar_files}
    print(f"Found {len(lidar_ids)} LiDAR files")
    
    # Get all radar file IDs  
    radar_files = list(radar_dir.glob('*.bin'))
    radar_ids = {f.stem for f in radar_files}
    print(f"Found {len(radar_ids)} radar files")
    
    # Find intersection (valid pairs)
    valid_ids = sorted(lidar_ids & radar_ids)
    missing_lidar = len(radar_ids - lidar_ids)
    missing_radar = len(lidar_ids - radar_ids)
    
    print(f"Valid pairs: {len(valid_ids)}")
    print(f"Missing LiDAR files: {missing_lidar}")
    print(f"Missing radar files: {missing_radar}")
    
    if len(valid_ids) == 0:
        raise ValueError("No valid file pairs found!")
    
    # Create splits (80% train, 20% test)
    import random
    random.seed(42)  # Reproducible splits
    
    shuffled_ids = valid_ids.copy()
    random.shuffle(shuffled_ids)
    
    split_idx = int(len(shuffled_ids) * 0.8)
    train_ids = shuffled_ids[:split_idx]
    test_ids = shuffled_ids[split_idx:]
    
    # Create val split (10% of train)
    val_split = int(len(train_ids) * 0.1)
    val_ids = train_ids[:val_split]
    train_ids = train_ids[val_split:]
    
    # Save corrected splits
    split_dir = data_path / 'split'
    split_dir.mkdir(exist_ok=True)
    
    # Backup originals if they exist
    backup_dir = split_dir / 'backup'
    backup_dir.mkdir(exist_ok=True)
    
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        original = split_dir / split_name
        if original.exists():
            import shutil
            shutil.copy(original, backup_dir / split_name)
    
    # Write new splits
    with open(split_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(split_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_ids))
    
    with open(split_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_ids))
    
    print(f"\nCorrected splits created:")
    print(f"  train.txt: {len(train_ids)} frames")
    print(f"  test.txt: {len(test_ids)} frames") 
    print(f"  val.txt: {len(val_ids)} frames")
    print(f"  Total: {len(valid_ids)} valid pairs")
    print(f"\nOriginal splits backed up to split/backup/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    args = parser.parse_args()
    
    fix_dataset_splits(args.data_root)
    print("\nDataset fixed! Ready for training.")


if __name__ == '__main__':
    main()
