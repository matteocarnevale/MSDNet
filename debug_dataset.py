#!/usr/bin/env python3
"""
Debug dataset issues causing high loss values.
Analyzes GT generation, occupancy rates, coordinate scales.
"""

import torch
import numpy as np
from pathlib import Path
from config import MSDNetConfig
from dataset import VoDDataset, collate_fn


def analyze_dataset_samples(data_root, n_samples=10):
    """Analyze real dataset samples in detail."""
    print("Dataset Analysis")
    print("=" * 50)
    
    cfg = MSDNetConfig()
    
    # Load dataset
    ds = VoDDataset(
        data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=False
    )
    
    print(f"Dataset size: {len(ds)} samples")
    print(f"Voxel size: {cfg.voxel.voxel_size}")
    print(f"Point cloud range: {cfg.voxel.point_cloud_range}")
    print()
    
    # Analyze multiple samples
    lidar_counts = []
    radar_counts = []
    occupancy_rates = {4: [], 2: [], 1: []}
    coordinate_ranges = {'lidar': [], 'radar': []}
    
    print(f"Analyzing {n_samples} samples...")
    
    for i in range(min(n_samples, len(ds))):
        try:
            sample = ds[i]
            
            lidar_pc = sample['lidar'].numpy()
            radar_pc = sample['radar'].numpy()
            
            lidar_counts.append(lidar_pc.shape[0])
            radar_counts.append(radar_pc.shape[0])
            
            # Coordinate range analysis
            coordinate_ranges['lidar'].append([
                lidar_pc[:, 0].min(), lidar_pc[:, 0].max(),  # x range
                lidar_pc[:, 1].min(), lidar_pc[:, 1].max(),  # y range
                lidar_pc[:, 2].min(), lidar_pc[:, 2].max()   # z range
            ])
            
            coordinate_ranges['radar'].append([
                radar_pc[:, 0].min(), radar_pc[:, 0].max(),
                radar_pc[:, 1].min(), radar_pc[:, 1].max(), 
                radar_pc[:, 2].min(), radar_pc[:, 2].max()
            ])
            
            # GT occupancy analysis
            for scale in [4, 2, 1]:
                occ = sample['gt_occ'][scale]
                total_voxels = occ.numel()
                occupied_voxels = (occ > 0).sum().item()
                occupancy_rate = occupied_voxels / total_voxels
                occupancy_rates[scale].append(occupancy_rate)
                
                if i == 0:  # Detailed analysis for first sample
                    print(f"Sample 0, Scale {scale}:")
                    print(f"  GT shape: {occ.shape}")
                    print(f"  Occupied: {occupied_voxels}/{total_voxels} ({occupancy_rate:.4f})")
                    print(f"  Offset stats: min={sample['gt_offset'][scale].min():.3f}, max={sample['gt_offset'][scale].max():.3f}")
                    
        except Exception as e:
            print(f"Error analyzing sample {i}: {e}")
    
    # Statistics
    print(f"\nDataset Statistics ({len(lidar_counts)} samples):")
    print(f"LiDAR points: {np.mean(lidar_counts):.0f} ± {np.std(lidar_counts):.0f} (range: {np.min(lidar_counts)}-{np.max(lidar_counts)})")
    print(f"Radar points: {np.mean(radar_counts):.0f} ± {np.std(radar_counts):.0f} (range: {np.min(radar_counts)}-{np.max(radar_counts)})")
    
    # Coordinate range analysis
    if coordinate_ranges['lidar']:
        lidar_coords = np.array(coordinate_ranges['lidar'])
        print(f"\nLiDAR coordinate ranges:")
        print(f"  X: [{lidar_coords[:, 0].min():.2f}, {lidar_coords[:, 1].max():.2f}]")
        print(f"  Y: [{lidar_coords[:, 2].min():.2f}, {lidar_coords[:, 3].max():.2f}]") 
        print(f"  Z: [{lidar_coords[:, 4].min():.2f}, {lidar_coords[:, 5].max():.2f}]")
        
        radar_coords = np.array(coordinate_ranges['radar'])
        print(f"\nRadar coordinate ranges:")
        print(f"  X: [{radar_coords[:, 0].min():.2f}, {radar_coords[:, 1].max():.2f}]")
        print(f"  Y: [{radar_coords[:, 2].min():.2f}, {radar_coords[:, 3].max():.2f}]")
        print(f"  Z: [{radar_coords[:, 4].min():.2f}, {radar_coords[:, 5].max():.2f}]")
    
    # Occupancy rate analysis
    print(f"\nGT Occupancy Rates:")
    for scale in [4, 2, 1]:
        if occupancy_rates[scale]:
            rates = np.array(occupancy_rates[scale])
            print(f"  Scale {scale}: {rates.mean():.5f} ± {rates.std():.5f}")
            
            if rates.mean() < 0.0001:
                print(f"    ✗ TOO SPARSE - voxel size too small for this data")
            elif rates.mean() > 0.1:
                print(f"    ✗ TOO DENSE - voxel size too large")
            else:
                print(f"    ✓ Occupancy rate reasonable")
    
    # Problem diagnosis
    print(f"\nProblem Diagnosis:")
    
    # Check if points are outside expected range
    expected_range = cfg.voxel.point_cloud_range
    if coordinate_ranges['lidar']:
        lidar_coords = np.array(coordinate_ranges['lidar'])
        outside_range = (
            (lidar_coords[:, 0] < expected_range[0]).any() or
            (lidar_coords[:, 1] > expected_range[3]).any() or
            (lidar_coords[:, 2] < expected_range[1]).any() or
            (lidar_coords[:, 3] > expected_range[4]).any() or
            (lidar_coords[:, 4] < expected_range[2]).any() or
            (lidar_coords[:, 5] > expected_range[5]).any()
        )
        
        if outside_range:
            print("1. ✗ Points outside expected voxel range - check preprocessing")
        else:
            print("1. ✓ Points within expected voxel range")
    
    # Check occupancy sparsity
    avg_occupancy = np.mean([np.mean(occupancy_rates[s]) for s in [4, 2, 1]])
    if avg_occupancy < 0.001:
        print("2. ✗ GT too sparse - loss will be dominated by empty voxels")
        print("   Solution: Increase occupancy threshold or adjust voxel size")
    elif avg_occupancy > 0.05:
        print("2. ✗ GT too dense - may cause overfitting")
    else:
        print("2. ✓ GT occupancy seems reasonable")
    
    # Check point density
    avg_lidar = np.mean(lidar_counts)
    avg_radar = np.mean(radar_counts)
    if avg_lidar < 100:
        print("3. ✗ Too few LiDAR points after preprocessing")
    elif avg_radar < 50:
        print("3. ✗ Too few radar points after preprocessing")
    else:
        print("3. ✓ Point densities reasonable")


def test_batch_loading_speed(data_root):
    """Test data loading speed."""
    print(f"\nBatch Loading Speed Test")
    print("-" * 30)
    
    from torch.utils.data import DataLoader
    from config import MSDNetConfig
    import time
    
    cfg = MSDNetConfig()
    
    ds = VoDDataset(
        data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=False
    )
    
    # Test different num_workers
    for num_workers in [0, 1, 4]:
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_fn)
        
        start_time = time.time()
        for i, batch in enumerate(loader):
            if i >= 5:  # Test first 5 batches
                break
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        print(f"Workers {num_workers}: {avg_time:.2f} sec/batch")
    
    print(f"Recommended: Use num_workers=1 for best balance")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()
    
    analyze_dataset_samples(args.data_root, args.n_samples)
    test_batch_loading_speed(args.data_root)


if __name__ == '__main__':
    main()
