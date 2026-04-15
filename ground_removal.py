#!/usr/bin/env python3
"""
Standard ground removal methods for LiDAR preprocessing.
"""

import numpy as np


def ground_removal_elevation_map(points, grid_size=0.2, height_threshold=0.5):
    """Elevation map method (common in autonomous driving)."""
    if points.shape[0] == 0:
        return points
        
    xyz = points[:, :3]
    
    # Create 2D elevation grid
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    
    if x_max - x_min < 0.1 or y_max - y_min < 0.1:
        # Too small area, use simple threshold
        return points[xyz[:, 2] > -2.0]
    
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    # Find minimum height per cell
    ground_heights = np.full((x_bins, y_bins), np.inf)
    x_indices = np.clip(((xyz[:, 0] - x_min) / grid_size).astype(int), 0, x_bins - 1)
    y_indices = np.clip(((xyz[:, 1] - y_min) / grid_size).astype(int), 0, y_bins - 1)
    
    for i in range(len(xyz)):
        x_idx, y_idx = x_indices[i], y_indices[i]
        ground_heights[x_idx, y_idx] = min(ground_heights[x_idx, y_idx], xyz[i, 2])
    
    # Classify points
    non_ground_mask = np.zeros(len(xyz), dtype=bool)
    for i in range(len(xyz)):
        x_idx, y_idx = x_indices[i], y_indices[i]
        ground_h = ground_heights[x_idx, y_idx]
        
        if ground_h == np.inf:
            non_ground_mask[i] = xyz[i, 2] > -1.5
        else:
            non_ground_mask[i] = (xyz[i, 2] - ground_h) > height_threshold
    
    return points[non_ground_mask]


def ground_removal_simple(points, threshold=-1.5):
    """Simple threshold (current method).""" 
    return points[points[:, 2] > threshold]
