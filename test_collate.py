#!/usr/bin/env python3
"""Quick test of collate_fn batch dimension handling."""

import torch
import numpy as np
from dataset import collate_fn


def test_collate_fn():
    """Test that collate_fn creates correct batch dimensions."""
    
    # Create fake samples like VoDDataset would produce
    batch_size = 2
    fake_batch = []
    
    for i in range(batch_size):
        # Create GT for single sample (what dataset produces)
        gt_occ = {}
        gt_offset = {}
        for scale in [4, 2, 1]:
            if scale == 4:
                dims = (1, 10, 40, 40)  # Single sample
            elif scale == 2:
                dims = (1, 20, 80, 80)
            else:
                dims = (1, 40, 160, 160)
            
            gt_occ[scale] = torch.zeros(dims[0], dims[1], dims[2], dims[3])
            gt_offset[scale] = torch.zeros(3, dims[1], dims[2], dims[3])
        
        fake_batch.append({
            "lidar": torch.randn(100 + i * 50, 4),
            "radar": torch.randn(50 + i * 20, 5),
            "gt_occ": gt_occ,
            "gt_offset": gt_offset,
            "frame_id": f"test_{i:06d}"
        })
    
    # Apply collate_fn
    collated = collate_fn(fake_batch)
    
    print(f"Batch size: {batch_size}")
    print("After collate_fn:")
    
    for scale in [4, 2, 1]:
        occ_shape = collated["gt_occ"][scale].shape
        off_shape = collated["gt_offset"][scale].shape
        print(f"  Scale {scale}: occ={occ_shape}, offset={off_shape}")
    
    # Check if first dimension is batch_size
    for scale in [4, 2, 1]:
        occ_batch_dim = collated["gt_occ"][scale].shape[0]
        off_batch_dim = collated["gt_offset"][scale].shape[0]
        
        if occ_batch_dim != batch_size or off_batch_dim != batch_size:
            print(f"ERROR: Scale {scale} doesn't have correct batch dimension")
            return False
    
    print("SUCCESS: collate_fn creates correct batch dimensions")
    return True


if __name__ == '__main__':
    test_collate_fn()
