# View of Delft (VoD) Dataset Management Guide

This guide explains how to understand, process, and use the **actual VoD dataset structure** for MSDNet training.

## Ÿ“ Actual VoD Dataset Structure

Based on the provided dataset structure, VoD follows a **KITTI-style organization**:

```
view_of_delft_PUBLIC/
”œâ”€â”€ lidar/
”‚   â”œâ”€â”€ ImageSets/                  # Pre-defined dataset splits
”‚   â”‚   â”œâ”€â”€ full.txt               # All frame IDs
”‚   â”‚   â”œâ”€â”€ train.txt              # Training frame IDs  
”‚   â”‚   â”œâ”€â”€ test.txt               # Test frame IDs
”‚   â”‚   â”œâ”€â”€ train_val.txt          # Training + validation IDs
”‚   â”‚   â””â”€â”€ val.txt                # Validation frame IDs
”‚   â”œâ”€â”€ testing/                   # Test set data
”‚   â””â”€â”€ training/                  # Training set data
”‚       â”œâ”€â”€ calib/                 # Calibration files (*.txt)
”‚       â”œâ”€â”€ image_2/               # Camera images (*.png)
”‚       â”œâ”€â”€ label_2/               # 3D object labels (*.txt)  
”‚       â”œâ”€â”€ pose/                  # Vehicle poses (*.txt)
”‚       â””â”€â”€ velodyne/              # LiDAR point clouds (*.bin)
”‚
”œâ”€â”€ radar/                         # Single-frame radar
”œâ”€â”€ radar_3frames/                 # 3-frame radar accumulation
””â”€â”€ radar_5frames/                 # 5-frame radar accumulation [RECOMMENDED]
    ”œâ”€â”€ ImageSets/                 # Same splits as LiDAR
    ”œâ”€â”€ testing/
    ””â”€â”€ training/
        ”œâ”€â”€ calib/                 # Radar-LiDAR calibration
        ”œâ”€â”€ image_2/               # Synchronized camera images
        ”œâ”€â”€ label_2/               # 3D object labels
        ”œâ”€â”€ pose/                  # Vehicle poses
        ””â”€â”€ velodyne/              # 4D radar point clouds (*.bin)
```

##  Key Insights

### 1. **Radar Variants**
- **`radar/`**: Single-frame 4D radar (sparse)
- **`radar_3frames/`**: 3-frame accumulation (denser)
- **`radar_5frames/`**: 5-frame accumulation (densest)  **RECOMMENDED for MSDNet**

### 2. **Data Synchronization**
- All sensors are **temporally synchronized**
- Same frame IDs across LiDAR, radar, camera
- Calibration files provide spatial alignment

### 3. **Pre-defined Splits**
- Splits already defined in `ImageSets/*.txt`
- **Use existing splits** to compare with literature
- Paper uses sequences 03, 04, 22 for testing

## Ÿ”„ Data Format Details

### LiDAR Points (`lidar/training/velodyne/*.bin`)
```python
# Binary format: (N, 4) float32
# [x, y, z, intensity]
lidar_points = np.frombuffer(file_data, dtype=np.float32).reshape(-1, 4)
```

### 4D Radar Points (`radar_5frames/training/velodyne/*.bin`)
```python  
# Binary format: (N, 5) float32
# [x, y, z, intensity, velocity] 
radar_points = np.frombuffer(file_data, dtype=np.float32).reshape(-1, 5)
```

### Calibration (`*/training/calib/*.txt`)
```
# Example calibration file content:
P0: 7.215377e+02 0.000000e+00 6.095593e+02 ...  # Camera projection
P1: 7.215377e+02 0.000000e+00 6.095593e+02 ...  # Camera projection
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 ... # LiDAR to camera transform
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 ... # IMU to LiDAR transform
```

##  Dataset Statistics

| Split | Purpose | Recommended Use |
|-------|---------|-----------------|
| `train.txt` | Training | MSDNet teacher + student training |
| `test.txt` | Testing | Final evaluation (sequences 03, 04, 22) |
| `val.txt` | Validation | Hyperparameter tuning |
| `train_val.txt` | Train+Val | Extended training set |

## Ÿ› ï¸ Preprocessing Pipeline

### 1. **LiDAR Preprocessing** (Paper Section IV-B)
```python
# Ground removal (z > -1.5m)
lidar_points = lidar_points[lidar_points[:, 2] > -1.5]

# FoV cropping (120Â° horizontal to match radar)
angles = np.arctan2(lidar_points[:, 1], lidar_points[:, 0])
fov_mask = np.abs(angles) <= np.deg2rad(60)  # Â±60Â° = 120Â° total
lidar_points = lidar_points[fov_mask]

# Range cropping [0,32] Ã— [-16,16] Ã— [-2,4] meters
range_mask = (
    (lidar_points[:, 0] >= 0) & (lidar_points[:, 0] < 32) &
    (lidar_points[:, 1] >= -16) & (lidar_points[:, 1] < 16) &
    (lidar_points[:, 2] >= -2) & (lidar_points[:, 2] < 4)
)
lidar_points = lidar_points[range_mask]
```

### 2. **4D Radar Preprocessing**
```python
# Only range cropping (no ground removal, no FoV filter)
range_mask = (
    (radar_points[:, 0] >= 0) & (radar_points[:, 0] < 32) &
    (radar_points[:, 1] >= -16) & (radar_points[:, 1] < 16) &
    (radar_points[:, 2] >= -2) & (radar_points[:, 2] < 4)
)
radar_points = radar_points[range_mask]
```

### 3. **Coordinate Alignment**
```python
def apply_calibration(points, calib_dict):
    """Apply calibration transformation if needed."""
    # Extract transformation matrix from calibration
    # Apply to align radar and LiDAR coordinate frames
    return transformed_points
```

##  Converting to MSDNet Format

Use the provided conversion script:

```bash
# Convert VoD to MSDNet format (recommended: radar_5frames)
python convert_vod_real.py \
    --vod_root /path/to/view_of_delft_PUBLIC \
    --output_dir data/vod \
    --radar_type radar_5frames

# Output structure will be:
data/vod/
”œâ”€â”€ lidar/          # Processed LiDAR (N,4) float32
”œâ”€â”€ radar/          # Processed radar (N,5) float32  
””â”€â”€ split/
    ”œâ”€â”€ train.txt
    ”œâ”€â”€ test.txt
    ””â”€â”€ val.txt
```

## Ÿ“‹ Verification Steps

### 1. **Check Data Integrity**
```python
import numpy as np
from pathlib import Path

def verify_conversion(data_dir):
    data_path = Path(data_dir)
    
    # Check splits exist
    splits = ['train.txt', 'test.txt']
    for split in splits:
        assert (data_path / 'split' / split).exists()
    
    # Check sample files
    with open(data_path / 'split' / 'train.txt') as f:
        frame_id = f.readline().strip()
    
    lidar_file = data_path / 'lidar' / f'{frame_id}.bin'
    radar_file = data_path / 'radar' / f'{frame_id}.bin'
    
    assert lidar_file.exists(), f"LiDAR file missing: {lidar_file}"
    assert radar_file.exists(), f"Radar file missing: {radar_file}"
    
    # Check data shapes
    lidar_data = np.frombuffer(lidar_file.read_bytes(), dtype=np.float32)
    radar_data = np.frombuffer(radar_file.read_bytes(), dtype=np.float32)
    
    assert lidar_data.size % 4 == 0, "LiDAR should have 4 features"
    assert radar_data.size % 5 == 0, "Radar should have 5 features"
    
    lidar_points = lidar_data.reshape(-1, 4)
    radar_points = radar_data.reshape(-1, 5)
    
    print(f" Frame {frame_id}:")
    print(f"  LiDAR: {lidar_points.shape[0]} points, shape {lidar_points.shape}")  
    print(f"  Radar: {radar_points.shape[0]} points, shape {radar_points.shape}")

# Verify conversion
verify_conversion('data/vod')
```

### 2. **Data Statistics**
```python
def analyze_dataset(data_dir):
    """Analyze converted dataset statistics."""
    import matplotlib.pyplot as plt
    
    data_path = Path(data_dir)
    
    with open(data_path / 'split' / 'train.txt') as f:
        train_ids = [line.strip() for line in f]
    
    lidar_counts = []
    radar_counts = []
    
    for frame_id in train_ids[:100]:  # Sample 100 frames
        lidar_file = data_path / 'lidar' / f'{frame_id}.bin'
        radar_file = data_path / 'radar' / f'{frame_id}.bin'
        
        if lidar_file.exists() and radar_file.exists():
            lidar_data = np.frombuffer(lidar_file.read_bytes(), dtype=np.float32)
            radar_data = np.frombuffer(radar_file.read_bytes(), dtype=np.float32)
            
            lidar_counts.append(lidar_data.size // 4)
            radar_counts.append(radar_data.size // 5)
    
    print(f"Dataset Statistics (n={len(lidar_counts)}):")
    print(f"LiDAR points: {np.mean(lidar_counts):.0f} Â± {np.std(lidar_counts):.0f}")
    print(f"Radar points: {np.mean(radar_counts):.0f} Â± {np.std(radar_counts):.0f}")
    print(f"Density ratio (LiDAR/Radar): {np.mean(lidar_counts)/np.mean(radar_counts):.1f}x")
```

##  Training with Converted Data

Once converted, use the standard MSDNet training pipeline:

```bash
# Stage 0: Train teacher
python train_teacher.py \
    --data_root data/vod \
    --epochs 60 \
    --val_interval 5

# Stage 1+2: Train student  
python train_student.py \
    --data_root data/vod \
    --teacher_ckpt checkpoints/teacher/teacher_best.pth \
    --epochs 90 \
    --val_interval 5

# Evaluation
python evaluate.py \
    --data_root data/vod \
    --student_ckpt checkpoints/student/student_best.pth
```

##  Important Notes

1. **Radar Type**: Use `radar_5frames` for best performance (densest point clouds)
2. **Calibration**: Ensure proper coordinate frame alignment between sensors
3. **Memory**: VoD is large (~15GB) - ensure sufficient disk space
4. **Splits**: Use existing splits for fair comparison with literature  
5. **Preprocessing**: Follow paper specifications exactly for reproducible results

##  Expected Performance

With proper VoD dataset processing, expect:
- **Training time**: ~20-27 hours (RTX 4090)  
- **Chamfer Distance**: ~5.16 (15% improvement over baselines)
- **F-score**: ~0.39 (44% improvement)
- **Memory usage**: <12GB GPU during training
