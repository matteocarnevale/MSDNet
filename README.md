# MSDNet — Multi-Stage Distillation for 4D Radar Super-Resolution

PyTorch implementation of **MSDNet** (Huang, Lu, Zheng et al., 2025), a
knowledge-distillation framework that turns sparse 4D radar point clouds into
dense, LiDAR-quality ones.  Training happens in two stages:

1. **Stage 0 — Teacher.**  A LiDAR-only network learns to encode point clouds
   into dense BEV features and reconstruct them back into 3D points.
2. **Stage 1+2 — Student.**  A radar network is trained to produce features
   that match the teacher's, first through *reconstruction-guided distillation*
   (RGFD), then through *diffusion-guided distillation* (DGFD).  At inference
   time, only the lightweight student runs — no LiDAR needed.

```
                        ┌──────────────────────────────────────────────────────┐
  LiDAR ─► VoxelEncoder ─► S2D Enhancement ─► F_l^D ─► Reconstruction ─► PC  │  TEACHER
                        └────────────────────────┬─────────────────────────────┘
                                                 │ distillation losses
                        ┌────────────────────────▼─────────────────────────────┐
  Radar ─► VoxelEncoder ─► RGFD ─► F_r^R ─► DGFD ─► F_r^D ─► Recon ─► PC    │  STUDENT
                        └──────────────────────────────────────────────────────┘
```

---

## 1. Project Structure

```
MSDNet/
├── config.py                  Hyperparameters (all in one place, as dataclasses)
├── dataset.py                 VoD dataset loader + ground-truth generation
├── losses.py                  All loss functions (reconstruction, distillation, diffusion)
│
├── models/
│   ├── modules.py             Shared building blocks (CBAM, ConvNeXt, BottleNeck, …)
│   ├── encoder.py             Point cloud → voxels → sparse 3D convs → BEV features
│   ├── enhancement.py         S2D module (teacher feature densification)
│   ├── rgfd.py                Stage 1 — Reconstruction-Guided Feature Distillation
│   ├── dgfd.py                Stage 2 — Diffusion-Guided Feature Distillation
│   ├── diffusion.py           Noise schedule, forward process, DDIM sampling
│   ├── reconstruction.py      BEV features → multi-scale 3D occupancy + offsets → point cloud
│   └── msdnet.py              Top-level MSDNetTeacher and MSDNetStudent classes
│
├── train_teacher.py           Script: train the teacher (Stage 0)
├── train_student.py           Script: train the student (Stage 1+2)
├── evaluate.py                Script: evaluate a trained student
├── test_smoke.py              Smoke tests (67 tests, no GPU needed)
└── requirements.txt
```

#### Key Improvements to the VoxelNet Implementation

This implementation includes several enhancements over a basic VoxelNet:

- **Multi-layer VFE**: Stacked VoxelNet Feature Encoding layers with element-wise max pooling (faithful to original Zhou & Tuzel paper)
- **Smart Voxelization**: Separate `max_voxels_train` and `max_voxels_eval` limits with random sampling during training for better regularization
- **Robust Point Sampling**: Random point selection within voxels during training to prevent overfitting to point ordering
- **Enhanced Training**: Validation loops, checkpoint resuming, and best model saving
- **Improved Configuration**: Configurable VFE channels and better hyperparameter organization

---

## 2. Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | >= 3.9 | |
| PyTorch | >= 2.0 | |
| torchvision | >= 0.15 | For `DeformConv2d` |
| spconv | >= 2.3 | Install the wheel matching your CUDA version (e.g. `pip install spconv-cu121`) |
| numpy | >= 1.24 | |
| scipy | >= 1.10 | For evaluation metrics |
| tqdm | >= 4.65 | Progress bars |
| tensorboard | >= 2.13 | Training logs |

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

MSDNet expects the **View-of-Delft (VoD)** dataset.  Prepare it in the
following directory layout:

```
data/vod/
├── lidar/              One .bin file per frame
│   ├── 000000.bin      Each file: (N, 4) float32 array — x, y, z, intensity
│   ├── 000001.bin
│   └── ...
├── radar/              One .bin file per frame
│   ├── 000000.bin      Each file: (N, 5) float32 array — x, y, z, intensity, velocity
│   ├── 000001.bin
│   └── ...
└── split/
    ├── train.txt       One frame ID per line (e.g. "000000")
    └── test.txt        Frame IDs for testing (sequences 03, 04, 22)
```

### How to convert VoD to this format

1. **Download VoD** from
   [the official repository](https://github.com/tudelft-iv/view-of-delft-dataset).
2. For each frame, save the LiDAR point cloud as a flat `float32` binary file
   with 4 columns `[x, y, z, intensity]`.  Do the same for radar with 5
   columns `[x, y, z, intensity, velocity]`.
3. Create `split/train.txt` and `split/test.txt` listing the frame IDs.
   Following 4DRVO-Net, sequences **03, 04, 22** go in `test.txt`; everything
   else in `train.txt`.

### Preprocessing (handled automatically)

The dataloader performs these steps at load time:

- **Ground removal:** LiDAR points below -1.5 m are discarded.
- **FoV cropping:** LiDAR points are clipped to the 120-degree horizontal
  field of view of the 4D radar.
- **Range cropping:** Both LiDAR and radar are cropped to the voxelization
  range `[0, 32] × [-16, 16] × [-2, 4]` meters.

---

## 4. Training

Training is a two-step process: teacher first, then student.

### Step 1 — Train the Teacher (Stage 0)

The teacher takes **LiDAR** point clouds, encodes them into BEV features,
enhances them with the S2D module, and learns to reconstruct 3D point clouds.

```bash
python train_teacher.py \
    --data_root  data/vod \
    --epochs     60 \
    --batch_size 4 \
    --lr         1e-3 \
    --ckpt_dir   checkpoints/teacher \
    --log_dir    runs/teacher
```

**Advanced Options:**
- `--resume checkpoints/teacher/teacher_epoch40.pth` — Resume from checkpoint
- `--val_interval 5` — Run validation every 5 epochs
- `--save_interval 10` — Save checkpoint every 10 epochs

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Path to the VoD dataset root |
| `--epochs` | 60 | Number of training epochs |
| `--batch_size` | 4 | Samples per batch |
| `--lr` | 1e-3 | Peak learning rate (OneCycleLR) |
| `--ckpt_dir` | `checkpoints/teacher` | Where to save checkpoints |
| `--log_dir` | `runs/teacher` | TensorBoard log directory |
| `--num_workers` | 4 | DataLoader workers |

**Output:**  `checkpoints/teacher/teacher_final.pth`

### Step 2 — Train the Student (Stage 1+2)

The student takes **4D radar** point clouds and learns to produce features
that match the frozen teacher's dense LiDAR features.  The reconstruction
module is shared (same weights) between teacher and student.

```bash
python train_student.py \
    --data_root    data/vod \
    --teacher_ckpt checkpoints/teacher/teacher_final.pth \
    --epochs       90 \
    --batch_size   4 \
    --lr           1e-3 \
    --ckpt_dir     checkpoints/student \
    --log_dir      runs/student
```

**Advanced Options:**
- `--teacher_ckpt checkpoints/teacher/teacher_best.pth` — Use best teacher checkpoint
- `--resume checkpoints/student/student_epoch50.pth` — Resume from checkpoint  
- `--val_interval 5` — Run validation every 5 epochs
- `--save_interval 10` — Save checkpoint every 10 epochs

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Path to the VoD dataset root |
| `--teacher_ckpt` | *(required)* | Path to the trained teacher checkpoint |
| `--epochs` | 90 | Number of training epochs |
| `--batch_size` | 4 | Samples per batch |
| `--lr` | 1e-3 | Peak learning rate |
| `--ckpt_dir` | `checkpoints/student` | Where to save checkpoints |
| `--log_dir` | `runs/student` | TensorBoard log directory |

**Output:**  `checkpoints/student/student_final.pth`

### Monitoring

```bash
tensorboard --logdir runs/
```

---

## 5. Evaluation

```bash
python evaluate.py \
    --data_root    data/vod \
    --teacher_ckpt checkpoints/teacher/teacher_final.pth \
    --student_ckpt checkpoints/student/student_final.pth \
    --threshold    0.5
```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Path to the VoD dataset root |
| `--student_ckpt` | *(required)* | Path to the trained student checkpoint |
| `--teacher_ckpt` | `None` | Teacher checkpoint (used to build the shared reconstruction head) |
| `--batch_size` | 1 | Evaluation batch size |
| `--threshold` | 0.5 | Occupancy threshold for point generation |

### Metrics

| Metric | What it measures | Lower/Higher is better |
|---|---|---|
| **CD** (Chamfer Distance) | Average bidirectional nearest-neighbor distance | Lower |
| **MHD** (Modified Hausdorff) | Worst-case average directional distance | Lower |
| **F-score** | Fraction of points within a distance threshold | Higher |
| **JSD** (Jensen-Shannon Discrepancy) | BEV spatial distribution similarity | Lower |
| **MMD** (Maximum Mean Discrepancy) | Distribution similarity via RBF kernel | Lower |

---

## 6. Smoke Tests

A comprehensive test suite verifies that every component works correctly
without requiring a GPU or the VoD dataset:

```bash
pip install pytest
python -m pytest test_smoke.py -v
```

The 67 tests cover: imports, config, dataset loading with synthetic data,
every model component (forward pass + gradient flow), all loss functions,
all evaluation metrics, and simulated training steps.

---

## 7. Configuration

All hyperparameters live in `config.py` as dataclasses.  The defaults match
the paper (Section IV-B).

### Voxelization

| Parameter | Default | Description |
|---|---|---|
| Point cloud range | `[0, -16, -2, 32, 16, 4]` m | Crop limits (x_min, y_min, z_min, x_max, y_max, z_max) |
| Voxel size | `0.1 × 0.1 × 0.15` m | Resolution per axis |
| Max points/voxel | 5 | Hard voxelization limit |

### Diffusion (DGFD)

| Parameter | Default | Description |
|---|---|---|
| Total timesteps (T) | 1000 | Noise schedule length |
| Start timestep (m) | 500 | Where the noise adapter targets |
| Sampling steps (T_m) | 50 | DDIM reverse steps |
| Sampling interval (n) | 10 | Step size between DDIM steps |

### Loss Weights (Eq. 15)

| Weight | Default | Loss term |
|---|---|---|
| lambda_1 | 1.0 | Reconstruction loss |
| lambda_2 | 0.01 | Reconstruction distillation loss |
| lambda_3 | 5.0 | Diffusion distillation loss |
| lambda_4 | 10.0 | Diffusion noise prediction loss |

---

## 8. Citation

```bibtex
@article{huang2025msdnet,
  title   = {MSDNet: Efficient 4D Radar Super-Resolution
             via Multi-Stage Distillation},
  author  = {Huang, Minqing and Lu, Shouyi and Zheng, Boyuan
             and Li, Ziyao and Tang, Xiao and Zhuo, Guirong},
  journal = {arXiv preprint arXiv:2509.13149},
  year    = {2025}
}
```
