"""RADIal Dataset for MSDNet.

Reads the `radar_lidar_pc_dataset` folder produced by build_radial_dataset.py.

Folder layout:
    <root>/
        radar_PCL/      pcl_{id:06d}.npy   [9, N]   radar point cloud
        laser_PCL/      pcl_{id:06d}.npy   [M, 3]   raw LiDAR XYZ
        laser_PCL_ng/   pcl_{id:06d}.npy   [M, 4]   ← auto-generated on first run
                                                       ground-removed LiDAR + intensity=1
        index.csv       sample manifest (sample_id, sequence, …)
        radar_columns.txt

Ground removal (RANSAC plane fitting):
    On first access each raw LiDAR frame is processed:
      1. Collect the lowest `candidate_percentile` % of points as ground candidates.
      2. RANSAC: sample 3 candidate points → fit plane → count inliers.
      3. Keep the plane with the most inliers after `n_iters` trials.
      4. Discard all points within `above_margin` metres of (or below) the plane.
    Result is saved to laser_PCL_ng/ and reused on every subsequent epoch.
    Fallback to a simple z-threshold if fewer than 3 points are available.

Radar PCL columns (stored as [9, N], rows = features):
    0 range_m  1 azimuth_rad  2 elevation_rad  3 power_db  4 doppler_bin
    5 x_m  6 y_m  7 z_m  8 v_bin_centered ∈ [-128, 127]

MSDNet features:
    radar → [x_m, y_m, z_m, power_norm, v_norm]     (N, 5)
    lidar → [x,   y,   z,   intensity=1]             (M, 4)

Split: sequence-based (whole drives). See get_splits() / print_split_info().
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Radar feature indices inside the [9, N] array
# ---------------------------------------------------------------------------

_COL_RANGE_M       = 0
_COL_AZIMUTH_RAD   = 1
_COL_ELEVATION_RAD = 2
_COL_POWER_DB      = 3
_COL_DOPPLER_BIN   = 4
_COL_X_M           = 5
_COL_Y_M           = 6
_COL_Z_M           = 7
_COL_V_BIN         = 8   # v_bin_centered ∈ [-128, 127]

_DOPPLER_MAX       = 128.0   # normalisation constant for v_bin_centered


# ---------------------------------------------------------------------------
# Splits  (sequence-based  —  no temporal / geographic leakage)
# ---------------------------------------------------------------------------

def get_splits(
    root_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Sequence-based train / val / test split from index.csv.

    Why sequence-based?
        Consecutive frames in the same drive are strongly correlated
        (same road, same weather, high overlap between frames).
        Splitting by frame would leak future/past frames from the same
        sequence into the validation set → inflated metrics.

    How it works:
        1. Read the unique 'sequence' values from index.csv.
        2. Shuffle them deterministically with `seed`.
        3. Assign the first 70 % to train, the next 15 % to val, the rest to test.
        4. Return the corresponding sample_ids for each split.

    Returns:
        (train_ids, val_ids, test_ids) — lists of integer sample_ids.
    """
    root = Path(root_dir)
    df   = pd.read_csv(root / "index.csv")

    _check_columns(df, required=["sample_id", "sequence"])

    sequences = sorted(df["sequence"].unique())
    rng  = np.random.default_rng(seed)
    seqs = np.array(sequences)
    rng.shuffle(seqs)

    n       = len(seqs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_seqs = set(seqs[:n_train])
    val_seqs   = set(seqs[n_train : n_train + n_val])
    test_seqs  = set(seqs[n_train + n_val :])

    def _ids(seq_set):
        return df[df["sequence"].isin(seq_set)]["sample_id"].tolist()

    return _ids(train_seqs), _ids(val_seqs), _ids(test_seqs)


def print_split_info(
    root_dir: str | Path,
    train_ids: List[int],
    val_ids:   List[int],
    test_ids:  List[int],
) -> None:
    """Print a human-readable summary of the split."""
    root = Path(root_dir)
    df   = pd.read_csv(root / "index.csv")

    def _seq_names(ids):
        return sorted(df[df["sample_id"].isin(ids)]["sequence"].unique())

    tr_seqs = _seq_names(train_ids)
    va_seqs = _seq_names(val_ids)
    te_seqs = _seq_names(test_ids)

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print("=" * 60)
    print(f"Dataset split summary  ({total} total frames)")
    print("=" * 60)
    print(f"  Train : {len(train_ids):>6}  frames  "
          f"({len(tr_seqs)} sequences, {100*len(train_ids)/total:.1f}%)")
    print(f"  Val   : {len(val_ids):>6}  frames  "
          f"({len(va_seqs)} sequences, {100*len(val_ids)/total:.1f}%)")
    print(f"  Test  : {len(test_ids):>6}  frames  "
          f"({len(te_seqs)} sequences, {100*len(test_ids)/total:.1f}%)")
    print()
    print("  Train sequences:")
    for s in tr_seqs: print(f"    {s}")
    print("  Val sequences:")
    for s in va_seqs: print(f"    {s}")
    print("  Test sequences:")
    for s in te_seqs: print(f"    {s}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# GT helpers  (self-contained)
# ---------------------------------------------------------------------------

def _gt_volume_shapes(
    bev_h: int, bev_w: int, grid_z: int
) -> Dict[int, Tuple[int, int, int]]:
    """
    (nz, nh, nw) of the GT tensors at each reconstruction scale.

    Must match PointCloudReconstruction output exactly:
      scale 4: (z//4, bev_h,   bev_w)
      scale 2: (z//2, bev_h*2, bev_w*2)
      scale 1: (z,    bev_h*4, bev_w*4)

    Args:
        bev_h: H_bev = Y_cells // 8  (row = lateral dimension)
        bev_w: W_bev = X_cells // 8  (col = forward dimension)
    """
    z4 = grid_z // 4
    return {
        4: (z4,     bev_h,     bev_w),
        2: (z4 * 2, bev_h * 2, bev_w * 2),
        1: (grid_z, bev_h * 4, bev_w * 4),
    }


def _generate_gt(
    lidar_xyz:     np.ndarray,
    pc_range:      List[float],
    volume_shapes: Dict[int, Tuple[int, int, int]],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Multi-scale occupancy and offset GT from LiDAR XYZ.

    Tensor ordering: (1, nz, nh, nw) = (1, Z, Y, X)
    Consistent with PointCloudReconstruction output.
    """
    pc_min  = np.array(pc_range[:3], dtype=np.float64)
    extent  = np.array(pc_range[3:], dtype=np.float64) - pc_min

    gt_occ, gt_off = {}, {}

    for scale, (nz, nh, nw) in volume_shapes.items():
        vx = extent[0] / nw    # X step  (col)
        vy = extent[1] / nh    # Y step  (row)
        vz = extent[2] / nz    # Z step  (depth)

        occ = np.zeros((1, nz, nh, nw), dtype=np.float32)
        off = np.zeros((3, nz, nh, nw), dtype=np.float32)

        if lidar_xyz.shape[0] > 0:
            xyz = lidar_xyz[:, :3]
            ic  = np.clip(((xyz[:, 0] - pc_min[0]) / vx).astype(int), 0, nw - 1)
            ir  = np.clip(((xyz[:, 1] - pc_min[1]) / vy).astype(int), 0, nh - 1)
            iz  = np.clip(((xyz[:, 2] - pc_min[2]) / vz).astype(int), 0, nz - 1)

            occ[0, iz, ir, ic] = 1.0

            cx = pc_min[0] + (ic + 0.5) * vx
            cy = pc_min[1] + (ir + 0.5) * vy
            cz = pc_min[2] + (iz + 0.5) * vz
            off[0, iz, ir, ic] = (xyz[:, 0] - cx).astype(np.float32)
            off[1, iz, ir, ic] = (xyz[:, 1] - cy).astype(np.float32)
            off[2, iz, ir, ic] = (xyz[:, 2] - cz).astype(np.float32)

        gt_occ[scale] = torch.from_numpy(occ)
        gt_off[scale] = torch.from_numpy(off)

    return gt_occ, gt_off


# ---------------------------------------------------------------------------
# RANSAC ground removal
# ---------------------------------------------------------------------------

def _ransac_ground_removal(
    pts:                  np.ndarray,
    n_iters:              int   = 100,
    dist_thresh:          float = 0.20,   # inlier distance from plane (metres)
    candidate_percentile: float = 20.0,   # sample candidates from lowest N% by z
    above_margin:         float = 0.10,   # keep points > this height above plane
    seed:                 int   = 0,
) -> np.ndarray:
    """
    Fit a ground plane with RANSAC and remove all points at or below it.

    Algorithm:
      1. Restrict RANSAC sampling to the lowest `candidate_percentile` % by z
         (avoids wasting iterations on obvious non-ground points).
      2. For each iteration: sample 3 candidates, compute plane normal via
         cross-product, count points within `dist_thresh` of the plane.
      3. After `n_iters` trials keep the best-supported plane.
      4. Discard points whose signed distance from the best plane is less
         than `above_margin` (i.e. on the ground or below it).

    Falls back to a simple z-threshold (-1.0 m) when fewer than 3 candidate
    points are available.

    Args:
        pts:    (M, 3) float32 XYZ array.
    Returns:
        (M', 3) float32 — points above the ground plane.
    """
    if pts.shape[0] < 3:
        return pts

    # Step 1: ground candidates = lowest percentile
    z_cut = np.percentile(pts[:, 2], candidate_percentile)
    cand  = pts[pts[:, 2] <= z_cut]

    if cand.shape[0] < 3:
        # Fallback: simple height threshold relative to min z
        z_floor = pts[:, 2].min() + 0.3
        return pts[pts[:, 2] > z_floor]

    rng          = np.random.default_rng(seed)
    best_normal  = None
    best_d       = 0.0
    best_count   = 0

    for _ in range(n_iters):
        idx = rng.choice(len(cand), 3, replace=False)
        p1, p2, p3 = cand[idx[0]], cand[idx[1]], cand[idx[2]]

        # Plane normal via cross-product
        normal = np.cross(p2 - p1, p3 - p1)
        mag    = np.linalg.norm(normal)
        if mag < 1e-8:
            continue           # degenerate — three collinear points
        normal /= mag
        d = -float(np.dot(normal, p1))

        # Inliers across ALL points
        signed = pts @ normal + d
        n_in   = int((np.abs(signed) < dist_thresh).sum())

        if n_in > best_count:
            best_count  = n_in
            best_normal = normal.copy()
            best_d      = d

    if best_normal is None:
        z_floor = pts[:, 2].min() + 0.3
        return pts[pts[:, 2] > z_floor]

    # Ensure the normal points upward (positive z component)
    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d      = -best_d

    # Keep points more than `above_margin` metres above the fitted plane
    signed_dist = pts @ best_normal + best_d
    return pts[signed_dist > above_margin]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _process_radar(
    raw: np.ndarray,
    pc_range: List[float],
) -> np.ndarray:
    """
    Convert raw [9, N] radar PCL → MSDNet [N, 5] feature tensor.

    Input rows: range_m, az_rad, el_rad, power_db, doppler_bin,
                x_m, y_m, z_m, v_bin_centered
    Output cols: x_m, y_m, z_m, power_norm ∈ [0,1], v_norm ∈ [-1,1]
    """
    if raw.size == 0 or raw.shape[1] == 0:
        return np.zeros((0, 5), dtype=np.float32)

    pcl = raw.T.astype(np.float32)   # (N, 9)

    x = pcl[:, _COL_X_M]
    y = pcl[:, _COL_Y_M]
    z = pcl[:, _COL_Z_M]
    p = pcl[:, _COL_POWER_DB]
    v = pcl[:, _COL_V_BIN]

    p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)   # per-frame [0,1]
    v_norm = v / _DOPPLER_MAX                               # [-1, 1]

    pts  = np.stack([x, y, z, p_norm, v_norm], axis=1)
    pc   = pc_range
    mask = (
        (pts[:, 0] >= pc[0]) & (pts[:, 0] < pc[3]) &
        (pts[:, 1] >= pc[1]) & (pts[:, 1] < pc[4]) &
        (pts[:, 2] >= pc[2]) & (pts[:, 2] < pc[5])
    )
    return pts[mask]


def _process_lidar(
    raw:      np.ndarray,
    pc_range: List[float],
    ransac_cfg: Optional[dict] = None,
) -> np.ndarray:
    """
    Process raw [M, 3] LiDAR XYZ:
      1. RANSAC ground removal.
      2. Crop to point_cloud_range.
      3. Pad intensity = 1.0.
    Returns [M', 4] float32.
    """
    if raw.ndim == 1:
        raw = raw.reshape(-1, 3)
    pts = raw[:, :3].astype(np.float32)

    if pts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Ground removal
    cfg = ransac_cfg or {}
    pts = _ransac_ground_removal(pts, **cfg)

    if pts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Range crop
    pc   = pc_range
    mask = (
        (pts[:, 0] >= pc[0]) & (pts[:, 0] < pc[3]) &
        (pts[:, 1] >= pc[1]) & (pts[:, 1] < pc[4]) &
        (pts[:, 2] >= pc[2]) & (pts[:, 2] < pc[5])
    )
    pts = pts[mask]
    if pts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    out = np.zeros((pts.shape[0], 4), dtype=np.float32)
    out[:, :3] = pts
    out[:,  3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RADIalDataset(Dataset):
    """
    RADIal dataset for MSDNet teacher and student training.

    Reads from the `radar_lidar_pc_dataset/` folder structure.

    Ground-removed LiDAR is cached in `<root>/laser_PCL_ng/` and reused
    on every subsequent epoch (RANSAC is run only once per frame).

    Args:
        root_dir:    Path to `radar_lidar_pc_dataset/`.
        sample_ids:  List of integer sample_ids (from get_splits()).
        cfg:         MSDNetConfig instance.
        ransac_cfg:  Optional dict of kwargs forwarded to
                     _ransac_ground_removal().  Defaults:
                       n_iters=100, dist_thresh=0.20,
                       candidate_percentile=20.0, above_margin=0.10

    Each sample returns:
        lidar:     (M, 4)  float32  [x, y, z, intensity=1]
        radar:     (N, 5)  float32  [x, y, z, power_norm, v_norm]
        gt_occ:    {4, 2, 1 → (1, Z_s, Y_s, X_s)}
        gt_offset: {4, 2, 1 → (3, Z_s, Y_s, X_s)}
        frame_id:  str
    """

    def __init__(
        self,
        root_dir:   str | Path,
        sample_ids: List[int],
        cfg,
        ransac_cfg: Optional[dict] = None,
    ) -> None:
        self.root       = Path(root_dir)
        self.sample_ids = sample_ids
        self.cfg        = cfg
        self.pc_range   = cfg.voxel.point_cloud_range
        self.ransac_cfg = ransac_cfg or {}

        # Processed LiDAR lives here — auto-created on first run
        self.lidar_ng_dir = self.root / "laser_PCL_ng"
        self.lidar_ng_dir.mkdir(exist_ok=True)

        # Pre-compute GT volume shapes for this config
        gx, gy, gz   = cfg.grid_size      # (X_cells, Y_cells, Z_cells)
        bev_h = gy // 8                   # H_bev = Y//8  (row, lateral)
        bev_w = gx // 8                   # W_bev = X//8  (col, forward)
        self._vol_shapes = _gt_volume_shapes(bev_h, bev_w, gz)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sid    = self.sample_ids[index]
        radar  = self._load_radar(sid)
        lidar  = self._load_lidar(sid)
        gt_occ, gt_off = _generate_gt(lidar, self.pc_range, self._vol_shapes)

        return {
            "lidar":     torch.from_numpy(lidar).float(),
            "radar":     torch.from_numpy(radar).float(),
            "gt_occ":    gt_occ,
            "gt_offset": gt_off,
            "frame_id":  f"radial_{sid:06d}",
        }

    # ── radar ──────────────────────────────────────────────────────────

    def _load_radar(self, sid: int) -> np.ndarray:
        path = self.root / "radar_PCL" / f"pcl_{sid:06d}.npy"
        if not path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        raw = np.load(path, allow_pickle=True)   # (9, N)
        return _process_radar(raw, self.pc_range)

    # ── LiDAR ──────────────────────────────────────────────────────────

    def _load_lidar(self, sid: int) -> np.ndarray:
        cache_f = self.lidar_ng_dir / f"pcl_{sid:06d}.npy"

        # 1. Return cached ground-removed result if available
        if cache_f.exists():
            return np.load(cache_f)

        # 2. Load raw LiDAR, apply RANSAC + crop + pad
        raw_path = self.root / "laser_PCL" / f"pcl_{sid:06d}.npy"
        if not raw_path.exists():
            return np.zeros((0, 4), dtype=np.float32)

        raw       = np.load(raw_path, allow_pickle=True)   # (M, 3)
        processed = _process_lidar(raw, self.pc_range, self.ransac_cfg)

        # 3. Save to laser_PCL_ng/ for all future epochs
        np.save(cache_f, processed)
        return processed


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> dict:
    """Custom collation: point clouds as lists, GT tensors stacked."""
    return {
        "lidar":     [b["lidar"]    for b in batch],
        "radar":     [b["radar"]    for b in batch],
        "gt_occ":    {s: torch.stack([b["gt_occ"][s]    for b in batch]) for s in [4, 2, 1]},
        "gt_offset": {s: torch.stack([b["gt_offset"][s] for b in batch]) for s in [4, 2, 1]},
        "frame_ids": [b["frame_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"index.csv is missing expected columns: {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            "Make sure the dataset was built with build_radial_dataset.py."
        )
