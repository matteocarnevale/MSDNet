"""Evaluate generated point clouds with MSDNet paper metrics and save a report.

Metrics (MSDNet paper, Sec. IV-C):
  - Chamfer Distance (CD)                 ↓
  - Modified Hausdorff Distance (MHD)     ↓
  - F-score                               ↑
  - Jensen–Shannon Discrepancy (JSD)      ↓   (BEV spatial distribution)
  - Maximum Mean Discrepancy (MMD)        ↓   (BEV spatial distribution)

This script can evaluate:
  1) Model outputs: run MSDNet student and generate point clouds.
  2) Pre-generated point clouds: load predictions from a directory.

It saves:
  - per-frame metrics CSV
  - summary CSV (mean/std/median/quantiles)
  - plots (histograms + boxplots) if matplotlib is installed

Examples:
  Evaluate by running the model:
    python evaluate_report.py --data_root /path/to/vod \
      --student_ckpt checkpoints/student/student_final.pth \
      --out_dir runs/eval_student

  Evaluate pre-generated predictions (one file per frame_id):
    python evaluate_report.py --data_root /path/to/vod \
      --pred_dir runs/preds_npy --pred_ext .npy \
      --out_dir runs/eval_preds
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MSDNetConfig
from dataset import VoDDataset, collate_fn


# ---------------------------------------------------------------------------
# Point cloud IO (predictions)
# ---------------------------------------------------------------------------

def _as_xyz(pc: np.ndarray) -> np.ndarray:
    """Return a float32 (N,3) xyz array, tolerant to extra columns."""
    if pc.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pc = np.asarray(pc)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"Invalid point cloud shape: {pc.shape} (expected (N,>=3))")
    return pc[:, :3].astype(np.float32, copy=False)


def load_pred_point_cloud(path: str) -> np.ndarray:
    """Load a prediction point cloud as (N,3) float32 from .npy/.npz/.bin."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return _as_xyz(np.load(path))
    if ext == ".npz":
        z = np.load(path)
        # common conventions: "points", "xyz"
        if "points" in z:
            return _as_xyz(z["points"])
        if "xyz" in z:
            return _as_xyz(z["xyz"])
        # fallback: first array
        keys = list(z.keys())
        if not keys:
            return np.zeros((0, 3), dtype=np.float32)
        return _as_xyz(z[keys[0]])
    if ext == ".bin":
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # Heuristics: many datasets store (N,4) or (N,5); predictions can be (N,3)
        for cols in (3, 4, 5, 6):
            if arr.size % cols == 0:
                pc = arr.reshape(-1, cols)
                return _as_xyz(pc)
        raise ValueError(f"Cannot infer .bin columns for {path} (size={arr.size})")
    raise ValueError(f"Unsupported prediction extension {ext!r} for {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def chamfer_distance(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    tree_pred = KDTree(pred_xyz)
    tree_gt = KDTree(gt_xyz)
    d_pred, _ = tree_gt.query(pred_xyz)
    d_gt, _ = tree_pred.query(gt_xyz)
    return float(d_pred.mean() + d_gt.mean())


def modified_hausdorff(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    tree_pred = KDTree(pred_xyz)
    tree_gt = KDTree(gt_xyz)
    d_pred, _ = tree_gt.query(pred_xyz)
    d_gt, _ = tree_pred.query(gt_xyz)
    return float(max(d_pred.mean(), d_gt.mean()))


def f_score(pred_xyz: np.ndarray, gt_xyz: np.ndarray, threshold: float) -> float:
    tree_pred = KDTree(pred_xyz)
    tree_gt = KDTree(gt_xyz)
    d_pred, _ = tree_gt.query(pred_xyz)
    d_gt, _ = tree_pred.query(gt_xyz)
    precision = float((d_pred < threshold).mean())
    recall = float((d_gt < threshold).mean())
    if precision + recall < 1e-8:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _bev_hist(
    xyz: np.ndarray,
    point_cloud_range: List[float],
    grid_res: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalized BEV histogram over (x,y) within the fixed range."""
    pc = point_cloud_range
    x_min, y_min, _ = pc[0], pc[1], pc[2]
    x_max, y_max, _ = pc[3], pc[4], pc[5]
    bins_x = int(np.ceil((x_max - x_min) / grid_res))
    bins_y = int(np.ceil((y_max - y_min) / grid_res))
    hist_range = [[x_min, x_max], [y_min, y_max]]
    h, _, _ = np.histogram2d(
        xyz[:, 0],
        xyz[:, 1],
        bins=[bins_x, bins_y],
        range=hist_range,
    )
    p = h.reshape(-1).astype(np.float64)
    p = p / (p.sum() + eps)
    return p


def jsd_bev(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    point_cloud_range: List[float],
    grid_res: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """Jensen–Shannon Discrepancy on BEV distributions."""
    p = _bev_hist(pred_xyz, point_cloud_range, grid_res, eps=eps)
    q = _bev_hist(gt_xyz, point_cloud_range, grid_res, eps=eps)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def mmd_rbf(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    sigma: float = 1.0,
    max_samples: int = 2048,
) -> float:
    """MMD with RBF kernel on 3D point clouds (sub-sampled for speed).

    Args:
      pred_xyz: (N, 3) predicted point cloud.
      gt_xyz:   (M, 3) ground-truth point cloud.
      sigma:    RBF bandwidth in metres (default 1.0, matching evaluate.py).
    """
    pred_xyz = pred_xyz.astype(np.float64, copy=False)
    gt_xyz = gt_xyz.astype(np.float64, copy=False)

    if pred_xyz.shape[0] > max_samples:
        idx = np.random.choice(pred_xyz.shape[0], max_samples, replace=False)
        pred_xyz = pred_xyz[idx]
    if gt_xyz.shape[0] > max_samples:
        idx = np.random.choice(gt_xyz.shape[0], max_samples, replace=False)
        gt_xyz = gt_xyz[idx]

    def _k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        return np.exp(-d / (2.0 * sigma ** 2))

    mmd2 = float(_k(pred_xyz, pred_xyz).mean() + _k(gt_xyz, gt_xyz).mean()
                 - 2.0 * _k(pred_xyz, gt_xyz).mean())
    return float(np.sqrt(max(mmd2, 0.0)))


# ---------------------------------------------------------------------------
# Reporting utils
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_id: str
    n_pred: int
    n_gt: int
    cd: float
    mhd: float
    fscore: float
    jsd: float
    mmd: float
    runtime_s: float


def _safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_csv(path: str, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_make_plots(out_dir: str, frame_rows: List[FrameResult]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # plots are optional
        return

    metrics = ["cd", "mhd", "fscore", "jsd", "mmd", "runtime_s"]
    data = {m: np.array([getattr(r, m) for r in frame_rows], dtype=np.float64) for m in metrics}

    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.reshape(-1)
    for ax, m in zip(axes, metrics):
        v = data[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.set_title(m)
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ax.hist(v, bins=40, alpha=0.85)
        ax.set_title(m)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_metrics.png"), dpi=160)
    plt.close(fig)

    # Boxplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    box_data = []
    labels = []
    for m in metrics:
        v = data[m]
        v = v[np.isfinite(v)]
        if v.size:
            box_data.append(v)
            labels.append(m)
    if box_data:
        ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_title("Metrics distribution (boxplot)")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "boxplot_metrics.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSDNet — evaluate & save report")
    p.add_argument("--data_root", type=str, required=True)

    mode = p.add_argument_group("Mode")
    mode.add_argument("--student_ckpt", type=str, default=None, help="Student checkpoint (model-eval mode)")
    mode.add_argument("--teacher_ckpt", type=str, default=None, help="Teacher checkpoint (optional; shared recon)")
    mode.add_argument("--pred_dir", type=str, default=None, help="Directory with pre-generated predictions")
    mode.add_argument("--pred_ext", type=str, default=".npy", help="Prediction extension (.npy/.npz/.bin)")

    evalg = p.add_argument_group("Evaluation")
    evalg.add_argument("--batch_size", type=int, default=1)
    evalg.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold for model generation")
    evalg.add_argument("--max_frames", type=int, default=0, help="If >0, limit number of evaluated frames")
    evalg.add_argument("--seed", type=int, default=42)
    evalg.add_argument(
        "--vod_sequence_filter",
        type=str,
        default="none",
        choices=("none", "4drvo_net"),
        help="Paper IV-A VoD test split when using 4drvo_net.",
    )

    met = p.add_argument_group("Metrics")
    met.add_argument("--fscore_thresh", type=float, default=-1.0, help="F-score threshold; default=voxel_size[x]")
    met.add_argument("--bev_grid_res", type=float, default=0.5, help="BEV histogram grid resolution (m)")
    met.add_argument("--mmd_sigma", type=float, default=-1.0, help="RBF bandwidth; <0 uses heuristic")

    out = p.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="runs/eval_report")
    return p.parse_args()


def _assert_mode(args: argparse.Namespace) -> None:
    if args.pred_dir and args.student_ckpt:
        raise SystemExit("Choose only one mode: either --pred_dir or --student_ckpt.")
    if not args.pred_dir and not args.student_ckpt:
        raise SystemExit("Choose a mode: provide --student_ckpt (run model) or --pred_dir (load predictions).")


def main() -> None:
    args = parse_args()
    _assert_mode(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = MSDNetConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vod_f = None if args.vod_sequence_filter == "none" else args.vod_sequence_filter
    test_ds = VoDDataset(
        args.data_root,
        "test",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=True,
        vod_sequence_filter=vod_f,
    )
    if len(test_ds) == 0:
        raise SystemExit("Dataset vuoto: controlla data_root / split.")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    student = None
    if args.student_ckpt:
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="spconv")
        from models.msdnet import MSDNetStudent, MSDNetTeacher

        shared_recon = None
        if args.teacher_ckpt:
            teacher = MSDNetTeacher(cfg).to(device)
            tckpt = torch.load(args.teacher_ckpt, map_location=device)
            teacher.load_state_dict(tckpt.get("model_state_dict", tckpt))
            shared_recon = teacher.reconstruction

        student = MSDNetStudent(cfg, shared_reconstruction=shared_recon).to(device)
        sckpt = torch.load(args.student_ckpt, map_location=device)
        student.load_state_dict(sckpt.get("model_state_dict", sckpt), strict=False)
        student.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    f_thresh = args.fscore_thresh if args.fscore_thresh > 0 else float(cfg.voxel.voxel_size[0])
    sigma = None if args.mmd_sigma < 0 else float(args.mmd_sigma)

    rows: List[FrameResult] = []
    n_seen = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        radar_list = batch["radar"]
        if args.student_ckpt:
            radar_list = [pc.to(device) for pc in radar_list]
        lidar_list = [pc.numpy() for pc in batch["lidar"]]
        frame_ids = batch.get("frame_ids", batch.get("frame_id", None)) or ["" for _ in lidar_list]

        start = time.time()
        pred_list: List[np.ndarray] = []
        if args.pred_dir:
            for fid in frame_ids:
                pred_path = os.path.join(args.pred_dir, f"{fid}{args.pred_ext}")
                if not os.path.exists(pred_path):
                    raise FileNotFoundError(f"Prediction not found for frame_id={fid!r}: {pred_path}")
                pred_list.append(load_pred_point_cloud(pred_path))
        else:
            assert student is not None
            with torch.no_grad():
                pred_pcs = student.generate_point_cloud(
                    radar_list,
                    args.batch_size,
                    threshold=args.threshold,
                    point_cloud_range=cfg.voxel.point_cloud_range,
                )
            for t in pred_pcs:
                pred_list.append(_as_xyz(t.detach().cpu().numpy()))
        runtime = time.time() - start

        for fid, pred_xyz, gt_np in zip(frame_ids, pred_list, lidar_list):
            gt_xyz = _as_xyz(gt_np)
            if pred_xyz.shape[0] < 2 or gt_xyz.shape[0] < 2:
                continue

            rows.append(
                FrameResult(
                    frame_id=str(fid),
                    n_pred=int(pred_xyz.shape[0]),
                    n_gt=int(gt_xyz.shape[0]),
                    cd=_safe_float(chamfer_distance(pred_xyz, gt_xyz)),
                    mhd=_safe_float(modified_hausdorff(pred_xyz, gt_xyz)),
                    fscore=_safe_float(f_score(pred_xyz, gt_xyz, threshold=f_thresh)),
                    jsd=_safe_float(jsd_bev(pred_xyz, gt_xyz, cfg.voxel.point_cloud_range, grid_res=args.bev_grid_res)),
                    mmd=_safe_float(mmd_rbf(pred_xyz, gt_xyz, sigma=sigma if sigma is not None else 1.0)),
                    runtime_s=_safe_float(runtime / max(len(frame_ids), 1)),
                )
            )
            n_seen += 1
            if args.max_frames and n_seen >= args.max_frames:
                break
        if args.max_frames and n_seen >= args.max_frames:
            break

    if not rows:
        raise SystemExit("Nessun frame valido valutato (pred/gt troppo piccoli?).")

    # Per-frame CSV
    per_frame_path = os.path.join(args.out_dir, "metrics_per_frame.csv")
    save_csv(
        per_frame_path,
        (r.__dict__ for r in rows),
        fieldnames=list(FrameResult.__dataclass_fields__.keys()),
    )

    # Summary CSV
    metrics = ["cd", "mhd", "fscore", "jsd", "mmd", "runtime_s", "n_pred", "n_gt"]
    summary_rows = []
    for m in metrics:
        arr = np.array([getattr(r, m) for r in rows], dtype=np.float64)
        s = summarize(arr)
        summary_rows.append({"metric": m, **s})
    summary_path = os.path.join(args.out_dir, "metrics_summary.csv")
    save_csv(
        summary_path,
        summary_rows,
        fieldnames=["metric", "mean", "std", "median", "q25", "q75", "min", "max"],
    )

    # Print a compact summary (paper-like)
    print("\n===== MSDNet metrics (report) =====")
    for m in ["cd", "mhd", "fscore", "jsd", "mmd"]:
        arr = np.array([getattr(r, m) for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        print(f"{m.upper():>8s}: {arr.mean():.4f} ± {arr.std():.4f}  (N={arr.size})")
    rt = np.array([r.runtime_s for r in rows], dtype=np.float64)
    print(f"{'Runtime':>8s}: {np.nanmean(rt):.3f}s per frame")
    print(f"{'Saved':>8s}: {per_frame_path}")
    print(f"{'Saved':>8s}: {summary_path}")

    # Optional plots
    try_make_plots(args.out_dir, rows)


if __name__ == "__main__":
    main()

