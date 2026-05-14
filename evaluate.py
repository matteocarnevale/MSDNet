"""MSDNet — evaluation on RADIal test split.

Metrics (Section IV-C):
    CD  – Chamfer Distance (lower is better)
    MHD – Modified Hausdorff Distance (lower is better)
    F1  – F-score at distance threshold = voxel size (higher is better)
    JSD – Jensen-Shannon Discrepancy on BEV 2D histograms (lower is better)
    MMD – Maximum Mean Discrepancy with RBF kernel (lower is better)

Usage:
    python evaluate.py \\
        --radial_root  /data/RADIal \\
        --student_ckpt checkpoints/student/best.pth \\
        --teacher_ckpt checkpoints/teacher/best.pth   # (optional, for shared head)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MSDNetConfig
from dataset import RADIalDataset, get_splits, collate_fn
from models.msdnet import MSDNetTeacher, MSDNetStudent


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    t_pred, t_gt = KDTree(pred), KDTree(gt)
    d_p, _ = t_gt.query(pred)
    d_g, _ = t_pred.query(gt)
    return float(d_p.mean() + d_g.mean())


def modified_hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    t_pred, t_gt = KDTree(pred), KDTree(gt)
    d_p, _ = t_gt.query(pred)
    d_g, _ = t_pred.query(gt)
    return float(max(d_p.mean(), d_g.mean()))


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.3) -> float:
    t_pred, t_gt = KDTree(pred), KDTree(gt)
    d_p, _ = t_gt.query(pred)
    d_g, _ = t_pred.query(gt)
    precision = (d_p < threshold).mean()
    recall    = (d_g < threshold).mean()
    denom     = precision + recall
    return float(2 * precision * recall / denom) if denom > 1e-8 else 0.0


def jsd_bev(pred: np.ndarray, gt: np.ndarray,
            grid_res: float = 0.5, eps: float = 1e-10) -> float:
    all_pts = np.concatenate([pred[:, :2], gt[:, :2]], axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    bx = max(int(np.ceil((xmax - xmin) / grid_res)) + 1, 2)
    by = max(int(np.ceil((ymax - ymin) / grid_res)) + 1, 2)
    rng = [[xmin, xmax + grid_res], [ymin, ymax + grid_res]]
    hp, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], bins=[bx, by], range=rng)
    hg, _, _ = np.histogram2d(gt[:, 0],   gt[:, 1],   bins=[bx, by], range=rng)
    pp = hp.flatten() / (hp.sum() + eps)
    pg = hg.flatten() / (hg.sum() + eps)
    pm = 0.5 * (pp + pg)
    kl = lambda a, b: np.sum(a[a > eps] * np.log(a[a > eps] / (b[a > eps] + eps)))
    return float(0.5 * kl(pp, pm) + 0.5 * kl(pg, pm))


def mmd_rbf(pred: np.ndarray, gt: np.ndarray,
            sigma: float = 1.0, max_pts: int = 2048) -> float:
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt   = gt[np.random.choice(gt.shape[0],   max_pts, replace=False)]
    k = lambda x, y: np.exp(-np.sum((x[:, None] - y[None, :]) ** 2, axis=-1) / (2 * sigma ** 2))
    return float(np.sqrt(max(k(pred, pred).mean() + k(gt, gt).mean() - 2 * k(pred, gt).mean(), 0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--radial_root",     required=True, type=str,
                   help="Path to radar_lidar_pc_dataset/")
    p.add_argument("--student_ckpt",    required=True, type=str)
    p.add_argument("--teacher_ckpt",    default=None,  type=str)
    # Ground-removed LiDAR is cached automatically in <radial_root>/laser_PCL_ng/
    p.add_argument("--batch_size",      default=1,     type=int)
    p.add_argument("--threshold",       default=0.5,   type=float)
    p.add_argument("--num_workers",     default=2,     type=int)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = MSDNetConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ids = get_splits(args.radial_root)
    test_ds = RADIalDataset(args.radial_root, test_ids, cfg)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    print(f"Test set: {len(test_ds)} samples")

    # Build model
    shared_recon = None
    if args.teacher_ckpt:
        teacher = MSDNetTeacher(cfg).to(device)
        ck = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ck["model"])
        shared_recon = teacher.reconstruction
        print(f"Teacher loaded: {args.teacher_ckpt}")

    student = MSDNetStudent(cfg, shared_reconstruction=shared_recon).to(device)
    ck = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(ck["model"], strict=False)
    student.eval()
    print(f"Student loaded: {args.student_ckpt}")

    results  = {m: [] for m in ("cd", "mhd", "f1", "jsd", "mmd")}
    runtimes = []
    thresh   = args.threshold
    pcr      = cfg.voxel.point_cloud_range

    for batch in tqdm(test_loader, desc="Evaluating"):
        radar_list = [pc.to(device) for pc in batch["radar"]]
        lidar_np   = [pc.numpy()    for pc in batch["lidar"]]

        t0 = time.perf_counter()
        pred_pcs = student.generate_point_cloud(
            radar_list, len(radar_list), threshold=thresh, point_cloud_range=pcr
        )
        runtimes.append(time.perf_counter() - t0)

        for pred_t, gt_np in zip(pred_pcs, lidar_np):
            pred_np = pred_t.cpu().numpy()
            if pred_np.shape[0] < 2 or gt_np.shape[0] < 2:
                continue
            gt_xyz = gt_np[:, :3]
            results["cd"].append(chamfer_distance(pred_np, gt_xyz))
            results["mhd"].append(modified_hausdorff(pred_np, gt_xyz))
            results["f1"].append(f_score(pred_np, gt_xyz,
                                         threshold=cfg.voxel.voxel_size[0]))
            results["jsd"].append(jsd_bev(pred_np, gt_xyz))
            results["mmd"].append(mmd_rbf(pred_np, gt_xyz))

    print("\n===== Test Results =====")
    for k, v in results.items():
        arr = np.array(v)
        print(f"  {k.upper():>5s}: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")
    print(f"  Latency: {np.mean(runtimes)*1000:.1f} ms / batch "
          f"({len(runtimes)} batches, bs={args.batch_size})")


if __name__ == "__main__":
    main()
