"""Evaluation script: reconstruct dense point clouds and compute metrics.

Metrics (Section IV-C):
    - Chamfer Distance (CD)
    - Modified Hausdorff Distance (MHD)
    - F-score (threshold = voxel size)
    - Jensen–Shannon Discrepancy (JSD) on BEV histograms
    - Maximum Mean Discrepancy (MMD)

Usage:
    python evaluate.py --data_root /path/to/vod \
        --student_ckpt checkpoints/student/student_final.pth
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial import KDTree
from tqdm import tqdm

from config import MSDNetConfig
from dataset import VoDDataset, collate_fn
from models.msdnet import MSDNetTeacher, MSDNetStudent


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)
    d_pred, _ = tree_gt.query(pred)
    d_gt, _ = tree_pred.query(gt)
    return float(d_pred.mean() + d_gt.mean())


def modified_hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)
    d_pred, _ = tree_gt.query(pred)
    d_gt, _ = tree_pred.query(gt)
    return float(max(d_pred.mean(), d_gt.mean()))


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.1) -> float:
    tree_pred = KDTree(pred)
    tree_gt = KDTree(gt)
    d_pred, _ = tree_gt.query(pred)
    d_gt, _ = tree_pred.query(gt)
    precision = (d_pred < threshold).mean()
    recall = (d_gt < threshold).mean()
    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def jsd_bev(pred: np.ndarray, gt: np.ndarray,
            grid_res: float = 0.5, eps: float = 1e-10) -> float:
    """Jensen-Shannon Discrepancy on BEV 2D histograms (x, y)."""
    all_pts = np.concatenate([pred[:, :2], gt[:, :2]], axis=0)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    bins_x = int(np.ceil((x_max - x_min) / grid_res)) + 1
    bins_y = int(np.ceil((y_max - y_min) / grid_res)) + 1
    hist_range = [[x_min, x_max + grid_res], [y_min, y_max + grid_res]]

    h_pred, _, _ = np.histogram2d(pred[:, 0], pred[:, 1],
                                  bins=[bins_x, bins_y], range=hist_range)
    h_gt, _, _ = np.histogram2d(gt[:, 0], gt[:, 1],
                                bins=[bins_x, bins_y], range=hist_range)

    prob_pred = h_pred.flatten() / (h_pred.sum() + eps)
    prob_gt = h_gt.flatten() / (h_gt.sum() + eps)
    prob_mix = 0.5 * (prob_pred + prob_gt)

    def _kl(a, b):
        mask = a > eps
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps)))

    return float(0.5 * _kl(prob_pred, prob_mix) + 0.5 * _kl(prob_gt, prob_mix))


def mmd_rbf(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel (sub-sampled)."""
    max_samples = 2048
    if pred.shape[0] > max_samples:
        idx = np.random.choice(pred.shape[0], max_samples, replace=False)
        pred = pred[idx]
    if gt.shape[0] > max_samples:
        idx = np.random.choice(gt.shape[0], max_samples, replace=False)
        gt = gt[idx]

    def _k(x, y):
        d = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        return np.exp(-d / (2 * sigma ** 2))

    return float(np.sqrt(
        max(_k(pred, pred).mean() + _k(gt, gt).mean() - 2 * _k(pred, gt).mean(), 0)
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MSDNet — evaluate")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, default=None,
                   help="Teacher checkpoint (needed to build shared recon head)")
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Occupancy threshold for point generation")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MSDNetConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    test_ds = VoDDataset(
        args.data_root, "test",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2,
        collate_fn=collate_fn,
    )

    # Model
    if args.teacher_ckpt:
        teacher = MSDNetTeacher(cfg).to(device)
        teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
        if 'model_state_dict' in teacher_ckpt:
            teacher.load_state_dict(teacher_ckpt['model_state_dict'])
        else:
            teacher.load_state_dict(teacher_ckpt)
        shared_recon = teacher.reconstruction
    else:
        shared_recon = None

    student = MSDNetStudent(cfg, shared_reconstruction=shared_recon).to(device)
    student_ckpt = torch.load(args.student_ckpt, map_location=device)
    if 'model_state_dict' in student_ckpt:
        student.load_state_dict(student_ckpt['model_state_dict'], strict=False)
    else:
        student.load_state_dict(student_ckpt, strict=False)
    student.eval()

    # Evaluate
    results = {"cd": [], "mhd": [], "fscore": [], "jsd": [], "mmd": []}
    runtimes = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        radar_list = [pc.to(device) for pc in batch["radar"]]
        lidar_list = [pc.numpy() for pc in batch["lidar"]]

        start_time = time.time()
        pred_pcs = student.generate_point_cloud(
            radar_list, args.batch_size,
            threshold=args.threshold,
            point_cloud_range=cfg.voxel.point_cloud_range,
        )
        runtimes.append(time.time() - start_time)

        for pred_t, gt_np in zip(pred_pcs, lidar_list):
            pred_np = pred_t.cpu().numpy()
            if pred_np.shape[0] < 2 or gt_np.shape[0] < 2:
                continue
            gt_xyz = gt_np[:, :3]

            results["cd"].append(chamfer_distance(pred_np, gt_xyz))
            results["mhd"].append(modified_hausdorff(pred_np, gt_xyz))
            results["fscore"].append(f_score(pred_np, gt_xyz,
                                             threshold=cfg.voxel.voxel_size[0]))
            results["jsd"].append(jsd_bev(pred_np, gt_xyz))
            results["mmd"].append(mmd_rbf(pred_np, gt_xyz))

    print("\n===== Results =====")
    for k, v in results.items():
        arr = np.array(v)
        print(f"  {k.upper():>8s}: {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"  {'Runtime':>8s}: {np.mean(runtimes):.3f}s  (per batch)")
    print(f"  {'Samples':>8s}: {len(results['cd'])}")


if __name__ == "__main__":
    main()
