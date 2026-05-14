"""MSDNet — visualisation for RADIal.

Plots a random test frame: LiDAR input, radar input, teacher and/or student
reconstructed point cloud.

Usage:
    # Teacher only (random weights or checkpoint)
    python visualize.py --radial_root /data/RADIal --mode teacher

    # Student inference
    python visualize.py --radial_root /data/RADIal --mode student \\
        --student_ckpt checkpoints/student/best.pth \\
        --teacher_ckpt checkpoints/teacher/best.pth

    # Both teacher + student side-by-side
    python visualize.py --radial_root /data/RADIal --mode both \\
        --teacher_ckpt checkpoints/teacher/best.pth \\
        --student_ckpt checkpoints/student/best.pth
"""

from __future__ import annotations

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import MSDNetConfig
from dataset import RADIalDataset, get_splits, collate_fn
from models.msdnet import MSDNetTeacher, MSDNetStudent


def _decode_occ(occ_logits, offset, pc_range, occ_thresh=0.5, max_pts=80_000):
    """(1,1,Z,Y,X) logits + (1,3,Z,Y,X) offset → (N, 3) numpy."""
    prob = torch.sigmoid(occ_logits[0, 0])
    mask = prob >= occ_thresh
    idx  = mask.nonzero(as_tuple=False).float()   # (N, 3): z,y,x indices
    if idx.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    _, _, Z, H, W = occ_logits.shape
    xmin, ymin, zmin = pc_range[:3]
    xmax, ymax, zmax = pc_range[3:]
    vx = (xmax - xmin) / W
    vy = (ymax - ymin) / H
    vz = (zmax - zmin) / Z

    zi, hi, wi = idx[:, 0].long(), idx[:, 1].long(), idx[:, 2].long()
    cx = xmin + (wi.float() + 0.5) * vx
    cy = ymin + (hi.float() + 0.5) * vy
    cz = zmin + (zi.float() + 0.5) * vz

    off = offset[0, :, zi, hi, wi].T          # (N, 3): dx,dy,dz
    pts = torch.stack([cx + off[:, 0],
                       cy + off[:, 1],
                       cz + off[:, 2]], dim=1)
    pts = pts.cpu().numpy()

    if pts.shape[0] > max_pts:
        pts = pts[np.random.choice(pts.shape[0], max_pts, replace=False)]
    return pts.astype(np.float32)


def _scatter3d(ax, pts, color="viridis", title="", box=None, elev=20, azim=-60):
    if pts.shape[0] > 0:
        c = pts[:, 2] if color == "z" else pts[:, 3] if pts.shape[1] > 3 else pts[:, 2]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=c, cmap="viridis", s=0.4, alpha=0.6, depthshade=False)
        # draw bounding box
        if box is not None:
            xs, ys, zs = [box[0], box[3]], [box[1], box[4]], [box[2], box[5]]
            for xi in xs:
                for yi in ys:
                    ax.plot([xi, xi], [yi, yi], zs, "gray", lw=0.4, alpha=0.5)
            for zi in zs:
                for xi in xs:
                    ax.plot([xi, xi], ys, [zi, zi], "gray", lw=0.4, alpha=0.5)
                for yi in ys:
                    ax.plot(xs, [yi, yi], [zi, zi], "gray", lw=0.4, alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radial_root",     required=True, type=str,
                   help="Path to radar_lidar_pc_dataset/")
    p.add_argument("--mode",            default="both",
                   choices=["teacher", "student", "both"])
    p.add_argument("--teacher_ckpt",    default=None, type=str)
    p.add_argument("--student_ckpt",    default=None, type=str)
    # Ground-removed LiDAR auto-cached in <radial_root>/laser_PCL_ng/
    p.add_argument("--out_dir",         default="runs/viz", type=str)
    p.add_argument("--num_samples",  default=2,    type=int)
    p.add_argument("--seed",         default=42,   type=int)
    p.add_argument("--occ_thresh",   default=0.5,  type=float)
    p.add_argument("--split",        default="test",
                   choices=["train", "val", "test"])
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg    = MSDNetConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids, test_ids = get_splits(args.radial_root)
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]

    ds = RADIalDataset(args.radial_root, split_ids, cfg)

    teacher = student = None

    if args.mode in ("teacher", "both"):
        teacher = MSDNetTeacher(cfg).to(device)
        if args.teacher_ckpt:
            ck = torch.load(args.teacher_ckpt, map_location=device)
            teacher.load_state_dict(ck["model"])
        teacher.eval()

    if args.mode in ("student", "both"):
        if args.student_ckpt is None:
            raise ValueError("--student_ckpt required for mode=student/both")
        shared_recon = teacher.reconstruction if teacher else None
        student = MSDNetStudent(cfg, shared_reconstruction=shared_recon).to(device)
        ck = torch.load(args.student_ckpt, map_location=device)
        student.load_state_dict(ck["model"], strict=False)
        student.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    pcr   = list(cfg.voxel.point_cloud_range)
    ncols = 2 + (1 if teacher else 0) + (1 if student else 0)

    for k in range(args.num_samples):
        idx    = random.randint(0, len(ds) - 1)
        sample = ds[idx]
        fid    = sample["frame_id"]
        lidar  = sample["lidar"]   # (M, 4)
        radar  = sample["radar"]   # (N, 5)

        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5),
                                 subplot_kw={"projection": "3d"})
        axes = list(axes)
        col  = 0

        lidar_np = lidar.numpy()
        radar_np = radar.numpy()

        _scatter3d(axes[col], lidar_np, title=f"LiDAR  ({lidar_np.shape[0]} pt)",
                   box=pcr)
        col += 1
        _scatter3d(axes[col], radar_np,
                   title=f"Radar  ({radar_np.shape[0]} pt)\nvel=ch4", box=pcr)
        col += 1

        with torch.no_grad():
            if teacher is not None:
                _, recon_t = teacher([lidar.to(device)], batch_size=1)
                pts_t = _decode_occ(recon_t["occ_1"], recon_t["offset_1"],
                                    pcr, args.occ_thresh)
                _scatter3d(axes[col],
                           pts_t if pts_t.shape[0] > 0 else np.zeros((1, 3)),
                           title=f"Teacher recon  ({pts_t.shape[0]} pt)", box=pcr)
                col += 1

            if student is not None:
                out_s = student([radar.to(device)], batch_size=1, training=False)
                pts_s = _decode_occ(out_s["recon_out"]["occ_1"],
                                    out_s["recon_out"]["offset_1"],
                                    pcr, args.occ_thresh)
                _scatter3d(axes[col],
                           pts_s if pts_s.shape[0] > 0 else np.zeros((1, 3)),
                           title=f"Student recon  ({pts_s.shape[0]} pt)", box=pcr)

        fig.suptitle(f"MSDNet RADIal — {fid}", fontsize=10)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{fid}_k{k:02d}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[{k+1}/{args.num_samples}] Saved {out}")


if __name__ == "__main__":
    main()
