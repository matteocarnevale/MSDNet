"""Visualizza LiDAR di un frame casuale dal VoDDataset e l'output del teacher (senza training).

Carica un .bin dalla split (stesso preprocessing di ``dataset.VoDDataset``), istanzia
``MSDNetTeacher`` con pesi casuali o da checkpoint, esegue un forward e salva figure PNG:
  - nuvola in input (dopo preprocess)
  - heatmap BEV delle feature dense (media sui canali)
  - massima proiezione sull'asse Z della prob. di occupazione (scala 1)
  - punti ricostruiti dai voxel occupati (sigmoid > soglia) a scala 1

Usage:
    python visualize_teacher_lidar_sample.py --data_root /path/to/vod
    python visualize_teacher_lidar_sample.py --data_root /path/to/vod --teacher_ckpt checkpoints/teacher/teacher_best.pth
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from config import MSDNetConfig
from dataset import VoDDataset
from models.msdnet import MSDNetTeacher
from plot_vod_raw_lidar import _draw_range_box_3d, _set_3d_equal_aspect


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _decode_points_from_finest(
    occ_logits: torch.Tensor,
    offset: torch.Tensor,
    pc_min: np.ndarray,
    voxel_size: np.ndarray,
    occ_thresh: float = 0.5,
    max_points: int = 80000,
) -> np.ndarray:
    """occ_logits: (1,1,Z,Y,X), offset: (1,3,Z,Y,X). Ritorna (N,3) numpy."""
    prob = torch.sigmoid(occ_logits[0, 0])
    device = prob.device
    mask = prob >= occ_thresh
    zi, yi, xi = torch.nonzero(mask, as_tuple=True)
    if zi.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    off = offset[0]
    pc_min_t = torch.tensor(pc_min, device=device, dtype=torch.float32)
    vs = torch.tensor(voxel_size, device=device, dtype=torch.float32)
    xi_f = xi.float()
    yi_f = yi.float()
    zi_f = zi.float()
    center = torch.stack(
        [
            pc_min_t[0] + (xi_f + 0.5) * vs[0],
            pc_min_t[1] + (yi_f + 0.5) * vs[1],
            pc_min_t[2] + (zi_f + 0.5) * vs[2],
        ],
        dim=1,
    )
    off_sel = off[:, zi, yi, xi].T
    pts = (center + off_sel).detach().cpu().numpy()
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]
    return pts.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="MSDNet — visualizza LiDAR + forward teacher")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=("train", "test"))
    p.add_argument("--out_dir", type=str, default="runs/teacher_lidar_viz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher_ckpt", type=str, default=None, help="Checkpoint teacher (opzionale)")
    p.add_argument("--vod_sequence_filter", type=str, default="none", choices=("none", "4drvo_net"))
    p.add_argument("--occ_thresh", type=float, default=0.5)
    args = p.parse_args()

    _set_seed(args.seed)
    cfg = MSDNetConfig()
    vod_f = None if args.vod_sequence_filter == "none" else args.vod_sequence_filter

    ds = VoDDataset(
        args.data_root,
        args.split,
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=True,
        vod_sequence_filter=vod_f,
    )
    if len(ds) == 0:
        raise SystemExit("Dataset vuoto: controlla data_root e split.")

    idx = random.randint(0, len(ds) - 1)
    sample = ds[idx]
    frame_id = sample["frame_id"]
    lidar = sample["lidar"]
    n_pts = lidar.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = MSDNetTeacher(cfg).to(device)
    if args.teacher_ckpt:
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        teacher.load_state_dict(state, strict=True)
        weights_tag = f"checkpoint: {os.path.basename(args.teacher_ckpt)}"
    else:
        weights_tag = "pesi casuali (nessun training)"

    teacher.eval()
    with torch.no_grad():
        lidar_list = [lidar.to(device)]
        f_dense, recon_out = teacher(lidar_list, batch_size=1)

    pc_min = np.array(cfg.voxel.point_cloud_range[:3], dtype=np.float32)
    vs = np.array(cfg.voxel.voxel_size, dtype=np.float32)

    occ1 = recon_out["occ_1"]
    off1 = recon_out["offset_1"]
    pred_pts = _decode_points_from_finest(occ1, off1, pc_min, vs, occ_thresh=args.occ_thresh)

    box6 = list(cfg.voxel.point_cloud_range)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"teacher_viz_{frame_id.replace('/', '_')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Installa matplotlib per generare le figure: pip install matplotlib") from e

    lidar_np = lidar.numpy()
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    if n_pts > 0:
        step = max(1, n_pts // 50000)
        sl = slice(None, None, step)
        xyz_in = lidar_np[sl, :3]
        ax1.scatter(
            xyz_in[:, 0],
            xyz_in[:, 1],
            xyz_in[:, 2],
            c=lidar_np[sl, 3],
            cmap="viridis",
            s=0.35,
            alpha=0.6,
            depthshade=False,
        )
        _set_3d_equal_aspect(ax1, xyz_in)
        _draw_range_box_3d(ax1, box6)
    ax1.set_title(f"Input LiDAR (preprocess)\n{frame_id}  ({n_pts} pt)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=20, azim=-60)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    if n_pts > 0:
        step = max(1, n_pts // 50000)
        sl = slice(None, None, step)
        xyz2 = lidar_np[sl, :3]
        ax2.scatter(
            xyz2[:, 0],
            xyz2[:, 1],
            xyz2[:, 2],
            c=lidar_np[sl, 3],
            cmap="viridis",
            s=0.35,
            alpha=0.6,
            depthshade=False,
        )
        _set_3d_equal_aspect(ax2, xyz2)
        _draw_range_box_3d(ax2, box6)
    ax2.set_title(f"Input LiDAR — altra vista\n{weights_tag}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.view_init(elev=8, azim=85)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    npred = pred_pts.shape[0]
    if npred > 0:
        st = max(1, npred // 50000)
        pr = pred_pts[::st]
        ax3.scatter(pr[:, 0], pr[:, 1], pr[:, 2], s=0.45, c="coral", alpha=0.55, depthshade=False)
        _set_3d_equal_aspect(ax3, pr)
        _draw_range_box_3d(ax3, box6)
    ax3.set_title(f"Ricostruzione occ/offset (τ={args.occ_thresh})\n{npred} pt")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=22, azim=-65)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    if npred > 0:
        st = max(1, npred // 50000)
        pr = pred_pts[::st]
        ax4.scatter(pr[:, 0], pr[:, 1], pr[:, 2], s=0.45, c="coral", alpha=0.55, depthshade=False)
        _set_3d_equal_aspect(ax4, pr)
        _draw_range_box_3d(ax4, box6)
    ax4.set_title("Ricostruzione — altra vista")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")
    ax4.view_init(elev=10, azim=70)

    fig.suptitle("MSDNet teacher — forward su un frame casuale", fontsize=12)
    fig.tight_layout()
    out_png = base + ".png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Testo riepilogo
    summary_path = base + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"frame_id: {frame_id}\n")
        f.write(f"dataset index: {idx}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"lidar points (after preprocess): {n_pts}\n")
        f.write(f"weights: {weights_tag}\n")
        f.write(f"f_dense shape: {tuple(f_dense.shape)}\n")
        f.write(f"occ_1 shape: {tuple(occ1.shape)}\n")
        f.write(f"decoded points (thresh={args.occ_thresh}): {npred}\n")
        f.write(f"figure: {out_png}\n")

    print(f"Frame casuale: {frame_id} (index {idx})")
    print(weights_tag)
    print(f"Figure salvate in: {out_png}")
    print(f"Riepilogo: {summary_path}")


if __name__ == "__main__":
    main()
