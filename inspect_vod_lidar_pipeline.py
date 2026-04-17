"""Ispezione visiva della pipeline LiDAR VoD: dal .bin grezzo al tensore usato dal dataset.

Usa gli stessi metodi di ``dataset.VoDDataset`` (ground removal → FoV → range) e salva una
figura **3D** (scatter x,y,z) per ogni stadio, con wireframe del ``point_cloud_range``.

Usage:
    python inspect_vod_lidar_pipeline.py --data_root /path/to/vod
    python inspect_vod_lidar_pipeline.py --data_root /path/to/vod --frame_id 000123
    python inspect_vod_lidar_pipeline.py --data_root /path/to/vod --bin_path /path/to/lidar/xxx.bin
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np

from config import MSDNetConfig
from dataset import VoDDataset
from plot_vod_raw_lidar import _draw_range_box_3d, _set_3d_equal_aspect


def load_lidar_bin(path: str) -> np.ndarray:
    """Carica un file VoD LiDAR ``(N,4)`` float32 ``x,y,z,intensity``."""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(
            f"{path}: dimensione file non multipla di 4 ({arr.size} float32)"
        )
    return arr.reshape(-1, 4)


def lidar_preprocessing_stages(ds: VoDDataset, raw: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Replica esattamente ``_preprocess_lidar`` con checkpoint intermedi.

    Chiavi: ``raw``, ``after_ground``, ``after_fov``, ``after_range`` (output del dataset).
    """
    if raw.shape[0] == 0:
        empty = raw.copy()
        return {
            "raw": empty,
            "after_ground": empty,
            "after_fov": empty,
            "after_range": empty,
        }
    g = ds._ground_removal_elevation_map(raw, grid_size=0.5, height_threshold=0.3)
    f = ds._crop_to_fov(g)
    r = ds._crop_to_range(f)
    return {
        "raw": raw.copy(),
        "after_ground": g,
        "after_fov": f,
        "after_range": r,
    }


def _subsample(pc: np.ndarray, max_pts: int) -> np.ndarray:
    n = pc.shape[0]
    if n <= max_pts:
        return pc
    idx = np.random.choice(n, max_pts, replace=False)
    return pc[idx]


def n_pts(pc: np.ndarray) -> int:
    return int(pc.shape[0]) if pc is not None else 0


def plot_stages_3d(
    stages: Dict[str, np.ndarray],
    out_png: str,
    frame_label: str,
    fov_deg: float,
    max_pts: int,
    point_cloud_range: list,
) -> None:
    """Figura 1×4: scatter 3D (x,y,z) colorato per z; wireframe rosso = ``point_cloud_range``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Serve matplotlib: pip install matplotlib") from e

    order = ["raw", "after_ground", "after_fov", "after_range"]
    title_map = {
        "raw": "Grezzo da .bin",
        "after_ground": "Dopo ground removal",
        "after_fov": f"Dopo crop FoV ({fov_deg:.0f}°)",
        "after_range": "Dopo crop range (dataset)",
    }
    n = len(order)
    fig = plt.figure(figsize=(4.2 * n, 4.8))
    views = [(22, -65), (18, 25), (12, 110), (8, -120)]
    for i, key in enumerate(order):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        pc = stages.get(key)
        if pc is None or pc.shape[0] == 0:
            ax.set_title(f"{title_map[key]}\n0 punti")
            continue
        s = _subsample(pc, max_pts)
        xyz = s[:, :3]
        sc = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=xyz[:, 2],
            cmap="viridis",
            s=0.2,
            alpha=0.5,
            depthshade=False,
            rasterized=True,
        )
        elev, azim = views[i % len(views)]
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        _set_3d_equal_aspect(ax, xyz)
        _draw_range_box_3d(ax, point_cloud_range)
        plt.colorbar(sc, ax=ax, fraction=0.04, shrink=0.7, label="z [m]")
        ax.set_title(f"{title_map[key]}\n{n_pts(pc)} pt (plot {s.shape[0]})")
    fig.suptitle(f"VoD LiDAR — 3D\n{frame_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def print_stats(name: str, pc: np.ndarray) -> None:
    if pc is None or pc.shape[0] == 0:
        print(f"  [{name}] 0 punti")
        return
    xyz = pc[:, :3]
    print(
        f"  [{name}] N={pc.shape[0]:7d}  "
        f"x∈[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}]  "
        f"y∈[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}]  "
        f"z∈[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]"
    )


def main():
    p = argparse.ArgumentParser(description="Plot VoD LiDAR raw vs preprocessing stages")
    p.add_argument("--data_root", type=str, required=True, help="Root VoD (lidar/, radar/, split/)")
    p.add_argument("--split", type=str, default="train", choices=("train", "test"))
    p.add_argument("--frame_id", type=str, default=None, help="Nome file senza .bin (es. 000000)")
    p.add_argument(
        "--bin_path",
        type=str,
        default=None,
        help="Percorso assoluto a un lidar .bin (usa questo frame; --data_root serve per config)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs/vod_lidar_inspect")
    p.add_argument("--max_pts_plot", type=int, default=120_000)
    p.add_argument("--vod_sequence_filter", type=str, default="none", choices=("none", "4drvo_net"))
    args = p.parse_args()

    np.random.seed(args.seed)
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

    if args.bin_path:
        lidar_path = os.path.abspath(args.bin_path)
        if not os.path.isfile(lidar_path):
            raise SystemExit(f"File non trovato: {lidar_path}")
        frame_id = os.path.splitext(os.path.basename(lidar_path))[0]
        raw = load_lidar_bin(lidar_path)
    else:
        if len(ds.frame_ids) == 0:
            raise SystemExit("Nessun frame nel dataset (split vuoto o file mancanti).")
        if args.frame_id is not None:
            frame_id = args.frame_id
            if frame_id not in ds.frame_ids:
                raise SystemExit(
                    f"frame_id {frame_id!r} non nella lista valida dello split "
                    f"(primi es.: {ds.frame_ids[:5]})"
                )
        else:
            frame_id = ds.frame_ids[np.random.randint(0, len(ds.frame_ids))]
        lidar_path = os.path.join(ds.root, "lidar", f"{frame_id}.bin")
        raw = load_lidar_bin(lidar_path)

    stages = lidar_preprocessing_stages(ds, raw)
    os.makedirs(args.out_dir, exist_ok=True)
    safe = frame_id.replace("/", "_").replace("\\", "_")
    out_png = os.path.join(args.out_dir, f"vod_lidar_3d_{safe}.png")
    summary_path = os.path.join(args.out_dir, f"vod_lidar_{safe}_stats.txt")

    print(f"File: {lidar_path}")
    print(f"frame_id: {frame_id}")
    print(f"point_cloud_range (config): {cfg.voxel.point_cloud_range}")
    print(f"radar_fov_deg: {ds.radar_fov_deg}")
    print("Conteggi / bounding box per stadio:")
    for k in ["raw", "after_ground", "after_fov", "after_range"]:
        print_stats(k, stages[k])

    plot_stages_3d(
        stages,
        out_png,
        frame_label=f"{frame_id}\n{lidar_path}",
        fov_deg=ds.radar_fov_deg,
        max_pts=args.max_pts_plot,
        point_cloud_range=cfg.voxel.point_cloud_range,
    )

    with open(summary_path, "w") as f:
        f.write(f"path: {lidar_path}\n")
        f.write(f"frame_id: {frame_id}\n")
        f.write(f"point_cloud_range: {cfg.voxel.point_cloud_range}\n")
        f.write(f"radar_fov_deg: {ds.radar_fov_deg}\n")
        for k in ["raw", "after_ground", "after_fov", "after_range"]:
            pc = stages[k]
            f.write(f"\n[{k}] N={pc.shape[0]}\n")
            if pc.shape[0] > 0:
                xyz = pc[:, :3]
                f.write(
                    f"  x min/max: {xyz[:,0].min():.4f} / {xyz[:,0].max():.4f}\n"
                    f"  y min/max: {xyz[:,1].min():.4f} / {xyz[:,1].max():.4f}\n"
                    f"  z min/max: {xyz[:,2].min():.4f} / {xyz[:,2].max():.4f}\n"
                )

    print(f"\nFigura 3D salvata: {out_png}")
    print(f"Statistiche: {summary_path}")


if __name__ == "__main__":
    main()
