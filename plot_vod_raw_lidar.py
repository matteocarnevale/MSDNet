#!/usr/bin/env python3
"""Visualizza il LiDAR VoD **esattamente come nel file .bin** (nessun preprocessing).

Non usa ``dataset.py``, ``convert_vod_fixed.py`` né crop: solo ``np.fromfile`` + reshape
``(N, 4)`` float32 ``[x, y, z, intensity]`` come da velodyne VoD.

Cerca il frame in percorsi tipici sotto ``--vod_root`` (dataset View-of-Delft non convertito).

Usage:
    python plot_vod_raw_lidar.py --vod_root /path/to/view_of_delft_PUBLIC
    python plot_vod_raw_lidar.py --vod_root ... --frame_id delft_03_xxx
    python plot_vod_raw_lidar.py --bin_path /assoluto/.../file.bin
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np


def find_velodyne_bin(vod_root: Path, frame_id: str) -> Optional[Path]:
    """Percorsi comuni per ``{frame_id}.bin`` LiDAR VoD originale."""
    rel = [
        Path("lidar") / "training" / "velodyne",
        Path("Lidar") / "training" / "velodyne",
        Path("training") / "velodyne",
        Path("velodyne"),
    ]
    for sub in rel:
        p = vod_root / sub / f"{frame_id}.bin"
        if p.is_file():
            return p
    return None


def _set_3d_equal_aspect(ax, xyz: np.ndarray) -> None:
    """Cubo di visualizzazione centrato sui punti (evita assi 3D schiacciati)."""
    if xyz.size == 0:
        return
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    c = (lo + hi) / 2.0
    r = max(float((hi - lo).max()) / 2.0, 0.25)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _draw_range_box_3d(ax, box6: List[float], **kw) -> None:
    """Wireframe del parallelepipedo ``[x0,y0,z0,x1,y1,z1]``."""
    x0, y0, z0, x1, y1, z1 = box6
    edges = (
        ((x0, y0, z0), (x1, y0, z0)),
        ((x0, y1, z0), (x1, y1, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x0, y1, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y1, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x0, y0, z1), (x0, y1, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
    )
    for a, b in edges:
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            **{"color": "red", "lw": 0.9, "alpha": 0.65, **kw},
        )


def load_raw_lidar_bin(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(
            f"{path}: {data.size} float32 non divisibile per 4 — formato non (N,4)?"
        )
    return data.reshape(-1, 4)


def pick_frame_from_split(vod_root: Path, split: str) -> Optional[str]:
    for sub in [
        Path("lidar") / "ImageSets" / f"{split}.txt",
        Path("ImageSets") / f"{split}.txt",
    ]:
        p = vod_root / sub
        if p.is_file():
            ids = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            return random.choice(ids) if ids else None
    return None


def plot_raw(
    pc: np.ndarray,
    out_png: str,
    title: str,
    max_pts: int,
    show_box: Optional[List[float]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("pip install matplotlib") from e

    n = pc.shape[0]
    if n == 0:
        raise SystemExit("Nuvola vuota.")

    rng = np.random.default_rng(0)
    if n > max_pts:
        sel = rng.choice(n, size=max_pts, replace=False)
        plot_pc = pc[sel]
        note = f" (campione {max_pts}/{n})"
    else:
        plot_pc = pc
        note = ""

    xyz = plot_pc[:, :3]
    xi = plot_pc[:, 3]

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=xyz[:, 2],
        cmap="turbo",
        s=0.35,
        alpha=0.55,
        depthshade=False,
        rasterized=True,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D — colore = z" + note)
    ax.view_init(elev=22, azim=-65)
    _set_3d_equal_aspect(ax, xyz)
    plt.colorbar(sc, ax=ax, fraction=0.04, shrink=0.65, label="z")

    if show_box is not None and len(show_box) == 6:
        _draw_range_box_3d(ax, show_box)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc2 = ax2.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=xi,
        cmap="magma",
        s=0.35,
        alpha=0.55,
        depthshade=False,
        rasterized=True,
    )
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")
    ax2.set_title("3D — colore = intensity")
    ax2.view_init(elev=12, azim=55)
    _set_3d_equal_aspect(ax2, xyz)
    plt.colorbar(sc2, ax=ax2, fraction=0.04, shrink=0.65, label="intensity")

    if show_box is not None and len(show_box) == 6:
        _draw_range_box_3d(ax2, show_box)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def print_stats(pc: np.ndarray, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"N = {pc.shape[0]}")
    if pc.shape[0] == 0:
        return
    xyz = pc[:, :3]
    for i, name in enumerate("xyz"):
        col = xyz[:, i]
        print(f"  {name}: min={col.min():.3f}  max={col.max():.3f}  mean={col.mean():.3f}")
    inten = pc[:, 3]
    print(f"  intensity: min={inten.min():.3f}  max={inten.max():.3f}")


def run_raw_preview(
    vod_root: Optional[str] = None,
    bin_path: Optional[str] = None,
    frame_id: Optional[str] = None,
    split: str = "train",
    seed: int = 0,
    out_dir: str = "runs/vod_raw_lidar",
    max_pts: int = 200_000,
    show_training_box: bool = False,
) -> tuple[str, str]:
    """
    Carica un .bin LiDAR senza processing, stampa statistiche, salva PNG + txt.

    Returns:
        (path_png, path_txt)
    """
    random.seed(seed)
    np.random.seed(seed)

    if bin_path:
        path = Path(bin_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File non trovato: {path}")
        frame_label = path.stem
        title = f"RAW LiDAR (solo lettura file)\n{path}"
    else:
        if not vod_root:
            raise ValueError("Serve vod_root oppure bin_path")
        vod_root_p = Path(vod_root).resolve()
        fid = frame_id
        if fid is None:
            fid = pick_frame_from_split(vod_root_p, split)
            if fid is None:
                raise FileNotFoundError(
                    "Impossibile trovare split sotto vod_root; passa frame_id o bin_path"
                )
        path = find_velodyne_bin(vod_root_p, fid)
        if path is None:
            raise FileNotFoundError(
                f"LiDAR non trovato per frame_id={fid!r} sotto {vod_root_p}. Usa --bin_path."
            )
        frame_label = fid
        title = f"RAW LiDAR VoD (nessun processing)\n{frame_label}\n{path}"

    pc = load_raw_lidar_bin(path)
    print_stats(pc, "Grezzo (N,4) float32")

    os.makedirs(out_dir, exist_ok=True)
    safe = frame_label.replace("/", "_").replace("\\", "_")
    out_png = os.path.join(out_dir, f"raw_lidar_{safe}.png")
    box = [0.0, -16.0, -2.0, 32.0, 16.0, 4.0] if show_training_box else None
    plot_raw(pc, out_png, title, max_pts=max_pts, show_box=box)

    txt = os.path.join(out_dir, f"raw_lidar_{safe}_stats.txt")
    with open(txt, "w") as f:
        f.write(f"path: {path}\nN: {pc.shape[0]}\n")
        if pc.shape[0]:
            xyz = pc[:, :3]
            for i, name in enumerate("xyz"):
                col = xyz[:, i]
                f.write(f"{name}: {col.min()} {col.max()} {col.mean()}\n")

    print(f"\nFigura salvata: {out_png}")
    print(f"Statistiche: {txt}")
    return out_png, txt


def main():
    p = argparse.ArgumentParser(description="Plot VoD LiDAR raw from .bin (no preprocessing)")
    p.add_argument("--vod_root", type=str, default=None, help="Root dataset VoD originale (non cartella MSDNet)")
    p.add_argument("--bin_path", type=str, default=None, help="Percorso diretto a un .bin (ignora vod_root per il file)")
    p.add_argument("--frame_id", type=str, default=None)
    p.add_argument("--split", type=str, default="train", help="Se --frame_id omesso, campiona da questo split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs/vod_raw_lidar")
    p.add_argument("--max_pts", type=int, default=200_000)
    p.add_argument(
        "--show_training_box",
        action="store_true",
        help="Sovrapponi il rettangolo xy di training [0,-16,-2,32,16,4] (solo riferimento, non filtra)",
    )
    args = p.parse_args()
    if not args.vod_root and not args.bin_path:
        p.error("Serve --vod_root oppure --bin_path")
    run_raw_preview(
        vod_root=args.vod_root,
        bin_path=args.bin_path,
        frame_id=args.frame_id,
        split=args.split,
        seed=args.seed,
        out_dir=args.out_dir,
        max_pts=args.max_pts,
        show_training_box=args.show_training_box,
    )


if __name__ == "__main__":
    main()
