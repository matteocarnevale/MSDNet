#!/usr/bin/env python3
"""VoD + MSDNet — due passi: anteprima grezza, poi ordine chiaro del ground removal.

Comandi:
    python vod_pipeline.py preview --vod_root /path/to/view_of_delft_PUBLIC
    python vod_pipeline.py preview --bin_path /path/to/xxx.bin
    python vod_pipeline.py pipeline

``preview`` = primo controllo dati (zero processing). ``pipeline`` = dove va il ground.
"""

from __future__ import annotations

import argparse

PIPELINE_IT = """
================================================================================
VoD → MSDNet  |  Ground removal: dove metterlo (scegli un percorso semplice)
================================================================================

1) PRIMO PASSO — controlla il grezzo (nessun processing)
      python vod_pipeline.py preview --vod_root .../view_of_delft_PUBLIC
   oppure
      python vod_pipeline.py preview --bin_path .../velodyne/xxx.bin

2) CONVERSIONE VoD → cartella MSDNet (opzionale)
      python convert_vod_fixed.py --vod_root ... --output_dir data/vod
   • --lidar_raw          → nessun ground nel converter; il suolo lo toglie SOLO il dataset.
   • --lidar_ground elevation (default) → ground nel converter, poi crop.
   Consiglio più lineare: --lidar_raw + ground solo nel training (punto 3).

3) TRAINING — VoDDataset.__getitem__ applica SEMPRE, in ordine:
      ground (elevation) → crop FoV radar → crop point_cloud_range → GT voxel
   Anche se i .bin sono già “puliti”, il ground step è idempotente nella pratica.

================================================================================
"""


def cmd_pipeline() -> None:
    print(PIPELINE_IT)


def cmd_preview(ns: argparse.Namespace) -> None:
    from plot_vod_raw_lidar import run_raw_preview

    run_raw_preview(
        vod_root=ns.vod_root,
        bin_path=ns.bin_path,
        frame_id=ns.frame_id,
        split=ns.split,
        seed=ns.seed,
        out_dir=ns.out_dir,
        max_pts=ns.max_pts,
        show_training_box=ns.show_training_box,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="vod_pipeline.py",
        description="VoD/MSDNet: anteprima grezza (preview) e ordine pipeline (pipeline).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pipe = sub.add_parser("pipeline", help="Stampa dove inserire il ground removal")

    p_prev = sub.add_parser("preview", help="Primo display: .bin senza alcun processing")
    p_prev.add_argument("--vod_root", type=str, default=None)
    p_prev.add_argument("--bin_path", type=str, default=None)
    p_prev.add_argument("--frame_id", type=str, default=None)
    p_prev.add_argument("--split", type=str, default="train")
    p_prev.add_argument("--seed", type=int, default=0)
    p_prev.add_argument("--out_dir", type=str, default="runs/vod_preview")
    p_prev.add_argument("--max_pts", type=int, default=200_000)
    p_prev.add_argument(
        "--show_training_box",
        action="store_true",
        help="Rettangolo xy del range MSDNet (solo guida visiva)",
    )

    args = p.parse_args()
    if args.cmd == "pipeline":
        cmd_pipeline()
    elif args.cmd == "preview":
        if not args.vod_root and not args.bin_path:
            p_prev.error("preview richiede --vod_root oppure --bin_path")
        cmd_preview(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
