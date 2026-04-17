#!/usr/bin/env python3
"""
Converter VoD → layout MSDNet (``lidar/``, ``radar/``, ``split/``).

**LiDAR:** con ``--lidar_ground elevation`` (consigliato) usa la stessa rimozione suolo di
``dataset.remove_ground_elevation_map`` (come in training). Con ``simple`` solo ``z > -1.5``.
Con ``--lidar_raw`` nessun filtro suolo. Poi (se non raw) si applica il crop ``point_cloud_range``.

**Radar:** normalizza il numero di colonne e applica il crop di range.

Ordine consigliato (anteprima grezza + dove va il ground): ``python vod_pipeline.py pipeline``.
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dataset import remove_ground_elevation_map


def apply_range_clipping(points, point_cloud_range, min_points=10):
    """Apply range clipping to keep points within voxelization range."""
    pc_min = np.array(point_cloud_range[:3])
    pc_max = np.array(point_cloud_range[3:])
    
    mask = ((points[:, :3] >= pc_min) & (points[:, :3] < pc_max)).all(axis=1)
    clipped = points[mask]
    
    return clipped if clipped.shape[0] >= min_points else None


def convert_vod_fixed(
    vod_root,
    output_dir,
    radar_type="radar_5frames",
    lidar_raw: bool = False,
    lidar_ground: str = "elevation",
    ground_height: float = -1.5,
    point_cloud_range=None,
):
    """
    Args:
        lidar_raw: se True, scrive il LiDAR velodyne così com'è (solo reshape N×4), senza
            filtro suolo né range clipping.
        lidar_ground: ``elevation`` | ``simple`` — ignorato se ``lidar_raw``.
            elevation = mappa di elevazione (come ``VoDDataset``); simple = solo ``z > -1.5``.
        ground_height: usato da ``remove_ground_elevation_map`` (default come il dataset).
        point_cloud_range: lista 6 float ``[x0,y0,z0,x1,y1,z1]``; default paper/MSDNet.
    """
    if point_cloud_range is None:
        point_cloud_range = [0, -16, -2, 32, 16, 4]
    vod_path = Path(vod_root)
    out_path = Path(output_dir)
    
    print("VoD → MSDNet layout")
    print(f"point_cloud_range: {point_cloud_range}")
    if lidar_raw:
        print("LiDAR: RAW (nessun ground removal, nessun crop)")
    else:
        print(f"LiDAR ground: {lidar_ground}  (ground_height={ground_height})")
    print(f"Source: {vod_path}")
    print(f"Output: {output_dir}")
    
    # Create structure
    (out_path / 'lidar').mkdir(parents=True, exist_ok=True)
    (out_path / 'radar').mkdir(parents=True, exist_ok=True)
    (out_path / 'split').mkdir(parents=True, exist_ok=True)
    
    # Copy splits
    lidar_dir = vod_path / 'lidar'
    imagesets = lidar_dir / 'ImageSets'
    
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        src = imagesets / split_name
        dst = out_path / 'split' / split_name
        if src.exists():
            shutil.copy(src, dst)
    
    # Get all frame IDs
    all_frame_ids = set()
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f if line.strip()]
                all_frame_ids.update(frame_ids)
    
    all_frame_ids = sorted(all_frame_ids)
    
    # Process LiDAR
    if lidar_raw:
        print("Processing LiDAR (RAW)...")
    elif lidar_ground == "elevation":
        print("Processing LiDAR (elevation-map ground removal + range clip)...")
    else:
        print("Processing LiDAR (simple z threshold + range clip)...")
    lidar_velodyne = lidar_dir / 'training' / 'velodyne'
    lidar_success = 0
    
    for frame_id in tqdm(all_frame_ids, desc="LiDAR"):
        src_file = lidar_velodyne / f'{frame_id}.bin'
        if not src_file.exists():
            continue
            
        try:
            data = np.fromfile(src_file, dtype=np.float32)
            if data.size % 4 == 0:
                points = data.reshape(-1, 4)

                if lidar_raw:
                    out_pts = points
                else:
                    if lidar_ground == "elevation":
                        points = remove_ground_elevation_map(
                            points,
                            ground_height=ground_height,
                            grid_size=0.5,
                            height_threshold=0.3,
                        )
                    else:
                        points = points[points[:, 2] > ground_height]
                    out_pts = apply_range_clipping(
                        points, point_cloud_range, min_points=20,
                    )

                if out_pts is not None and out_pts.shape[0] > 0:
                    dst_file = out_path / 'lidar' / f'{frame_id}.bin'
                    out_pts.astype(np.float32).tofile(dst_file)
                    lidar_success += 1
                    
        except Exception:
            pass
    
    # Process radar with RANGE CLIPPING (this was missing!)
    print("Processing radar (format handling + range clipping)...")
    radar_dir_path = vod_path / radar_type
    radar_velodyne = radar_dir_path / 'training' / 'velodyne'
    radar_success = 0
    
    for frame_id in tqdm(all_frame_ids, desc="Radar"):
        src_file = radar_velodyne / f'{frame_id}.bin'
        if not src_file.exists():
            continue
            
        try:
            data = np.fromfile(src_file, dtype=np.float32)
            points = None
            
            # Handle your format distribution
            if data.size % 7 == 0:
                points = data.reshape(-1, 7)[:, :5]
            elif data.size % 5 == 0:
                points = data.reshape(-1, 5)
            elif data.size % 4 == 0:
                temp = data.reshape(-1, 4)
                velocity = np.zeros((temp.shape[0], 1), dtype=np.float32)
                points = np.hstack([temp, velocity])
            elif data.size % 3 == 0:
                temp = data.reshape(-1, 3)
                intensity = np.ones((temp.shape[0], 1), dtype=np.float32)
                velocity = np.zeros((temp.shape[0], 1), dtype=np.float32)
                points = np.hstack([temp, intensity, velocity])
            
            # CRITICAL: Apply same range clipping as LiDAR
            if points is not None:
                points = apply_range_clipping(points, point_cloud_range, min_points=10)
                
                if points is not None:
                    dst_file = out_path / 'radar' / f'{frame_id}.bin'
                    points.astype(np.float32).tofile(dst_file)
                    radar_success += 1
                    
        except Exception:
            pass
    
    # Update splits
    final_pairs = set()
    for f in (out_path / 'lidar').glob('*.bin'):
        frame_id = f.stem
        if (out_path / 'radar' / f'{frame_id}.bin').exists():
            final_pairs.add(frame_id)
    
    print(f"\nFixed conversion results:")
    print(f"LiDAR processed: {lidar_success}")
    print(f"Radar processed: {radar_success}")  
    print(f"Final pairs: {len(final_pairs)}")
    
    # Update splits
    for split_name in ['train.txt', 'test.txt', 'val.txt']:
        split_file = out_path / 'split' / split_name
        if split_file.exists():
            with open(split_file, 'r') as f:
                original_ids = [line.strip() for line in f if line.strip()]
            
            valid_ids = [fid for fid in original_ids if fid in final_pairs]
            
            with open(split_file, 'w') as f:
                f.write('\n'.join(valid_ids))
            
            print(f"Updated {split_name}: {len(original_ids)} -> {len(valid_ids)}")
    
    if lidar_raw:
        print("\nLiDAR: output = velodyne grezzo (nessun crop). Radar: ancora con range clip se applicabile.")
    else:
        print(f"\nLiDAR+radar (dove usato): punti filtrati dentro point_cloud_range = {point_cloud_range}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vod_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--radar_type', default='radar_5frames')
    parser.add_argument(
        '--lidar_raw',
        action='store_true',
        help='LiDAR: copia velodyne (N,4) senza ground removal e senza crop.',
    )
    parser.add_argument(
        '--lidar_ground',
        choices=('elevation', 'simple'),
        default='elevation',
        help='Come rimuovere il suolo prima del crop: elevation = come VoDDataset; simple = z > ground_height.',
    )
    parser.add_argument(
        '--ground_height',
        type=float,
        default=-1.5,
        help='Soglia z per simple mode e fallback in elevation (come VoDDataset.ground_height).',
    )
    args = parser.parse_args()

    convert_vod_fixed(
        args.vod_root,
        args.output_dir,
        args.radar_type,
        lidar_raw=args.lidar_raw,
        lidar_ground=args.lidar_ground,
        ground_height=args.ground_height,
    )
    print("Ready for training with fixed coordinates!")


if __name__ == '__main__':
    main()
