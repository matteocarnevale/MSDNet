"""MSDNet — unified training script for RADIal.

Usage:
    # Stage 1: train teacher (LiDAR encoder + reconstruction)
    python train.py --mode teacher \\
        --radial_root /data/radar_lidar_pc_dataset \\
        --lidar_cache_dir /data/lidar_processed \\
        --epochs 60 --batch_size 4

    # Stage 2: train student (requires trained teacher)
    python train.py --mode student \\
        --radial_root /data/radar_lidar_pc_dataset \\
        --lidar_cache_dir /data/lidar_processed \\
        --teacher_ckpt checkpoints/teacher/best.pth \\
        --epochs 90 --batch_size 4

    # Ablate Doppler conditioning
    python train.py --mode student ... --no_doppler

Workflow:
    1. Build dataset with build_radial_dataset.py (produces radar_lidar_pc_dataset/)
    2. Train teacher:  python train.py --mode teacher ...
    3. Train student:  python train.py --mode student --teacher_ckpt checkpoints/teacher/best.pth ...
    4. Evaluate:       python evaluate.py --radial_root ... --student_ckpt checkpoints/student/best.pth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import MSDNetConfig
from dataset import RADIalDataset, get_splits, collate_fn
from models.msdnet import MSDNetTeacher, MSDNetStudent
from losses import (TeacherLoss, StudentLoss,
                    reconstruction_loss_breakdown)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MSDNet training on RADIal")
    p.add_argument("--mode",           required=True, choices=["teacher", "student"])
    p.add_argument("--radial_root",    required=True, type=str,
                   help="Path to radar_lidar_pc_dataset/ folder")
    # Ground-removed LiDAR is cached automatically in <radial_root>/laser_PCL_ng/
    p.add_argument("--teacher_ckpt",   default=None, type=str,
                   help="[student mode] Path to trained teacher checkpoint")
    p.add_argument("--epochs",       default=None,   type=int,
                   help="Override default epoch count from config")
    p.add_argument("--batch_size",   default=None,   type=int)
    p.add_argument("--lr",           default=None,   type=float)
    p.add_argument("--ckpt_dir",     default=None,   type=str)
    p.add_argument("--log_dir",      default=None,   type=str)
    p.add_argument("--num_workers",  default=4,      type=int)
    p.add_argument("--val_interval", default=5,      type=int)
    p.add_argument("--resume",       default=None,   type=str)
    p.add_argument("--ddim_backprop",action="store_true",
                   help="[student] Backprop through DDIM (high VRAM)")
    p.add_argument("--no_doppler",   action="store_true",
                   help="Disable Doppler conditioning (ablation)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_teacher(cfg, train_loader, val_loader, args, device):
    model     = MSDNetTeacher(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = TeacherLoss(cfg.loss.rho, cfg.loss.zeta)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(args.log_dir)
    step     = 0
    best_val = float("inf")
    start_ep = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_ep = ck["epoch"] + 1
        print(f"Resumed teacher from epoch {start_ep}")

    for epoch in range(start_ep, args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[teacher] epoch {epoch}/{args.epochs}")

        for batch in pbar:
            lidar  = [x.to(device) for x in batch["lidar"]]
            gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
            gt_off = {s: v.to(device) for s, v in batch["gt_offset"].items()}

            optimizer.zero_grad()
            _, recon_out = model(lidar, batch_size=len(lidar))

            # reconstruction_loss_breakdown returns (tensor, dict)
            loss, breakdown = reconstruction_loss_breakdown(
                recon_out, gt_occ, gt_off,
                rho=cfg.loss.rho, zeta=cfg.loss.zeta,
            )
            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("teacher/loss_step", loss.item(), step)
            for k, v in breakdown.items():
                writer.add_scalar(f"teacher/{k}", v, step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

        avg_train = running / len(train_loader)
        writer.add_scalar("teacher/loss_epoch", avg_train, epoch)

        if epoch % args.val_interval == 0:
            avg_val = _val_teacher(model, val_loader, cfg, device)
            writer.add_scalar("teacher/val_epoch", avg_val, epoch)
            print(f"  epoch {epoch}: train={avg_train:.4f}  val={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                torch.save({"epoch": epoch, "model": model.state_dict()},
                           ckpt_dir / "best.pth")
                print(f"  → new best val ({best_val:.4f}) saved to best.pth")
        else:
            print(f"  epoch {epoch}: train={avg_train:.4f}")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   ckpt_dir / "last.pth")

    writer.close()
    print(f"Teacher training done. Best val loss: {best_val:.4f}")
    return model


@torch.no_grad()
def _val_teacher(model, loader, cfg, device):
    model.eval()
    total = 0.0
    for batch in loader:
        lidar  = [x.to(device) for x in batch["lidar"]]
        gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
        gt_off = {s: v.to(device) for s, v in batch["gt_offset"].items()}
        _, recon_out = model(lidar, batch_size=len(lidar))
        loss, _ = reconstruction_loss_breakdown(
            recon_out, gt_occ, gt_off, cfg.loss.rho, cfg.loss.zeta
        )
        total += loss.item()
    model.train()
    return total / max(len(loader), 1)


def train_student(cfg, train_loader, val_loader, args, device):
    if args.teacher_ckpt is None:
        raise ValueError("--teacher_ckpt required for student training")

    teacher = MSDNetTeacher(cfg).to(device)
    ck = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ck["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher loaded from {args.teacher_ckpt}")

    student   = MSDNetStudent(cfg, shared_reconstruction=teacher.reconstruction).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    criterion = StudentLoss(cfg.loss)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(args.log_dir)
    step     = 0
    best_val = float("inf")
    start_ep = 0

    if args.resume:
        ck2 = torch.load(args.resume, map_location=device)
        student.load_state_dict(ck2["model"])
        optimizer.load_state_dict(ck2["optimizer"])
        start_ep = ck2["epoch"] + 1
        print(f"Resumed student from epoch {start_ep}")

    for epoch in range(start_ep, args.epochs):
        student.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[student] epoch {epoch}/{args.epochs}")

        for batch in pbar:
            lidar  = [x.to(device) for x in batch["lidar"]]
            radar  = [x.to(device) for x in batch["radar"]]
            gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
            gt_off = {s: v.to(device) for s, v in batch["gt_offset"].items()}
            B = len(radar)

            optimizer.zero_grad()

            with torch.no_grad():
                f_teacher, _ = teacher(lidar, batch_size=B)

            out = student(radar, batch_size=B, f_teacher=f_teacher, training=True)
            loss, ld = criterion(out, f_teacher, gt_occ, gt_off)

            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("student/loss_step", loss.item(), step)
            for k, v in ld.items():
                writer.add_scalar(f"student/{k}", v, step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

        avg_train = running / len(train_loader)
        writer.add_scalar("student/loss_epoch", avg_train, epoch)

        if epoch % args.val_interval == 0:
            avg_val = _val_student(student, teacher, val_loader, cfg, criterion, device)
            writer.add_scalar("student/val_epoch", avg_val, epoch)
            print(f"  epoch {epoch}: train={avg_train:.4f}  val={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                torch.save({"epoch": epoch, "model": student.state_dict()},
                           ckpt_dir / "best.pth")
                print(f"  → new best val ({best_val:.4f}) saved to best.pth")
        else:
            print(f"  epoch {epoch}: train={avg_train:.4f}")

        torch.save({"epoch": epoch, "model": student.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   ckpt_dir / "last.pth")

    writer.close()
    print(f"Student training done. Best val loss: {best_val:.4f}")


@torch.no_grad()
def _val_student(student, teacher, loader, cfg, criterion, device):
    student.eval()
    total = 0.0
    for batch in loader:
        lidar  = [x.to(device) for x in batch["lidar"]]
        radar  = [x.to(device) for x in batch["radar"]]
        gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
        gt_off = {s: v.to(device) for s, v in batch["gt_offset"].items()}
        B = len(radar)
        f_t, _ = teacher(lidar, batch_size=B)
        out  = student(radar, batch_size=B, f_teacher=f_t, training=True)
        loss, _ = criterion(out, f_t, gt_occ, gt_off)
        total += loss.item()
    student.train()
    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = MSDNetConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override config from CLI
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.lr:         cfg.training.lr         = args.lr

    if args.no_doppler:
        cfg.encoder.doppler_channels = 0
        print("Doppler conditioning disabled (ablation)")

    if args.mode == "teacher":
        cfg.diffusion.ddim_backprop_in_training = False
        epochs   = args.epochs or cfg.training.teacher_epochs
        ckpt_dir = args.ckpt_dir or "checkpoints/teacher"
        log_dir  = args.log_dir  or "runs/teacher"
    else:
        cfg.diffusion.ddim_backprop_in_training = args.ddim_backprop
        epochs   = args.epochs or cfg.training.student_epochs
        ckpt_dir = args.ckpt_dir or "checkpoints/student"
        log_dir  = args.log_dir  or "runs/student"

    args.epochs   = epochs
    args.ckpt_dir = ckpt_dir
    args.log_dir  = log_dir
    args.lr       = cfg.training.lr

    # Dataset
    print(f"Loading RADIal splits from {args.radial_root} …")
    from dataset import print_split_info
    train_ids, val_ids, test_ids = get_splits(args.radial_root)
    print_split_info(args.radial_root, train_ids, val_ids, test_ids)

    train_ds = RADIalDataset(args.radial_root, train_ids, cfg)
    val_ds   = RADIalDataset(args.radial_root, val_ids,   cfg)

    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    print(f"Training {args.mode} for {args.epochs} epochs (bs={cfg.training.batch_size})")
    print(f"  BEV size: {cfg.bev_size}   Grid: {cfg.grid_size}")

    if args.mode == "teacher":
        train_teacher(cfg, train_loader, val_loader, args, device)
    else:
        train_student(cfg, train_loader, val_loader, args, device)


if __name__ == "__main__":
    main()
