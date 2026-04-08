"""Stage 0: Train the teacher network (LiDAR encoder + enhancement + reconstruction).

Usage:
    python train_teacher.py --data_root /path/to/vod --epochs 60
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import MSDNetConfig
from dataset import VoDDataset, collate_fn
from models.msdnet import MSDNetTeacher
from losses import TeacherLoss


def parse_args():
    p = argparse.ArgumentParser(description="MSDNet — train teacher")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/teacher")
    p.add_argument("--log_dir", type=str, default="runs/teacher")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MSDNetConfig()
    if args.epochs:
        cfg.training.teacher_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.lr = args.lr

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_dir)

    # Data
    train_ds = VoDDataset(
        args.data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )

    # Model
    model = MSDNetTeacher(cfg).to(device)
    criterion = TeacherLoss(
        rho=cfg.loss.rho, zeta=cfg.loss.zeta,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.lr,
        epochs=cfg.training.teacher_epochs,
        steps_per_epoch=len(train_loader),
    )

    # Training loop
    global_step = 0
    for epoch in range(cfg.training.teacher_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Teacher epoch {epoch+1}/{cfg.training.teacher_epochs}")

        for batch in pbar:
            lidar_list = [pc.to(device) for pc in batch["lidar"]]
            gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
            gt_offset = {s: v.to(device) for s, v in batch["gt_offset"].items()}

            _, recon_out = model(lidar_list, cfg.training.batch_size)
            loss = criterion(recon_out, gt_occ, gt_offset)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("teacher/loss", loss.item(), global_step)
            global_step += 1

        # Save checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f"teacher_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    # Save final
    torch.save(model.state_dict(),
               os.path.join(args.ckpt_dir, "teacher_final.pth"))
    print("Teacher training complete.")
    writer.close()


if __name__ == "__main__":
    main()
