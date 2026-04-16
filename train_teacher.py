"""Stage 0: Train the teacher network (LiDAR encoder + enhancement + reconstruction).

Usage:
    python train_teacher.py --data_root /path/to/vod --epochs 60
"""

import argparse
import os
import warnings

import torch

# Suppress spconv deprecation warnings (non-critical)
warnings.filterwarnings("ignore", category=UserWarning, module="spconv")
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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/teacher")
    p.add_argument("--log_dir", type=str, default="runs/teacher")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--val_interval", type=int, default=10, help="Validation interval")
    p.add_argument("--save_interval", type=int, default=20, help="Checkpoint save interval")
    p.add_argument(
        "--vod_sequence_filter",
        type=str,
        default="none",
        choices=("none", "4drvo_net"),
        help="none: use split files. 4drvo_net: paper IV-A VoD split.",
    )
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

    vod_f = None if args.vod_sequence_filter == "none" else args.vod_sequence_filter
    # Data
    train_ds = VoDDataset(
        args.data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=True,  # Verify files exist and skip missing ones
        vod_sequence_filter=vod_f,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    
    # Validation data
    try:
        val_ds = VoDDataset(
            args.data_root, "test",
            point_cloud_range=cfg.voxel.point_cloud_range,
            voxel_size=cfg.voxel.voxel_size,
            verify_files=True,
            vod_sequence_filter=vod_f,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True, drop_last=False,
        )
        print(f"Validation dataset: {len(val_ds)} samples")
    except:
        val_loader = None
        print("No validation dataset found, skipping validation")

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

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        best_loss = ckpt.get('best_loss', float('inf'))

    def validate():
        if val_loader is None:
            return float('inf')
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                lidar_list = [pc.to(device) for pc in batch["lidar"]]
                gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
                gt_off = {s: v.to(device) for s, v in batch["gt_offset"].items()}
                
                _, recon_out = model(lidar_list, len(lidar_list))
                loss = criterion(recon_out, gt_occ, gt_off)
                total_loss += loss.item()
        model.train()
        return total_loss / len(val_loader)

    # Training loop
    for epoch in range(start_epoch, cfg.training.teacher_epochs):
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

        # Validation
        val_loss = None
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate()
            writer.add_scalar("teacher/val_loss", val_loss, epoch)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                }, os.path.join(args.ckpt_dir, "teacher_best.pth"))

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"teacher_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }, ckpt_path)

    # Save final
    torch.save({
        "epoch": cfg.training.teacher_epochs,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }, os.path.join(args.ckpt_dir, "teacher_final.pth"))
    print("Teacher training complete.")
    print(f"Best validation loss: {best_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
