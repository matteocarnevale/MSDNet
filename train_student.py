"""Stage 1+2: Train the student network with multi-stage distillation.

Requires a pre-trained teacher checkpoint.  The teacher is frozen and used
to compute F_l^D, which supervises RGFD and DGFD distillation losses.
The point-cloud reconstruction module is shared (same weights).

Usage:
    python train_student.py --data_root /path/to/vod \
        --teacher_ckpt checkpoints/teacher/teacher_final.pth --epochs 90
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
from models.msdnet import MSDNetTeacher, MSDNetStudent
from losses import StudentLoss


def parse_args():
    p = argparse.ArgumentParser(description="MSDNet — train student")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/student")
    p.add_argument("--log_dir", type=str, default="runs/student")
    p.add_argument("--num_workers", type=int, default=4)
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
    p.add_argument(
        "--ddim_backprop",
        action="store_true",
        help="Backprop through DDIM during training (high VRAM).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MSDNetConfig()
    if args.epochs:
        cfg.training.student_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.lr = args.lr
    if args.ddim_backprop:
        cfg.diffusion.ddim_backprop_in_training = True

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_dir)

    vod_f = None if args.vod_sequence_filter == "none" else args.vod_sequence_filter
    # Data
    train_ds = VoDDataset(
        args.data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=True,
        vod_sequence_filter=vod_f,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )

    # Teacher (frozen)
    teacher = MSDNetTeacher(cfg).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        teacher.load_state_dict(ckpt['model_state_dict'])
    else:
        teacher.load_state_dict(ckpt)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
        
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

    # Student (shares reconstruction head with teacher)
    student = MSDNetStudent(
        cfg, shared_reconstruction=teacher.reconstruction,
    ).to(device)

    criterion = StudentLoss(
        rho=cfg.loss.rho, zeta=cfg.loss.zeta,
        alpha=cfg.loss.alpha, gamma=cfg.loss.gamma,
        lambda_recon=cfg.loss.lambda_recon,
        lambda_rec_distill=cfg.loss.lambda_rec_distill,
        lambda_diff_distill=cfg.loss.lambda_diff_distill,
        lambda_diff=cfg.loss.lambda_diff,
    )

    # Only optimise student-specific parameters (encoder, RGFD, DGFD)
    # plus the shared reconstruction head
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.lr,
        epochs=cfg.training.student_epochs,
        steps_per_epoch=len(train_loader),
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        best_loss = ckpt.get('best_loss', float('inf'))

    def validate():
        if val_loader is None:
            return float('inf')
        student.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                lidar_list = [pc.to(device) for pc in batch["lidar"]]
                radar_list = [pc.to(device) for pc in batch["radar"]]
                gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
                gt_off = {s: v.to(device) for s, v in batch["gt_offset"].items()}

                f_teacher, _ = teacher(lidar_list, len(lidar_list))
                student_out = student(
                    radar_list, len(radar_list),
                    f_teacher=f_teacher, training=True,
                )
                total_loss_val, _ = criterion(student_out, f_teacher, gt_occ, gt_off)
                total_loss += total_loss_val.item()
        student.train()
        return total_loss / len(val_loader)

    loss_ema = None
    ema_decay = 0.99

    # Training loop
    for epoch in range(start_epoch, cfg.training.student_epochs):
        student.train()
        pbar = tqdm(train_loader,
                    desc=f"Student epoch {epoch+1}/{cfg.training.student_epochs}")
        sums = {"total": 0.0, "recon": 0.0, "rec_distill": 0.0,
                "diff_distill": 0.0, "diff": 0.0}
        n_train_batches = 0

        for batch in pbar:
            lidar_list = [pc.to(device) for pc in batch["lidar"]]
            radar_list = [pc.to(device) for pc in batch["radar"]]
            gt_occ = {s: v.to(device) for s, v in batch["gt_occ"].items()}
            gt_offset = {s: v.to(device) for s, v in batch["gt_offset"].items()}

            # Frozen teacher forward → F_l^D
            with torch.no_grad():
                f_teacher, _ = teacher(lidar_list, cfg.training.batch_size)

            # Student forward
            student_out = student(
                radar_list, cfg.training.batch_size,
                f_teacher=f_teacher, training=True,
            )

            total_loss, loss_dict = criterion(
                student_out, f_teacher, gt_occ, gt_offset,
            )

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            for k in sums:
                sums[k] += loss_dict[k]
            n_train_batches += 1

            tot = loss_dict["total"]
            if loss_ema is None:
                loss_ema = tot
            else:
                loss_ema = ema_decay * loss_ema + (1.0 - ema_decay) * tot

            gn = float(grad_norm)
            run_ep = sums["total"] / n_train_batches
            pbar.set_postfix(
                tot=f"{tot:.3f}",
                ema=f"{loss_ema:.3f}",
                ep=f"{run_ep:.3f}",
                lr=f"{lr:.1e}",
                gn=f"{gn:.1f}",
                rec=f"{loss_dict['recon']:.2f}",
                rd=f"{loss_dict['rec_distill']:.2f}",
                dd=f"{loss_dict['diff_distill']:.2f}",
                df=f"{loss_dict['diff']:.2f}",
            )
            for k, v in loss_dict.items():
                writer.add_scalar(f"student/{k}", v, global_step)
            writer.add_scalar("student/total_ema", loss_ema, global_step)
            writer.add_scalar("student/lr", lr, global_step)
            writer.add_scalar("student/grad_norm", grad_norm, global_step)
            global_step += 1

        for k, s in sums.items():
            writer.add_scalar(f"student/train_{k}_epoch", s / max(n_train_batches, 1), epoch)

        # Validation
        val_loss = None
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate()
            writer.add_scalar("student/val_loss", val_loss, epoch)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                }, os.path.join(args.ckpt_dir, "student_best.pth"))

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"student_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }, ckpt_path)

    torch.save({
        "epoch": cfg.training.student_epochs,
        "global_step": global_step,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }, os.path.join(args.ckpt_dir, "student_final.pth"))
    print("Student training complete.")
    print(f"Best validation loss: {best_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
