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

import torch
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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/student")
    p.add_argument("--log_dir", type=str, default="runs/student")
    p.add_argument("--num_workers", type=int, default=4)
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

    # Teacher (frozen)
    teacher = MSDNetTeacher(cfg).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

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

    # Training loop
    global_step = 0
    for epoch in range(cfg.training.student_epochs):
        student.train()
        pbar = tqdm(train_loader,
                    desc=f"Student epoch {epoch+1}/{cfg.training.student_epochs}")

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
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(
                total=f"{loss_dict['total']:.3f}",
                rec=f"{loss_dict['recon']:.3f}",
                rd=f"{loss_dict['rec_distill']:.3f}",
                dd=f"{loss_dict['diff_distill']:.3f}",
                df=f"{loss_dict['diff']:.3f}",
            )
            for k, v in loss_dict.items():
                writer.add_scalar(f"student/{k}", v, global_step)
            global_step += 1

        # Save checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f"student_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    torch.save(student.state_dict(),
               os.path.join(args.ckpt_dir, "student_final.pth"))
    print("Student training complete.")
    writer.close()


if __name__ == "__main__":
    main()
