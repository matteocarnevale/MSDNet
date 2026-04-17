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
from losses import TeacherLoss, reconstruction_loss_breakdown


def parse_args():
    p = argparse.ArgumentParser(description="MSDNet — train teacher")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/teacher")
    p.add_argument("--log_dir", type=str, default="runs/teacher")
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Worker DataLoader (0 = solo processo principale, lento). 4–8 tipico su GPU.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Mixed precision (float16) — spesso ~1.3–1.8× più veloce su GPU; se NaN/instabilità, ometti.",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log TensorBoard ogni N step (1 = ogni batch). Es. 5 riduce I/O.",
    )
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    writer = SummaryWriter(args.log_dir)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    print(
        f"Device: {device} | DataLoader workers: {args.num_workers} | "
        f"AMP: {use_amp} | log_every: {args.log_every}"
    )

    vod_f = None if args.vod_sequence_filter == "none" else args.vod_sequence_filter
    # Data
    train_ds = VoDDataset(
        args.data_root, "train",
        point_cloud_range=cfg.voxel.point_cloud_range,
        voxel_size=cfg.voxel.voxel_size,
        verify_files=True,  # Verify files exist and skip missing ones
        vod_sequence_filter=vod_f,
    )
    _dl_kw = dict(
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if args.num_workers > 0:
        _dl_kw["persistent_workers"] = True
        _dl_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **_dl_kw)
    
    # Validation data
    try:
        val_ds = VoDDataset(
            args.data_root, "test",
            point_cloud_range=cfg.voxel.point_cloud_range,
            voxel_size=cfg.voxel.voxel_size,
            verify_files=True,
            vod_sequence_filter=vod_f,
        )
        _vkw = dict(
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if args.num_workers > 0:
            _vkw["persistent_workers"] = True
            _vkw["prefetch_factor"] = 2
        val_loader = DataLoader(val_ds, **_vkw)
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
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        best_loss = ckpt.get('best_loss', float('inf'))

    def _ckpt_state():
        d = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
        }
        if scaler is not None:
            d["scaler_state_dict"] = scaler.state_dict()
        return d

    def validate():
        if val_loader is None:
            return float("inf"), {}
        model.eval()
        total_loss = 0.0
        bd_sum = None
        n_batches = 0
        rho, zeta = cfg.loss.rho, cfg.loss.zeta
        with torch.no_grad():
            for batch in val_loader:
                lidar_list = [pc.to(device, non_blocking=True) for pc in batch["lidar"]]
                gt_occ = {s: v.to(device, non_blocking=True) for s, v in batch["gt_occ"].items()}
                gt_off = {s: v.to(device, non_blocking=True) for s, v in batch["gt_offset"].items()}

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    _, recon_out = model(lidar_list, len(lidar_list))
                    loss = criterion(recon_out, gt_occ, gt_off)
                total_loss += loss.item()
                bd = reconstruction_loss_breakdown(recon_out, gt_occ, gt_off, rho, zeta)
                if bd_sum is None:
                    bd_sum = {k: 0.0 for k in bd}
                for k in bd:
                    bd_sum[k] += bd[k]
                n_batches += 1
        model.train()
        for k in bd_sum:
            bd_sum[k] /= max(n_batches, 1)
        return total_loss / len(val_loader), bd_sum

    loss_ema = None
    ema_decay = 0.99
    last_val_loss = None
    rho, zeta = cfg.loss.rho, cfg.loss.zeta

    # Training loop
    for epoch in range(start_epoch, cfg.training.teacher_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Teacher epoch {epoch+1}/{cfg.training.teacher_epochs}")
        train_loss_sum = 0.0
        n_train_batches = 0
        epoch_loss_min = float("inf")
        epoch_loss_max = float("-inf")
        bd_epoch_sum = None

        for batch in pbar:
            lidar_list = [pc.to(device, non_blocking=True) for pc in batch["lidar"]]
            gt_occ = {s: v.to(device, non_blocking=True) for s, v in batch["gt_occ"].items()}
            gt_offset = {s: v.to(device, non_blocking=True) for s, v in batch["gt_offset"].items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, recon_out = model(lidar_list, cfg.training.batch_size)
                loss = criterion(recon_out, gt_occ, gt_offset)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            scheduler.step()

            li = loss.item()
            train_loss_sum += li
            n_train_batches += 1
            epoch_loss_min = min(epoch_loss_min, li)
            epoch_loss_max = max(epoch_loss_max, li)
            if loss_ema is None:
                loss_ema = li
            else:
                loss_ema = ema_decay * loss_ema + (1.0 - ema_decay) * li

            bd = reconstruction_loss_breakdown(recon_out, gt_occ, gt_offset, rho, zeta)
            if bd_epoch_sum is None:
                bd_epoch_sum = {k: 0.0 for k in bd}
            for k in bd:
                bd_epoch_sum[k] += bd[k]
            if args.log_every <= 1 or global_step % args.log_every == 0:
                for k, v in bd.items():
                    writer.add_scalar(f"teacher/train_{k}", v, global_step)
                writer.add_scalar(
                    "teacher/train_total_approx",
                    bd["w_occ_total"] + bd["w_off_total"],
                    global_step,
                )

            lr = optimizer.param_groups[0]["lr"]
            gn = float(grad_norm)
            run_ep = train_loss_sum / n_train_batches
            val_str = f"{last_val_loss:.4f}" if last_val_loss is not None else "—"
            best_str = f"{best_loss:.4f}" if best_loss < float("inf") else "—"
            pbar.set_postfix(
                loss=f"{li:.3f}",
                ema=f"{loss_ema:.3f}",
                ep=f"{run_ep:.3f}",
                occ=f"{bd['w_occ_total']:.2f}",
                off=f"{bd['w_off_total']:.2f}",
                lr=f"{lr:.1e}",
                gn=f"{gn:.1f}",
                vmin=f"{epoch_loss_min:.2f}",
                vmax=f"{epoch_loss_max:.2f}",
                val=val_str,
                best=best_str,
            )
            if args.log_every <= 1 or global_step % args.log_every == 0:
                writer.add_scalar("teacher/loss", li, global_step)
                writer.add_scalar("teacher/loss_ema", loss_ema, global_step)
                writer.add_scalar("teacher/lr", lr, global_step)
                writer.add_scalar("teacher/grad_norm", grad_norm, global_step)
            global_step += 1

        train_epoch_mean = train_loss_sum / max(n_train_batches, 1)
        writer.add_scalar("teacher/train_loss_epoch", train_epoch_mean, epoch)
        writer.add_scalar("teacher/train_loss_epoch_min", epoch_loss_min, epoch)
        writer.add_scalar("teacher/train_loss_epoch_max", epoch_loss_max, epoch)
        if bd_epoch_sum is not None and n_train_batches > 0:
            for k in bd_epoch_sum:
                writer.add_scalar(
                    f"teacher/train_epoch_mean_{k}",
                    bd_epoch_sum[k] / n_train_batches,
                    epoch,
                )

        # Validation
        val_loss = None
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_bd = validate()
            last_val_loss = val_loss
            writer.add_scalar("teacher/val_loss", val_loss, epoch)
            for k, v in val_bd.items():
                writer.add_scalar(f"teacher/val_{k}", v, epoch)
            print(
                f"Epoch {epoch+1} - val_loss={val_loss:.6f}  "
                f"w_occ={val_bd.get('w_occ_total', 0):.4f}  "
                f"w_off={val_bd.get('w_off_total', 0):.4f}"
            )
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                payload = _ckpt_state()
                payload.update({"epoch": epoch + 1, "global_step": global_step})
                torch.save(payload, os.path.join(args.ckpt_dir, "teacher_best.pth"))

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"teacher_epoch{epoch+1}.pth")
            payload = _ckpt_state()
            payload.update({"epoch": epoch + 1, "global_step": global_step})
            torch.save(payload, ckpt_path)

    # Save final
    payload = _ckpt_state()
    payload.update({"epoch": cfg.training.teacher_epochs, "global_step": global_step})
    torch.save(payload, os.path.join(args.ckpt_dir, "teacher_final.pth"))
    print("Teacher training complete.")
    print(f"Best validation loss: {best_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
