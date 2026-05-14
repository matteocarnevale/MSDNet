"""Loss functions for MSDNet (Eqs. 5–8, 10, 14–15).

Teacher loss:   L_teacher = Σ_s ρ(s) L_occ(s) + ζ(s) L_off(s)
Student loss:   L_student = λ1 L_recon + λ2 L_rec_distill
                          + λ3 L_diff_distill + λ4 L_diff

Bug-fixes vs original:
  - reconstruction_loss_breakdown now returns (total_tensor, breakdown_dict)
    so `loss.backward()` works correctly.
  - StudentLoss accepts a LossConfig dataclass directly.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reconstruction losses (Eqs. 5-7)
# ---------------------------------------------------------------------------

def occupancy_loss(pred_logits: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for voxel occupancy (Eq. 5)."""
    return F.binary_cross_entropy_with_logits(pred_logits, target)


def offset_loss(pred_offset: torch.Tensor,
                target_offset: torch.Tensor,
                occ_mask: torch.Tensor) -> torch.Tensor:
    """L1 offset loss only over occupied voxels (Eq. 6)."""
    mask = occ_mask.expand_as(pred_offset)
    num_occupied = mask.sum().clamp(min=1)
    return (mask * (pred_offset - target_offset).abs()).sum() / num_occupied


def reconstruction_loss(recon_out: dict,
                        gt_occ: dict, gt_offset: dict,
                        rho: list, zeta: list) -> torch.Tensor:
    """Multi-scale reconstruction loss (Eq. 7). Returns differentiable tensor."""
    total = None
    for i, s in enumerate([4, 2, 1]):
        lo = occupancy_loss(recon_out[f"occ_{s}"], gt_occ[s])
        lf = offset_loss(recon_out[f"offset_{s}"], gt_offset[s], gt_occ[s])
        term = rho[i] * lo + zeta[i] * lf
        total = term if total is None else total + term
    return total


def reconstruction_loss_breakdown(
    recon_out: dict,
    gt_occ: dict,
    gt_offset: dict,
    rho: list,
    zeta: list,
) -> Tuple[torch.Tensor, dict]:
    """
    Multi-scale reconstruction loss that returns BOTH:
      - total differentiable tensor (for loss.backward())
      - breakdown dict of floats (for TensorBoard logging)

    Fixed: original version returned only a dict of detached floats,
    making loss.backward() impossible.
    """
    total = None
    out = {}
    for i, s in enumerate([4, 2, 1]):
        lo = occupancy_loss(recon_out[f"occ_{s}"], gt_occ[s])
        lf = offset_loss(recon_out[f"offset_{s}"], gt_offset[s], gt_occ[s])
        wo = rho[i] * lo
        wz = zeta[i] * lf
        term = wo + wz
        total = term if total is None else total + term
        out[f"L_occ_s{s}"] = float(lo)
        out[f"L_off_s{s}"] = float(lf)
    return total, out


# ---------------------------------------------------------------------------
# Feature distillation losses (Eqs. 8 and 14)
# ---------------------------------------------------------------------------

def bev_nonempty_mask(gt_occ_fine: torch.Tensor,
                      bev_hw: tuple) -> torch.Tensor:
    """
    Ω_ne: BEV mask of non-empty columns derived from finest-scale LiDAR occupancy.
    Collapses Z dimension then pools to teacher BEV resolution (Eqs. 8, 14).
    """
    collapsed = gt_occ_fine.amax(dim=2)
    pooled = F.adaptive_max_pool2d(collapsed, output_size=bev_hw)
    return (pooled > 0.5).float()


def feature_distillation_loss(f_student: torch.Tensor,
                              f_teacher: torch.Tensor,
                              alpha: float = 10.0,
                              gamma: float = 20.0,
                              omega_ne: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted MSE distillation (Eqs. 8, 14)."""
    se = (f_teacher.detach() - f_student).pow(2).mean(dim=1, keepdim=True)

    if omega_ne is not None:
        non_empty = omega_ne
    else:
        teacher_norm = f_teacher.detach().pow(2).sum(dim=1, keepdim=True)
        non_empty = (teacher_norm > 0).float()
    empty = 1.0 - non_empty

    num_ne = non_empty.sum().clamp(min=1)
    num_e = empty.sum().clamp(min=1)

    return (alpha * (se * non_empty).sum() / num_ne
            + gamma * (se * empty).sum() / num_e)


# ---------------------------------------------------------------------------
# Diffusion loss (Eq. 10)
# ---------------------------------------------------------------------------

def diffusion_loss(noise_pred: torch.Tensor,
                   noise_gt: torch.Tensor) -> torch.Tensor:
    """L2 noise prediction loss (Eq. 10)."""
    return F.mse_loss(noise_pred, noise_gt)


# ---------------------------------------------------------------------------
# Composite wrappers
# ---------------------------------------------------------------------------

class TeacherLoss(nn.Module):
    """L_teacher = Σ_s ρ(s) L_occ(s) + ζ(s) L_off(s)  (Eq. 7)."""

    def __init__(self, rho=(1., 1., 1.), zeta=(10., 10., 10.)):
        super().__init__()
        self.rho = list(rho)
        self.zeta = list(zeta)

    def forward(self, recon_out, gt_occ, gt_offset):
        return reconstruction_loss(recon_out, gt_occ, gt_offset,
                                   self.rho, self.zeta)


class StudentLoss(nn.Module):
    """
    L_student = λ1 L_recon + λ2 L_rec_distill + λ3 L_diff_distill + λ4 L_diff
    (Eq. 15).

    Accepts either a LossConfig dataclass or explicit keyword arguments.
    Fixed: original accepted LossConfig as `rho` which caused a TypeError.
    """

    def __init__(self, cfg_or_rho=(1., 1., 1.), zeta=(10., 10., 10.),
                 alpha=10., gamma=20.,
                 lambda_recon=1., lambda_rec_distill=0.01,
                 lambda_diff_distill=5., lambda_diff=10.):
        super().__init__()
        if hasattr(cfg_or_rho, 'rho'):
            # Called as StudentLoss(loss_config) ← the common case
            cfg = cfg_or_rho
            self.rho = list(cfg.rho)
            self.zeta = list(cfg.zeta)
            self.alpha = cfg.alpha
            self.gamma = cfg.gamma
            self.weights = dict(
                recon=cfg.lambda_recon,
                rec_distill=cfg.lambda_rec_distill,
                diff_distill=cfg.lambda_diff_distill,
                diff=cfg.lambda_diff,
            )
        else:
            # Called as StudentLoss(rho=[...], zeta=[...], ...)
            self.rho = list(cfg_or_rho)
            self.zeta = list(zeta)
            self.alpha = alpha
            self.gamma = gamma
            self.weights = dict(
                recon=lambda_recon,
                rec_distill=lambda_rec_distill,
                diff_distill=lambda_diff_distill,
                diff=lambda_diff,
            )

    def forward(self, student_out: dict,
                f_teacher: torch.Tensor,
                gt_occ: dict, gt_offset: dict):
        """
        Args:
            student_out:  dict from MSDNetStudent.forward(training=True)
            f_teacher:    frozen teacher BEV features F_l^D
            gt_occ:       multi-scale occupancy GT
            gt_offset:    multi-scale offset GT
        Returns:
            total_loss (tensor), loss_dict (floats for logging)
        """
        loss_recon = reconstruction_loss(
            student_out["recon_out"], gt_occ, gt_offset,
            self.rho, self.zeta,
        )

        omega = bev_nonempty_mask(gt_occ[1], f_teacher.shape[-2:])

        loss_rec_distill = feature_distillation_loss(
            student_out["f_recon"], f_teacher,
            self.alpha, self.gamma, omega_ne=omega,
        )
        loss_diff_distill = feature_distillation_loss(
            student_out["f_denoised"], f_teacher,
            self.alpha, self.gamma, omega_ne=omega,
        )

        eps_pred, eps_gt, _ = student_out["diff_loss_inputs"]
        loss_diff = diffusion_loss(eps_pred, eps_gt)

        total = (self.weights["recon"]        * loss_recon
               + self.weights["rec_distill"]  * loss_rec_distill
               + self.weights["diff_distill"] * loss_diff_distill
               + self.weights["diff"]         * loss_diff)

        return total, {
            "recon":        float(loss_recon),
            "rec_distill":  float(loss_rec_distill),
            "diff_distill": float(loss_diff_distill),
            "diff":         float(loss_diff),
            "total":        float(total),
        }
