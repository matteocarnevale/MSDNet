"""Loss functions for MSDNet (Eqs. 5–8, 10, 14–15).

Teacher loss:   L_teacher = Σ_s ρ(s) L_occ(s) + ζ(s) L_off(s)
Student loss:   L_student = λ1 L_recon + λ2 L_rec_distill
                          + λ3 L_diff_distill + λ4 L_diff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reconstruction losses (Eqs. 5-7)
# ---------------------------------------------------------------------------

def occupancy_loss(pred_logits: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for voxel occupancy (Eq. 5).

    Args:
        pred_logits: (B, 1, Z, Y, X) raw logits
        target:      (B, 1, Z, Y, X) binary ground truth {0, 1}
    """
    return F.binary_cross_entropy_with_logits(pred_logits, target)


def offset_loss(pred_offset: torch.Tensor,
                target_offset: torch.Tensor,
                occ_mask: torch.Tensor) -> torch.Tensor:
    """L1 offset loss computed only over occupied voxels (Eq. 6).

    Args:
        pred_offset:   (B, 3, Z, Y, X)
        target_offset: (B, 3, Z, Y, X)
        occ_mask:      (B, 1, Z, Y, X) binary mask of occupied voxels
    """
    mask = occ_mask.expand_as(pred_offset)
    num_occupied = mask.sum().clamp(min=1)
    return (mask * (pred_offset - target_offset).abs()).sum() / num_occupied


def reconstruction_loss(recon_out: dict,
                        gt_occ: dict, gt_offset: dict,
                        rho: list, zeta: list) -> torch.Tensor:
    """Multi-scale teacher/student reconstruction loss (Eq. 7).

    Args:
        recon_out:   dict with keys occ_{4,2,1}, offset_{4,2,1}
        gt_occ:      dict with keys 4, 2, 1 → (B,1,Z_s,Y_s,X_s)
        gt_offset:   dict with keys 4, 2, 1 → (B,3,Z_s,Y_s,X_s)
        rho, zeta:   per-scale loss weights
    """
    total = torch.tensor(0.0, device=recon_out["occ_4"].device)
    for i, s in enumerate([4, 2, 1]):
        occ_pred = recon_out[f"occ_{s}"]
        off_pred = recon_out[f"offset_{s}"]
        occ_gt = gt_occ[s]
        off_gt = gt_offset[s]

        l_occ = occupancy_loss(occ_pred, occ_gt)
        l_off = offset_loss(off_pred, off_gt, occ_gt)
        total = total + rho[i] * l_occ + zeta[i] * l_off
    return total


# ---------------------------------------------------------------------------
# Feature distillation losses (Eqs. 8 and 14)
# ---------------------------------------------------------------------------

def feature_distillation_loss(f_student: torch.Tensor,
                              f_teacher: torch.Tensor,
                              alpha: float = 10.0,
                              gamma: float = 20.0) -> torch.Tensor:
    """
    Weighted MSE distillation loss with separate weights for non-empty
    and empty BEV grid cells (Eqs. 8, 14).

    Non-empty cells are those where the teacher feature has non-zero
    activations (proxy for non-empty voxels after height compression).

    Args:
        f_student: (B, C, H, W) student features (F_r^R or F_r^D)
        f_teacher: (B, C, H, W) dense LiDAR features F_l^D
        alpha:     weight for non-empty cells
        gamma:     weight for empty cells
    """
    squared_error = (f_teacher - f_student).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)

    # Heuristic: non-empty where teacher feature norm > 0
    teacher_norm = f_teacher.detach().pow(2).sum(dim=1, keepdim=True)
    non_empty = (teacher_norm > 0).float()
    empty = 1.0 - non_empty

    num_nonempty = non_empty.sum().clamp(min=1)
    num_empty = empty.sum().clamp(min=1)

    loss = alpha * (squared_error * non_empty).sum() / num_nonempty \
         + gamma * (squared_error * empty).sum() / num_empty
    return loss


# ---------------------------------------------------------------------------
# Diffusion loss (Eq. 10)
# ---------------------------------------------------------------------------

def diffusion_loss(noise_pred: torch.Tensor,
                   noise_gt: torch.Tensor) -> torch.Tensor:
    """L2 noise prediction loss (Eq. 10)."""
    return F.mse_loss(noise_pred, noise_gt)


# ---------------------------------------------------------------------------
# Composite loss wrappers
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
    """

    def __init__(self, rho=(1., 1., 1.), zeta=(10., 10., 10.),
                 alpha=10., gamma=20.,
                 lambda_recon=1., lambda_rec_distill=0.01,
                 lambda_diff_distill=5., lambda_diff=10.):
        super().__init__()
        self.rho = list(rho)
        self.zeta = list(zeta)
        self.alpha = alpha
        self.gamma = gamma
        self.weights = {
            "recon": lambda_recon,
            "rec_distill": lambda_rec_distill,
            "diff_distill": lambda_diff_distill,
            "diff": lambda_diff,
        }

    def forward(self, student_out: dict,
                f_teacher: torch.Tensor,
                gt_occ: dict, gt_offset: dict):
        """
        Args:
            student_out: dict from MSDNetStudent.forward(training=True)
            f_teacher:   frozen teacher features F_l^D
            gt_occ:      multi-scale ground-truth occupancy
            gt_offset:   multi-scale ground-truth offsets
        Returns:
            total_loss, loss_dict
        """
        loss_recon = reconstruction_loss(
            student_out["recon_out"], gt_occ, gt_offset,
            self.rho, self.zeta,
        )
        loss_rec_distill = feature_distillation_loss(
            student_out["f_recon"], f_teacher, self.alpha, self.gamma,
        )
        loss_diff_distill = feature_distillation_loss(
            student_out["f_denoised"], f_teacher, self.alpha, self.gamma,
        )

        eps_pred, eps_gt, _ = student_out["diff_loss_inputs"]
        loss_diff = diffusion_loss(eps_pred, eps_gt)

        total = (self.weights["recon"] * loss_recon
                 + self.weights["rec_distill"] * loss_rec_distill
                 + self.weights["diff_distill"] * loss_diff_distill
                 + self.weights["diff"] * loss_diff)

        loss_dict = {
            "recon": loss_recon.item(),
            "rec_distill": loss_rec_distill.item(),
            "diff_distill": loss_diff_distill.item(),
            "diff": loss_diff.item(),
            "total": total.item(),
        }
        return total, loss_dict
