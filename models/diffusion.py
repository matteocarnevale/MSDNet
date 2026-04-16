"""Diffusion utilities: noise schedules, forward diffusion, DDIM sampling.

Implements the Markovian forward process (Eq. 9) and deterministic DDIM
reverse sampling (Song et al., ICLR 2021) used in MSDNet's DGFD stage.
"""

import torch
import torch.nn as nn


class DiffusionSchedule:
    """Linear beta schedule and precomputed alpha-bar products."""

    def __init__(self, total_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: torch.device = torch.device("cuda")):
        self.T = total_timesteps
        betas = torch.linspace(beta_start, beta_end, total_timesteps,
                               dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.alpha_bar = alpha_bar.float().to(device)
        self.sqrt_alpha_bar = alpha_bar.sqrt().float().to(device)
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().float().to(device)

    def to(self, device):
        for attr in ("betas", "alphas", "alpha_bar",
                      "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ----- forward diffusion (Eq. 9) -----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._gather(self.sqrt_alpha_bar, t, x0)
        sqrt_1mab = self._gather(self.sqrt_one_minus_alpha_bar, t, x0)
        return sqrt_ab * x0 + sqrt_1mab * noise

    # ----- DDIM deterministic sampling step -----
    def ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int,
                  noise_pred: torch.Tensor) -> torch.Tensor:
        """One step of DDIM (eta=0, deterministic)."""
        ab_t = self.alpha_bar[t]
        if t_prev >= 0:
            ab_prev = self.alpha_bar[t_prev]
        else:
            ab_prev = torch.tensor(1.0, device=x_t.device, dtype=ab_t.dtype)

        # predicted x_0
        x0_pred = (x_t - (1 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1 - ab_prev).sqrt() * noise_pred
        return ab_prev.sqrt() * x0_pred + dir_xt

    # ----- full reverse sampling loop (DDIM) -----
    def ddim_sample(self, model, x_start: torch.Tensor,
                    start_t: int, interval: int, num_steps: int,
                    enable_grad: bool = False) -> torch.Tensor:
        """
        DDIM reverse sampling from timestep `start_t` down to 0 (Eq. 13).

        Args:
            enable_grad: if True, run under torch.enable_grad() (training).
        """
        timesteps = list(range(start_t, start_t - interval * num_steps, -interval))
        ctx = torch.enable_grad if enable_grad else torch.no_grad
        with ctx():
            x_t = x_start
            for i, t in enumerate(timesteps):
                t_prev = t - interval if (i + 1) < num_steps else 0
                t_tensor = torch.full((x_t.shape[0],), t,
                                      device=x_t.device, dtype=torch.long)
                noise_pred = model(x_t, t_tensor)
                x_t = self.ddim_step(x_t, t, max(t_prev, 0), noise_pred)

        return x_t

    # ----- helpers -----
    @staticmethod
    def _gather(coeff, t, x):
        """Index `coeff` by timestep `t` and reshape for broadcasting with `x`."""
        out = coeff[t]
        while out.dim() < x.dim():
            out = out.unsqueeze(-1)
        return out.to(x.device)
