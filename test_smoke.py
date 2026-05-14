"""Smoke tests for MSDNet.

Verifies imports, config, dataset loading, model component forward passes,
loss computation, backward passes, and evaluation metrics using small
synthetic data. Full-pipeline tests that require spconv are automatically
skipped when spconv is not available.

Usage:
    conda activate venv
    python -m pytest test_smoke.py -v
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import spconv.pytorch as spconv

    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

requires_spconv = pytest.mark.skipif(not HAS_SPCONV, reason="spconv not installed")

B = 2
C = 128
BEV_H = BEV_W = 8
GRID_Z = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    """MSDNetConfig with a reduced spatial grid and fast diffusion schedule."""
    from config import MSDNetConfig

    cfg = MSDNetConfig()
    cfg.voxel.point_cloud_range = [0.0, -3.2, -0.6, 6.4, 3.2, 0.6]
    cfg.diffusion.total_timesteps = 10
    cfg.diffusion.start_timestep = 5
    cfg.diffusion.sampling_steps = 5
    cfg.diffusion.sampling_interval = 1
    cfg.diffusion.ddim_backprop_in_training = False
    cfg.training.batch_size = B
    return cfg


@pytest.fixture
def bev():
    """Synthetic BEV feature tensor."""
    return torch.randn(B, C, BEV_H, BEV_W)


@pytest.fixture
def recon_gt():
    """GT occupancy / offset dicts whose shapes match reconstruction output."""
    z4 = GRID_Z // 4
    dims = {4: (z4, BEV_H, BEV_W),
            2: (z4 * 2, BEV_H * 2, BEV_W * 2),
            1: (z4 * 4, BEV_H * 4, BEV_W * 4)}
    occ, off = {}, {}
    for s, (z, h, w) in dims.items():
        occ[s] = torch.zeros(B, 1, z, h, w)
        off[s] = torch.zeros(B, 3, z, h, w)
        occ[s][:, 0, 0, 0, 0] = 1.0
        off[s][:, :, 0, 0, 0] = 0.01
    return occ, off


@pytest.fixture
def radial_dir():
    """Temporary directory with synthetic RADIal-format npy data."""
    d = tempfile.mkdtemp(prefix="msdnet_smoke_")
    pc_dir = os.path.join(d, "radar_pc_cache")
    for sub in ("radar_FFT", "laser_PCL", pc_dir):
        os.makedirs(os.path.join(d, sub) if sub != pc_dir else sub)

    rng = np.random.RandomState(42)
    for sid in (0, 1):
        fft = (rng.randn(512, 256, 16) + 1j * rng.randn(512, 256, 16)).astype(np.complex64)
        np.save(os.path.join(d, "radar_FFT", f"fft_{sid:06d}.npy"), fft)
        lidar = rng.uniform([0.5, -2, -0.3], [5, 2, 0.3], (200, 3)).astype(np.float32)
        np.save(os.path.join(d, "laser_PCL", f"pcl_{sid:06d}.npy"), lidar)
        radar_pc = rng.uniform(
            [0.5, -2, -0.3, 0, -5], [5, 2, 0.3, 1, 5], (80, 5)
        ).astype(np.float32)
        np.save(os.path.join(pc_dir, f"radar_{sid:06d}.npy"), radar_pc)

    # Return (root, cache_dir) tuple
    yield d, pc_dir
    shutil.rmtree(d)


# Keep old name as alias for backward compat within this file
vod_dir = radial_dir


@pytest.fixture
def point_clouds():
    """Pair of random point clouds for metric tests."""
    rng = np.random.RandomState(0)
    return rng.randn(200, 3), rng.randn(300, 3)


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_config(self):
        from config import MSDNetConfig

        assert MSDNetConfig() is not None

    def test_models(self):
        from models import (
            MSDNetTeacher,
            MSDNetStudent,
            VoxelEncoder,
            FeatureEnhancement,
            RGFD,
            DGFD,
            PointCloudReconstruction,
        )

    def test_losses(self):
        from losses import (
            TeacherLoss,
            StudentLoss,
            occupancy_loss,
            offset_loss,
            reconstruction_loss,
            feature_distillation_loss,
            diffusion_loss,
        )

    def test_dataset(self):
        from dataset import RADIalDataset, collate_fn

    def test_metrics(self):
        from evaluate import (
            chamfer_distance,
            modified_hausdorff,
            f_score,
            jsd_bev,
            mmd_rbf,
        )


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        from config import MSDNetConfig

        cfg = MSDNetConfig()
        assert cfg.encoder.lidar_in_features == 4
        assert cfg.encoder.radar_in_features == 5
        assert cfg.encoder.bev_channels == 128
        assert cfg.training.batch_size == 4

    def test_grid_size(self, small_cfg):
        assert small_cfg.grid_size == (64, 64, 8)

    def test_bev_size(self, small_cfg):
        assert small_cfg.bev_size == (8, 8)

    def test_loss_weights(self):
        from config import MSDNetConfig

        cfg = MSDNetConfig()
        assert len(cfg.loss.rho) == 3
        assert len(cfg.loss.zeta) == 3
        assert cfg.loss.lambda_recon > 0


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def _make_ds(self, radial_dir, small_cfg):
        from dataset import RADIalDataset
        root, pc_dir = radial_dir
        return RADIalDataset(root, [0, 1], small_cfg, radar_pc_dir=pc_dir)

    def test_load(self, radial_dir, small_cfg):
        ds = self._make_ds(radial_dir, small_cfg)
        assert len(ds) == 2

    def test_getitem_keys(self, radial_dir, small_cfg):
        ds = self._make_ds(radial_dir, small_cfg)
        sample = ds[0]
        for key in ("lidar", "radar", "gt_occ", "gt_offset", "frame_id"):
            assert key in sample

    def test_getitem_shapes(self, radial_dir, small_cfg):
        ds = self._make_ds(radial_dir, small_cfg)
        sample = ds[0]
        assert sample["lidar"].dim() == 2 and sample["lidar"].shape[1] == 4
        assert sample["radar"].dim() == 2 and sample["radar"].shape[1] == 5

    def test_gt_scales(self, radial_dir, small_cfg):
        ds = self._make_ds(radial_dir, small_cfg)
        sample = ds[0]
        for scale in (4, 2, 1):
            assert scale in sample["gt_occ"]
            assert scale in sample["gt_offset"]
            assert sample["gt_occ"][scale].shape[0] == 1
            assert sample["gt_offset"][scale].shape[0] == 3

    def test_collate(self, radial_dir, small_cfg):
        from dataset import collate_fn
        ds = self._make_ds(radial_dir, small_cfg)
        batch = collate_fn([ds[0], ds[1]])
        assert isinstance(batch["lidar"], list) and len(batch["lidar"]) == 2
        assert isinstance(batch["radar"], list) and len(batch["radar"]) == 2
        for s in (4, 2, 1):
            assert batch["gt_occ"][s].shape[0] == 2
            assert batch["gt_offset"][s].shape[0] == 2

    def test_get_splits(self, radial_dir, small_cfg):
        from dataset import get_splits
        root, _ = radial_dir
        train, val, test = get_splits(root, train_ratio=0.5, val_ratio=0.25)
        assert len(train) + len(val) + len(test) == 2


# ---------------------------------------------------------------------------
# 4. Building Blocks
# ---------------------------------------------------------------------------


class TestBlocks:
    def test_cbam(self):
        from models.modules import CBAM

        x = torch.randn(2, 64, 8, 8)
        assert CBAM(64)(x).shape == x.shape

    def test_convnext_block(self):
        from models.modules import ConvNeXtBlock

        x = torch.randn(2, 64, 8, 8)
        assert ConvNeXtBlock(64)(x).shape == x.shape

    def test_bottleneck(self):
        from models.modules import BottleNeck

        x = torch.randn(2, 64, 8, 8)
        assert BottleNeck(64)(x).shape == x.shape

    def test_deformable_conv(self):
        from models.modules import DeformableConvBlock

        x = torch.randn(2, 64, 8, 8)
        assert DeformableConvBlock(64, 64)(x).shape == x.shape

    def test_deformable_conv_strided(self):
        from models.modules import DeformableConvBlock

        x = torch.randn(2, 64, 8, 8)
        out = DeformableConvBlock(64, 64, stride=2)(x)
        assert out.shape == (2, 64, 4, 4)

    def test_conv_bn_act(self):
        from models.modules import ConvBNAct

        x = torch.randn(2, 64, 8, 8)
        assert ConvBNAct(64, 128)(x).shape == (2, 128, 8, 8)

    def test_conv_bn_act_transposed(self):
        from models.modules import ConvBNAct

        x = torch.randn(2, 64, 4, 4)
        out = ConvBNAct(64, 64, kernel_size=4, stride=2, padding=1, transposed=True)(x)
        assert out.shape == (2, 64, 8, 8)

    def test_sinusoidal_embedding(self):
        from models.modules import SinusoidalTimestepEmbedding

        emb = SinusoidalTimestepEmbedding(128)
        out = emb(torch.tensor([0, 50, 999]))
        assert out.shape == (3, 128)
        assert torch.isfinite(out).all()

    def test_timestep_embedding(self):
        from models.modules import TimestepEmbedding

        assert TimestepEmbedding(128)(torch.tensor([10, 50])).shape == (2, 128)


# ---------------------------------------------------------------------------
# 5. Model Components (no spconv required)
# ---------------------------------------------------------------------------


class TestFeatureEnhancement:
    def test_shape(self, bev):
        from models.enhancement import FeatureEnhancement

        assert FeatureEnhancement(C)(bev).shape == bev.shape

    def test_gradient_flow(self, bev):
        from models.enhancement import FeatureEnhancement

        bev = bev.clone().requires_grad_(True)
        FeatureEnhancement(C)(bev).sum().backward()
        assert bev.grad is not None


class TestRGFD:
    def test_shape(self, bev):
        from models.rgfd import RGFD

        assert RGFD(C)(bev).shape == bev.shape

    def test_gradient_flow(self, bev):
        from models.rgfd import RGFD

        bev = bev.clone().requires_grad_(True)
        RGFD(C)(bev).sum().backward()
        assert bev.grad is not None


class TestDGFD:
    def test_student_forward(self, bev):
        from models.dgfd import DGFD
        from models.diffusion import DiffusionSchedule

        out = DGFD(C, C).student_forward(bev, 5, DiffusionSchedule(10), 5, 1)
        assert out.shape == bev.shape

    def test_diffusion_net(self, bev):
        from models.dgfd import DGFD

        t = torch.randint(0, 10, (B,))
        eps = DGFD(C, C).diffusion_net(bev, t)
        assert eps.shape == bev.shape

    def test_noise_adapter(self, bev):
        from models.dgfd import NoiseAdapter

        na = NoiseAdapter(C, C)
        noise = torch.randn_like(bev)
        t_m = torch.full((B,), 5, dtype=torch.long)
        out = na(bev, t_m, noise)
        assert out.shape == bev.shape


class TestReconstruction:
    def test_forward_keys(self, bev):
        from models.reconstruction import PointCloudReconstruction

        out = PointCloudReconstruction(C, 64, GRID_Z)(bev)
        for s in (4, 2, 1):
            assert f"occ_{s}" in out and f"offset_{s}" in out

    def test_forward_shapes(self, bev):
        from models.reconstruction import PointCloudReconstruction

        out = PointCloudReconstruction(C, 64, GRID_Z)(bev)
        z4 = GRID_Z // 4
        assert out["occ_4"].shape == (B, 1, z4, BEV_H, BEV_W)
        assert out["offset_4"].shape == (B, 3, z4, BEV_H, BEV_W)
        assert out["occ_2"].shape == (B, 1, z4 * 2, BEV_H * 2, BEV_W * 2)
        assert out["offset_2"].shape == (B, 3, z4 * 2, BEV_H * 2, BEV_W * 2)
        assert out["occ_1"].shape == (B, 1, z4 * 4, BEV_H * 4, BEV_W * 4)
        assert out["offset_1"].shape == (B, 3, z4 * 4, BEV_H * 4, BEV_W * 4)

    def test_gradient_flow(self, bev):
        from models.reconstruction import PointCloudReconstruction

        bev = bev.clone().requires_grad_(True)
        out = PointCloudReconstruction(C, 64, GRID_Z)(bev)
        out["occ_1"].sum().backward()
        assert bev.grad is not None

    def test_generate_point_cloud(self, bev):
        from models.reconstruction import PointCloudReconstruction

        pcs = PointCloudReconstruction(C, 64, GRID_Z).generate_point_cloud(
            bev, 0.5, [0, -3.2, -0.6, 6.4, 3.2, 0.6]
        )
        assert len(pcs) == B
        for pc in pcs:
            assert pc.dim() == 2
            if pc.shape[0] > 0:
                assert pc.shape[1] == 3


class TestDiffusion:
    def test_schedule_creation(self):
        from models.diffusion import DiffusionSchedule

        s = DiffusionSchedule(100)
        assert s.T == 100
        assert s.alpha_bar.shape == (100,)
        assert (s.alpha_bar > 0).all()
        assert s.alpha_bar[0] > s.alpha_bar[-1]

    def test_q_sample(self):
        from models.diffusion import DiffusionSchedule

        s = DiffusionSchedule(100)
        x = torch.randn(2, 4, 8, 8)
        x_t = s.q_sample(x, torch.tensor([10, 50]))
        assert x_t.shape == x.shape

    def test_q_sample_noise_scaling(self):
        from models.diffusion import DiffusionSchedule

        s = DiffusionSchedule(100)
        x = torch.ones(1, 1, 4, 4)
        noise = torch.zeros(1, 1, 4, 4)
        x_t = s.q_sample(x, torch.tensor([0]), noise)
        expected = s.sqrt_alpha_bar[0] * x
        assert torch.allclose(x_t, expected, atol=1e-6)

    def test_ddim_step(self):
        from models.diffusion import DiffusionSchedule

        s = DiffusionSchedule(100)
        x_t = torch.randn(2, 4, 8, 8)
        x_prev = s.ddim_step(x_t, t=50, t_prev=40, noise_pred=torch.randn_like(x_t))
        assert x_prev.shape == x_t.shape

    def test_device_transfer(self):
        from models.diffusion import DiffusionSchedule

        s = DiffusionSchedule(10)
        s = s.to("cpu")
        assert s.alpha_bar.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# 6. Losses
# ---------------------------------------------------------------------------


class TestLosses:
    def test_occupancy_loss(self):
        from losses import occupancy_loss

        loss = occupancy_loss(torch.randn(B, 1, 4, 8, 8), torch.zeros(B, 1, 4, 8, 8))
        assert loss.dim() == 0 and loss.item() >= 0

    def test_occupancy_loss_perfect(self):
        from losses import occupancy_loss

        target = torch.zeros(B, 1, 4, 8, 8)
        pred = torch.full_like(target, -100.0)
        loss = occupancy_loss(pred, target)
        assert loss.item() < 0.01

    def test_offset_loss(self):
        from losses import offset_loss

        mask = torch.zeros(B, 1, 4, 8, 8)
        mask[:, 0, 0, 0, 0] = 1
        loss = offset_loss(torch.randn(B, 3, 4, 8, 8), torch.zeros(B, 3, 4, 8, 8), mask)
        assert loss.dim() == 0 and loss.item() >= 0

    def test_offset_loss_empty_mask(self):
        from losses import offset_loss

        mask = torch.zeros(B, 1, 4, 8, 8)
        loss = offset_loss(torch.randn(B, 3, 4, 8, 8), torch.zeros(B, 3, 4, 8, 8), mask)
        assert loss.item() == 0.0

    def test_teacher_loss(self, bev, recon_gt):
        from losses import TeacherLoss
        from models.reconstruction import PointCloudReconstruction

        out = PointCloudReconstruction(C, 64, GRID_Z)(bev)
        occ, off = recon_gt
        loss = TeacherLoss()(out, occ, off)
        assert loss.dim() == 0 and loss.requires_grad

    def test_teacher_loss_backward(self, bev, recon_gt):
        from losses import TeacherLoss
        from models.reconstruction import PointCloudReconstruction

        recon = PointCloudReconstruction(C, 64, GRID_Z)
        out = recon(bev)
        occ, off = recon_gt
        loss = TeacherLoss()(out, occ, off)
        loss.backward()
        grads = [p.grad for p in recon.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_feature_distillation_loss(self, bev, recon_gt):
        from losses import bev_nonempty_mask, feature_distillation_loss

        occ, _ = recon_gt
        mask = bev_nonempty_mask(occ[1], bev.shape[-2:])
        loss = feature_distillation_loss(
            torch.randn_like(bev), bev, omega_ne=mask,
        )
        assert loss.dim() == 0 and loss.item() >= 0

    def test_diffusion_loss(self):
        from losses import diffusion_loss

        loss = diffusion_loss(torch.randn(B, C, 8, 8), torch.randn(B, C, 8, 8))
        assert loss.dim() == 0 and loss.item() >= 0

    def test_student_loss(self, bev, recon_gt):
        from losses import StudentLoss
        from models.dgfd import DGFD
        from models.diffusion import DiffusionSchedule
        from models.reconstruction import PointCloudReconstruction
        from models.rgfd import RGFD

        f_teacher = bev.detach()
        schedule = DiffusionSchedule(10)
        rgfd = RGFD(C)
        dgfd = DGFD(C, C)
        recon = PointCloudReconstruction(C, 64, GRID_Z)

        f_recon = rgfd(torch.randn(B, C, BEV_H, BEV_W))
        f_denoised = dgfd.student_forward(f_recon, 5, schedule, 5, 1)
        recon_out = recon(f_denoised)

        t = torch.randint(0, 10, (B,))
        noise_gt = torch.randn_like(f_teacher)
        noise_pred = dgfd.diffusion_net(schedule.q_sample(f_teacher, t, noise_gt), t)

        student_out = {
            "f_recon": f_recon,
            "f_denoised": f_denoised,
            "recon_out": recon_out,
            "diff_loss_inputs": (noise_pred, noise_gt, t),
        }
        occ, off = recon_gt
        total, dct = StudentLoss()(student_out, f_teacher, occ, off)
        assert total.dim() == 0
        assert all(k in dct for k in ("recon", "rec_distill", "diff_distill", "diff", "total"))

    def test_student_loss_backward(self, bev, recon_gt):
        from losses import StudentLoss
        from models.dgfd import DGFD
        from models.diffusion import DiffusionSchedule
        from models.reconstruction import PointCloudReconstruction
        from models.rgfd import RGFD

        f_teacher = bev.detach()
        schedule = DiffusionSchedule(10)
        rgfd = RGFD(C)
        dgfd = DGFD(C, C)
        recon = PointCloudReconstruction(C, 64, GRID_Z)

        f_recon = rgfd(torch.randn(B, C, BEV_H, BEV_W))
        f_denoised = dgfd.student_forward(f_recon, 5, schedule, 5, 1)
        recon_out = recon(f_denoised)

        t = torch.randint(0, 10, (B,))
        noise_gt = torch.randn_like(f_teacher)
        noise_pred = dgfd.diffusion_net(schedule.q_sample(f_teacher, t, noise_gt), t)

        student_out = {
            "f_recon": f_recon,
            "f_denoised": f_denoised,
            "recon_out": recon_out,
            "diff_loss_inputs": (noise_pred, noise_gt, t),
        }
        occ, off = recon_gt
        total, _ = StudentLoss()(student_out, f_teacher, occ, off)
        total.backward()

        rgfd_grads = sum(1 for p in rgfd.parameters() if p.grad is not None)
        dgfd_grads = sum(1 for p in dgfd.parameters() if p.grad is not None)
        assert rgfd_grads > 0
        assert dgfd_grads > 0


# ---------------------------------------------------------------------------
# 7. Evaluation Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_chamfer_distance(self, point_clouds):
        from evaluate import chamfer_distance

        cd = chamfer_distance(*point_clouds)
        assert isinstance(cd, float) and cd >= 0

    def test_chamfer_distance_identical(self):
        from evaluate import chamfer_distance

        pts = np.random.randn(100, 3)
        assert chamfer_distance(pts, pts) < 1e-10

    def test_modified_hausdorff(self, point_clouds):
        from evaluate import modified_hausdorff

        mhd = modified_hausdorff(*point_clouds)
        assert isinstance(mhd, float) and mhd >= 0

    def test_f_score(self, point_clouds):
        from evaluate import f_score

        fs = f_score(*point_clouds, threshold=2.0)
        assert 0.0 <= fs <= 1.0

    def test_f_score_identical(self):
        from evaluate import f_score

        pts = np.random.randn(100, 3)
        assert f_score(pts, pts, threshold=0.01) == 1.0

    def test_jsd_bev(self, point_clouds):
        from evaluate import jsd_bev

        jsd = jsd_bev(*point_clouds)
        assert isinstance(jsd, float) and jsd >= 0

    def test_mmd_rbf(self, point_clouds):
        from evaluate import mmd_rbf

        mmd = mmd_rbf(*point_clouds)
        assert isinstance(mmd, float) and mmd >= 0

    def test_mmd_identical(self):
        from evaluate import mmd_rbf

        pts = np.random.randn(100, 3)
        assert mmd_rbf(pts, pts) < 1e-6


# ---------------------------------------------------------------------------
# 8. Component-level Training Step (no spconv)
# ---------------------------------------------------------------------------


class TestComponentTrainingStep:
    """Simulates one teacher and one student training step using only
    post-encoder components (no spconv)."""

    def test_teacher_step(self, bev, recon_gt):
        from losses import TeacherLoss
        from models.enhancement import FeatureEnhancement
        from models.reconstruction import PointCloudReconstruction

        enhance = FeatureEnhancement(C)
        recon = PointCloudReconstruction(C, 64, GRID_Z)
        params = list(enhance.parameters()) + list(recon.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        f_dense = enhance(bev)
        recon_out = recon(f_dense)
        occ, off = recon_gt
        loss = TeacherLoss()(recon_out, occ, off)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()

        assert loss.item() >= 0

    def test_student_step(self, bev, recon_gt):
        from losses import StudentLoss
        from models.dgfd import DGFD
        from models.diffusion import DiffusionSchedule
        from models.reconstruction import PointCloudReconstruction
        from models.rgfd import RGFD

        f_teacher = bev.detach()
        schedule = DiffusionSchedule(10)
        rgfd = RGFD(C)
        dgfd = DGFD(C, C)
        recon = PointCloudReconstruction(C, 64, GRID_Z)

        params = (
            list(rgfd.parameters())
            + list(dgfd.parameters())
            + list(recon.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)

        f_sparse = torch.randn(B, C, BEV_H, BEV_W)
        f_recon = rgfd(f_sparse)
        f_denoised = dgfd.student_forward(f_recon, 5, schedule, 5, 1)
        recon_out = recon(f_denoised)

        t = torch.randint(0, 10, (B,))
        noise_gt = torch.randn_like(f_teacher)
        noise_pred = dgfd.diffusion_net(schedule.q_sample(f_teacher, t, noise_gt), t)

        student_out = {
            "f_recon": f_recon,
            "f_denoised": f_denoised,
            "recon_out": recon_out,
            "diff_loss_inputs": (noise_pred, noise_gt, t),
        }
        occ, off = recon_gt
        total, loss_dict = StudentLoss()(student_out, f_teacher, occ, off)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()

        assert total.item() >= 0
        assert all(v >= 0 for v in loss_dict.values())


# ---------------------------------------------------------------------------
# 9. Full Pipeline (requires spconv)
# ---------------------------------------------------------------------------


@requires_spconv
class TestTeacherPipeline:
    def test_instantiation(self, small_cfg):
        from models.msdnet import MSDNetTeacher

        model = MSDNetTeacher(small_cfg)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward(self, small_cfg, radial_dir):
        from dataset import RADIalDataset, collate_fn
        from models.msdnet import MSDNetTeacher

        root, pc_dir = radial_dir
        model = MSDNetTeacher(small_cfg).eval()
        ds = RADIalDataset(root, [0, 1], small_cfg, radar_pc_dir=pc_dir)
        batch = collate_fn([ds[0], ds[1]])
        with torch.no_grad():
            f_dense, recon_out = model(batch["lidar"], B)
        assert f_dense.shape[0] == B
        assert f_dense.shape[1] == small_cfg.encoder.bev_channels
        for s in (4, 2, 1):
            assert f"occ_{s}" in recon_out

    def test_training_step(self, small_cfg, radial_dir):
        from dataset import RADIalDataset, collate_fn
        from losses import TeacherLoss
        from models.msdnet import MSDNetTeacher

        root, pc_dir = radial_dir
        model = MSDNetTeacher(small_cfg).train()
        ds = RADIalDataset(root, [0, 1], small_cfg, radar_pc_dir=pc_dir)
        batch = collate_fn([ds[0], ds[1]])
        _, recon_out = model(batch["lidar"], B)
        loss = TeacherLoss()(recon_out, batch["gt_occ"], batch["gt_offset"])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() >= 0


@requires_spconv
class TestStudentPipeline:
    def test_instantiation(self, small_cfg):
        from models.msdnet import MSDNetStudent

        model = MSDNetStudent(small_cfg)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward(self, small_cfg, radial_dir):
        from dataset import RADIalDataset, collate_fn
        from models.msdnet import MSDNetStudent, MSDNetTeacher

        root, pc_dir = radial_dir
        teacher = MSDNetTeacher(small_cfg).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = MSDNetStudent(small_cfg, teacher.reconstruction)

        ds = RADIalDataset(root, [0, 1], small_cfg, radar_pc_dir=pc_dir)
        batch = collate_fn([ds[0], ds[1]])
        with torch.no_grad():
            f_teacher, _ = teacher(batch["lidar"], B)

        out = student(batch["radar"], B, f_teacher=f_teacher, training=True)
        for key in ("f_recon", "f_denoised", "recon_out", "diff_loss_inputs"):
            assert key in out

    def test_training_step(self, small_cfg, radial_dir):
        from dataset import RADIalDataset, collate_fn
        from losses import StudentLoss
        from models.msdnet import MSDNetStudent, MSDNetTeacher

        root, pc_dir = radial_dir
        teacher = MSDNetTeacher(small_cfg).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = MSDNetStudent(small_cfg, teacher.reconstruction)

        ds = RADIalDataset(root, [0, 1], small_cfg, radar_pc_dir=pc_dir)
        batch = collate_fn([ds[0], ds[1]])
        with torch.no_grad():
            f_teacher, _ = teacher(batch["lidar"], B)

        out = student(batch["radar"], B, f_teacher=f_teacher, training=True)
        loss, loss_dict = StudentLoss(small_cfg.loss)(
            out, f_teacher, batch["gt_occ"], batch["gt_offset"]
        )

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() >= 0

    def test_inference(self, small_cfg, radial_dir):
        from dataset import RADIalDataset, collate_fn
        from models.msdnet import MSDNetStudent

        root, pc_dir = radial_dir
        student = MSDNetStudent(small_cfg).eval()
        ds = RADIalDataset(root, [0, 1], small_cfg, radar_pc_dir=pc_dir)
        batch = collate_fn([ds[0], ds[1]])
        pcs = student.generate_point_cloud(
            batch["radar"],
            B,
            threshold=0.5,
            point_cloud_range=small_cfg.voxel.point_cloud_range,
        )
        assert len(pcs) == B
        for pc in pcs:
            assert pc.dim() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
