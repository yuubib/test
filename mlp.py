# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import math
import glob
import shutil
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import imageio.v2 as imageio


# =============================================================================
# 0) Utils
# =============================================================================

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def atomic_save(obj: Any, path: str) -> None:
    """
    Atomic-ish save: write to temp then replace.
    """
    ensure_dir(path)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_uint8_srgb(x: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    x = x * (2.0 ** exposure)          # exposure
    x = x / (1.0 + x)                  # Reinhard
    x = np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)
    return (x * 255.0 + 0.5).astype(np.uint8)


def flatten_hw(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    lead_shape = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), lead_shape


def unflatten_hw(x: torch.Tensor, lead_shape: Tuple[int, ...]) -> torch.Tensor:
    return x.view(*lead_shape, x.shape[-1])


def get_rng_state(device: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.startswith("cuda") and torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if "cpu" in state and state["cpu"] is not None:
        torch.set_rng_state(state["cpu"])
    if "cuda_all" in state and state["cuda_all"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_all"])


def dtype_to_str(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def str_to_dtype(s: str) -> torch.dtype:
    return getattr(torch, s)


# =============================================================================
# 1) Configs
# =============================================================================

class DataConfig:
    def __init__(self, root: str = "./dataset"):
        self.root = root

    def to_dict(self) -> Dict[str, Any]:
        return {"root": self.root}


class ModelConfig:
    def __init__(
        self,
        theta_dim: int = 32,
        theta_tex_res: Tuple[int, int] = (256, 256),
        probe_grid_res: Tuple[int, int, int] = (128, 128, 128),
        sh_order: int = 3,  # L=3 -> 16 coeffs
        aabb_min: Tuple[float, float, float] = (-9.64634, -0.01676, -7.6),
        aabb_max: Tuple[float, float, float] = (7.43, 3.8, 0.6),
        mlp_hidden_dim: int = 256,
        mlp_num_hidden_layers: int = 2,
        mlp_use_skip: bool = True,
        mlp_out_activation: Optional[str] = None,  # None | "sigmoid" | "softplus"
        init_scale: float = 0.01,
        probe_dtype: torch.dtype = torch.float32,
    ):
        self.theta_dim = theta_dim
        self.theta_tex_res = theta_tex_res
        self.probe_grid_res = probe_grid_res
        self.sh_order = sh_order
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_use_skip = mlp_use_skip
        self.mlp_out_activation = mlp_out_activation
        self.init_scale = init_scale
        self.probe_dtype = probe_dtype

    @property
    def n_sh(self) -> int:
        return (self.sh_order + 1) ** 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "theta_tex_res": list(self.theta_tex_res),
            "probe_grid_res": list(self.probe_grid_res),
            "sh_order": self.sh_order,
            "aabb_min": list(self.aabb_min),
            "aabb_max": list(self.aabb_max),
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "mlp_num_hidden_layers": self.mlp_num_hidden_layers,
            "mlp_use_skip": self.mlp_use_skip,
            "mlp_out_activation": self.mlp_out_activation,
            "init_scale": self.init_scale,
            "probe_dtype": dtype_to_str(self.probe_dtype),
        }


class CheckpointConfig:
    def __init__(
        self,
        ckpt_dir: str = "./checkpoints",
        save_every_epochs: int = 1,
        keep_last: int = 5,
        resume_from: Optional[str] = None,  # path to .pt
        strict_load: bool = True,
    ):
        self.ckpt_dir = ckpt_dir
        self.save_every_epochs = save_every_epochs
        self.keep_last = keep_last
        self.resume_from = resume_from
        self.strict_load = strict_load

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ckpt_dir": self.ckpt_dir,
            "save_every_epochs": self.save_every_epochs,
            "keep_last": self.keep_last,
            "resume_from": self.resume_from,
            "strict_load": self.strict_load,
        }


class TrainConfig:
    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 1,
        lr: float = 1e-3,
        num_workers: int = 0,
        pin_memory: bool = True,
        log_every: int = 10,
        step_lr_step_size: int = 1,
        step_lr_gamma: float = 0.5,
        use_amp: bool = False,
        grad_clip_norm: Optional[float] = None,
        device: Optional[str] = None,
        ckpt: Optional[CheckpointConfig] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.log_every = log_every
        self.step_lr_step_size = step_lr_step_size
        self.step_lr_gamma = step_lr_gamma
        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = ckpt or CheckpointConfig()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "log_every": self.log_every,
            "step_lr_step_size": self.step_lr_step_size,
            "step_lr_gamma": self.step_lr_gamma,
            "use_amp": self.use_amp,
            "grad_clip_norm": self.grad_clip_norm,
            "device": self.device,
            "ckpt": self.ckpt.to_dict(),
        }


# =============================================================================
# 2) Dataset
# =============================================================================

class MitsubaNPZDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        meta_path = os.path.join(root, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.frames = meta["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel = self.frames[idx]["file"]
        path = os.path.join(self.root, rel)
        data = np.load(path)

        rgb = torch.from_numpy(data["rgb"]).float()            # [H,W,3]
        pos = torch.from_numpy(data["position"]).float()       # [H,W,3]
        nrm = torch.from_numpy(data["sh_normal"]).float()      # [H,W,3]
        alb = torch.from_numpy(data["albedo"]).float()         # [H,W,3]
        uv  = torch.from_numpy(data["uv"]).float()             # [H,W,2]
        cam_origin = torch.from_numpy(data["cam_origin"]).float()  # [3]
        cam_target = torch.from_numpy(data["cam_target"]).float()  # [3]

        view6 = torch.cat([cam_origin, cam_target], dim=0)     # [6]
        view6 = view6.view(1, 1, 6).expand(pos.shape[0], pos.shape[1], 6)

        return {
            "rgb": rgb,
            "position": pos,
            "normal": nrm,
            "albedo": alb,
            "uv": uv,
            "view": view6,
        }


# =============================================================================
# 3) Model components
# =============================================================================

def fc_layer(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True),
    )


class PerPixelMLP(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 6,
        out_features: int = 3,
        use_skip: bool = True,
        out_activation: Optional[str] = None,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.out_features = out_features
        self.use_skip = use_skip
        self.out_activation = out_activation

        self.in_dim = 3 + 3 + 3 + 6 + theta_dim

        self.inner = fc_layer(self.in_dim, hidden_dim)

        self.hidden = nn.ModuleList()
        self._skip_mask = []
        for i in range(num_hidden_layers):
            do_skip = use_skip and (i % 2 == 1)
            self._skip_mask.append(do_skip)
            layer_in = hidden_dim + self.in_dim if do_skip else hidden_dim
            self.hidden.append(fc_layer(layer_in, hidden_dim))

        self.outer = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        normal: torch.Tensor,
        position: torch.Tensor,
        albedo: torch.Tensor,
        view6: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([normal, position, albedo, view6, theta], dim=-1)
        x0, lead = flatten_hw(x)

        h = self.inner(x0)
        for layer, do_skip in zip(self.hidden, self._skip_mask):
            h = layer(torch.cat([h, x0], dim=-1) if do_skip else h)

        y = self.outer(h)

        if self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation == "softplus":
            y = F.softplus(y)

        return unflatten_hw(y, lead)


class ProbeGridSH(nn.Module):
    def __init__(
        self,
        grid_res: Tuple[int, int, int],
        n_sh: int,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        init_scale: float = 0.01,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        Nx, Ny, Nz = grid_res
        self.grid_res = grid_res
        self.n_sh = n_sh

        self.register_buffer("aabb_min", aabb_min.to(device=device, dtype=dtype))
        self.register_buffer("aabb_max", aabb_max.to(device=device, dtype=dtype))

        coeffs = init_scale * torch.randn(Nx, Ny, Nz, n_sh, 3, device=device, dtype=dtype)
        self.coeffs = nn.Parameter(coeffs)

    def sample_coeffs_trilinear(self, p_world: torch.Tensor) -> torch.Tensor:
        p = (p_world - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8)
        p = p.clamp(0.0, 1.0)

        Nx, Ny, Nz = self.grid_res
        gx = p[..., 0] * (Nx - 1)
        gy = p[..., 1] * (Ny - 1)
        gz = p[..., 2] * (Nz - 1)

        x0 = torch.floor(gx).long()
        y0 = torch.floor(gy).long()
        z0 = torch.floor(gz).long()
        x1 = (x0 + 1).clamp(0, Nx - 1)
        y1 = (y0 + 1).clamp(0, Ny - 1)
        z1 = (z0 + 1).clamp(0, Nz - 1)

        tx = (gx - x0.float()).unsqueeze(-1).unsqueeze(-1)
        ty = (gy - y0.float()).unsqueeze(-1).unsqueeze(-1)
        tz = (gz - z0.float()).unsqueeze(-1).unsqueeze(-1)

        c000 = self.coeffs[x0, y0, z0]
        c100 = self.coeffs[x1, y0, z0]
        c010 = self.coeffs[x0, y1, z0]
        c110 = self.coeffs[x1, y1, z0]
        c001 = self.coeffs[x0, y0, z1]
        c101 = self.coeffs[x1, y0, z1]
        c011 = self.coeffs[x0, y1, z1]
        c111 = self.coeffs[x1, y1, z1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tz) + c1 * tz

    @staticmethod
    def sh_basis_L3(d: torch.Tensor) -> torch.Tensor:
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        one = torch.ones_like(x)

        Y = []
        Y.append(0.28209479177387814 * one)
        Y.append(0.4886025119029199 * y)
        Y.append(0.4886025119029199 * z)
        Y.append(0.4886025119029199 * x)
        Y.append(1.0925484305920792 * x * y)
        Y.append(1.0925484305920792 * y * z)
        Y.append(0.31539156525252005 * (3.0 * z * z - 1.0))
        Y.append(1.0925484305920792 * x * z)
        Y.append(0.5462742152960396 * (x * x - y * y))
        Y.append(0.5900435899266435 * y * (3.0 * x * x - y * y))
        Y.append(2.890611442640554 * x * y * z)
        Y.append(0.4570457994644658 * y * (5.0 * z * z - 1.0))
        Y.append(0.3731763325901154 * (5.0 * z * z * z - 3.0 * z))
        Y.append(0.4570457994644658 * x * (5.0 * z * z - 1.0))
        Y.append(1.445305721320277 * z * (x * x - y * y))
        Y.append(0.5900435899266435 * x * (x * x - 3.0 * y * y))

        return torch.stack(Y, dim=-1)

    @staticmethod
    def lambert_irradiance_from_sh_L3(C_sh: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        if C_sh.shape[-2] != 16:
            raise ValueError(f"C_sh must have 16 SH coeffs (L=3), got {C_sh.shape[-2]}")

        n = F.normalize(normal, dim=-1, eps=1e-8)
        Y_n = ProbeGridSH.sh_basis_L3(n)

        A0 = math.pi
        A1 = 2.0 * math.pi / 3.0
        A2 = math.pi / 4.0
        A3 = 0.0

        k = torch.tensor(
            [A0] * 1 + [A1] * 3 + [A2] * 5 + [A3] * 7,
            device=C_sh.device, dtype=C_sh.dtype
        )

        Cw = C_sh * k.view(*([1] * (C_sh.dim() - 2)), 16, 1)
        E = torch.einsum("...jc,...j->...c", Cw, Y_n)
        return E


class ThetaTexture(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        tex_res: Tuple[int, int],
        init_scale: float = 0.01,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        Ht, Wt = tex_res
        tex = init_scale * torch.randn(1, theta_dim, Ht, Wt, device=device, dtype=dtype)
        self.tex = nn.Parameter(tex)

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        grid = uv * 2.0 - 1.0
        lead = uv.shape[:-1]
        grid = grid.reshape(1, -1, 1, 2)

        feat = F.grid_sample(
            self.tex,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        feat = feat.squeeze(0).squeeze(-1).transpose(0, 1)
        return feat.view(*lead, -1)


class FactorizedRenderer(nn.Module):
    def __init__(self, probe_grid: ProbeGridSH, theta_tex: ThetaTexture, mlp: PerPixelMLP):
        super().__init__()
        self.probe_grid = probe_grid
        self.theta_tex = theta_tex
        self.mlp = mlp

    def forward(
        self,
        normal: torch.Tensor,
        position: torch.Tensor,
        albedo: torch.Tensor,
        view6: torch.Tensor,
        uv: torch.Tensor,
    ) -> torch.Tensor:
        c_sh = self.probe_grid.sample_coeffs_trilinear(position)  # [...,16,3]
        E = ProbeGridSH.lambert_irradiance_from_sh_L3(c_sh, normal)  # [...,3]
        theta = self.theta_tex.sample(uv)  # [...,D]
        F_hat = self.mlp(normal, position, albedo, view6, theta)  # [...,3]
        return E * F_hat


# =============================================================================
# 4) Factory
# =============================================================================

class RendererFactory:
    @staticmethod
    def build(cfg: ModelConfig, device: str) -> FactorizedRenderer:
        if cfg.sh_order != 3:
            raise ValueError("This implementation includes explicit L=3 SH basis; set sh_order=3.")

        probe = ProbeGridSH(
            grid_res=cfg.probe_grid_res,
            n_sh=cfg.n_sh,
            aabb_min=torch.tensor(cfg.aabb_min),
            aabb_max=torch.tensor(cfg.aabb_max),
            init_scale=cfg.init_scale,
            device=device,
            dtype=str_to_dtype(dtype_to_str(cfg.probe_dtype)),
        ).to(device)

        theta_tex = ThetaTexture(
            theta_dim=cfg.theta_dim,
            tex_res=cfg.theta_tex_res,
            init_scale=cfg.init_scale,
            device=device,
            dtype=torch.float32,
        ).to(device)

        mlp = PerPixelMLP(
            theta_dim=cfg.theta_dim,
            hidden_dim=cfg.mlp_hidden_dim,
            num_hidden_layers=cfg.mlp_num_hidden_layers,
            out_features=3,
            use_skip=cfg.mlp_use_skip,
            out_activation=cfg.mlp_out_activation,
        ).to(device)

        return FactorizedRenderer(probe, theta_tex, mlp).to(device)


# =============================================================================
# 5) Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Saves and loads:
      - model state_dict
      - optimizer state_dict
      - scheduler state_dict
      - scaler state_dict (if AMP)
      - epoch, best_loss
      - RNG state
      - configs (as dict)
    File types:
      - epoch_{epoch:04d}.pt   (periodic)
      - latest.pt
      - best.pt
      - final.pt
    """

    def __init__(self, cfg: CheckpointConfig, device: str):
        self.cfg = cfg
        self.device = device
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.cfg.ckpt_dir, name)

    def save(
        self,
        name: str,
        payload: Dict[str, Any],
    ) -> str:
        path = self._path(name)
        atomic_save(payload, path)
        return path

    def save_epoch(
        self,
        epoch: int,
        payload: Dict[str, Any],
    ) -> str:
        name = f"epoch_{epoch:04d}.pt"
        path = self.save(name, payload)
        self.cleanup_keep_last()
        return path

    def cleanup_keep_last(self) -> None:
        """
        Keep only last K epoch_*.pt files. Do not delete best/latest/final.
        """
        k = int(self.cfg.keep_last)
        if k <= 0:
            return

        files = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "epoch_*.pt")))
        if len(files) <= k:
            return

        to_delete = files[:-k]
        for f in to_delete:
            try:
                os.remove(f)
            except OSError:
                pass

    def load(self, path: str) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu")


# =============================================================================
# 6) Trainer
# =============================================================================

class Trainer:
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = train_cfg.device

        self.ds = MitsubaNPZDataset(data_cfg.root)
        self.dl = DataLoader(
            self.ds,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=(train_cfg.pin_memory and self.device == "cuda"),
            drop_last=False,
        )

        self.renderer = RendererFactory.build(model_cfg, self.device)

        self.optim = torch.optim.Adam(self.renderer.parameters(), lr=train_cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=train_cfg.step_lr_step_size,
            gamma=train_cfg.step_lr_gamma,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.use_amp and self.device == "cuda"))

        self.ckpt_mgr = CheckpointManager(train_cfg.ckpt, self.device)
        self.start_epoch = 0
        self.best_loss = float("inf")

        # Optional resume
        if train_cfg.ckpt.resume_from:
            self._resume(train_cfg.ckpt.resume_from)

    def _make_payload(self, epoch: int, mean_loss: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "mean_loss": float(mean_loss),
            "best_loss": float(self.best_loss),
            "model": self.renderer.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "rng": get_rng_state(self.device),
            "config": {
                "data": self.data_cfg.to_dict(),
                "model": self.model_cfg.to_dict(),
                "train": self.train_cfg.to_dict(),
            },
        }
        return payload

    def _resume(self, ckpt_path: str) -> None:
        ckpt = self.ckpt_mgr.load(ckpt_path)
        strict = bool(self.train_cfg.ckpt.strict_load)

        self.renderer.load_state_dict(ckpt["model"], strict=strict)
        self.optim.load_state_dict(ckpt["optim"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        if ckpt.get("scaler") is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])

        # epoch is "last completed epoch" in this design
        last_epoch = int(ckpt.get("epoch", -1))
        self.start_epoch = last_epoch + 1
        self.best_loss = float(ckpt.get("best_loss", float("inf")))

        rng = ckpt.get("rng", None)
        if rng is not None:
            set_rng_state(rng)

        print(f"Resumed from: {ckpt_path}")
        print(f" -> start_epoch={self.start_epoch}, best_loss={self.best_loss:.6f}")

    def train_one_epoch(self, epoch: int) -> float:
        self.renderer.train()
        running = 0.0
        n_batches = 0

        for it, batch in enumerate(self.dl):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            alb = batch["albedo"].to(self.device, non_blocking=True)
            pos = batch["position"].to(self.device, non_blocking=True)
            nrm = batch["normal"].to(self.device, non_blocking=True)
            uv  = batch["uv"].to(self.device, non_blocking=True)
            view6 = batch["view"].to(self.device, non_blocking=True)

            self.optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                pred = self.renderer(nrm, pos, alb, view6, uv)
                loss = F.mse_loss(pred, rgb)

            self.scaler.scale(loss).backward()

            if self.train_cfg.grad_clip_norm is not None:
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.renderer.parameters(), self.train_cfg.grad_clip_norm)

            self.scaler.step(self.optim)
            self.scaler.update()

            running += float(loss.item())
            n_batches += 1

            if self.train_cfg.log_every and ((it + 1) % self.train_cfg.log_every == 0):
                print(f"epoch {epoch:03d} iter {it+1:04d}/{len(self.dl)} loss={loss.item():.6f}")

        return running / max(1, n_batches)

    @torch.no_grad()
    def save_preview(self, frame_idx: int, out_path: str, exposure: float = 0.0) -> None:
        self.renderer.eval()

        sample = self.ds[frame_idx]
        rgb = sample["rgb"].unsqueeze(0).to(self.device)
        alb = sample["albedo"].unsqueeze(0).to(self.device)
        pos = sample["position"].unsqueeze(0).to(self.device)
        nrm = sample["normal"].unsqueeze(0).to(self.device)
        uv  = sample["uv"].unsqueeze(0).to(self.device)
        view6 = sample["view"].unsqueeze(0).to(self.device)

        pred = self.renderer(nrm, pos, alb, view6, uv)

        pred_np = pred.squeeze(0).detach().cpu().numpy()
        gt_np   = rgb.squeeze(0).detach().cpu().numpy()

        pred_u8 = to_uint8_srgb(pred_np, exposure=exposure)
        gt_u8   = to_uint8_srgb(gt_np, exposure=exposure)
        vis = np.concatenate([pred_u8, gt_u8], axis=1)

        ensure_dir(out_path)
        imageio.imwrite(out_path, vis)
        imageio.imwrite(os.path.join(os.path.dirname(out_path) or ".", "pred.png"), pred_u8)
        imageio.imwrite(os.path.join(os.path.dirname(out_path) or ".", "gt.png"), gt_u8)
        print("Saved preview:", out_path, "(left=pred, right=gt)")

    def fit(
        self,
        preview_every: int = 1,
        preview_frame_idx: int = 0,
        preview_out: str = "./preview.png",
    ) -> FactorizedRenderer:
        ckpt_cfg = self.train_cfg.ckpt

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            mean_loss = self.train_one_epoch(epoch)

            # scheduler: per-epoch step
            self.scheduler.step()

            lr_now = self.optim.param_groups[0]["lr"]
            print(f"[epoch {epoch:03d}] mean_loss={mean_loss:.6f} lr={lr_now:.6e}")

            # update best
            is_best = mean_loss < self.best_loss
            if is_best:
                self.best_loss = float(mean_loss)

            # build payload once (contains best_loss updated)
            payload = self._make_payload(epoch=epoch, mean_loss=mean_loss)

            # always save latest
            self.ckpt_mgr.save("latest.pt", payload)

            # save best
            if is_best:
                self.ckpt_mgr.save("best.pt", payload)

            # periodic epoch checkpoint
            if ckpt_cfg.save_every_epochs > 0 and ((epoch + 1) % ckpt_cfg.save_every_epochs == 0):
                self.ckpt_mgr.save_epoch(epoch, payload)

            # preview
            if preview_every > 0 and ((epoch + 1) % preview_every == 0):
                self.save_preview(preview_frame_idx, preview_out)

        # final
        final_payload = self._make_payload(epoch=self.train_cfg.epochs - 1, mean_loss=self.best_loss)
        self.ckpt_mgr.save("final.pt", final_payload)
        print("Saved final model to:", os.path.join(ckpt_cfg.ckpt_dir, "final.pt"))
        export_realtime_assets(self.renderer, out_dir="./realtime_assets", dtype=torch.float16, export_mlp_torchscript=True)
        return self.renderer



def export_realtime_assets(
    renderer,
    out_dir: str = "./realtime_assets",
    dtype: torch.dtype = torch.float16,
    export_mlp_torchscript: bool = True,
) -> None:
    """
    导出用于实时渲染的“资产”：
      - theta texture (UV feature map)
      - probe SH coeffs
      - metadata
      - (optional) MLP TorchScript

    输出：
      realtime_assets/
        assets.npz
        meta.json
        mlp_ts.pt   (optional)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- theta texture: [1,D,H,W] -> 保存为 [H,W,D] 方便图形管线（按像素取向量）
    theta = renderer.theta_tex.tex.detach().to("cpu")
    theta = theta.squeeze(0)                 # [D,H,W]
    theta = theta.permute(1, 2, 0).contiguous()  # [H,W,D]

    # --- probe SH coeffs: [Nx,Ny,Nz,16,3]
    probe = renderer.probe_grid.coeffs.detach().to("cpu")

    # --- metadata
    aabb_min = renderer.probe_grid.aabb_min.detach().cpu().numpy().tolist()
    aabb_max = renderer.probe_grid.aabb_max.detach().cpu().numpy().tolist()
    Nx, Ny, Nz = renderer.probe_grid.grid_res
    n_sh = int(renderer.probe_grid.n_sh)
    Ht, Wt, D = int(theta.shape[0]), int(theta.shape[1]), int(theta.shape[2])

    # cast
    theta_np = theta.to(dtype).numpy()
    probe_np = probe.to(dtype).numpy()

    # save compressed npz
    npz_path = os.path.join(out_dir, "assets.npz")
    np.savez_compressed(
        npz_path,
        theta_hwd=theta_np,           # [H,W,D]
        probe_xyzjc=probe_np,         # [Nx,Ny,Nz,16,3]
        aabb_min=np.array(aabb_min, np.float32),
        aabb_max=np.array(aabb_max, np.float32),
    )

    meta = {
        "theta": {
            "layout": "HWD",
            "theta_dim": D,
            "tex_res": [Ht, Wt],
            "dtype": str(dtype).replace("torch.", ""),
            "file": "assets.npz::theta_hwd",
        },
        "probe": {
            "layout": "XYZJRGB",
            "grid_res": [Nx, Ny, Nz],
            "n_sh": n_sh,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "dtype": str(dtype).replace("torch.", ""),
            "file": "assets.npz::probe_xyzjc",
        },
        "notes": {
            "sh_order": 3,
            "sh_basis": "real_sloan_L3_ordering",
            "lambert_kernel": {"A0": math.pi, "A1": 2.0 * math.pi / 3.0, "A2": math.pi / 4.0, "A3": 0.0},
        },
    }

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # --- optional: export MLP as TorchScript for realtime inference (PyTorch runtime)
    if export_mlp_torchscript:
        renderer.mlp.eval()
        mlp_ts = torch.jit.script(renderer.mlp.to("cpu"))
        mlp_ts_path = os.path.join(out_dir, "mlp_ts.pt")
        mlp_ts.save(mlp_ts_path)
        meta["mlp"] = {"format": "torchscript", "file": "mlp_ts.pt"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Exported realtime assets to:", out_dir)

# =============================================================================
# 7) Entry
# =============================================================================

def main() -> None:
    data_cfg = DataConfig(root="./dataset")

    model_cfg = ModelConfig(
        theta_dim=32,
        theta_tex_res=(256, 256),
        probe_grid_res=(128, 128, 128),
        sh_order=3,
        aabb_min=(-9.64634, -0.01676, -7.6),
        aabb_max=(7.43, 3.8, 0.6),
        mlp_hidden_dim=256,
        mlp_num_hidden_layers=2,
        mlp_use_skip=True,
        mlp_out_activation=None,
        init_scale=0.01,
        probe_dtype=torch.float32,
    )

    ckpt_cfg = CheckpointConfig(
        ckpt_dir="./checkpoints",
        save_every_epochs=1,
        keep_last=5,
        resume_from=None,     # e.g. "./checkpoints/latest.pt"
        strict_load=True,
    )

    train_cfg = TrainConfig(
        epochs=2,
        batch_size=4,
        lr=1e-3,
        num_workers=0,
        log_every=10,
        step_lr_step_size=10,
        step_lr_gamma=0.5,
        use_amp=False,
        grad_clip_norm=None,
        ckpt=ckpt_cfg,
    )

    trainer = Trainer(data_cfg, model_cfg, train_cfg)
    trainer.fit(preview_every=1, preview_frame_idx=0, preview_out="./preview.png")
    


if __name__ == "__main__":
    main()
