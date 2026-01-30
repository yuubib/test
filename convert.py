"""
realtime_renderer.py

Load pre-trained GI assets (theta texture, probe SH grid, traced MLP)
and perform real-time rendering from a G-buffer.

Assets are assumed to come from an export like:

    export_realtime_assets(
        renderer,
        out_dir="./realtime_assets",
        dtype=torch.float16,
        export_mlp_torchscript=True,
    )

Directory layout:

    realtime_assets/
        assets.npz      # theta_hwd, probe_xyzjc, aabb_min, aabb_max
        meta.json       # metadata (grid_res, theta_dim, etc.)
        mlp_ts.pt       # traced TorchScript MLP
"""

import os
import json
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# SH basis and Lambert kernel (L = 3)
# ------------------------------------------------------------

def sh_basis_L3(d: torch.Tensor) -> torch.Tensor:
    """
    Real SH basis, Sloan L=3 ordering.

    d: [...,3] normalized direction (x,y,z)
    return: [...,16]
    """
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    one = torch.ones_like(x)

    Y = []

    # l=0
    Y.append(0.28209479177387814 * one)

    # l=1
    Y.append(0.4886025119029199 * y)
    Y.append(0.4886025119029199 * z)
    Y.append(0.4886025119029199 * x)

    # l=2
    Y.append(1.0925484305920792 * x * y)
    Y.append(1.0925484305920792 * y * z)
    Y.append(0.31539156525252005 * (3.0 * z * z - 1.0))
    Y.append(1.0925484305920792 * x * z)
    Y.append(0.5462742152960396 * (x * x - y * y))

    # l=3
    Y.append(0.5900435899266435 * y * (3.0 * x * x - y * y))
    Y.append(2.890611442640554 * x * y * z)
    Y.append(0.4570457994644658 * y * (5.0 * z * z - 1.0))
    Y.append(0.3731763325901154 * (5.0 * z * z * z - 3.0 * z))
    Y.append(0.4570457994644658 * x * (5.0 * z * z - 1.0))
    Y.append(1.445305721320277 * z * (x * x - y * y))
    Y.append(0.5900435899266435 * x * (x * x - 3.0 * y * y))

    return torch.stack(Y, dim=-1)  # [...,16]


def lambert_irradiance_from_sh_L3(C_sh: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """
    Lambert irradiance from SH L=3:

        E(n) = ∫_{Ω+} L(ω) max(0, n·ω) dω
             ≈ Σ (A_l * C_lm) Y_lm(n)

    For clamped-cosine kernel:
        A0 = π, A1 = 2π/3, A2 = π/4, A3 = 0

    C_sh:   [...,16,3]
    normal: [...,3]
    return: [...,3]
    """
    if C_sh.shape[-2] != 16:
        raise ValueError(f"C_sh must have 16 coeffs for L=3, got {C_sh.shape[-2]}")

    n = F.normalize(normal, dim=-1, eps=1e-8)
    Y_n = sh_basis_L3(n)  # [...,16]

    A0 = math.pi
    A1 = 2.0 * math.pi / 3.0
    A2 = math.pi / 4.0
    A3 = 0.0

    k = torch.tensor(
        [A0] * 1 + [A1] * 3 + [A2] * 5 + [A3] * 7,
        device=C_sh.device,
        dtype=C_sh.dtype,
    )  # [16]

    # Cw[...,j,c] = C_sh[...,j,c] * k[j]
    Cw = C_sh * k.view(*([1] * (C_sh.dim() - 2)), 16, 1)

    # E[...,c] = Σ_j Cw[...,j,c] * Y_n[...,j]
    E = torch.einsum("...jc,...j->...c", Cw, Y_n)
    return E


# ------------------------------------------------------------
# Assets: theta texture + probe SH grid + traced MLP
# ------------------------------------------------------------

class RealTimeAssets:
    """
    Load exported assets and provide sampling utilities:

      - theta_tex: [1, D, Ht, Wt]
      - probe_coeffs: [Nx, Ny, Nz, 16, 3]
      - aabb_min / aabb_max: world AABB for probe grid
      - mlp: torch.jit traced MLP
    """

    def __init__(self, assets_dir: str, device: str = "cuda"):
        self.assets_dir = assets_dir
        self.device = torch.device(device)

        # --- meta.json ---
        meta_path = os.path.join(assets_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # --- assets.npz ---
        npz_path = os.path.join(assets_dir, "assets.npz")
        data = np.load(npz_path)

        theta_hwd = data["theta_hwd"]          # [H,W,D]
        probe_xyzjc = data["probe_xyzjc"]      # [Nx,Ny,Nz,16,3]
        aabb_min = data["aabb_min"]            # [3]
        aabb_max = data["aabb_max"]            # [3]

        # theta: [H,W,D] -> [1,D,H,W]
        theta = torch.from_numpy(theta_hwd).permute(2, 0, 1)   # [D,H,W]
        self.theta_tex = theta.unsqueeze(0).to(self.device)    # [1,D,H,W]

        # probe coeffs: [Nx,Ny,Nz,16,3]
        self.probe_coeffs = torch.from_numpy(probe_xyzjc).to(self.device)

        # AABB
        self.aabb_min = torch.from_numpy(aabb_min).to(self.device)
        self.aabb_max = torch.from_numpy(aabb_max).to(self.device)

        # metadata
        self.grid_res: Tuple[int, int, int] = tuple(self.meta["probe"]["grid_res"])
        self.n_sh: int = int(self.meta["probe"]["n_sh"])
        assert self.n_sh == 16, "Implementation assumes L=3 (16 SH coeffs)."

        self.theta_dim: int = int(self.meta["theta"]["theta_dim"])
        self.theta_res: Tuple[int, int] = tuple(self.meta["theta"]["tex_res"])  # (H,W)

        # --- MLP (traced TorchScript) ---
        mlp_info = self.meta.get("mlp", None)
        if mlp_info is None:
            raise RuntimeError("meta.json has no 'mlp' entry; export_mlp_torchscript must be True.")

        mlp_path = os.path.join(assets_dir, mlp_info["file"])
        self.mlp = torch.jit.load(mlp_path, map_location=self.device).eval()

        print(f"[RealTimeAssets] loaded from {assets_dir}")
        print(f"  theta_tex:  [1, {self.theta_dim}, {self.theta_res[0]}, {self.theta_res[1]}]")
        print(f"  probe_grid: {self.grid_res}, n_sh={self.n_sh}")

    # ---------------------------
    # Sampling theta(uv) and SH(p)
    # ---------------------------

    def sample_theta(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: [...,2] in [0,1]
        return: [..., D]
        """
        uv = uv.to(self.device, dtype=self.theta_tex.dtype)

        # grid_sample expects [-1,1]
        grid = uv * 2.0 - 1.0                    # [...,2]
        orig_shape = uv.shape[:-1]
        grid = grid.view(1, -1, 1, 2)            # [1,N,1,2]

        feat = F.grid_sample(
            self.theta_tex,                      # [1,D,Ht,Wt]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                        # [1,D,N,1]
        feat = feat.squeeze(0).squeeze(-1).transpose(0, 1)  # [N,D]
        return feat.view(*orig_shape, -1)        # [...,D]

    def sample_probe_sh(self, p_world: torch.Tensor) -> torch.Tensor:
        """
        p_world: [...,3] world space position
        return: [...,16,3] SH coeffs at p (trilinear interpolation)
        """
        p_world = p_world.to(self.device)
        Nx, Ny, Nz = self.grid_res

        # Normalize to [0,1]
        p = (p_world - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8)
        p = p.clamp(0.0, 1.0)

        gx = p[..., 0] * (Nx - 1)
        gy = p[..., 1] * (Ny - 1)
        gz = p[..., 2] * (Nz - 1)

        x0 = torch.floor(gx).long()
        y0 = torch.floor(gy).long()
        z0 = torch.floor(gz).long()

        x1 = (x0 + 1).clamp(0, Nx - 1)
        y1 = (y0 + 1).clamp(0, Ny - 1)
        z1 = (z0 + 1).clamp(0, Nz - 1)

        tx = (gx - x0.float())[..., None, None]
        ty = (gy - y0.float())[..., None, None]
        tz = (gz - z0.float())[..., None, None]

        C = self.probe_coeffs  # [Nx,Ny,Nz,16,3]

        c000 = C[x0, y0, z0]
        c100 = C[x1, y0, z0]
        c010 = C[x0, y1, z0]
        c110 = C[x1, y1, z0]
        c001 = C[x0, y0, z1]
        c101 = C[x1, y0, z1]
        c011 = C[x0, y1, z1]
        c111 = C[x1, y1, z1]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx

        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty

        c = c0 * (1.0 - tz) + c1 * tz         # [...,16,3]
        return c


# ------------------------------------------------------------
# Real-time renderer: Lo = E(n,p) * F_hat(...)
# ------------------------------------------------------------

class RealTimeRenderer:
    """
    Evaluate:

        E(p,n) from probe SH
        theta(uv) from UV feature map
        F_hat = MLP(normal, position, albedo, view6, theta)
        Lo = E * F_hat
    """

    def __init__(self, assets: RealTimeAssets):
        self.assets = assets
        self.mlp = assets.mlp

    @torch.no_grad()
    def render(
        self,
        normal: torch.Tensor,
        position: torch.Tensor,
        albedo: torch.Tensor,
        view6: torch.Tensor,
        uv: torch.Tensor,
    ) -> torch.Tensor:
        """
        normal:   [...,3]
        position: [...,3]
        albedo:   [...,3]
        view6:    [...,6]  (cam_origin + cam_target or any 6D view encoding)
        uv:       [...,2] in [0,1]

        return Lo: [...,3]  (linear RGB)
        """
        device = self.assets.device

        normal = normal.to(device)
        position = position.to(device)
        albedo = albedo.to(device)
        view6 = view6.to(device)
        uv = uv.to(device)

        # 1) Probe SH → irradiance
        C_sh = self.assets.sample_probe_sh(position)          # [...,16,3]
        E = lambert_irradiance_from_sh_L3(C_sh, normal)       # [...,3]

        # 2) theta(uv)
        theta = self.assets.sample_theta(uv)                  # [...,D]

        # 3) Network factor
        F_hat = self.mlp(normal, position, albedo, view6, theta)  # [...,3]

        # 4) Final outgoing radiance
        Lo = E * F_hat
        return Lo


# ------------------------------------------------------------
# Utility: simple tonemap to uint8 sRGB for preview
# ------------------------------------------------------------

def to_uint8_srgb(x: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """
    x: HxWx3 linear RGB float32 (can be >1)
    exposure: exposure in stops
    return: uint8 sRGB image
    """
    x = x * (2.0 ** exposure)
    x = x / (1.0 + x)                # simple Reinhard tonemap
    x = np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)
    return (x * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------
# Example usage stub
# ------------------------------------------------------------

if __name__ == "__main__":
    """
    This is just a stub showing how to call the renderer.
    Replace the dummy G-buffer with your own data.
    """
    import imageio.v2 as imageio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assets_dir = "./realtime_assets"  # adjust if needed

    assets = RealTimeAssets(assets_dir, device=device)
    renderer = RealTimeRenderer(assets)

    # Dummy G-buffer: 128x128 grey quad as an example
    H, W = 128, 128
    nrm = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 3).expand(H, W, 3)
    pos = torch.zeros(H, W, 3)
    alb = torch.ones(H, W, 3) * 0.5
    uv = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, H),
            torch.linspace(0.0, 1.0, W),
            indexing="ij",
        ),
        dim=-1,
    )  # [H,W,2]
    cam_origin = torch.tensor([0.0, 1.0, 5.0])
    cam_target = torch.tensor([0.0, 1.0, 0.0])
    view6 = torch.cat([cam_origin, cam_target], dim=0).view(1, 1, 6).expand(H, W, 6)

    Lo = renderer.render(nrm, pos, alb, view6, uv)  # [H,W,3]
    Lo_np = Lo.cpu().numpy()

    img_u8 = to_uint8_srgb(Lo_np, exposure=0.0)
    os.makedirs("./rt_preview", exist_ok=True)
    out_path = "./rt_preview/demo.png"
    imageio.imwrite(out_path, img_u8)
    print("Saved preview to:", out_path)
