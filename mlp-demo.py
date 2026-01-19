import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import mitsuba as mi
import os, json
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------
# 1) 基础 MLP：\hat F(normal, position, view, theta)
# -------------------------
def fc_layer(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True),
    )


class PerPixelMLP(nn.Module):
    """
    per-pixel MLP: 逐像素独立，不做卷积邻域聚合
    输入: normal(3), position(3), view(3), theta(D)
    输出: RGB(3) 或标量(1)，用于近似你草稿里的第二项积分/因子
    """
    def __init__(
        self,
        theta_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 6,
        out_features: int = 3,
        use_skip: bool = True,
        out_activation: str | None = None,  # None | "sigmoid" | "softplus"
    ):
        super().__init__()
        in_dim = 3 + 3 + 3 + 6 + theta_dim
        self.in_dim = in_dim
        self.out_features = out_features
        self.use_skip = use_skip
        self.out_activation = out_activation

        self.inner = fc_layer(in_dim, hidden_dim)

        self.hidden = nn.ModuleList()
        self._skip_mask = []
        for i in range(num_hidden_layers):
            do_skip = use_skip and (i % 2 == 1)
            self._skip_mask.append(do_skip)
            layer_in = hidden_dim + in_dim if do_skip else hidden_dim
            self.hidden.append(fc_layer(layer_in, hidden_dim))

        self.outer = nn.Linear(hidden_dim, out_features)

    def forward(self, normal, position, albedo, view, theta):
        # 支持 [...,C] 任意 batch 形状；最后一维是通道
        x = torch.cat([normal, position, albedo, view,theta], dim=-1)  # [..., in_dim]
        x0 = x.reshape(-1, self.in_dim)  # [N, in_dim]

        h = self.inner(x0)
        for layer, do_skip in zip(self.hidden, self._skip_mask):
            h = layer(torch.cat([h, x0], dim=-1) if do_skip else h)

        y = self.outer(h)  # [N, out_features]

        if self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation == "softplus":
            y = F.softplus(y)

        # reshape 回原形状（把最后一维换成 out_features）
        out_shape = list(x.shape[:-1]) + [self.out_features]
        return y.view(*out_shape)

# class PerPixelMLP(nn.Module):
#     def __init__(self, theta_dim=32, hidden_dim=256, num_hidden=6, out_dim=3):
#         super().__init__()
#         in_dim = 3 + 3 + 6 + theta_dim  # normal + position + view + theta
#         self.inner = fc_layer(in_dim, hidden_dim)
#         self.hidden = nn.ModuleList([fc_layer(hidden_dim, hidden_dim) for _ in range(num_hidden)])
#         self.outer = nn.Linear(hidden_dim, out_dim)

#     def forward(self, normal, position, view, theta):
#         x = torch.cat([normal, position, view, theta], dim=-1)  # [B,H,W,C]
#         B, H, W, C = x.shape
#         x = x.view(B * H * W, C)
#         h = self.inner(x)
#         for layer in self.hidden:
#             h = layer(h)
#         y = self.outer(h).view(B, H, W, 3)
#         return y

# -------------------------
# 2) Learnable 探针网格（SH 系数）+ 三线性插值：C_j(p)=Σ w_i C_j^(i)
# -------------------------
import math
import torch
import torch.nn as nn


class ProbeGridSH(nn.Module):
    """
    探针网格参数化：
      coeffs[x,y,z, j, rgb] 为每个探针的 SH 系数（可训练）
    对任意点 p，做 trilinear 得到 C_j(p)。
    """

    def __init__(
        self,
        grid_res: tuple[int, int, int],   # (Nx, Ny, Nz)
        n_sh: int,                        # L=3 -> 16
        aabb_min: torch.Tensor,           # [3]
        aabb_max: torch.Tensor,           # [3]
        init_scale: float = 0.01,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        Nx, Ny, Nz = grid_res
        self.grid_res = grid_res
        self.n_sh = n_sh

        if self.n_sh != 16:
            raise ValueError(f"For L=3, n_sh must be 16, got {self.n_sh}")

        self.register_buffer("aabb_min", aabb_min.to(device=device, dtype=dtype))
        self.register_buffer("aabb_max", aabb_max.to(device=device, dtype=dtype))

        coeffs = init_scale * torch.randn(Nx, Ny, Nz, n_sh, 3, device=device, dtype=dtype)
        self.coeffs = nn.Parameter(coeffs)  # <- learnable probe SH coeffs

    def sample_coeffs_trilinear(self, p_world: torch.Tensor) -> torch.Tensor:
        """
        p_world: [...,3] 世界坐标
        return:  [..., n_sh, 3] 插值得到的 C_j(p)
        """
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

        return c0 * (1 - tz) + c1 * tz  # [...,16,3]

    # -------------------------
    # L=3 (16 coeffs) Real SH basis (Sloan-style ordering)
    # -------------------------
    @staticmethod
    def sh_basis_L3(d: torch.Tensor) -> torch.Tensor:
        """
        d: [...,3] normalized direction (x,y,z)
        returns Y: [...,16] real SH basis values with ordering:
          l=0:  0
          l=1:  1..3   (m=-1,0,1)
          l=2:  4..8   (m=-2..2)
          l=3:  9..15  (m=-3..3)
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

    @staticmethod
    def lambert_irradiance_from_sh_L3(C_sh: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """
        Lambert irradiance:
          E(n) = ∫_{Ω+} L(ω) (n·ω) dω
        With SH lighting L(ω)=Σ C_j Y_j(ω), this becomes:
          E(n) = Σ (A_l * C_lm) Y_lm(n), where A_l are clamped-cosine kernel coeffs.
        For this kernel: A0=π, A1=2π/3, A2=π/4, A3=0  (so L=3 adds no extra irradiance term).

        C_sh:   [...,16,3]  SH coeffs at point p (after trilinear)
        normal: [...,3]     surface normal (will be normalized)
        return: [...,3]     irradiance RGB
        """
        if C_sh.shape[-2] != 16:
            raise ValueError(f"C_sh must have 16 SH coeffs for L=3, got {C_sh.shape[-2]}")

        n = torch.nn.functional.normalize(normal, dim=-1, eps=1e-8)
        Y_n = ProbeGridSH.sh_basis_L3(n)  # [...,16]

        A0 = math.pi
        A1 = 2.0 * math.pi / 3.0
        A2 = math.pi / 4.0
        A3 = 0.0

        k = torch.tensor(
            [A0] * 1 + [A1] * 3 + [A2] * 5 + [A3] * 7,
            device=C_sh.device, dtype=C_sh.dtype
        )  # [16]

        # 加权系数：Cw[...,j,c] = C_sh[...,j,c] * k[j]
        Cw = C_sh * k.view(*([1] * (C_sh.dim() - 2)), 16, 1)

        # E[...,c] = Σ_j Cw[...,j,c] * Y_n[...,j]
        E = torch.einsum("...jc,...j->...c", Cw, Y_n)
        return E



# -------------------------
# 3) Learnable θ：UV 高维特征贴图 + grid_sample 双线性采样
# -------------------------
class ThetaTexture(nn.Module):
    """
    theta_tex: [1, D, Ht, Wt] 作为 nn.Parameter
    输入 uv ∈ [0,1]，输出 theta(uv) ∈ R^D
    """
    def __init__(
        self,
        theta_dim: int,
        tex_res: tuple[int, int],  # (Ht, Wt)
        init_scale: float = 0.01,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        Ht, Wt = tex_res
        tex = torch.zeros(1, theta_dim, Ht, Wt, device=device, dtype=dtype)
        tex = tex + init_scale * torch.randn_like(tex)
        self.tex = nn.Parameter(tex)  # <- 可优化变量（θ）

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """
        uv: [...,2] in [0,1]
        return: [..., D]
        """
        # grid_sample 需要 [-1,1]
        grid = uv * 2.0 - 1.0  # [...,2]
        # 组织成 [N=1, H_out, W_out, 2] 的 grid
        # 这里把所有点展平到 H_out=Npts, W_out=1
        orig_shape = uv.shape[:-1]
        grid = grid.reshape(1, -1, 1, 2)

        # 双线性采样：输出 [1, D, Npts, 1]
        feat = F.grid_sample(
            self.tex,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        feat = feat.squeeze(0).squeeze(-1).transpose(0, 1)  # [Npts, D]
        return feat.view(*orig_shape, -1)


# -------------------------
# 4) 把整条链拼起来：L_o ≈ (∫ L_i dω) * \hat F(...)
# -------------------------
class FactorizedRenderer(nn.Module):
    def __init__(self, probe_grid: ProbeGridSH, theta_tex: ThetaTexture, mlp: PerPixelMLP):
        super().__init__()
        self.probe_grid = probe_grid
        self.theta_tex = theta_tex
        self.mlp = mlp

    def forward(self, normal, position, albedo, view, uv):
        """
        normal:   [...,3]
        position: [...,3]
        view:     [...,3]
        uv:       [...,2] in [0,1]
        return Lo: [...,3]
        """
        # 1) probe trilinear -> SH coeffs at p
        c_sh = self.probe_grid.sample_coeffs_trilinear(position)     # [..., n_sh, 3]

        # 2) light factor: ∫ L_i dω （用 DC 项闭式）
        E = self.probe_grid.lambert_irradiance_from_sh_L3(c_sh, normal)  # [...,3]

        # 3) theta(uv) from learnable texture
        theta = self.theta_tex.sample(uv)  # [..., D]

        # 4) reflectance/network factor
        F_hat = self.mlp(normal, position, albedo, view, theta)  # [...,3] (or [...,1])

        # 5) factorized output
        Lo = E * F_hat
        return Lo



class MitsubaNPZDataset(Dataset):
    def __init__(self, root):
        self.root = root
        meta_path = os.path.join(root, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.frames = meta["frames"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        rel = self.frames[idx]["file"]
        path = os.path.join(self.root, rel)
        data = np.load(path)

        rgb = torch.from_numpy(data["rgb"]).float()           # [H,W,3]
        pos = torch.from_numpy(data["position"]).float()      # [H,W,3]
        nrm = torch.from_numpy(data["sh_normal"]).float()     # [H,W,3]
        alb = torch.from_numpy(data["albedo"]).float()     # [H,W,3]
        uv  = torch.from_numpy(data["uv"]).float()            # [H,W,2]
        cam_origin = torch.from_numpy(data["cam_origin"]).float()  # [3]
        cam_target = torch.from_numpy(data["cam_target"]).float()  # [3]

        # view = normalize(cam_origin - position)
        view = torch.cat([cam_origin, cam_target], dim=0)         # [6]
        view = view.view(1, 1, 6).expand(pos.shape[0], pos.shape[1], 6)  # [H,W,6]
        # 规范化 normal（以防渲染输出非单位长度）
        nrm = F.normalize(nrm, dim=-1, eps=1e-6)

        return {
            "rgb": rgb,
            "position": pos,
            "normal": nrm,
            "albedo": alb,
            "uv": uv,
            "view": view,
        }

# class CompositeRenderer(nn.Module):
#     """
#     final_pred = indirect_renderer(...) + s * albedo
#     s: learnable RGB vector
#     """
#     def __init__(self, indirect_renderer: nn.Module, s_init=1, constrain_nonneg=False):
#         super().__init__()
#         self.indirect = indirect_renderer
#         self.constrain_nonneg = constrain_nonneg

#         if constrain_nonneg:
#             # 用 softplus 约束 s >= 0（可选）
#             s0 = torch.full((3,), float(s_init))
#             self.s_raw = nn.Parameter(torch.log(torch.exp(s0) - 1.0))  # inverse softplus
#         else:
#             self.s = nn.Parameter(torch.full((3,), float(s_init)))

#     def get_s(self):
#         if self.constrain_nonneg:
#             return F.softplus(self.s_raw)
#         return self.s

#     def forward(self, nrm, pos, view, uv, albedo):
#         indirect = self.indirect(nrm, pos, view, uv)  # [B,H,W,3]
#         s = self.get_s().view(1, 1, 1, 3)             # broadcast
#         pred = indirect + s * albedo                  # [B,H,W,3]
#         return pred, indirect
# -------------------------
# 5) 训练：把探针和 θ 加进 optimizer，靠渲染损失反传更新
# -------------------------

def train_epochs(
    root: str = "./dataset",
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-3,
    num_workers: int = 0,
    log_every: int = 10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset / DataLoader
    ds = MitsubaNPZDataset(root)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # --- 可训练对象：probe coeffs & theta texture & mlp weights
    n_sh = 16  # L=3 -> 16
    probe = ProbeGridSH(
        grid_res=(128, 128, 128),
        n_sh=n_sh,
        aabb_min=torch.tensor([-9.64634, -0.01676, -7.6]),
        aabb_max=torch.tensor([7.43, 3.8, 0.599999]),
        device=device,
    ).to(device)

    theta_tex = ThetaTexture(theta_dim=32, tex_res=(256, 256)).to(device)

    # PerPixelMLP 必须已改为 view 输入
    mlp = PerPixelMLP(
        theta_dim=32, hidden_dim=256, num_hidden_layers=2, out_features=3, use_skip=True
    ).to(device)

    renderer = FactorizedRenderer(probe, theta_tex, mlp).to(device)

    optim = torch.optim.Adam(renderer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=1, gamma=0.5   # 每 10 个 epoch，lr *= 0.5
        )
    for epoch in range(epochs):
        renderer.train()
        running = 0.0
        n_batches = 0

        for it, batch in enumerate(dl):
            rgb   = batch["rgb"].to(device, non_blocking=True)        # [B,H,W,3]
            albedo   = batch["albedo"].to(device, non_blocking=True)        # [B,H,W,3]
            pos   = batch["position"].to(device, non_blocking=True)   # [B,H,W,3]
            nrm   = batch["normal"].to(device, non_blocking=True)     # [B,H,W,3]
            uv    = batch["uv"].to(device, non_blocking=True)         # [B,H,W,2]
            view = batch["view"].to(device, non_blocking=True)      # [B,H,W,6]

            pred = renderer(nrm, pos, albedo, view, uv)                  # [B,H,W,3]
            loss = F.mse_loss(pred, rgb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.item())
            n_batches += 1
            scheduler.step()  # 关键：epoch 结束后 step
            if log_every and ((it + 1) % log_every == 0):
                print(f"epoch {epoch:03d} iter {it+1:04d}/{len(dl)} loss={loss.item():.6f}")

        epoch_loss = running / max(1, n_batches)
        print(f"[epoch {epoch:03d}] mean_loss={epoch_loss:.6f}")

    return renderer
import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

def _to_uint8_srgb(x: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """
    x: HxWx3 linear RGB float32 (可能 >1)
    exposure: stops, e.g. 0.0, 1.0, -1.0
    return: uint8 sRGB
    """
    # exposure
    x = x * (2.0 ** exposure)

    # tone map (Reinhard) to [0,1]
    x = x / (1.0 + x)

    # gamma to sRGB-ish
    x = np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)
    return (x * 255.0 + 0.5).astype(np.uint8)

@torch.no_grad()
def save_preview(renderer, dataset_root="./dataset", frame_idx=0, out_path="./preview.png", exposure=0.0):
    renderer.eval()
    device = next(renderer.parameters()).device

    ds = MitsubaNPZDataset(dataset_root)
    sample = ds[frame_idx]

    rgb   = sample["rgb"].unsqueeze(0).to(device)        # [1,H,W,3] GT
    albedo   = sample["albedo"].unsqueeze(0).to(device)        # [1,H,W,3] GT
    pos   = sample["position"].unsqueeze(0).to(device)   # [1,H,W,3]
    nrm   = sample["normal"].unsqueeze(0).to(device)     # [1,H,W,3]
    uv    = sample["uv"].unsqueeze(0).to(device)         # [1,H,W,2]
    view = sample["view"].unsqueeze(0).to(device)      # [1,H,W,6]

    pred = renderer(nrm, pos, albedo, view, uv)         # [1,H,W,3]

    pred_np = pred.squeeze(0).detach().cpu().numpy()
    gt_np   = rgb.squeeze(0).detach().cpu().numpy()

    pred_u8 = _to_uint8_srgb(pred_np, exposure=exposure)
    gt_u8   = _to_uint8_srgb(gt_np, exposure=exposure)

    # side-by-side: [pred | gt]
    vis = np.concatenate([pred_u8, gt_u8], axis=1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.imwrite(out_path, vis)
    imageio.imwrite("./pr.png", pred_u8)
    imageio.imwrite("./gt.png", gt_u8)
    print("Saved:", out_path, "(left=pred, right=gt)")

if __name__ == "__main__":
    renderer = train_epochs(root="./dataset", epochs=2, batch_size=4, lr=1e-3)
    save_preview(renderer, "./dataset", frame_idx=0, out_path="./preview.png", exposure=0.0)