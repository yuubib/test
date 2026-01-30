import os
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

# ============================================================
# 配置
# ============================================================

@dataclass(frozen=True)
class RenderConfig:
    scene_path: str = "veach-ajar/scene.xml"
    out_dir: str = "./dataset"
    n_frames: int = 1
    spp: int = 1024

    tex_res: int = 512
    pad_texels: int = 4

    center: np.ndarray = field(default_factory=lambda: np.array([4.05402, 1.61647, -2.30652], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.array([3.06401, 1.58417, -2.44373], dtype=np.float32))
    up:     np.ndarray = field(default_factory=lambda: np.array([-0.0319925, 0.999478, -0.00443408], dtype=np.float32))
    radius: float = 1.0

    variant: str = "cuda_ad_rgb"
    apply_camera_to_scene: bool = False

# ============================================================
# 工具：Bitmap -> numpy
# ============================================================

class BitmapUtils:
    @staticmethod
    def to_numpy(
            bmp: mi.Bitmap,
            pixel_format: mi.Bitmap.PixelFormat = mi.Bitmap.PixelFormat.RGBA,
            float_type: mi.Struct.Type = mi.Struct.Type.Float32,
            srgb_gamma: bool = False,
    ) -> np.ndarray:
        b = bmp.convert(srgb_gamma=srgb_gamma)
        arr = np.array(b, copy=False)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def split_film_components(film: mi.Film, raw: bool = False) -> Dict[str, mi.Bitmap]:
        comps = film.bitmap(raw=raw).split()
        return {name: bmp for name, bmp in comps}


# ============================================================
# UV 退化检测 + 基于 positions 重建 uv01 + Atlas Packing
# ============================================================

class AtlasPacker:
    def __init__(self, params: mi.SceneParameters, tex_res: int, pad_texels: int):
        self.params = params
        self.tex_res = int(tex_res)
        self.pad_texels = int(pad_texels)

    def _rebuild_uv_from_positions(self, pos_flat) -> mi.Vector2f:
        P = dr.unravel(mi.Point3f, pos_flat)  # Point3f[N]
        xmin, ymin, zmin = dr.min(P.x), dr.min(P.y), dr.min(P.z)
        xmax, ymax, zmax = dr.max(P.x), dr.max(P.y), dr.max(P.z)
        ex, ey, ez = (xmax - xmin), (ymax - ymin), (zmax - zmin)

        # 选跨度最小的轴作为“丢弃轴”，用另外两轴做平面展开
        if ex <= ey and ex <= ez:
            u = (P.y - ymin) / dr.maximum(ey, 1e-8)
            v = (P.z - zmin) / dr.maximum(ez, 1e-8)
        elif ey <= ex and ey <= ez:
            u = (P.x - xmin) / dr.maximum(ex, 1e-8)
            v = (P.z - zmin) / dr.maximum(ez, 1e-8)
        else:
            u = (P.x - xmin) / dr.maximum(ex, 1e-8)
            v = (P.y - ymin) / dr.maximum(ey, 1e-8)

        return mi.Vector2f(u, v)  # [0,1]（近似）

    def _is_degenerate_uv(self, uv_flat, eps: float = 1e-8) -> bool:
        UV = dr.unravel(mi.Vector2f, uv_flat)
        du = dr.max(UV.x) - dr.min(UV.x)
        dv = dr.max(UV.y) - dr.min(UV.y)
        m = (du < eps) & (dv < eps)
        return bool(m)

    def _find_mesh_pairs(self) -> List[Tuple[str, str]]:
        # vertex_texcoords <-> vertex_positions
        uv_keys = sorted([k for k in self.params.keys() if k.endswith("vertex_texcoords")])
        pairs: List[Tuple[str, str]] = []
        for uk in uv_keys:
            pk = uk.replace("vertex_texcoords", "vertex_positions")
            if pk in self.params:
                pairs.append((pk, uk))
        return pairs

    def pack(self) -> Dict[str, Any]:
        pairs = self._find_mesh_pairs()
        if not pairs:
            raise RuntimeError("未找到可编辑的 mesh vertex_positions/vertex_texcoords（可能是解析几何或网格无UV）。")

        n_mesh = len(pairs)
        K = int(math.ceil(math.sqrt(n_mesh)))
        tile = 1.0 / K

        pad = self.pad_texels / float(self.tex_res)
        inside = tile - 2.0 * pad
        if inside <= 0:
            raise RuntimeError("padding 过大：inside<=0")

        atlas_meta: Dict[str, Any] = {
            "K": K,
            "tile": tile,
            "pad": pad,
            "inside": inside,
            "tex_res": self.tex_res,
            "pad_texels": self.pad_texels,
            "meshes": []
        }

        for i, (pk, uk) in enumerate(pairs):
            tx, ty = i % K, i // K
            u0 = tx * tile + pad
            v0 = ty * tile + pad

            pos_flat = self.params[pk]
            uv_flat = self.params[uk]

            if self._is_degenerate_uv(uv_flat):
                uv01 = self._rebuild_uv_from_positions(pos_flat)
            else:
                uv01 = dr.unravel(mi.Vector2f, uv_flat)

            uv_atlas = uv01 * inside + mi.Vector2f(u0, v0)
            self.params[uk] = dr.ravel(uv_atlas)

            atlas_meta["meshes"].append({
                "i": i,
                "pk": pk,
                "uk": uk,
                "tile_xy": [int(tx), int(ty)],
                "uv_offset": [float(u0), float(v0)],
            })

        return atlas_meta


# ============================================================
# 渲染与 AOV 提取
# ============================================================
x = 1
class AOVDatasetRenderer:
    def __init__(
            self,
            scene: mi.Scene,
            params: mi.SceneParameters,
            required_aovs: List[str],
            integrator_aov_spec: str,
    ):
        self.scene = scene
        self.params = params
        self.required_aovs = required_aovs
        self.integrator = mi.load_dict({
            "type": "aov",
            "aovs": integrator_aov_spec,  # e.g. "uv:uv, position:position, sh_normal:sh_normal, albedo:albedo"
            "img": {"type": "path"}
        })

        self.cam_key = self._find_camera_to_world_key()

    def _find_camera_to_world_key(self) -> str:
        cam_keys = [k for k in self.params.keys() if k.endswith("to_world") and "sensor" in k]
        if not cam_keys:
            raise RuntimeError("未找到 sensor.to_world")
        return cam_keys[0]

    @staticmethod
    def _pick_main_img_key(comp: Dict[str, mi.Bitmap], aov_names: List[str]) -> str:
        # 常见主输出 key
        for cand in ["img", "color", "beauty", "root", "rgba"]:
            if cand in comp:
                return cand
        aov_set = set(aov_names)
        for k in comp.keys():
            if k not in aov_set:
                return k
        raise RuntimeError(f"Cannot find main RGB output. Keys: {list(comp.keys())}")

    def render_frame(
            self,
            spp: int,
            film_raw: bool = False,
    ) -> Dict[str, np.ndarray]:
        film = self.scene.sensors()[0].film()
        film.clear()

        _ = mi.render(self.scene, integrator=self.integrator, spp=int(spp))

        comp = BitmapUtils.split_film_components(film, raw=film_raw)

        # AOV 检查
        for k in self.required_aovs:
            if k not in comp:
                raise RuntimeError(f"Missing AOV '{k}'. Available keys: {list(comp.keys())}")

        img_key = self._pick_main_img_key(comp, self.required_aovs)
        global x
        mi.util.write_bitmap(f"./xxx/rgb_{x}.exr", comp[img_key])
        x= x + 1
        # 输出统一为 float32 HxWx3 / HxWx2
        # rgb 做log1p处理
        rgb = np.log1p(BitmapUtils.to_numpy(comp[img_key], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3])

        # pos进行归一化
        pos = BitmapUtils.to_numpy(comp["position"], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3]
        pos = (pos - np.array(self.scene.bbox().min + self.scene.bbox().max).reshape(1, 1, 3)) / (
                np.array(self.scene.bbox().max - self.scene.bbox().min) * 0.5).reshape(1, 1, 3)

        nrm = BitmapUtils.to_numpy(comp["sh_normal"], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3]
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)

        alb = BitmapUtils.to_numpy(comp["albedo"], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3]

        uv_rgba = np.array(comp["uv"], copy=False).astype(np.float32, copy=False)
        uv = uv_rgba[..., :2]

        return {
            "img_key": np.array([img_key], dtype=object),  # 仅用于调试（可选）
            "rgb": rgb,
            "position": pos,
            "sh_normal": nrm,
            "albedo": alb,
            "uv": uv,
        }


class AOVRuntimeRenderer:
    def __init__(
            self,
            scene: mi.Scene,
            params: mi.SceneParameters,
            required_aovs: List[str],
            integrator_aov_spec: str,
    ):
        self.scene = scene
        self.params = params
        self.required_aovs = required_aovs
        self.integrator = mi.load_dict({
            "type": "aov",
            "aovs": integrator_aov_spec,  # e.g. "uv:uv, position:position, sh_normal:sh_normal, albedo:albedo"
        })

        self.cam_key = self._find_camera_to_world_key()

    def _find_camera_to_world_key(self) -> str:
        cam_keys = [k for k in self.params.keys() if k.endswith("to_world") and "sensor" in k]
        if not cam_keys:
            raise RuntimeError("未找到 sensor.to_world")
        return cam_keys[0]

    @staticmethod
    def _pick_main_img_key(comp: Dict[str, mi.Bitmap], aov_names: List[str]) -> str:
        # 常见主输出 key
        for cand in ["img", "color", "beauty", "root", "rgba"]:
            if cand in comp:
                return cand
        aov_set = set(aov_names)
        for k in comp.keys():
            if k not in aov_set:
                return k
        raise RuntimeError(f"Cannot find main RGB output. Keys: {list(comp.keys())}")

    def render_frame(
            self,
            spp: int,
            film_raw: bool = False,
    ) -> Dict[str, np.ndarray]:
        film = self.scene.sensors()[0].film()
        film.clear()

        _ = mi.render(self.scene, integrator=self.integrator, spp=int(spp))

        comp = BitmapUtils.split_film_components(film, raw=film_raw)

        # AOV 检查
        for k in self.required_aovs:
            if k not in comp:
                raise RuntimeError(f"Missing AOV '{k}'. Available keys: {list(comp.keys())}")
        # 输出统一为 float32 HxWx3 / HxWx2
        # rgb 做log1p处理
        # pos进行归一化
        pos = BitmapUtils.to_numpy(comp["position"], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3]

        nrm = BitmapUtils.to_numpy(comp["sh_normal"], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3]
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)

        alb = BitmapUtils.to_numpy(comp["albedo"], mi.Bitmap.PixelFormat.RGBA, srgb_gamma=False)[..., :3]

        uv_rgba = np.array(comp["uv"], copy=False).astype(np.float32, copy=False)
        #uv = uv_rgba[..., :2]
        # 不知名bug
        uv = uv_rgba[..., :2][..., [1, 0]]  

        return {
            "position": pos,
            "sh_normal": nrm,
            "albedo": alb,
            "uv": uv,
        }



# ============================================================
# 轨迹
# ============================================================

class CameraTrajectory:
    def __init__(self, center: np.ndarray, target: np.ndarray, up: np.ndarray, radius: float):
        self.center = center.astype(np.float32)
        self.target = target.astype(np.float32)
        self.up = up.astype(np.float32)
        self.radius = float(radius)

    def pose(self, f: int, n_frames: int) -> Tuple[np.ndarray, mi.ScalarTransform4f]:
        theta = 0.5 * np.pi * (f / max(n_frames, 1))

        origin = self.center + np.array(
            [self.radius * np.sin(theta), 0.0, self.radius * (np.cos(theta)-np.pi / 3)],
            dtype=np.float32
        )
        target = self.target

        T = mi.ScalarTransform4f.look_at(
            origin=mi.ScalarPoint3f(*origin),
            target=mi.ScalarPoint3f(*target),
            up=mi.ScalarVector3f(*self.up),
        )
        return origin, T


# ============================================================
# 数据集写入
# ============================================================

class DatasetWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

    def write_frame_npz(
            self,
            frame_idx: int,
            data: Dict[str, np.ndarray],
            cam_to_world: np.ndarray,
            cam_origin: np.ndarray,
            cam_target: np.ndarray,
            cam_up: np.ndarray,
    ) -> str:
        frame_path = os.path.join(self.frames_dir, f"frame_{frame_idx:04d}.npz")
        np.savez_compressed(
            frame_path,
            rgb=data["rgb"],
            position=data["position"],
            sh_normal=data["sh_normal"],
            albedo=data["albedo"],
            uv=data["uv"],
            cam_to_world=cam_to_world.astype(np.float32, copy=False),
            cam_origin=cam_origin.astype(np.float32, copy=False),
            cam_target=cam_target.astype(np.float32, copy=False),
            cam_up=cam_up.astype(np.float32, copy=False),
        )
        return frame_path

    def write_meta_json(self, meta: Dict[str, Any]) -> None:
        path = os.path.join(self.out_dir, "meta.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)


# ============================================================
# 主流程
# ============================================================

class DatasetPipeline:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

        mi.set_variant(cfg.variant)
        self.scene = mi.load_file(cfg.scene_path)
        self.params = mi.traverse(self.scene)

        self.writer = DatasetWriter(cfg.out_dir)

    def _pack_uv_atlas(self) -> Dict[str, Any]:
        packer = AtlasPacker(self.params, tex_res=self.cfg.tex_res, pad_texels=self.cfg.pad_texels)
        atlas_meta = packer.pack()
        self.params.update()
        return atlas_meta
    
    def runtime(self, param, traj:CameraTrajectory):
        renderer = AOVRuntimeRenderer(
            scene=self.scene,
            params=self.params,
            required_aovs=["uv", "position", "sh_normal", "albedo"],
            integrator_aov_spec="uv:uv, position:position, sh_normal:sh_normal, albedo:albedo",
        )   
        
        theta = 0.5 * np.pi * param
        origin = traj.center + np.array(
            [traj.radius * np.sin(theta), 0.0, traj.radius * (np.cos(theta)-np.pi / 3)],
            dtype=np.float32
        )
        target = traj.target

        T = mi.ScalarTransform4f.look_at(
            origin=mi.ScalarPoint3f(*origin),
            target=mi.ScalarPoint3f(*target),
            up=mi.ScalarVector3f(*self.cfg.up),
        )
        self.params[renderer.cam_key] = T
        self.params.update()

        frame_data = renderer.render_frame(spp=1, film_raw=False)       
        frame_data["origin"] = origin         
        frame_data["target"] = target          
        return frame_data
    
    def run(self) -> None:
        atlas_meta = self._pack_uv_atlas()

        renderer = AOVDatasetRenderer(
            scene=self.scene,
            params=self.params,
            required_aovs=["uv", "position", "sh_normal", "albedo"],
            integrator_aov_spec="uv:uv, position:position, sh_normal:sh_normal, albedo:albedo",
        )

        traj = CameraTrajectory(
            center=self.cfg.center,
            target=self.cfg.target,
            up=self.cfg.up,
            radius=self.cfg.radius,
        )

        film = self.scene.sensors()[0].film()
        size = film.size()

        meta: Dict[str, Any] = {
            "scene": self.cfg.scene_path,
            "variant": self.cfg.variant,
            "n_frames": int(self.cfg.n_frames),
            "spp": int(self.cfg.spp),
            "camera_key": renderer.cam_key,
            "center": self.cfg.center.tolist(),
            "target": self.cfg.target.tolist(),
            "up": self.cfg.up.tolist(),
            "radius": float(self.cfg.radius),
            "atlas": atlas_meta,
            "res": [int(size[0]), int(size[1])],
            "frames": [],
        }

        for f in range(self.cfg.n_frames):
            origin, T = traj.pose(f, self.cfg.n_frames)

            if self.cfg.apply_camera_to_scene:
                self.params[renderer.cam_key] = T
                print(T)
                self.params.update()

            frame_data = renderer.render_frame(spp=self.cfg.spp, film_raw=False)

            T_np = np.array(T.matrix, dtype=np.float32)
            frame_path = self.writer.write_frame_npz(
                frame_idx=f,
                data=frame_data,
                cam_to_world=T_np,
                cam_origin=origin,
                cam_target=self.cfg.target,
                cam_up=self.cfg.up,
            )

            # 记录帧信息
            img_key = str(frame_data["img_key"][0]) if "img_key" in frame_data else "unknown"
            meta["frames"].append({
                "frame": int(f),
                "file": f"frames/frame_{f:04d}.npz",
                "img_key": img_key,
            })

            print(f"[{f:04d}] saved -> {frame_path} (img_key={img_key})")

        self.writer.write_meta_json(meta)
        print("Done. Dataset written to:", self.cfg.out_dir)


def main():
    cfg = RenderConfig(
        scene_path="veach-ajar/scene.xml",
        out_dir="./dataset",
        n_frames=1,
        spp=1024,
        tex_res=512,
        pad_texels=4,
        center=np.array([4.05402, 1.61647, -2.30652], dtype=np.float32),
        target=np.array([3.06401, 1.58417, -2.44373], dtype=np.float32),
        up=np.array([-0.0319925, 0.999478, -0.00443408], dtype=np.float32),
        radius=0.2,
        variant="cuda_ad_rgb",
        apply_camera_to_scene=True,  # 若你确实要改 scene 相机（而非仅保存 pose），设 True
    )
    DatasetPipeline(cfg).run()


if __name__ == "__main__":
    main()
