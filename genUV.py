import os, json, math
import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")

# -----------------------------
# 配置
# -----------------------------
SCENE_PATH = "veach-ajar/scene.xml"
OUT_DIR = "./dataset"
N_FRAMES = 1
SPP = 1024

TEX_RES = 512
PAD_TEXELS = 4

center = np.array([4.05402, 1.61647, -2.30652], dtype=np.float32)
target = np.array([3.06401, 1.58417, -2.44373], dtype=np.float32)
up     = np.array([-0.0319925, 0.999478, -0.00443408], dtype=np.float32)
radius = 1.0

os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)

# -----------------------------
# 工具：Bitmap -> numpy
# -----------------------------
def bitmap_to_numpy(bmp, pixel_format=mi.Bitmap.PixelFormat.RGBA):
    #b = bmp.convert(pixel_format, mi.Struct.Type.Float32, srgb_gamma=False)
    arr = np.array(bmp, copy=False)  # 通常为 HxWxC float32
    return arr.astype(np.float32, copy=False)

def get_component_dict(film):
    comps = film.bitmap(raw=False).split()
    d = {}
    for name, bmp in comps:
        d[name] = bmp
    return d

# -----------------------------
# UV 退化检测 + 基于 positions 重建 uv01
# -----------------------------
def rebuild_uv_from_positions(pos_flat):
    P = dr.unravel(mi.Point3f, pos_flat)  # Point3f[N]
    xmin, ymin, zmin = dr.min(P.x), dr.min(P.y), dr.min(P.z)
    xmax, ymax, zmax = dr.max(P.x), dr.max(P.y), dr.max(P.z)
    ex, ey, ez = (xmax - xmin), (ymax - ymin), (zmax - zmin)

    if ex <= ey and ex <= ez:
        u = (P.y - ymin) / dr.maximum(ey, 1e-8)
        v = (P.z - zmin) / dr.maximum(ez, 1e-8)
    elif ey <= ex and ey <= ez:
        u = (P.x - xmin) / dr.maximum(ex, 1e-8)
        v = (P.z - zmin) / dr.maximum(ez, 1e-8)
    else:
        u = (P.x - xmin) / dr.maximum(ex, 1e-8)
        v = (P.y - ymin) / dr.maximum(ey, 1e-8)

    return mi.Vector2f(u, v)  # uv in [0,1]

def is_degenerate_uv(uv_flat, eps=1e-8):
    UV = dr.unravel(mi.Vector2f, uv_flat)
    du = dr.max(UV.x) - dr.min(UV.x)
    dv = dr.max(UV.y) - dr.min(UV.y)
    return (du < eps) & (dv < eps)

# -----------------------------
# 主流程：加载场景 + pack UV + 渲染 AOV + 写 NPZ
# -----------------------------
scene = mi.load_file(SCENE_PATH)
params = mi.traverse(scene)

# 找 mesh keys
pos_keys = sorted([k for k in params.keys() if k.endswith("vertex_positions")])
uv_keys  = sorted([k for k in params.keys() if k.endswith("vertex_texcoords")])
print(uv_keys)
pairs = []
for uk in uv_keys:
    pk = uk.replace("vertex_texcoords", "vertex_positions")
    if pk in params:
        pairs.append((pk, uk))

if not pairs:
    raise RuntimeError("未找到可编辑的 mesh vertex_positions/vertex_texcoords（可能是解析几何或网格无UV）。")

n_mesh = len(pairs)
K = math.ceil(math.sqrt(n_mesh))
tile = 1.0 / K

pad = PAD_TEXELS / TEX_RES
inside = tile - 2.0 * pad
if inside <= 0:
    raise RuntimeError("padding 过大：inside<=0")

# 记录 atlas 信息（可用于调试/复现）
atlas_meta = {
    "K": K,
    "tile": tile,
    "pad": pad,
    "inside": inside,
    "tex_res": TEX_RES,
    "pad_texels": PAD_TEXELS,
    "meshes": []
}

# pack UV（原逻辑保留）
for i, (pk, uk) in enumerate(pairs):
    tx = i % K
    ty = i // K
    u0 = (tx * tile + pad)
    v0 = (ty * tile + pad)

    pos_flat = params[pk]
    uv_flat  = params[uk]

    if bool(is_degenerate_uv(uv_flat)):
        uv01 = rebuild_uv_from_positions(pos_flat)
    else:
        uv01 = dr.unravel(mi.Vector2f, uv_flat)

    uv_atlas = uv01 * inside + mi.Vector2f(u0, v0)
    params[uk] = dr.ravel(uv_atlas)

    atlas_meta["meshes"].append({
        "i": i, "pk": pk, "uk": uk, "tile_xy": [int(tx), int(ty)], "uv_offset": [float(u0), float(v0)]
    })

params.update()
print(f"UV rebuilt(if needed) + packed {n_mesh} meshes into {K}x{K} atlas.")

# AOV integrator（修正拼写与命名）
uv_integrator = mi.load_dict({
    "type": "aov",
    # 常用训练输入：uv/position/sh_normal/shape_index（你也可加 albedo/depth 等）
    "aovs": "uv:uv, position:position, sh_normal:sh_normal, albedo:albedo",
    "img": {"type": "path"}
})

# 相机 to_world key
cam_keys = [k for k in params.keys() if k.endswith("to_world") and "sensor" in k]
if not cam_keys:
    raise RuntimeError("未找到 sensor.to_world")
cam_key = cam_keys[0]

# 保存 meta.json：相机轨迹、场景、渲染参数、atlas
meta = {
    "scene": SCENE_PATH,
    "variant": "cuda_ad_rgb",
    "n_frames": N_FRAMES,
    "spp": SPP,
    "camera_key": cam_key,
    "center": center.tolist(),
    "target": target.tolist(),
    "up": up.tolist(),
    "radius": float(radius),
    "atlas": atlas_meta,
    "frames": []
}

film = scene.sensors()[0].film()
size = film.size()
meta["res"] = [int(size[0]), int(size[1])]

for f in range(N_FRAMES):
    ang = np.pi * (f / N_FRAMES)
    #origin = center + np.array([radius*np.sin(ang), 0.6, radius*np.cos(ang)], dtype=np.float32)
    origin = center

    T = mi.ScalarTransform4f.look_at(
        origin=mi.ScalarPoint3f(*origin),
        target=mi.ScalarPoint3f(*target),
        up=mi.ScalarVector3f(*up)
    )

    # params[cam_key] = T
    # params.update()

    # 确保 film 清空
    film.clear()

    _ = mi.render(scene, integrator=uv_integrator, spp=SPP)

    # 组件命名在不同版本可能略有差异；这里做“必需项检查 + 自动取图像输出”
    # AOV 部分通常就是 uv/position/sh_normal/shape_index
    comp = get_component_dict(film)
    need = ["uv", "position", "sh_normal", "albedo"]
    for k in need:
        if k not in comp:
            raise RuntimeError(f"Missing AOV '{k}'. Available keys: {list(comp.keys())}")

    # 主图像输出：优先找 img / color / beauty / root；找不到就取“第一个非 AOV”
    img_key = None
    for cand in ["img", "color", "beauty", "root", "rgba"]:
        if cand in comp:
            img_key = cand
            break
    if img_key is None:
        aov_set = set(need)
        for k in comp.keys():
            if k not in aov_set:
                img_key = k
                break
    if img_key is None:
        raise RuntimeError(f"Cannot find main RGB output. Keys: {list(comp.keys())}")

    rgb = bitmap_to_numpy(comp[img_key], mi.Bitmap.PixelFormat.RGBA)[..., :3]  # HxWx3
    pos = bitmap_to_numpy(comp["position"], mi.Bitmap.PixelFormat.RGBA)[..., :3]
    nrm = bitmap_to_numpy(comp["sh_normal"], mi.Bitmap.PixelFormat.RGBA)[..., :3]
    alb = bitmap_to_numpy(comp["albedo"], mi.Bitmap.PixelFormat.RGBA)[..., :3]

    uv_rgba = bitmap_to_numpy(comp["uv"], mi.Bitmap.PixelFormat.YA)
    uv = uv_rgba[..., :2]  # uv 只取前两维

    # 保存相机矩阵（4x4）
    T_np = np.array(T.matrix, dtype=np.float32)  # ScalarTransform4f.matrix -> 4x4
    frame_path = os.path.join(OUT_DIR, "frames", f"frame_{f:04d}.npz")

    np.savez_compressed(
        frame_path,
        rgb=rgb,
        position=pos,
        sh_normal=nrm,
        albedo = alb,
        uv=uv,
        cam_to_world=T_np,
        cam_origin=np.array(origin, dtype=np.float32),
        cam_target=np.array(target, dtype=np.float32),
        cam_up=np.array(up, dtype=np.float32),
    )

    meta["frames"].append({
        "frame": f,
        "file": f"frames/frame_{f:04d}.npz",
        "img_key": img_key
    })

    print(f"[{f:04d}] saved -> {frame_path} (img_key={img_key})")

with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as fp:
    json.dump(meta, fp, indent=2, ensure_ascii=False)

print("Done. Dataset written to:", OUT_DIR)
