import os
import numpy as np
import xml.etree.ElementTree as ET

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

# -----------------------------
# A. 自定义 Integrator：probe 网格三线性插值着色
# -----------------------------
class ProbeGridIntegrator(mi.SamplingIntegrator):
    """
    在表面交点 p 上，根据 probe 网格做三线性插值，得到 C(p)，输出：
        L = strength * albedo * C(p)
    probe 数据来自 .npz 文件（由本脚本生成）。
    """
    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.probe_file = props['probe_file'] if props.has_property('probe_file') else None
        if self.probe_file is None:
            raise RuntimeError('ProbeGridIntegrator: missing string property "probe_file".')

        self.strength = float(props['strength']) if props.has_property('strength') else 1.0

        # 读取 probe 数据（numpy）
        data = np.load(self.probe_file)
        bmin = data['bbox_min'].astype(np.float32)
        bmax = data['bbox_max'].astype(np.float32)
        dims = data['dims'].astype(np.int32)  # (nx, ny, nz)
        colors = data['colors'].astype(np.float32)  # shape: (nz, ny, nx, 3) or flat (N,3)

        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])
        if colors.ndim == 4:
            # (nz, ny, nx, 3) -> (N,3), C-order: x fastest, then y, then z
            flat = colors.reshape(-1, 3)
        else:
            flat = colors
        if flat.shape[0] != self.nx * self.ny * self.nz or flat.shape[1] != 3:
            raise RuntimeError("ProbeGridIntegrator: colors shape mismatch with dims.")

        # 存为 Dr.Jit 数组，便于 gather
        self.col_r = mi.Float(flat[:, 0])
        self.col_g = mi.Float(flat[:, 1])
        self.col_b = mi.Float(flat[:, 2])

        self.nx_u = mi.UInt32(self.nx)
        self.ny_u = mi.UInt32(self.ny)
        self.nxy_u = mi.UInt32(self.nx * self.ny)

        self.bmin = mi.Point3f(float(bmin[0]), float(bmin[1]), float(bmin[2]))
        self.bmax = mi.Point3f(float(bmax[0]), float(bmax[1]), float(bmax[2]))
        extent = self.bmax - self.bmin
        extent = dr.maximum(extent, 1e-6)
        self.inv_extent = 1.0 / extent

    def _lin_index(self, ix: mi.UInt32, iy: mi.UInt32, iz: mi.UInt32) -> mi.UInt32:
        # idx = x + y*nx + z*nx*ny
        return ix + iy * self.nx_u + iz * self.nxy_u

    def _gather_color(self, idx: mi.UInt32) -> mi.Color3f:
        r = dr.gather(mi.Float, self.col_r, idx)
        g = dr.gather(mi.Float, self.col_g, idx)
        b = dr.gather(mi.Float, self.col_b, idx)
        return mi.Color3f(r, g, b)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        si = scene.ray_intersect(ray, active=active)
        valid = si.is_valid()

        # miss -> black
        L = mi.Color3f(0.0)

        # 命中点
        p = si.p

        # p -> [0,1)
        u = (p - self.bmin) * self.inv_extent
        u = dr.clamp(u, 0.0, 0.999999)

        # -> grid coord
        gx = u.x * (self.nx - 1)
        gy = u.y * (self.ny - 1)
        gz = u.z * (self.nz - 1)

        ix0 = mi.UInt32(dr.floor(gx))
        iy0 = mi.UInt32(dr.floor(gy))
        iz0 = mi.UInt32(dr.floor(gz))

        fx = gx - mi.Float(ix0)
        fy = gy - mi.Float(iy0)
        fz = gz - mi.Float(iz0)

        ix1 = dr.minimum(ix0 + 1, mi.UInt32(self.nx - 1))
        iy1 = dr.minimum(iy0 + 1, mi.UInt32(self.ny - 1))
        iz1 = dr.minimum(iz0 + 1, mi.UInt32(self.nz - 1))

        # 8 corner indices
        i000 = self._lin_index(ix0, iy0, iz0)
        i100 = self._lin_index(ix1, iy0, iz0)
        i010 = self._lin_index(ix0, iy1, iz0)
        i110 = self._lin_index(ix1, iy1, iz0)
        i001 = self._lin_index(ix0, iy0, iz1)
        i101 = self._lin_index(ix1, iy0, iz1)
        i011 = self._lin_index(ix0, iy1, iz1)
        i111 = self._lin_index(ix1, iy1, iz1)

        # trilinear weights
        w000 = (1 - fx) * (1 - fy) * (1 - fz)
        w100 = fx * (1 - fy) * (1 - fz)
        w010 = (1 - fx) * fy * (1 - fz)
        w110 = fx * fy * (1 - fz)
        w001 = (1 - fx) * (1 - fy) * fz
        w101 = fx * (1 - fy) * fz
        w011 = (1 - fx) * fy * fz
        w111 = fx * fy * fz

        C = (w000 * self._gather_color(i000) +
             w100 * self._gather_color(i100) +
             w010 * self._gather_color(i010) +
             w110 * self._gather_color(i110) +
             w001 * self._gather_color(i001) +
             w101 * self._gather_color(i101) +
             w011 * self._gather_color(i011) +
             w111 * self._gather_color(i111))

        # 漫反射反照率（近似）
        bsdf = si.bsdf(ray)
        albedo = bsdf.eval_diffuse_reflectance(si)

        L_hit = mi.Color3f(self.strength) * albedo * C
        L = dr.select(valid, L_hit, L)

        return L, valid, []

    def aov_names(self):
        return []

    def to_string(self):
        return f"ProbeGridIntegrator[file={self.probe_file}, n=({self.nx},{self.ny},{self.nz}), strength={self.strength}]"


def register_probe_integrator():
    mi.register_integrator("probe_grid", lambda props: ProbeGridIntegrator(props))


# -----------------------------
# B. 生成 probe 网格（固定颜色）并保存 npz
# -----------------------------
def make_probe_grid(scene: mi.Scene, nx=10, ny=10, nz=10, margin_frac=0.03, out_npz="probes.npz"):
    bbox = scene.bbox()
    bmin = np.array([bbox.min[0], bbox.min[1], bbox.min[2]], dtype=np.float32)
    bmax = np.array([bbox.max[0], bbox.max[1], bbox.max[2]], dtype=np.float32)

    extent = bmax - bmin
    margin = extent * float(margin_frac)
    bmin2 = bmin + margin
    bmax2 = bmax - margin

    # 规则网格坐标
    xs = np.linspace(bmin2[0], bmax2[0], nx, dtype=np.float32)
    ys = np.linspace(bmin2[1], bmax2[1], ny, dtype=np.float32)
    zs = np.linspace(bmin2[2], bmax2[2], nz, dtype=np.float32)

    # 固定颜色：用归一化网格坐标做可复现的 RGB（也可换成随机/调色板）
    # colors shape: (nz, ny, nx, 3)
    colors = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                colors[k, j, i, 0] = i / max(nx - 1, 1)
                colors[k, j, i, 1] = j / max(ny - 1, 1)
                colors[k, j, i, 2] = k / max(nz - 1, 1)

    np.savez(
        out_npz,
        bbox_min=bmin2,
        bbox_max=bmax2,
        dims=np.array([nx, ny, nz], dtype=np.int32),
        colors=colors
    )
    return out_npz, bmin2, bmax2


# -----------------------------
# C. 把你的 XML integrator 段替换为 probe_grid，并写出新 XML
# -----------------------------
def patch_xml_integrator(xml_in: str, xml_out: str, probe_file: str, strength=1.0):
    tree = ET.parse(xml_in)
    root = tree.getroot()

    # 找到 <integrator ...>
    integrator = None
    for child in root:
        if child.tag == 'integrator':
            integrator = child
            break
    if integrator is None:
        raise RuntimeError("XML has no <integrator> node.")

    # 清空 integrator 子节点
    for c in list(integrator):
        integrator.remove(c)

    # 改 integrator type（不依赖 default 变量，直接写死）
    integrator.set('type', 'probe_grid')

    # 添加参数
    s_node = ET.SubElement(integrator, 'string')
    s_node.set('name', 'probe_file')
    s_node.set('value', probe_file)

    f_node = ET.SubElement(integrator, 'float')
    f_node.set('name', 'strength')
    f_node.set('value', str(float(strength)))

    tree.write(xml_out, encoding='utf-8', xml_declaration=False)

def append_probe_spheres_to_xml(xml_in: str, xml_out: str, probe_npz: str, radius=0.02):
    tree = ET.parse(xml_in)
    root = tree.getroot()

    data = np.load(probe_npz)
    bmin = data['bbox_min'].astype(np.float32)
    bmax = data['bbox_max'].astype(np.float32)
    nx, ny, nz = data['dims'].astype(np.int32).tolist()
    colors = data['colors'].astype(np.float32)  # (nz, ny, nx, 3)

    xs = np.linspace(bmin[0], bmax[0], nx, dtype=np.float32)
    ys = np.linspace(bmin[1], bmax[1], ny, dtype=np.float32)
    zs = np.linspace(bmin[2], bmax[2], nz, dtype=np.float32)

    # 追加很多 sphere shape
    idx = 0
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                c = colors[k, j, i]
                shape = ET.SubElement(root, "shape", {"type": "sphere", "id": f"Probe_{idx}"})
                ET.SubElement(shape, "float", {"name": "radius", "value": str(float(radius))})
                tr = ET.SubElement(shape, "transform", {"name": "to_world"})
                ET.SubElement(tr, "translate", {"x": str(float(x)), "y": str(float(y)), "z": str(float(z))})

                bsdf = ET.SubElement(shape, "bsdf", {"type": "diffuse"})
                ET.SubElement(bsdf, "rgb", {"name": "reflectance",
                                            "value": f"{c[0]},{c[1]},{c[2]}"})
                idx += 1

    tree.write(xml_out, encoding="utf-8", xml_declaration=False)
def main():
    # 你可以改这里
    xml_path = "veach-ajar/scene.xml"
    out_npz = "probes.npz"
    patched_xml = "veach-ajar/scene_probe_grid.xml"
    out_exr = "probe_interp.exr"
    out_png = "probe_interp1.exr"

    nx, ny, nz = (32, 32, 16)     # probe 密度
    margin_frac = 0.04         # 从 bbox 内缩比例，避免 probe 贴墙
    strength = 1.0             # probe 着色强度

    # 选择 JIT 变体（建议）
    mi.set_variant("cuda_ad_rgb")  # 或 "cuda_ad_rgb"

    # 先注册 Python integrator（重要：必须在 load_file 前完成）
    register_probe_integrator()

    # 加载原始 XML 场景
    scene = mi.load_file(xml_path)

    # 生成 probe 网格文件
    make_probe_grid(scene, nx=nx, ny=ny, nz=nz, margin_frac=margin_frac, out_npz=out_npz)

    # 生成替换 integrator 后的新 XML
    #patch_xml_integrator(xml_path, patched_xml, probe_file=out_npz, strength=strength)
    append_probe_spheres_to_xml(xml_path, patched_xml, out_npz)
    # 加载新 XML 并渲染
    scene2 = mi.load_file(patched_xml)
    img = mi.render(scene2, spp=64)  # spp 可自行改；原 XML 的 sampler 不再起作用（因为 integrator 已替换）
    mi.Bitmap(img).write(out_exr)
    mi.Bitmap(img).convert(srgb_gamma=True).write(out_png)

    print(f"Wrote: {out_npz}, {patched_xml}, {out_exr}, {out_png}")


if __name__ == "__main__":
    main()
