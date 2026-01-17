import numpy as np
import drjit as dr
import mitsuba as mi

# 建议用 JIT variant（官方也强调 Python 插件应优先 JIT variant）:contentReference[oaicite:3]{index=3}
mi.set_variant('cuda_ad_rgb')  # 或 'cuda_ad_rgb'

# ---------------------------
# 1) 生成一个“固定颜色”的 probe grid（仅数据，不往 XML 塞 sphere）
# ---------------------------
def save_debug_probe_grid(scene: mi.Scene,
                          out_npz: str,
                          dims=(10, 10, 10),
                          margin=0.02):
    bbox = scene.bbox()
    bmin = np.array([bbox.min.x, bbox.min.y, bbox.min.z], dtype=np.float32)
    bmax = np.array([bbox.max.x, bbox.max.y, bbox.max.z], dtype=np.float32)

    # 缩一下，避免 probe 正好贴墙导致插值/数值不稳定
    extent = bmax - bmin
    bmin = bmin + margin * extent
    bmax = bmax - margin * extent

    nx, ny, nz = map(int, dims)
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    zs = np.linspace(0.0, 1.0, nz, dtype=np.float32)

    colors = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                # 固定颜色：直接用归一化坐标（debug 最直观）
                colors[ix, iy, iz] = [x, y, z]

    np.savez(
        out_npz,
        bbox_min=bmin,
        bbox_max=bmax,
        dims=np.array([nx, ny, nz], dtype=np.int32),
        colors=colors
    )

# ---------------------------
# 2) “完美版”Probe Integrator：透明/镜面不终止，直到命中可着色表面
# ---------------------------
class ProbeGridIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        mi.SamplingIntegrator.__init__(self, props)

        # 参数
        self.max_depth = int(props['max_depth']) if props.has_property('max_depth') else 32
        self.rr_depth  = int(props['rr_depth'])  if props.has_property('rr_depth')  else 8
        self.follow_glossy = bool(props['follow_glossy']) if props.has_property('follow_glossy') else True
        self.use_emitters  = bool(props['use_emitters'])  if props.has_property('use_emitters')  else False
        self.strength = float(props['strength']) if props.has_property('strength') else 1.0
        self.background = str(props['background']) if props.has_property('background') else 'probe'  # 'probe' or 'black'

        if not props.has_property('probe_npz'):
            raise RuntimeError("ProbeGridIntegrator requires 'probe_npz' property.")
        data = np.load(str(props['probe_npz']))

        bmin = data['bbox_min'].astype(np.float32)
        bmax = data['bbox_max'].astype(np.float32)
        dims = data['dims'].astype(np.int32)
        cols = data['colors'].astype(np.float32)  # (nx, ny, nz, 3)

        self.bmin = mi.Point3f(float(bmin[0]), float(bmin[1]), float(bmin[2]))
        self.bmax = mi.Point3f(float(bmax[0]), float(bmax[1]), float(bmax[2]))

        self.nx = mi.UInt32(int(dims[0]))
        self.ny = mi.UInt32(int(dims[1]))
        self.nz = mi.UInt32(int(dims[2]))

        cols_flat = cols.reshape(-1, 3)
        self.cr = mi.Float(cols_flat[:, 0])
        self.cg = mi.Float(cols_flat[:, 1])
        self.cb = mi.Float(cols_flat[:, 2])

    def aov_names(self):
        # 只给一个 valid AOV，方便 debug
        return ['probe.valid']

    # ---- probe trilinear interpolation
    def _probe_color(self, p: mi.Point3f) -> mi.Color3f:
        # 归一化到 [0, 1]
        extent = self.bmax - self.bmin
        u = (p - self.bmin) / extent
        u = dr.clamp(u, 0.0, 1.0)

        gx = u.x * (mi.Float(self.nx) - 1.0)
        gy = u.y * (mi.Float(self.ny) - 1.0)
        gz = u.z * (mi.Float(self.nz) - 1.0)

        ix0f = dr.floor(gx); tx = gx - ix0f
        iy0f = dr.floor(gy); ty = gy - iy0f
        iz0f = dr.floor(gz); tz = gz - iz0f

        ix0 = mi.UInt32(ix0f); iy0 = mi.UInt32(iy0f); iz0 = mi.UInt32(iz0f)
        ix1 = dr.minimum(ix0 + 1, self.nx - 1)
        iy1 = dr.minimum(iy0 + 1, self.ny - 1)
        iz1 = dr.minimum(iz0 + 1, self.nz - 1)

        def idx(ix, iy, iz):
            return ix + self.nx * (iy + self.ny * iz)

        def gather_color(ix, iy, iz):
            linear = idx(ix, iy, iz)
            r = dr.gather(mi.Float, self.cr, linear)
            g = dr.gather(mi.Float, self.cg, linear)
            b = dr.gather(mi.Float, self.cb, linear)
            return mi.Color3f(r, g, b)

        c000 = gather_color(ix0, iy0, iz0)
        c100 = gather_color(ix1, iy0, iz0)
        c010 = gather_color(ix0, iy1, iz0)
        c110 = gather_color(ix1, iy1, iz0)
        c001 = gather_color(ix0, iy0, iz1)
        c101 = gather_color(ix1, iy0, iz1)
        c011 = gather_color(ix0, iy1, iz1)
        c111 = gather_color(ix1, iy1, iz1)

        c00 = dr.lerp(c000, c100, tx)
        c10 = dr.lerp(c010, c110, tx)
        c01 = dr.lerp(c001, c101, tx)
        c11 = dr.lerp(c011, c111, tx)

        c0 = dr.lerp(c00, c10, ty)
        c1 = dr.lerp(c01, c11, ty)

        return dr.lerp(c0, c1, tz)

    # ---- ray hits probe bbox? used for background sampling
    def _probe_on_ray_exit(self, ray: mi.Ray3f) -> mi.Color3f:
        o = ray.o
        d = ray.d
        # 安全倒数（避免 d=0）
        inv_d = 1.0 / dr.select(dr.abs(d) > 1e-12, d, dr.sign(d) * 1e-12)

        t0 = (self.bmin - o) * inv_d
        t1 = (self.bmax - o) * inv_d

        tmin = dr.maximum(dr.maximum(dr.minimum(t0.x, t1.x),
                                     dr.minimum(t0.y, t1.y)),
                          dr.minimum(t0.z, t1.z))
        tmax = dr.minimum(dr.minimum(dr.maximum(t0.x, t1.x),
                                     dr.maximum(t0.y, t1.y)),
                          dr.maximum(t0.z, t1.z))

        hit = (tmax >= dr.maximum(tmin, 0.0))
        # 取射线“离开盒子”的点
        t_exit = dr.select(hit, tmax, 0.0)
        p_exit = o + t_exit * d
        return dr.select(hit, self._probe_color(p_exit), mi.Color3f(0.0))

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.RayDifferential3f,
               medium: mi.Medium = None,
               active: bool = True):
        ctx = mi.BSDFContext()

        L = mi.Color3f(0.0)
        throughput = mi.Color3f(1.0)

        ray_c = mi.RayDifferential3f(ray)

        # 主循环：追踪透明/镜面链，直到“可着色表面”
        for depth in range(self.max_depth):
            si = scene.ray_intersect(ray_c, active)

            hit = si.is_valid()
            if not dr.any(hit):
                if self.background == 'probe':
                    L += throughput * self.strength * self._probe_on_ray_exit(mi.Ray3f(ray_c))
                break

            # 可选：叠加场景真实发光体
            if self.use_emitters:
                emitter = si.emitter(scene)
                if emitter is not None:
                    L += throughput * emitter.eval(si)

            bsdf = si.bsdf(ray_c)
            flags = bsdf.flags()

            # 1) Null：无散射透射，必须 eval_null_transmission 并继续，否则就是“黑洞”
            is_null = mi.has_flag(flags, mi.BSDFFlags.Null)
            if dr.any(is_null):
                tr = bsdf.eval_null_transmission(si, active=hit)
                throughput *= tr
                ray_c = si.spawn_ray(ray_c.d)
                # rr
                if depth >= self.rr_depth:
                    q = dr.minimum(dr.maximum(dr.max(throughput), 0.05), 0.95)
                    survive = sampler.next_1d() < q
                    throughput /= q
                    active = hit & survive
                else:
                    active = hit
                continue

            # 2) 是否继续追踪（透明/镜面/可选 glossy）
            is_delta = mi.has_flag(flags, mi.BSDFFlags.Delta)
            is_glossy = mi.has_flag(flags, mi.BSDFFlags.Glossy)

            follow = is_delta | (is_glossy & self.follow_glossy)
            albedo = bsdf.eval_diffuse_reflectance(si, active=hit)
            spec = bsdf.eval_null_transmission(si, active=hit)
            probe = self._probe_color(si.p)
            L += throughput * self.strength * (albedo + spec) * probe

            # 4) 继续追踪：按 BSDF 采样得到下一条射线方向与权重
            #    sample() 返回的 value 是典型的 (f * cos / pdf) 形式权重
            bs, w = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d(), active=hit)
            throughput *= w
            ray_c = si.spawn_ray(bs.wo)

            # Russian roulette
            if depth >= self.rr_depth:
                q = dr.minimum(dr.maximum(dr.max(throughput), 0.05), 0.95)
                survive = sampler.next_1d() < q
                throughput /= q
                active = hit & survive
            else:
                active = hit

        valid_aov = [dr.select(dr.any(throughput != 0.0), 1.0, 0.0)]
        return L, True, valid_aov

    def to_string(self):
        return (f"ProbeGridIntegrator[max_depth={self.max_depth}, rr_depth={self.rr_depth}, "
                f"follow_glossy={self.follow_glossy}, use_emitters={self.use_emitters}, "
                f"strength={self.strength}, background={self.background}]")

# 注册 integrator（官方说明“同类注册函数也包括 register_integrator”）
mi.register_integrator("probe_grid", lambda props: ProbeGridIntegrator(props))

# ---------------------------
# 3) 使用方式：不改 XML，只替换 integrator
# ---------------------------
if __name__ == "__main__":
    scene = mi.load_file("./veach-ajar/scene.xml")

    # 生成 probe 数据（一次即可）
    save_debug_probe_grid(scene, "probes_debug.npz", dims=(12, 12, 12), margin=0.02)

    my_integrator = mi.load_dict({
        'type': 'probe_grid',
        'probe_npz': 'probes_debug.npz',
        'max_depth': 1,
        'rr_depth': 8,
        'follow_glossy': True,
        'use_emitters': False,     # 若想叠加顶灯，改 True
        'strength': 1.0,
        'background': 'probe',     # 'black' 或 'probe'
    })

    out = mi.render(scene, integrator=my_integrator, spp=256)  # 你要求的调用方式
    mi.Bitmap(out).write("probe_debug.exr")


