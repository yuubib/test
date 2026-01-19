import mitsuba as mi

mi.set_variant("cuda_ad_rgb")  # 任选你的 variant
scene = mi.load_file("cornell-box/scene.xml")

integrator_full = mi.load_dict({"type": "path", "max_depth": -1, "hide_emitters": True})
integrator_direct = mi.load_dict({"type": "path", "max_depth": 2, "hide_emitters": True})  # depth=2 => direct-only

spp = 1024
seed = 0  # 固定 seed 便于可复现 :contentReference[oaicite:2]{index=2}

img_full = mi.render(scene, integrator=integrator_full, spp=spp, seed=seed)
img_direct = mi.render(scene, integrator=integrator_direct, spp=spp, seed=seed)

img_indirect = img_full - img_direct

mi.util.write_bitmap("full.exr", img_full)
mi.util.write_bitmap("direct.exr", img_direct)
mi.util.write_bitmap("indirect.exr", img_indirect)
