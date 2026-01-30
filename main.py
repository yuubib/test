import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")
scene = mi.load_file("./veach-ajar/scene.xml")

scene.integrator().render(scene, scene.sensors()[0])
bmp = scene.sensors()[0].film().bitmap(raw=False)
components = scene.sensors()[0].film().bitmap(raw=False).split()

for i in range(len(components)):
    if 'root' in components[i][0]:
        buffer = components[i][1].convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
        #denoised_bmp = denoiser(bmp, albedo_ch="albedo", normals_ch="sh_normal", noisy_ch="<root>").convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
        mi.util.write_bitmap("root.exr", buffer)
    if 'img' in components[i][0]:
        print(components[i][1])
        buffer = components[i][1].convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
        mi.util.write_bitmap("img.exr", buffer)
    if 'sh_normal' in components[i][0]:
        mi.util.write_bitmap("sh_normal.exr", components[i][1])
    if 'albedo' in components[i][0]:
        mi.util.write_bitmap("albedo.exr", components[i][1])
    if 'position' in components[i][0]:
        mi.util.write_bitmap("position.exr", components[i][1])
    if 'uv' in components[i][0]:
        mi.util.write_bitmap("uv.exr", components[i][1])
