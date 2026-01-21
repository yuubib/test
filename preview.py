import sys
import os


import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('cuda_ad_rgb')

import numpy as np
import random
import configargparse
import cv2
import glfw
import OpenGL.GL as gl
import imgui
import time
from imgui.integrations.glfw import GlfwRenderer

diff_exposure = 1
exposure = [0.5]
preview_resolution = 600
resolution = 300


def preview():
    # conf = configargparse.ArgumentParser()

    # conf.add('--model_path', required=True, help='Path to model which will be used for tha path generation')
    # conf.add('--scene_path', required=True, help='Path to the scene to be preview')
    # conf.add('--scene_buffers_path', required=True, help='Path to the buffers version of scene to preview')

    # # Generators (default: Pixel Generator)
    # conf.add('--arch', default='pixel', choices=['pixel', 'ppixel'])
    # conf.add('--device', type=str, default='cuda', help='Cuda device to use')
    # conf.add('--tonemap', default='log1p', choices=['log1p'])
    # conf.add('--metric', default='dssim', choices=['l1', 'l2', 'lpips', 'dssim', 'mape', 'smape', 'mrse'])
    # conf.add('--hidden_features', type=int, default=700, help='Number of hidden features for the generator')
    # conf.add('--hidden_layers', type=int, default=8, help='Number of hidden layers for the generator')

    # Set random seeds
    random.seed(0)

    # conf = conf.parse_args()
    custom_values = dict()
    custom_values["sensor"] = [1,2,3,4,5,6]

    # Initialize window
    window = impl_glfw_init()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize images
    prediction_img = np.zeros((preview_resolution, preview_resolution, 3)).astype(np.float16)
    gt_img = np.zeros((resolution, resolution, 3))
    closest_img = np.zeros((resolution, resolution, 3))
    diff_img = np.zeros((resolution, resolution, 3))

    # Bind prediction texture
    prediction_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, prediction_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16F, preview_resolution, preview_resolution, 0, gl.GL_RGB, gl.GL_FLOAT, prediction_img)

    # Bind ground truth texture
    gt_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, gt_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, gt_img)

    # Bind closest data point texture
    closest_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, closest_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, closest_img)

    # Bind diff texture
    diff_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, diff_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, resolution, resolution, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, diff_img)

    # Enable key event callback
    glfw.set_key_callback(window, key_event)

    # Images are in linear space transform them to sRGB
    gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
    gl.glEnable(gl.GL_DITHER)

    loss = 0

    frames = 0
    fps = 0

    start_frame = time.time()

    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.set_next_window_position(600, 0)
        imgui.set_next_window_size(600, 300)
        imgui.begin("Configurable parameters", flags=imgui.WINDOW_NO_MOVE)

        imgui.text('FPS: ' + str(fps))

        imgui.push_item_width(500)

        # Sensor sliders
        changed, values = imgui.slider_float3("sensor" + '_origin', *custom_values["sensor"][0:3], 0, 1)
        changed2, values2 = imgui.slider_float3("sensor" + '_target', *custom_values["sensor"][3:6], 0, 1)
        values = values + values2
        custom_values["sensor"] = list(values)


        imgui.spacing()

        imgui.end()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(600, 920)
        imgui.begin("Preview", flags=imgui.WINDOW_NO_MOVE)

        imgui.image(prediction_id, preview_resolution, preview_resolution)

        # Draw Ground Truth and Diff
        imgui.begin_group()

        if imgui.button("Generate GT"):
            gt = cv2.resize(gt, (resolution, resolution), cv2.INTER_NEAREST)

            gt_img = gt * exposure[0]

            gl.glBindTexture(gl.GL_TEXTURE_2D, gt_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution, resolution, gl.GL_RGB, gl.GL_FLOAT, gt_img)

        imgui.image(gt_id, resolution, resolution)

        imgui.end_group()

        imgui.same_line()

        imgui.begin_group()

        imgui.text('loss : %.4f' % loss)

        imgui.image(diff_id, resolution, resolution)

        imgui.end_group()

        imgui.end()

        gl.glClearColor(0.00, 0.00, 0.00, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        frames += 1

        if (time.time() - start_frame) > 1.0:
            fps = frames
            frames = 0
            start_frame = time.time()

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1200, 920
    window_name = "Neural Rendering"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def key_event(window, key, scancode, action, mods):
    if (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_P:
        exposure[0] = exposure[0] + 0.1
    elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_O:
        exposure[0] = exposure[0] - 0.1


preview()
