import pygfx as gfx
import pylinalg as la
import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import glfw
import imutils
import wgpu
import pygame
import pygame as pg
from pygame._sdl2 import Window, Texture, Image, Renderer, get_drivers
from typing import Coroutine, NoReturn, Tuple, TypedDict, Union, List, Dict, Any, Optional, Callable, Awaitable, Generator
import anyio
from pygfx.cameras._perspective import fov_distance_factor
from controller.controller import CameraControl, KeyRegister, ZoomParam, OrbitParam, Action
from entity.my_box import MyBox

WIDTH = 640
HEIGHT = 480
FPS = 30

# https://wgpu.rs
# https://github.com/AlexElvers/pygame-with-asyncio


async def render_task():
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight(intensity=1))
    scene.add(gfx.DirectionalLight(cast_shadow=True))

    # only init display module
    # https://github.com/pygame/pygame/blob/main/examples/video.py
    pg.display.init()
    clock = pg.time.Clock()
    main_win = Window("main", size=(WIDTH, HEIGHT))
    main_win_renderer = Renderer(main_win)
    aux_win = Window("aux", size=(WIDTH, HEIGHT))
    aux_win_renderer = Renderer(aux_win)

    camera = gfx.PerspectiveCamera(90)
    cube = MyBox((50, 100, 50), "#ff0000")
    camera.show_object(cube.mesh)
    cam_ctrl = CameraControl(camera, clock)

    auto = cam_ctrl.auto_id
    cam_ctrl.register_key(KeyRegister(
        auto(), pg.K_q, Action.Press, ZoomParam(0.02, -0.02)))
    cam_ctrl.register_key(KeyRegister(
        auto(), pg.K_e, Action.Press, ZoomParam(-0.02, 0.02)))
    cam_ctrl.register_key(KeyRegister(
        auto(), pg.K_w, Action.Press, OrbitParam(0.025, 0)))
    cam_ctrl.register_key(KeyRegister(
        auto(), pg.K_s, Action.Press, OrbitParam(-0.025, 0)))
    cam_ctrl.register_key(KeyRegister(
        auto(), pg.K_a, Action.Press, OrbitParam(0, 0.025)))
    cam_ctrl.register_key(KeyRegister(
        auto(), pg.K_d, Action.Press, OrbitParam(0, -0.025)))

    camera.look_at(cube.mesh.world.position)
    scene.add(cube.mesh)
    scene.add(gfx.AxesHelper(size=125))
    # a dummy np data to init the texture. Texture would inference the shape from the data
    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.float32)
    # https://docs.rs/wgpu/latest/wgpu/enum.TextureFormat.html
    texture = gfx.Texture(
        data=data, dim=2, format=wgpu.TextureFormat.rgba32float)  # type: ignore
    renderer = gfx.WgpuRenderer(texture, show_fps=False)
    # write a similar function to rotate the cube but handled in pygame event
    # https://github.com/pygfx/pygfx/blob/295435d9bd99008c2f0c472242720b805ae793c7/pygfx/controllers/_orbit.py#L49-L64
    # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/examples/other/post_processing1.py
    # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/examples/other/post_processing2.py
    # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/pygfx/renderers/wgpu/_renderer.py#L678
    # https://stackoverflow.com/questions/19554059/opencv-opengl-get-opengl-image-as-opencv-camera
    # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/examples/introductory/offscreen.py
    # https://github.com/pygfx/pygfx/issues/264
    # https://www.youtube.com/watch?v=QQ3jr-9Rc1o

    def iter():
        cam_ctrl.iter()
        cube.iter()
        renderer.render(scene, camera)

    def get_frame():
        while True:
            iter()
            yield renderer.snapshot()

    # maybe not efficient, at least once copy happens
    def refresh_window(renderer: Renderer, frame: cv.Mat):
        # make sure the mat is 8UC3 or 8UC1
        assert frame.dtype == np.uint8
        if frame.ndim == 3:
            assert frame.shape[2] == 3
        else:
            assert frame.ndim == 2
        surf = pygame.surfarray.make_surface(frame)
        texture = Texture.from_surface(renderer, surf)
        renderer.blit(texture)
        renderer.present()

    for frame in get_frame():
        rgb = cv.cvtColor(frame, cv.COLOR_RGBA2RGB)
        u = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX,
                         cv.CV_8UC3)  # type: ignore
        #  Height is at index 0, Width is at index 1; and number of channels at index 2
        resized = cv.resize(u, (HEIGHT, WIDTH), interpolation=cv.INTER_AREA)
        canny = cv.Canny(resized, 100, 200)
        refresh_window(main_win_renderer, resized)
        refresh_window(aux_win_renderer, canny)
        clock.tick(FPS)
        main_win.title = f"main: {clock.get_fps():.2f}FPS"
        # you need this to make pygame to start windows and respond to events
        # Maybe need multiple focus
        # https://gamedev.stackexchange.com/questions/162732/how-do-i-check-if-a-window-has-focus-in-sdl2
        for ev in pygame.event.get():
            cam_ctrl.poll_event(ev)
            match ev.type:
                case pg.QUIT:
                    pygame.quit()
                    return
                case pg.KEYDOWN:
                    match ev.key:
                        case pg.K_ESCAPE:
                            pygame.quit()
                            return


async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(render_task)

# https://github.com/pygfx/pygfx/issues/260
if __name__ == "__main__":
    anyio.run(main)
