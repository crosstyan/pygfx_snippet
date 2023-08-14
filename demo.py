import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas
import pylinalg as la
import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime, timedelta
import glfw
import imutils
import wgpu
import pygame
import pygame as pg
from pygame._sdl2 import Window, Texture, Image, Renderer, get_drivers
from typing import Coroutine, NoReturn, Tuple, Union, List, Dict, Any, Optional, Callable, Awaitable, Generator
import anyio

WIDTH = 640
HEIGHT = 480
FPS = 60

# https://wgpu.rs
# https://github.com/AlexElvers/pygame-with-asyncio
async def render_task() -> Coroutine[None, None, NoReturn]:
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

  camera = gfx.PerspectiveCamera(110, 4 / 3)

  geometry = gfx.box_geometry(200, 200, 200)
  material = gfx.MeshPhongMaterial(color="#336699")
  cube = gfx.Mesh(geometry, material)
  camera.show_object(cube)
  camera.local.z = 400
  scene.add(cube)
  scene.add(gfx.AxesHelper(size=250))
  # https://docs.rs/wgpu/latest/wgpu/enum.TextureFormat.html
  data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.float32)
  texture = gfx.Texture(data=data, dim=2, format=wgpu.TextureFormat.rgba32float)
  renderer = gfx.WgpuRenderer(texture, show_fps=False)
  is_rotating = True
  def rotate_cube():
    nonlocal is_rotating
    if is_rotating:
      rot = la.quat_from_euler((0.005, 0.01 ), order="XY")
      cube.local.rotation = la.quat_mul(rot, cube.local.rotation)
  # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/examples/other/post_processing1.py
  # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/examples/other/post_processing2.py
  # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/pygfx/renderers/wgpu/_renderer.py#L678
  # https://stackoverflow.com/questions/19554059/opencv-opengl-get-opengl-image-as-opencv-camera
  # https://github.com/pygfx/pygfx/blob/78280bcdb9be8648974653acb48fb3e9df583acd/examples/introductory/offscreen.py
  # https://github.com/pygfx/pygfx/issues/264
  # https://www.youtube.com/watch?v=QQ3jr-9Rc1o
  def r():
    rotate_cube()
    renderer.render(scene, camera)

  def get_frame():
    while True:
      r()
      yield renderer.snapshot()
  
  for frame in get_frame():
    rgb = cv.cvtColor(frame, cv.COLOR_RGBA2RGB)
    u = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
    #  Height is at index 0, Width is at index 1; and number of channels at index 2
    resized = cv.resize(u, (HEIGHT, WIDTH), interpolation=cv.INTER_AREA)
    canny = cv.Canny(resized, 100, 200)
    # maybe not efficient, at least once copy happens
    surf = pygame.surfarray.make_surface(canny)
    surf_p = pygame.surfarray.make_surface(resized)
    t = Texture.from_surface(main_win_renderer, surf)
    t2 = Texture.from_surface(aux_win_renderer, surf_p)
    main_win_renderer.blit(t)
    aux_win_renderer.blit(t2)
    main_win_renderer.present()
    aux_win_renderer.present()
    clock.tick(FPS)
    main_win.title = f"main: {clock.get_fps():.2f}FPS"
    # you need this to make pygame to start windows and respond to events
    for ev in pygame.event.get():
      pass

async def main():
  async with anyio.create_task_group() as tg:
    tg.start_soon(render_task)

# https://github.com/pygfx/pygfx/issues/260
if __name__ == "__main__":
  anyio.run(main)
