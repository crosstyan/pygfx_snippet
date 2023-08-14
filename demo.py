import pygfx as gfx
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

class MyBox:
  _mesh: gfx.Mesh
  is_rotating: bool

  @property
  def mesh(self) -> gfx.Mesh:
    return self._mesh

  def __init__(self) -> None:
    geometry = gfx.box_geometry(200, 200, 200)
    material = gfx.MeshPhongMaterial(color="#336699")
    self.is_rotating = False
    self._mesh = gfx.Mesh(geometry, material)

  def try_rotate(self):
    if self.is_rotating:
      rot = la.quat_from_euler((0.005, 0.01 ), order="XY")
      self._mesh.local.rotation = la.quat_mul(rot, self._mesh.local.rotation)

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
  cube = MyBox()
  camera.show_object(cube.mesh)
  camera.local.z = 400
  scene.add(cube.mesh)
  scene.add(gfx.AxesHelper(size=250))
  # https://docs.rs/wgpu/latest/wgpu/enum.TextureFormat.html
  data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.float32)
  texture = gfx.Texture(data=data, dim=2, format=wgpu.TextureFormat.rgba32float)
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
  def r():
    cube.try_rotate()
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
    # Maybe need multiple focus
    # https://gamedev.stackexchange.com/questions/162732/how-do-i-check-if-a-window-has-focus-in-sdl2
    for ev in pygame.event.get():
      match ev.type:
        case pg.QUIT:
          pygame.quit()
          return
        case pg.KEYDOWN:
          match ev.key:
            case pg.K_r:
              cube.is_rotating = not cube.is_rotating
            case pg.K_ESCAPE:
              pygame.quit()
              return

async def main():
  async with anyio.create_task_group() as tg:
    tg.start_soon(render_task)

# https://github.com/pygfx/pygfx/issues/260
if __name__ == "__main__":
  anyio.run(main)
