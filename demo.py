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
import anyio

class Instant:
  _value: datetime

  @property
  def value(self) -> datetime:
    return self._value

  def __init__(self):
    self._value = datetime.now()
  
  def elapsed(self) -> timedelta:
    return datetime.now() - self._value
  
  def reset(self):
    self._value = datetime.now()
  
  def __repr__(self):
    return str(self.value)

  def elapsed_reset(self) -> timedelta:
    now = datetime.now()
    elapsed = now - self._value
    self._value = now
    return elapsed

WIDTH = 640
HEIGHT = 480

# https://wgpu.rs
# https://github.com/AlexElvers/pygame-with-asyncio
def render_task():
  scene = gfx.Scene()
  scene.add(gfx.AmbientLight(intensity=1))
  scene.add(gfx.DirectionalLight(cast_shadow=True))

  ok, err = pygame.init()
  logger.info(f"Pygame init: {ok} successes and {err} failures")
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("PyGFX")
  clock = pygame.time.Clock()

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
  
  FPS = 30
  for frame in get_frame():
    rgb = cv.cvtColor(frame, cv.COLOR_RGBA2RGB)
    u = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
    resized = cv.resize(u, (HEIGHT, WIDTH), interpolation=cv.INTER_AREA)
    surf = pygame.surfarray.make_surface(resized)
    screen.blit(surf, (0, 0))
    pygame.display.update()
    clock.tick(FPS)
    for ev in pygame.event.get():
      pass

async def main():
  async with anyio.create_task_group() as tg:
    tg.start_soon(render_task)

# https://anyio.readthedocs.io/en/3.x/streams.html
# https://github.com/pygfx/pygfx/issues/260
if __name__ == "__main__":
  render_task()
