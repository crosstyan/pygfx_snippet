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
import wgpu
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


# https://wgpu.rs
async def render_task():
  scene = gfx.Scene()
  scene.add(gfx.AmbientLight(intensity=1))
  scene.add(gfx.DirectionalLight(cast_shadow=True))

  camera = gfx.PerspectiveCamera(110, 4 / 3)

  geometry = gfx.box_geometry(200, 200, 200)
  material = gfx.MeshPhongMaterial(color="#336699")
  cube = gfx.Mesh(geometry, material)
  camera.show_object(cube)
  camera.local.z = 400
  scene.add(cube)
  scene.add(gfx.AxesHelper(size=250))
  # https://docs.rs/wgpu/latest/wgpu/enum.TextureFormat.html
  data = np.zeros((640, 800, 4), dtype=np.float32)
  # texture = gfx.Texture(data=data, dim=2, format="TextureFormat.rgba32float")
  texture = gfx.Texture(data=data, dim=2, format=wgpu.TextureFormat.rgba32float)
  # canvas = WgpuCanvas()
  # close the canvas
  # We only need the opencv window
  # canvas._on_close()
  renderer = gfx.WgpuRenderer(texture, show_fps=False)
  controller = gfx.OrbitController(camera, register_events=renderer)
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
  frame_interval = timedelta(seconds=1/FPS)
  instant = Instant()
  window = cv.namedWindow("frame", cv.WINDOW_NORMAL)
  for frame in get_frame():
    # print(frame)
    bgr = cv.cvtColor(frame, cv.COLOR_RGBA2BGRA)
    normalized = cv.normalize(bgr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC4)
    cv.imshow("frame", normalized)
    cv.waitKey(1)
    if instant.elapsed() < frame_interval:
      delay = (frame_interval - instant.elapsed()).total_seconds()
      await anyio.sleep(delay)
    else:
      logger.warning("Frame took too long: {}".format(instant.elapsed()))
    instant.reset()

async def main():
  async with anyio.create_task_group() as tg:
    tg.start_soon(render_task)

# https://anyio.readthedocs.io/en/3.x/streams.html
# https://github.com/pygfx/pygfx/issues/260
if __name__ == "__main__":
  anyio.run(main)
