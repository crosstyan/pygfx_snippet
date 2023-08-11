import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run
import pylinalg as la
from loguru import logger

# https://wgpu.rs
def main():
  scene = gfx.Scene()
  scene.add(gfx.AmbientLight(intensity=1))
  scene.add(gfx.DirectionalLight(cast_shadow=True))

  camera = gfx.PerspectiveCamera(110, 4 / 3)

  geometry = gfx.box_geometry(200, 200, 200)
  material = gfx.MeshPhongMaterial(color="#336699")
  cube = gfx.Mesh(geometry, material)
  # InterpolatedCamera
  camera.show_object(cube)
  scene.add(cube)
  scene.add(gfx.AxesHelper(size=250))
  canvas = WgpuCanvas()
  renderer = gfx.WgpuRenderer(canvas)
  controller = gfx.OrbitController(camera, register_events=renderer)
  st = camera.get_state()
  is_rotating = False
  def rotate_cube():
    nonlocal is_rotating
    if is_rotating:
      rot = la.quat_from_euler((0.005, 0.01 ), order="XY")
      cube.local.rotation = la.quat_mul(rot, cube.local.rotation)
  def on_key_down(event):
      if event.key == "q":
        nonlocal st
        st = camera.get_state()
        logger.info("Save camera state to {}".format(st))
      elif event.key == "e":
        camera.set_state(st)
        logger.info("Restore camera state from {}".format(st))
      elif event.key == "r":
        nonlocal is_rotating
        is_rotating = True
  def on_key_up(event):
    if event.key == "r":
      nonlocal is_rotating
      is_rotating = False
  renderer.add_event_handler(on_key_down, "key_down")
  renderer.add_event_handler(on_key_up, "key_up")
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
    canvas.request_draw()
  canvas.request_draw(r)
  run()
  # gfx.show(scene, camera=camera, renderer=renderer)


if __name__ == "__main__":
  main()