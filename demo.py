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

WIDTH = 640
HEIGHT = 480
FPS = 30


class MyBox:
    _mesh: gfx.Mesh
    is_rotating: bool

    @property
    def mesh(self) -> gfx.Mesh:
        return self._mesh

    # width, height, depth
    def __init__(self, dim: Tuple[float, float, float] = (200, 200, 200), color: str = "#336699") -> None:
        geometry = gfx.box_geometry(*dim)
        material = gfx.MeshPhongMaterial(color=color)
        self.is_rotating = False
        self._mesh = gfx.Mesh(geometry, material)

    def iter(self):
        if self.is_rotating:
            rot = la.quat_from_euler((0.005, 0.01), order="XY")
            self._mesh.local.rotation = la.quat_mul(
                rot, self._mesh.local.rotation)

# some type annotations for the magic camera state


class CameraState(TypedDict):
    width: float
    height: float
    fov: float
    position: Tuple[float, float, float]
    maintain_aspect: bool
    rotation: Tuple[float, float, float, float]


class CameraControl:
    _camera: gfx.PerspectiveCamera
    _clock: pg.time.Clock
    # euler angle in radians (x, y, z)
    # https://pylinalg.readthedocs.io/en/latest/reference.html#pylinalg.quat_from_euler
    _orbit_delta: Tuple[float, float]
    _zoom_delta: Tuple[float, float]

    def __init__(self, camera: gfx.PerspectiveCamera, clock: pg.time.Clock) -> None:
        self._clock = clock
        self._camera = camera
        self._orbit_delta = (0, 0)
        self._zoom_delta = (0, 0)

    def _get_target_vec(self, camera_state, **kwargs):
        """Method used by the controler implementations to determine the "target"."""
        rotation = kwargs.get("rotation", camera_state["rotation"])
        extent = 0.5 * (camera_state["width"] + camera_state["height"])
        extent = kwargs.get("extent", extent)
        fov = kwargs.get("fov", camera_state.get("fov"))

        distance = fov_distance_factor(fov) * extent
        return la.vec_transform_quat((0, 0, -distance), rotation)

    def delta_orbit(self, delta: Tuple[float, float]):
        azumith, elevation = delta
        self._orbit_delta = (
            self._orbit_delta[0] + azumith, self._orbit_delta[1] + elevation)

    def delta_zoom(self, delta: Tuple[float, float]):
        self._zoom_delta = (
            self._zoom_delta[0] + delta[0], self._zoom_delta[1] + delta[1])

    def _set_camera_state(self, camera_state: CameraState):
        self._camera.set_state(camera_state)

    def _get_camera_state(self) -> CameraState:
        return self._camera.get_state()  # type: ignore

    def _update_zoom(self, delta):
        if isinstance(delta, (int, float)):
            delta = (delta, delta)
        assert isinstance(delta, tuple) and len(delta) == 2

        fx = 2 ** delta[0]
        fy = 2 ** delta[1]
        new_cam_state = self._zoom(fx, fy, self._get_camera_state())
        self._set_camera_state(new_cam_state)

    def _zoom(self, fx, fy, cam_state: CameraState) -> CameraState:
        position = cam_state["position"]
        maintain_aspect = cam_state["maintain_aspect"]
        width = cam_state["width"]
        height = cam_state["height"]
        extent = 0.5 * (width + height)

        # Scale width and height equally, or use width and height.
        if maintain_aspect:
            new_width = width / fy
            new_height = height / fy
        else:
            new_width = width / fx
            new_height = height / fy

        # Get new position
        new_extent = 0.5 * (new_width + new_height)
        pos2target1 = self._get_target_vec(cam_state, extent=extent)
        pos2target2 = self._get_target_vec(cam_state, extent=new_extent)
        new_position = position + pos2target1 - pos2target2

        return {
            **cam_state,
            "width": new_width,
            "height": new_height,
            "position": new_position,
            "fov": cam_state["fov"],
        }

    def _update_orbit(self, delta: Tuple[float, float]) -> None:
        assert isinstance(delta, tuple) and len(delta) == 2

        delta_azimuth, delta_elevation = delta
        camera_state = self._camera.get_state()

        # Note: this code does not use la.vec_euclidian_to_spherical and
        # la.vec_spherical_to_euclidian, because those functions currently
        # have no way to specify a different up vector.

        position = camera_state["position"]
        rotation = camera_state["rotation"]
        up = camera_state["reference_up"]

        # Where is the camera looking at right now
        forward = la.vec_transform_quat((0, 0, -1), rotation)

        # # Get a reference vector, that is orthogonal to up, in a deterministic way.
        # # Might need this if we ever want the azimuth
        # aligned_up = _get_axis_aligned_up_vector(up)
        # orthogonal_vec = np.cross(up, np.roll(aligned_up, 1))

        # Get current elevation, so we can clip it.
        # We don't need the azimuth. When we do, it'd need more care to get a proper 0..2pi range
        elevation = la.vec_angle(forward, up) - 0.5 * np.pi

        # Apply boundaries to the elevation
        new_elevation = elevation + delta_elevation
        bounds = -89 * np.pi / 180, 89 * np.pi / 180
        if new_elevation < bounds[0]:
            delta_elevation = bounds[0] - elevation
        elif new_elevation > bounds[1]:
            delta_elevation = bounds[1] - elevation

        r_azimuth = la.quat_from_axis_angle(up, -delta_azimuth)
        r_elevation = la.quat_from_axis_angle((1, 0, 0), -delta_elevation)

        # Get rotations
        rot1 = rotation
        rot2 = la.quat_mul(r_azimuth, la.quat_mul(rot1, r_elevation))

        # Calculate new position
        pos1 = position
        pos2target1 = self._get_target_vec(camera_state, rotation=rot1)
        pos2target2 = self._get_target_vec(camera_state, rotation=rot2)
        pos2 = pos1 + pos2target1 - pos2target2

        # Apply new state
        new_camera_state = {"position": pos2, "rotation": rot2}
        self._camera.set_state(new_camera_state)

    def reset(self):
        self._zoom_delta = (0, 0)
        self._orbit_delta = (0, 0)

    def iter(self):
        self._update_zoom(self._zoom_delta)
        self._update_orbit(self._orbit_delta)


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

    camera = gfx.PerspectiveCamera(90, 4 / 3)
    cube = MyBox((50, 100, 50), "#ff0000")
    camera.show_object(cube.mesh)
    cam_ctrl = CameraControl(camera, clock)
    camera.look_at(cube.mesh.world.position)
    scene.add(cube.mesh)
    scene.add(gfx.AxesHelper(size=125))
    # a dummy np data to init the texture. Texture would inference the shape from the data
    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.float32)
    # https://docs.rs/wgpu/latest/wgpu/enum.TextureFormat.html
    texture = gfx.Texture(
        data=data, dim=2, format=wgpu.TextureFormat.rgba32float)
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
        u = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
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
                        case pg.K_q:
                            cam_ctrl.delta_zoom((0.02, -0.02))
                        case pg.K_e:
                            cam_ctrl.delta_zoom((-0.02, 0.02))
                        case pg.K_w:
                            cam_ctrl.delta_orbit((0.025, 0))
                        case pg.K_s:
                            cam_ctrl.delta_orbit((-0.025, 0))
                        case pg.K_a:
                            cam_ctrl.delta_orbit((0, 0.025))
                        case pg.K_d:
                            cam_ctrl.delta_orbit((0, -0.025))
                case pg.KEYUP:
                    match ev.key:
                        case pg.K_s:
                            cam_ctrl.reset()
                        case pg.K_w:
                            cam_ctrl.reset()
                        case pg.K_d:
                            cam_ctrl.reset()
                        case pg.K_a:
                            cam_ctrl.reset()
                        case pg.K_q:
                            cam_ctrl.reset()
                        case pg.K_e:
                            cam_ctrl.reset()


async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(render_task)

# https://github.com/pygfx/pygfx/issues/260
if __name__ == "__main__":
    anyio.run(main)
