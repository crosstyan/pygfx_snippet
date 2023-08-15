import pygfx as gfx
import pylinalg as la
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import pygame as pg
from pygame._sdl2 import Window, Texture, Image, Renderer, get_drivers
from typing import Coroutine, NoReturn, Tuple, TypedDict, Union, List, Dict, Any, Optional, Callable, Awaitable, Generator, Any
from pygfx.cameras._perspective import fov_distance_factor
from numpy.typing import NDArray


# some type annotations for the magic camera state
class CameraState(TypedDict):
    width: float
    height: float
    fov: float
    # not sure about dimensions
    position: NDArray
    maintain_aspect: bool
    rotation: NDArray
    reference_up: NDArray


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
        camera_state = self._get_camera_state()

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
        new_camera_state:CameraState = {**camera_state,"position": pos2, "rotation": rot2}
        self._set_camera_state(new_camera_state)

    def reset(self):
        self._zoom_delta = (0, 0)
        self._orbit_delta = (0, 0)

    def iter(self):
        self._update_zoom(self._zoom_delta)
        self._update_orbit(self._orbit_delta)
