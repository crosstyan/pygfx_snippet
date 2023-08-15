import pygfx as gfx
import pylinalg as la

from typing import Coroutine, NoReturn, Tuple, TypedDict, Union, List, Dict, Any, Optional, Callable, Awaitable, Generator

class MyBox:
    _mesh: gfx.Mesh
    is_rotating: bool

    @property
    def mesh(self) -> gfx.Mesh:
        return self._mesh

    # width, height, depth
    def __init__(self, dim: Tuple[int, int, int] = (200, 200, 200), color: str = "#336699") -> None:
        geometry = gfx.box_geometry(*dim)
        material = gfx.MeshPhongMaterial(color=color)
        self.is_rotating = False
        self._mesh = gfx.Mesh(geometry, material)

    def iter(self):
        if self.is_rotating:
            rot = la.quat_from_euler((0.005, 0.01), order="XY")
            self._mesh.local.rotation = la.quat_mul(
                rot, self._mesh.local.rotation)
