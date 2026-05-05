"""Terrain height scanner using MuJoCo ray casting.

Casts rays downward from probe points defined in the robot body frame
and returns sampled terrain heights (world-frame Z coordinates).

This module is designed to be lightweight and zero-copy — it reuses
the existing MuJoCo model and data from ``MuJoCoRobot``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

# Default probe grid: a 5×5 grid spanning ±0.4 m in X/Y, centered under the robot.
_DEFAULT_PROBE_GRID: list[tuple[float, float]] = [
    (x, y)
    for x in np.linspace(-0.4, 0.4, 5)
    for y in np.linspace(-0.4, 0.4, 5)
]


def _quat_rotate_vec(quat_wxyz: Float64Array, vec: Float64Array) -> Float64Array:
    """Rotate *vec* by quaternion *quat_wxyz* (w,x,y,z)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    qn = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])

    # Rotate: q * v_quat * q_conj
    # Compute using Rodrigues-like formula for efficiency
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])

    rx = v[0] + w * tx + (y * tz - z * ty)
    ry = v[1] + w * ty + (z * tx - x * tz)
    rz = v[2] + w * tz + (x * ty - y * tx)

    return np.array([rx, ry, rz], dtype=np.float64)


class TerrainHeightScanner:
    """Scans terrain heights at probe points using MuJoCo ray casting.

    Probe points are defined as 2D (x, y) offsets in the robot body frame
    (X-forward, Y-left).  Rays are cast straight down (world -Z) from each
    world-space probe position.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model (must contain hfield geom(s) for terrain).
    data : mujoco.MjData
        MuJoCo data (must be up-to-date with scene state).
    probe_offsets : sequence of (float, float)
        2D offsets in robot body frame where heights are sampled.
        Default: 5×5 grid over ±0.4 m.
    ray_start_height : float
        Height (m) above the base from which to start each ray.
        Ensures rays always start above terrain.  Default: 1.0 m.
    ray_length : float
        Maximum ray length downward (m).  Default: 5.0 m.
    geom_group_filter : int | None
        If set, only geoms in this group are considered for ray hits.
        Use MuJoCo's geom group mechanism to exclude robot bodies.
        Default: None (no filtering).

    Notes
    -----
    - Terrain is expected to be represented as MuJoCo hfield geoms.
    - The scanner uses ``mujoco.mj_ray`` which queries the full scene
      including hfields.
    - Heights are returned as world-frame Z coordinates of hit points.
    - If a ray misses (no hit), the height is set to 0.0 (ground plane).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        probe_offsets: Sequence[tuple[float, float]] | None = None,
        ray_start_height: float = 1.0,
        ray_length: float = 5.0,
        geom_group_filter: int | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._probe_offsets: list[tuple[float, float]] = list(
            probe_offsets if probe_offsets is not None else _DEFAULT_PROBE_GRID
        )
        self._ray_start_height = float(ray_start_height)
        self._ray_length = float(ray_length)
        self._geom_group_filter = geom_group_filter
        self.num_points: int = len(self._probe_offsets)

        _logger.info(
            "TerrainHeightScanner initialized: %d probe points, "
            "ray_start_height=%.2f, ray_length=%.2f",
            self.num_points,
            self._ray_start_height,
            self._ray_length,
        )

    @property
    def probe_offsets(self) -> list[tuple[float, float]]:
        return list(self._probe_offsets)

    def scan(
        self,
        base_pos: Float64Array,
        base_quat: Float64Array,
    ) -> Float32Array:
        """Scan terrain heights at all probe points.

        Parameters
        ----------
        base_pos : ndarray of shape (3,)
            Robot base position in world frame.
        base_quat : ndarray of shape (4,)
            Robot base orientation quaternion (w, x, y, z).

        Returns
        -------
        heights : ndarray of shape (num_points,)
            World-frame Z coordinates at each probe point.
            Rays that miss all geometry return 0.0.
        """
        base_pos_arr = np.asarray(base_pos, dtype=np.float64).reshape(3)
        base_quat_arr = np.asarray(base_quat, dtype=np.float64).reshape(4)

        heights = np.zeros(self.num_points, dtype=np.float32)

        for i, (ox, oy) in enumerate(self._probe_offsets):
            # Local probe offset in body frame
            local_offset = np.array([ox, oy, self._ray_start_height], dtype=np.float64)

            # Rotate offset to world frame and add base position
            world_offset = _quat_rotate_vec(base_quat_arr, local_offset)
            ray_start = base_pos_arr + world_offset

            # Ray direction: straight down (world -Z)
            ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

            # Perform ray cast
            # mujoco.mj_ray returns distance to nearest geom, -1 if no hit
            distance = mujoco.mj_ray(
                self._model,
                self._data,
                ray_start,       # pnt  — ray origin
                ray_dir,         # vec  — ray direction
                None,            # geomgroup — optional filter
                -1,              # flg_static — include static geoms
                -1,              # bodyexclude — exclude no body
                None,            # geomid output
            )

            if distance >= 0.0 and distance <= self._ray_length:
                # Hit point Z = ray_start.z - distance
                hit_z = float(ray_start[2] - distance)
                heights[i] = np.float32(hit_z)
            else:
                # No hit — default to ground plane (Z=0)
                heights[i] = np.float32(0.0)

        return heights
