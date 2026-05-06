"""Terrain height scanner using MuJoCo ray casting.

Casts rays downward from probe points defined in the robot body frame
and returns sampled terrain heights (world-frame Z coordinates).

This module is designed to be lightweight and zero-copy — it reuses
the existing MuJoCo model and data from ``MuJoCoRobot``.

Ray casting uses a manual implementation (``_ray_cast``) to avoid
numpy 2.x / pybind11 compatibility issues with ``mujoco.mj_ray``.
Supported geom types: plane, hfield, box.
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

# MuJoCo geom type constants (mjtGeom)
_mjGEOM_PLANE = 0
_mjGEOM_HFIELD = 1
_mjGEOM_SPHERE = 2
_mjGEOM_CAPSULE = 3
_mjGEOM_ELLIPSOID = 4
_mjGEOM_CYLINDER = 5
_mjGEOM_BOX = 6
_mjGEOM_MESH = 7

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


def _quat_to_mat33(quat_wxyz: Float64Array) -> Float64Array:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    qn = np.asarray(quat_wxyz, dtype=np.float64)
    qn = qn / max(np.linalg.norm(qn), 1e-8)
    w, x, y, z = float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _ray_plane_intersection(
    ray_origin: Float64Array,
    ray_dir: Float64Array,
    geom_xpos: Float64Array,
    geom_xmat: Float64Array,
) -> float:
    """Compute ray-plane intersection distance.

    MuJoCo plane geoms have infinite extent.  The plane normal is the geom's
    local +Z (transformed by geom_xmat), and the plane passes through geom_xpos.

    Returns distance along ray, or -1 if parallel / no intersection.
    """
    # Plane normal: rotate geom local Z (0,0,1) by geom_xmat
    normal = np.array([
        geom_xmat[2], geom_xmat[5], geom_xmat[8],
    ], dtype=np.float64)

    # Ray-plane intersection: t = (p0 - o)·n / (d·n)
    denom = float(np.dot(ray_dir, normal))
    if abs(denom) < 1e-12:
        return -1.0  # parallel

    t = float(np.dot(geom_xpos - ray_origin, normal)) / denom
    if t < 0.0:
        return -1.0  # behind ray origin

    return t


def _ray_box_intersection(
    ray_origin: Float64Array,
    ray_dir: Float64Array,
    geom_xpos: Float64Array,
    geom_xmat: Float64Array,
    geom_size: Float64Array,
) -> float:
    """Compute ray-AABB intersection distance (slab method).

    The box is centered at geom_xpos with half-extents *geom_size*,
    oriented by geom_xmat.

    Returns nearest positive distance, or -1 if no intersection.
    """
    # Transform ray to box-local coordinates
    center = np.asarray(geom_xpos, dtype=np.float64).reshape(3)
    half = np.asarray(geom_size, dtype=np.float64).reshape(3)
    mat33 = np.asarray(geom_xmat, dtype=np.float64).reshape(3, 3)

    # Inverse rotate: mat33^T = mat33^{-1} for rotation matrix
    local_origin = mat33.T @ (ray_origin - center)
    local_dir = mat33.T @ ray_dir

    tmin = -1e30
    tmax = 1e30

    for i in range(3):
        if abs(local_dir[i]) < 1e-12:
            # Ray parallel to this slab
            if local_origin[i] < -half[i] or local_origin[i] > half[i]:
                return -1.0
        else:
            inv_d = 1.0 / local_dir[i]
            t1 = (-half[i] - local_origin[i]) * inv_d
            t2 = (half[i] - local_origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return -1.0

    if tmin < 0.0:
        if tmax < 0.0:
            return -1.0
        return tmax

    return tmin


def _ray_hfield_intersection(
    ray_origin: Float64Array,
    ray_dir: Float64Array,
    model: mujoco.MjModel,
    hfield_id: int,
) -> float:
    """Compute ray-hfield intersection by stepping along the ray.

    MuJoCo hfields are elevation grids in the X-Y plane.
    This performs a simple ray-march through the hfield grid.

    Returns distance along ray, or -1 if no intersection.
    """
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    size = np.array(model.hfield_size[hfield_id, :4], dtype=np.float64)
    # size = [xmin, ymin, xmax, ymax] for the hfield footprint
    adr = model.hfield_adr[hfield_id]
    data_view = model.hfield_data[adr:adr + nrow * ncol]

    dx = (size[2] - size[0]) / (ncol - 1) if ncol > 1 else 1.0
    dy = (size[3] - size[1]) / (nrow - 1) if nrow > 1 else 1.0

    # Step along ray in small increments
    step_size = 0.02  # 2cm steps
    max_dist = 10.0
    t = 0.0
    while t < max_dist:
        p = ray_origin + ray_dir * t
        # Check if point is within hfield XY bounds
        if size[0] <= p[0] <= size[2] and size[1] <= p[1] <= size[3]:
            # Bilinear interpolate hfield height at this XY
            col_f = (p[0] - size[0]) / dx
            row_f = (p[1] - size[1]) / dy
            col0 = int(np.clip(np.floor(col_f), 0, ncol - 1))
            row0 = int(np.clip(np.floor(row_f), 0, nrow - 1))
            col1 = min(col0 + 1, ncol - 1)
            row1 = min(row0 + 1, nrow - 1)
            fx = col_f - col0
            fy = row_f - row1  # note: MuJoCo hfield row 0 is at ymax

            h00 = float(data_view[row0 * ncol + col0])
            h10 = float(data_view[row0 * ncol + col1])
            h01 = float(data_view[row1 * ncol + col0])
            h11 = float(data_view[row1 * ncol + col1])

            h = (1 - fx) * (1 - fy) * h00 + fx * (1 - fy) * h10 + \
                (1 - fx) * fy * h01 + fx * fy * h11

            if p[2] <= h:
                return t

        t += step_size

    return -1.0


def _ray_cast(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ray_origin: Float64Array,
    ray_dir: Float64Array,
    ray_length: float = 5.0,
    geom_group_filter: int | None = None,
) -> float:
    """Cast a ray against the scene and return distance to nearest geom.

    Manual implementation that works around numpy 2.x / pybind11
    compatibility issues with ``mujoco.mj_ray`` in MuJoCo 3.8.0.

    Supported geom types: plane, hfield, box.

    Returns distance along ray, or -1 if no intersection.
    """
    best_t = ray_length + 1.0
    origin = np.asarray(ray_origin, dtype=np.float64).reshape(3)
    direction = np.asarray(ray_dir, dtype=np.float64).reshape(3)
    direction = direction / max(np.linalg.norm(direction), 1e-8)

    for i in range(model.ngeom):
        geom_type = model.geom_type[i]
        # Skip if geom is invisible or filtered out
        if geom_group_filter is not None:
            if not (model.geom_group[i] & geom_group_filter):
                continue

        # Skip robot body geoms — we only want static/terrain geoms
        body_id = model.geom_bodyid[i]
        if body_id > 0:
            # Check if this is a static body (body 0 is world)
            # Static geoms have bodyid == 0 in MuJoCo
            continue

        xpos = data.geom_xpos[i].copy()
        xmat = data.geom_xmat[i].copy()
        size = model.geom_size[i].copy()

        t = -1.0
        if geom_type == _mjGEOM_PLANE:
            t = _ray_plane_intersection(origin, direction, xpos, xmat)
        elif geom_type == _mjGEOM_BOX:
            t = _ray_box_intersection(origin, direction, xpos, xmat, size)
        elif geom_type == _mjGEOM_HFIELD:
            t = _ray_hfield_intersection(origin, direction, model, i)
        # Other geom types (sphere, capsule, cylinder, mesh) — skip for now

        if 0.0 <= t < best_t:
            best_t = t

    if best_t <= ray_length:
        return best_t
    return -1.0


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

            # Perform ray cast using manual implementation
            # (avoids numpy 2.x / pybind11 incompatibility with mujoco.mj_ray)
            distance = _ray_cast(
                self._model,
                self._data,
                ray_start,
                ray_dir,
                ray_length=self._ray_length,
                geom_group_filter=self._geom_group_filter,
            )

            if distance >= 0.0 and distance <= self._ray_length:
                # Hit point Z = ray_start.z - distance
                hit_z = float(ray_start[2] - distance)
                heights[i] = np.float32(hit_z)
            else:
                # No hit — default to ground plane (Z=0)
                heights[i] = np.float32(0.0)

        return heights
