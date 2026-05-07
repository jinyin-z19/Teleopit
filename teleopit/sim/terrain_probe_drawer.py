"""Terrain probe point renderer for MuJoCo viewer / render_sim.

Draws coloured spheres at each terrain height probe point so users can
visually verify the scanner coverage and sampled heights.

Provides three rendering paths:

1. ``draw_via_visualizer()`` — for ``DebugVisualizer`` (play.py native/viser/video)
2. ``init_scene_geoms()`` / ``update_scene_geoms()`` — for raw ``MjvScene`` (render_sim.py)
3. ``make_update_callback()`` — convenience factory for play.py ``env.update_visualizers``
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]

# Lazy import to avoid hard dependency on mujoco when the module is first parsed.
_mujoco = None


def _get_mujoco():
    global _mujoco
    if _mujoco is None:
        import mujoco as _mj
        _mujoco = _mj
    return _mujoco


def _quat_rotate_vec(quat_wxyz: Float64Array, vec: Float64Array) -> Float64Array:
    """Rotate *vec* by quaternion *quat_wxyz* (w,x,y,z)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    qn = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    rx = v[0] + w * tx + (y * tz - z * ty)
    ry = v[1] + w * ty + (z * tx - x * tz)
    rz = v[2] + w * tz + (x * ty - y * tx)
    return np.array([rx, ry, rz], dtype=np.float64)


def _height_color(h: float, h_min: float = -0.3, h_max: float = 0.3) -> tuple[float, float, float, float]:
    """Map height to RGBA: blue (low) → green (mid) → red (high)."""
    t = max(0.0, min(1.0, (h - h_min) / max(h_max - h_min, 1e-6)))
    if t < 0.5:
        # blue → green
        s = t * 2.0
        return (0.0, s, 1.0 - s, 0.85)
    else:
        # green → red
        s = (t - 0.5) * 2.0
        return (s, 1.0 - s, 0.0, 0.85)


class TerrainProbeDrawer:
    """Renders terrain scanner probe points as coloured spheres.

    Parameters
    ----------
    probe_offsets : list of (float, float)
        2D offsets in robot body frame (X-forward, Y-left).
    ray_start_height : float
        Height above base from which rays are cast.
    fixed_color : tuple or None
        If a 4-tuple (r, g, b, a), all probes use this color.
        If None, uses the height-based blue→green→red gradient.
    """

    def __init__(
        self,
        probe_offsets: list[tuple[float, float]],
        ray_start_height: float = 1.0,
        fixed_color: tuple[float, float, float, float] | None = (
            1.0, 0.0, 0.0, 0.85,
        ),
    ) -> None:
        self._probe_offsets = list(probe_offsets)
        self._ray_start_height = float(ray_start_height)
        self._fixed_color = fixed_color
        self.num_points = len(self._probe_offsets)

    # ── mjlab DebugVisualizer path (play.py native / viser) ──────────

    def draw_via_visualizer(
        self,
        visualizer,  # DebugVisualizer
        base_pos: Float64Array,
        base_quat: Float64Array,
        heights: Float32Array | None = None,
        h_min: float = -0.3,
        h_max: float = 0.3,
    ) -> None:
        """Draw probe spheres using a DebugVisualizer."""
        bp = np.asarray(base_pos, dtype=np.float64).reshape(3)
        bq = np.asarray(base_quat, dtype=np.float64).reshape(4)
        num = len(self._probe_offsets)
        h = np.zeros(num, dtype=np.float32)
        if heights is not None:
            h = np.asarray(heights, dtype=np.float32).reshape(-1)[:num]
        for i, (ox, oy) in enumerate(self._probe_offsets):
            offset_w = _quat_rotate_vec(bq, [ox, oy, self._ray_start_height])
            center = (bp + offset_w).astype(np.float64)
            # Shift probe sphere to the actual terrain height
            center[2] = float(h[i]) + 0.02  # small z-offset above surface
            color = (
                self._fixed_color
                if self._fixed_color is not None
                else _height_color(float(h[i]), h_min, h_max)
            )
            visualizer.add_sphere(center, 0.03, color)

    # ── Raw MuJoCo scene path (render_sim.py, mocap viewer) ──────────

    _mjv_geoms: list[int] = []  # geom indices allocated in MjvScene

    def init_scene_geoms(self, scene: Any, max_geoms: int = 25) -> None:
        """Pre-allocate geom slots in *scene* for probe spheres."""
        mujoco = _get_mujoco()
        self._mjv_geoms = []
        for _ in range(max_geoms):
            idx = scene.ngeom
            if idx >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.03, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).ravel(),
                rgba=np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),
            )
            scene.ngeom = idx + 1
            self._mjv_geoms.append(idx)

    def update_scene_geoms(
        self,
        scene: Any,
        base_pos: Float64Array,
        base_quat: Float64Array,
        heights: Float32Array | None = None,
        h_min: float = -0.3,
        h_max: float = 0.3,
    ) -> None:
        """Update pre-allocated geoms to current probe positions."""
        bp = np.asarray(base_pos, dtype=np.float64).reshape(3)
        bq = np.asarray(base_quat, dtype=np.float64).reshape(4)
        num = min(len(self._probe_offsets), len(self._mjv_geoms))
        h = np.zeros(num, dtype=np.float32)
        if heights is not None:
            h = np.asarray(heights, dtype=np.float32).reshape(-1)[:num]
        for i, (ox, oy) in enumerate(self._probe_offsets[:num]):
            offset_w = _quat_rotate_vec(bq, [ox, oy, self._ray_start_height])
            center = bp + offset_w
            center[2] = float(h[i]) + 0.02
            color = (
                self._fixed_color
                if self._fixed_color is not None
                else _height_color(float(h[i]), h_min, h_max)
            )
            g = scene.geoms[self._mjv_geoms[i]]
            g.pos[:] = center.astype(np.float64)
            g.rgba[:] = np.array(color, dtype=np.float32)

    # ── Convenience factory for play.py ─────────────────────────────

    def make_update_callback(
        self,
        env: Any,
        num_height_points: int = 25,
    ) -> Callable[[Any], None]:
        """Return a callable suitable for ``env.update_visualizers``.

        The returned function reads base pose and terrain heights from *env*
        each frame, then calls ``draw_via_visualizer()``.

        Usage in play.py::

            drawer = TerrainProbeDrawer(probe_offsets, ray_start_height=1.0)
            env.update_visualizers = drawer.make_update_callback(env)
        """
        # Import here to avoid circular deps when this module is first loaded.
        from train_mimic.tasks.tracking.mdp.observations import terrain_heights

        _drawer = self

        def _callback(visualizer: Any) -> None:
            try:
                # Unwrap RslRlVecEnvWrapper to reach ManagerBasedRlEnv.
                inner = env
                while hasattr(inner, "env") and not hasattr(inner, "scene"):
                    inner = getattr(inner, "env", inner)
                    if inner is env:  # safety against self-referencing wrappers
                        break

                robot = inner.scene["robot"]
                base_pos = robot.data.root_link_pos_w[0].cpu().numpy()
                base_quat = robot.data.root_link_quat_w[0].cpu().numpy()

                heights_t = terrain_heights(
                    inner, "motion", num_height_points=num_height_points,
                )
                heights_np = heights_t[0].cpu().numpy()

                _drawer.draw_via_visualizer(
                    visualizer, base_pos, base_quat, heights_np,
                )
            except Exception:
                _logger.debug("Terrain probe render skipped", exc_info=True)

        return _callback
