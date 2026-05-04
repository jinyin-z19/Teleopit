"""AzureLoong V9 robot constants for mjlab training.

Reads kp / kd / torque / default_angles / root height from
``teleopit/configs/robot/azureloong_v9.yaml`` — the yaml is the single source of truth.
Joints are grouped by identical (kp, kd, torque) into minimal actuator configs.
"""

import re
from collections import defaultdict
from pathlib import Path

import mujoco
import yaml

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# ---------------------------------------------------------------------------
# Paths — resolve through installed teleopit package to avoid case-sensitivity issues
# ---------------------------------------------------------------------------

import teleopit
_PROJECT = Path(teleopit.__file__).resolve().parent.parent
_XML_PATH = _PROJECT / "teleopit" / "retargeting" / "gmr" / "assets" / "azureloong_v9" / "azureloong_v9.xml"
_YAML_PATH = _PROJECT / "teleopit" / "configs" / "robot" / "azureloong_v9.yaml"
assert _XML_PATH.exists(), f"XML not found: {_XML_PATH}"
assert _YAML_PATH.exists(), f"YAML not found: {_YAML_PATH}"

_Y = yaml.safe_load(_YAML_PATH.read_text())

# ---------------------------------------------------------------------------
# Joint names — must match order in azureloong_v9.xml (and yaml kps/kds arrays)
# ---------------------------------------------------------------------------

_JOINTS = (
    *(f"J_arm_l_0{i}" for i in range(1, 8)),
    *(f"J_arm_r_0{i}" for i in range(1, 8)),
    "J_head_yaw", "J_head_pitch",
    "J_waist_roll", "J_waist_yaw",
    "J_hip_l_pitch", "J_hip_l_roll", "J_hip_l_yaw",
    "J_knee_l_pitch", "J_ankle_l_pitch", "J_ankle_l_roll",
    "J_hip_r_pitch", "J_hip_r_roll", "J_hip_r_yaw",
    "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll",
)
assert len(_JOINTS) == len(_Y["kps"]) == 30
_KPS = _Y["kps"]
_KDS = _Y["kds"]
_TORQUES = _Y["torque_limits"]
_Z = float(_Y["mujoco_default_qpos"][2])


# ---------------------------------------------------------------------------
# get_spec — strips built-in actuators so articulation can inject them
# ---------------------------------------------------------------------------

def get_spec() -> mujoco.MjSpec:
    """Load XML for mjlab, stripping built-in actuators (articulation injects them)."""
    raw = _XML_PATH.read_text()
    # Make meshdir absolute so MjSpec.from_string() can find STL files
    raw = raw.replace(
        'meshdir="meshes/"',
        f'meshdir="{_XML_PATH.parent / "meshes"}/"',
    )
    raw = re.sub(r"<actuator>.*?</actuator>", "", raw, flags=re.DOTALL)
    return mujoco.MjSpec.from_string(raw)

# ---------------------------------------------------------------------------
# Actuators — grouped by identical (kp, kd, torque) from yaml
# ---------------------------------------------------------------------------

def _act(kp: float, kd: float, tq: float, *names: str) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=names, stiffness=kp, damping=kd, effort_limit=tq,
    )

_groups: dict[tuple[float, float, float], list[str]] = defaultdict(list)
for jn, kp, kd, tq in zip(_JOINTS, _KPS, _KDS, _TORQUES):
    _groups[(kp, kd, tq)].append(jn)

ACTUATORS = tuple(
    _act(kp, kd, tq, *names) for (kp, kd, tq), names in _groups.items()
)

AZURELOONG_V9_ARTICULATION = EntityArticulationInfoCfg(
    actuators=ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

# ---------------------------------------------------------------------------
# Action scale (0.25 * torque / stiffness — same formula as G1)
# ---------------------------------------------------------------------------

AZURELOONG_V9_ACTION_SCALE: dict[str, float] = {
    jn: 0.25 * tq / kp for jn, kp, tq in zip(_JOINTS, _KPS, _TORQUES)
}

# ---------------------------------------------------------------------------
# Initial state (from yaml default_angles + mujoco_default_qpos z)
# ---------------------------------------------------------------------------

_defaults = _Y["default_angles"]
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, _Z),
    joint_pos=dict(zip(_JOINTS, _defaults)) if any(a != 0.0 for a in _defaults) else {},
    joint_vel={".*": 0.0},
)

# ---------------------------------------------------------------------------
# Collision config
# ---------------------------------------------------------------------------

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={"base_link_collision": 1, ".*_collision": 3},
    priority={"link_ankle_.*_collision": 1},
    friction={"link_ankle_.*_collision": (0.6,)},
)

# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_azureloong_v9_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=AZURELOONG_V9_ARTICULATION,
    )
