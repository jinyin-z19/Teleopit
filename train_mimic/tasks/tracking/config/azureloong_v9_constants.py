"""AzureLoong V9 robot constants for mjlab training.

Adapted from G1 constants. Actuator specs are estimated defaults since exact
motor specs for azureloong_v9 are not available in the training pipeline yet.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

AZURELOONG_V9_XML: Path = (
    Path(__file__).resolve().parents[4]
    / "teleopit"
    / "retargeting"
    / "gmr"
    / "assets"
    / "azureloong_v9"
    / "azureloong_v9_mjlab.xml"
)
assert AZURELOONG_V9_XML.exists(), f"XML not found: {AZURELOONG_V9_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(AZURELOONG_V9_XML))


##
# Actuator config.
##

# Arm / head / waist actuators: computed from motor specs via natural-frequency PD.
_MEDIUM = ElectricActuator(
    reflected_inertia=0.01,
    velocity_limit=20.0,
    effort_limit=50.0,
)
_SMALL = ElectricActuator(
    reflected_inertia=0.003,
    velocity_limit=30.0,
    effort_limit=15.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _make_actuator(
    names_expr: tuple[str, ...],
    motor: ElectricActuator,
) -> BuiltinPositionActuatorCfg:
    """Create a BuiltinPositionActuatorCfg with PD gains from motor specs."""
    armature = motor.reflected_inertia
    stiffness = armature * NATURAL_FREQ**2
    damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
    return BuiltinPositionActuatorCfg(
        target_names_expr=names_expr,
        stiffness=stiffness,
        damping=damping,
        effort_limit=motor.effort_limit,
        armature=armature,
    )


def _make_explicit_actuator(
    names_expr: tuple[str, ...],
    stiffness: float,
    damping: float,
    effort_limit: float,
) -> BuiltinPositionActuatorCfg:
    """Create a BuiltinPositionActuatorCfg with explicit PD gains."""
    return BuiltinPositionActuatorCfg(
        target_names_expr=names_expr,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        armature=stiffness / NATURAL_FREQ**2,
    )


# Arm joints: J_arm_l_01 through J_arm_l_07 (same for right)
ARM_JOINT_NAMES_L = tuple(f"J_arm_l_0{i}" for i in range(1, 8))
ARM_JOINT_NAMES_R = tuple(f"J_arm_r_0{i}" for i in range(1, 8))

# Waist joints
WAIST_JOINT_NAMES = ("J_waist_roll", "J_waist_yaw")

# Head joints
HEAD_JOINT_NAMES = ("J_head_yaw", "J_head_pitch")

AZURELOONG_V9_ACTUATOR_ARM_L = _make_actuator(ARM_JOINT_NAMES_L, _MEDIUM)
AZURELOONG_V9_ACTUATOR_ARM_R = _make_actuator(ARM_JOINT_NAMES_R, _MEDIUM)
AZURELOONG_V9_ACTUATOR_WAIST = _make_actuator(WAIST_JOINT_NAMES, _MEDIUM)
AZURELOONG_V9_ACTUATOR_HEAD = _make_actuator(HEAD_JOINT_NAMES, _SMALL)

# Leg actuators: per-joint explicit stiffness/damping from hardware reference.
# Reference values (stiffness, damping, effort_limit):
#   hip_roll:    (300, 1.0, 100)   hip_yaw:   (200, 0.5, 100)
#   hip_pitch:   (200, 1.0, 100)   knee:      (400, 4.0, 100)
#   ankle_pitch: (120, 1.0,  15)   ankle_roll:(120, 1.0,  15)
AZURELOONG_V9_ACTUATOR_HIP_ROLL = _make_explicit_actuator(
    ("J_hip_l_roll", "J_hip_r_roll"), 300.0, 1.0, 100.0,
)
AZURELOONG_V9_ACTUATOR_HIP_YAW = _make_explicit_actuator(
    ("J_hip_l_yaw", "J_hip_r_yaw"), 200.0, 0.5, 100.0,
)
AZURELOONG_V9_ACTUATOR_HIP_PITCH = _make_explicit_actuator(
    ("J_hip_l_pitch", "J_hip_r_pitch"), 200.0, 1.0, 100.0,
)
AZURELOONG_V9_ACTUATOR_KNEE = _make_explicit_actuator(
    ("J_knee_l_pitch", "J_knee_r_pitch"), 400.0, 4.0, 100.0,
)
AZURELOONG_V9_ACTUATOR_ANKLE = _make_explicit_actuator(
    ("J_ankle_l_pitch", "J_ankle_l_roll", "J_ankle_r_pitch", "J_ankle_r_roll"),
    120.0, 1.0, 15.0,
)

##
# Keyframe config.
##

# Home standing pose for azureloong_v9: straight leg at XML default height 1.25m.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 1.25),
    joint_pos={},
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# azureloong_v9 collision geoms: base_link_collision, link_ankle_l_roll_collision,
# link_ankle_r_roll_collision (minimal set in the mjlab XML).
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={"base_link_collision": 1, ".*_collision": 3},
    priority={"link_ankle_.*_collision": 1},
    friction={"link_ankle_.*_collision": (0.6,)},
)

##
# Final config.
##

AZURELOONG_V9_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        AZURELOONG_V9_ACTUATOR_ARM_L,
        AZURELOONG_V9_ACTUATOR_ARM_R,
        AZURELOONG_V9_ACTUATOR_HIP_ROLL,
        AZURELOONG_V9_ACTUATOR_HIP_YAW,
        AZURELOONG_V9_ACTUATOR_HIP_PITCH,
        AZURELOONG_V9_ACTUATOR_KNEE,
        AZURELOONG_V9_ACTUATOR_ANKLE,
        AZURELOONG_V9_ACTUATOR_WAIST,
        AZURELOONG_V9_ACTUATOR_HEAD,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_azureloong_v9_robot_cfg() -> EntityCfg:
    """Get a fresh AzureLoong V9 robot configuration instance."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=AZURELOONG_V9_ARTICULATION,
    )


AZURELOONG_V9_ACTION_SCALE: dict[str, float] = {}
for a in AZURELOONG_V9_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        AZURELOONG_V9_ACTION_SCALE[n] = 0.25 * e / s
