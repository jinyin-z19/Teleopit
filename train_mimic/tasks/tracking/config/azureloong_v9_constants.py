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

# Estimated actuator specs. These are reasonable defaults for a humanoid robot.
# Joint naming convention for azureloong_v9:
#   Arms:  J_arm_l_0[1-7], J_arm_r_0[1-7]  (7 DOF each)
#   Head:  J_head_yaw, J_head_pitch
#   Waist: J_waist_roll, J_waist_yaw
#   Legs:  J_hip_[lr]_(pitch|roll|yaw), J_knee_[lr]_pitch,
#          J_ankle_[lr]_(pitch|roll)  (6 DOF each leg)

# Use default actuator properties:
#   MEDIUM: for most arm/waist joints
#   LARGE:  for hip/knee joints (higher load)
#   SMALL:  for ankle/head/wrist joints
_MEDIUM = ElectricActuator(
    reflected_inertia=0.01,
    velocity_limit=20.0,
    effort_limit=50.0,
)
_LARGE = ElectricActuator(
    reflected_inertia=0.03,
    velocity_limit=15.0,
    effort_limit=100.0,
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


# Arm joints: J_arm_l_01 through J_arm_l_07 (same for right)
ARM_JOINT_NAMES_L = tuple(f"J_arm_l_0{i}" for i in range(1, 8))
ARM_JOINT_NAMES_R = tuple(f"J_arm_r_0{i}" for i in range(1, 8))

# Leg joints
LEG_JOINT_NAMES = (
    "J_hip_l_pitch",
    "J_hip_l_roll",
    "J_hip_l_yaw",
    "J_knee_l_pitch",
    "J_hip_r_pitch",
    "J_hip_r_roll",
    "J_hip_r_yaw",
    "J_knee_r_pitch",
)

# Ankle joints
ANKLE_JOINT_NAMES = (
    "J_ankle_l_pitch",
    "J_ankle_l_roll",
    "J_ankle_r_pitch",
    "J_ankle_r_roll",
)

# Waist joints
WAIST_JOINT_NAMES = ("J_waist_roll", "J_waist_yaw")

# Head joints (not actuated in G1 either, but present in model)
HEAD_JOINT_NAMES = ("J_head_yaw", "J_head_pitch")

AZURELOONG_V9_ACTUATOR_ARM_L = _make_actuator(ARM_JOINT_NAMES_L, _MEDIUM)
AZURELOONG_V9_ACTUATOR_ARM_R = _make_actuator(ARM_JOINT_NAMES_R, _MEDIUM)
AZURELOONG_V9_ACTUATOR_LEG = _make_actuator(LEG_JOINT_NAMES, _LARGE)
AZURELOONG_V9_ACTUATOR_ANKLE = _make_actuator(ANKLE_JOINT_NAMES, _SMALL)
AZURELOONG_V9_ACTUATOR_WAIST = _make_actuator(WAIST_JOINT_NAMES, _MEDIUM)
AZURELOONG_V9_ACTUATOR_HEAD = _make_actuator(HEAD_JOINT_NAMES, _SMALL)

##
# Keyframe config.
##

# Home standing pose for azureloong_v9: all zeros (default standing position).
# The default qpos from the XML gives a standing pose at height 1.25.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 1.10),  # Slightly lower than default 1.25 for stability
    joint_pos={
        "J_hip_l_pitch": -0.1,
        "J_knee_l_pitch": 0.3,
        "J_ankle_l_pitch": -0.2,
        "J_hip_r_pitch": -0.1,
        "J_knee_r_pitch": 0.3,
        "J_ankle_r_pitch": -0.2,
        "J_arm_l_02": 0.2,
        "J_arm_l_04": 1.0,
        "J_arm_r_02": 0.2,
        "J_arm_r_04": 1.0,
    },
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
        AZURELOONG_V9_ACTUATOR_LEG,
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
