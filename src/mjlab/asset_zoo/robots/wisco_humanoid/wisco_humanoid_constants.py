"""Wisco Humanoid robot constants.

Scaffold configuration for a robot whose model is not yet defined. The structure and
the derived-gain formulas are carried over from the Asimov reference implementation;
every hardware-specific number below is a placeholder that must be replaced once the
real MJCF and motor datasheets are available.

Joint naming convention expected by the task configs (all lookups are regex based):

  - ``left_*_joint`` / ``right_*_joint``
  - ``*_hip_pitch_joint``, ``*_hip_roll_joint``, ``*_hip_yaw_joint``,
    ``*_knee_joint``, ``*_ankle_pitch_joint``, ``*_ankle_roll_joint``
  - foot sites: ``left_foot``, ``right_foot``
  - collision geoms: ``*_collision``
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

WISCO_HUMANOID_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "wisco_humanoid"
  / "xmls"
  / "wisco_humanoid.xml"
)


def get_spec() -> mujoco.MjSpec:
  """Load the robot MJCF.

  The file check is deliberately lazy: it runs when the model is first built, not at
  import time, so the task registry and ``list-envs`` keep working before the robot
  model has been added to the repository.
  """
  if not WISCO_HUMANOID_XML.exists():
    raise FileNotFoundError(
      f"Robot MJCF not found at {WISCO_HUMANOID_XML}.\n"
      "Add the robot model to "
      "src/mjlab/asset_zoo/robots/wisco_humanoid/xmls/wisco_humanoid.xml and its "
      "meshes under src/mjlab/asset_zoo/robots/wisco_humanoid/xmls/assets/. "
      "See the package README for the expected naming conventions."
    )
  return mujoco.MjSpec.from_file(str(WISCO_HUMANOID_XML))


##
# Actuator config.
#
# Physics-based PD gains, kept from the reference implementation:
#   KP = J_reflected * omega_n^2
#   KD = 2 * zeta * J_reflected * omega_n
# KD is capped at the hardware maximum so the learned policy transfers to the real
# robot instead of exploiting a stiffer simulated controller.
#
# TODO(wisco-humanoid): every number below is an Asimov placeholder. Replace with the
# real datasheets, where J_reflected = J_rotor * gear_ratio^2.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz.
DAMPING_RATIO = 2.0
KD_MAX = 5.0  # Hardware cap [Nm s/rad].

# Reflected inertias [kg m^2]. J_rotor * gear_ratio^2.
ARMATURE_HIP_PITCH = 0.0652
ARMATURE_HIP_ROLL = 0.100
ARMATURE_HIP_YAW = 0.0343
ARMATURE_KNEE = 0.0330
ARMATURE_ANKLE_PITCH = 0.0236
ARMATURE_ANKLE_ROLL = 0.0236

# Stiffness [Nm/rad].
STIFFNESS_HIP_PITCH = ARMATURE_HIP_PITCH * NATURAL_FREQ**2
STIFFNESS_HIP_ROLL = ARMATURE_HIP_ROLL * NATURAL_FREQ**2
STIFFNESS_HIP_YAW = ARMATURE_HIP_YAW * NATURAL_FREQ**2
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE_PITCH = ARMATURE_ANKLE_PITCH * NATURAL_FREQ**2
STIFFNESS_ANKLE_ROLL = ARMATURE_ANKLE_ROLL * NATURAL_FREQ**2

# Damping [Nm s/rad].
DAMPING_HIP_PITCH = KD_MAX
DAMPING_HIP_ROLL = KD_MAX
DAMPING_HIP_YAW = KD_MAX
DAMPING_KNEE = KD_MAX
DAMPING_ANKLE = KD_MAX

# Effort limits [Nm], peak torque from the motor datasheets.
EFFORT_HIP_PITCH = 120.0
EFFORT_HIP_ROLL = 90.0
EFFORT_HIP_YAW = 60.0
EFFORT_KNEE = 75.0
EFFORT_ANKLE = 36.0

WISCO_HUMANOID_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=STIFFNESS_HIP_PITCH,
  damping=DAMPING_HIP_PITCH,
  effort_limit=EFFORT_HIP_PITCH,
  armature=ARMATURE_HIP_PITCH,
)

WISCO_HUMANOID_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=STIFFNESS_HIP_ROLL,
  damping=DAMPING_HIP_ROLL,
  effort_limit=EFFORT_HIP_ROLL,
  armature=ARMATURE_HIP_ROLL,
)

WISCO_HUMANOID_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=STIFFNESS_HIP_YAW,
  damping=DAMPING_HIP_YAW,
  effort_limit=EFFORT_HIP_YAW,
  armature=ARMATURE_HIP_YAW,
)

WISCO_HUMANOID_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=EFFORT_KNEE,
  armature=ARMATURE_KNEE,
)

WISCO_HUMANOID_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_ANKLE_PITCH,
  damping=DAMPING_ANKLE,
  effort_limit=EFFORT_ANKLE,
  armature=ARMATURE_ANKLE_PITCH,
)

##
# Keyframe config.
##

# TODO(wisco-humanoid): set the standing height to match the real model's kinematics.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.75),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

# Example of a bent-knee stance. Note that reference implementations sometimes define
# mirrored joint axes for the left and right legs, in which case the same physical
# pose needs opposite signs per side. Adapt or delete once the model exists.
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.70),
  joint_pos={
    "left_hip_pitch_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,
    "right_knee_joint": -0.4,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": 0.2,
    ".*_ankle_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Full collision, including self-collisions. Foot geoms get condim=3 so they can
# generate friction; everything else is contact-only along the normal.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={
    r"^(left|right)_ankle_roll_link_collision$": 3,
    ".*_collision": 1,
  },
  priority={
    r"^(left|right)_ankle_roll_link_collision$": 1,
    ".*_collision": 0,
  },
  friction={r"^(left|right)_ankle_roll_link_collision$": (0.6,)},
)

# Feet-only collision (self-collision disabled).
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_ankle_roll_link_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

WISCO_HUMANOID_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    WISCO_HUMANOID_ACTUATOR_HIP_PITCH,
    WISCO_HUMANOID_ACTUATOR_HIP_ROLL,
    WISCO_HUMANOID_ACTUATOR_HIP_YAW,
    WISCO_HUMANOID_ACTUATOR_KNEE,
    WISCO_HUMANOID_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_wisco_humanoid_robot_cfg() -> EntityCfg:
  """Get a fresh robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when the config
  is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=WISCO_HUMANOID_ARTICULATION,
  )


# Action scale: how much a unit action shifts the joint position target. Scaled by the
# effort budget so every joint uses a comparable fraction of its torque range.
WISCO_HUMANOID_ACTION_SCALE: dict[str, float] = {}
for _actuator in WISCO_HUMANOID_ARTICULATION.actuators:
  assert isinstance(_actuator, BuiltinPositionActuatorCfg)
  _effort = _actuator.effort_limit
  _stiffness = _actuator.stiffness
  assert _effort is not None
  assert _stiffness is not None
  for _name in _actuator.target_names_expr:
    WISCO_HUMANOID_ACTION_SCALE[_name] = 0.3 * _effort / _stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_wisco_humanoid_robot_cfg())

  viewer.launch(robot.spec.compile())
