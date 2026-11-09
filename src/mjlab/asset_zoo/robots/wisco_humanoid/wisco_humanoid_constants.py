"""Wisco Humanoid robot constants.

Everything the simulation needs to know about the hardware lives here. The MJCF is
produced by the mechanical team; this file only layers simulation configuration on top of
it.

Five things, in the order they appear below:

  Spec           the MuJoCo model, loaded from the MJCF
  Actuators      one group per joint family: stiffness, damping, effort, armature
  Initial state  where the robot spawns: base height and default joint pose
  Collisions     which geoms touch, and with what contact properties
  Action scale   how much one unit of policy output moves each joint

A complete worked example, including two-stage gearbox inertia, is
``asset_zoo/robots/unitree_g1/g1_constants.py``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# Spec.
#
# The MJCF comes from the mechanical team. Load it as-is: per-robot simulation settings
# (actuators, collisions, initial state) belong in the configs below, not in the XML.
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

  The existence check is lazy on purpose: it runs when the model is first built, not at
  import time, so the task registry and ``list-envs`` work before the model exists.
  """
  if not WISCO_HUMANOID_XML.exists():
    raise FileNotFoundError(
      f"Robot MJCF not found at {WISCO_HUMANOID_XML}.\n"
      "Add the model to "
      "src/mjlab/asset_zoo/robots/wisco_humanoid/xmls/wisco_humanoid.xml and its "
      "meshes under xmls/assets/. See the package README for the naming conventions."
    )
  return mujoco.MjSpec.from_file(str(WISCO_HUMANOID_XML))


##
# Actuators.
#
# One BuiltinPositionActuatorCfg per joint family (hip pitch, hip roll, hip yaw, knee,
# ankle pitch/roll). Values come from the motor datasheets:
#
#   armature     = J_rotor * gear_ratio^2          reflected inertia at the joint
#   stiffness    = armature * (2*pi*10)^2          10 Hz bandwidth
#   damping      = 2 * zeta * armature * omega     capped at the motor's maximum
#   effort_limit = peak torque
#
# TODO(wisco-humanoid): define the groups once the motors are chosen. Shape:
#
#   WISCO_HUMANOID_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
#     target_names_expr=(".*_hip_pitch_joint",),
#     stiffness=...,
#     damping=...,
#     effort_limit=...,
#     armature=...,
#   )
##


##
# Initial state.
#
# TODO(wisco-humanoid): set the standing height and a stable default pose. ``joint_pos``
# maps regex to angle, so ".*" applies to every joint.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),  # Standing height in metres.
  joint_pos={".*": 0.0},  # Default joint pose.
  joint_vel={".*": 0.0},
)

##
# Collisions.
#
# TODO(wisco-humanoid): decide the contact policy. Start with the feet alone, then enable
# self-collision once the model is stable. Shape:
#
#   FEET_ONLY_COLLISION = CollisionCfg(
#     geom_names_expr=(r"^(left|right)_foot_collision$",),
#     contype=0,
#     conaffinity=1,
#     condim=3,      # 3 gives friction; use 1 for normal-only self-collision.
#     priority=1,
#     friction=(0.6,),
#   )
##


##
# Articulation.
#
# Assemble the actuator groups and point the entity at this file's spec, initial state and
# collision policy.
##

WISCO_HUMANOID_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    # TODO(wisco-humanoid): one entry per group, e.g.
    # WISCO_HUMANOID_ACTUATOR_HIP_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_wisco_humanoid_robot_cfg() -> EntityCfg:
  """Get a fresh robot configuration instance.

  Returns a new EntityCfg each call to avoid mutation issues when the config is shared
  across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(),  # TODO(wisco-humanoid): add the chosen CollisionCfg.
    spec_fn=get_spec,
    articulation=WISCO_HUMANOID_ARTICULATION,
  )


##
# Action scale.
#
# Gives the policy's action a physical dimension:
#
#   target_joint_pos = default_joint_pos + action * action_scale
#
# A larger scale means one unit of policy output moves the joint further. The usual choice
# is ``action_scale = k * effort_limit / stiffness`` with ``k`` around 0.25-0.3, so a full
# action asks for roughly that fraction of the joint's torque budget.
#
# TODO(wisco-humanoid): derive one entry per joint from the actuator groups above.
##

WISCO_HUMANOID_ACTION_SCALE: dict[str, float] = {}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_wisco_humanoid_robot_cfg())

  viewer.launch(robot.spec.compile())
