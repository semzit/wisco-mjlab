"""Wisco Humanoid velocity environment configurations.

Every task starts from mjlab's velocity task factory and only overrides what this robot
needs. The skeleton keeps the tasks registerable and loadable while leaving each
robot-specific decision as a TODO.

Fill in, in this order:

  1. Names        base body, foot sites, foot collision geoms
  2. Sensors      feet-ground contact, self-collision, foot-height scan
  3. Observations what the real robot can measure (start with base_lin_vel)
  4. Commands     velocity ranges this robot can actually reach
  5. Rewards      pose tolerances and penalty weights
  6. Events       friction and centre-of-mass randomization

Worked examples: ``config/g1/env_cfgs.py`` and ``config/go1/env_cfgs.py``.
"""

from mjlab.asset_zoo.robots import (
  WISCO_HUMANOID_ACTION_SCALE,
  get_wisco_humanoid_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

##
# Robot-specific names.
#
# TODO(wisco-humanoid): replace the placeholders with the names used by the MJCF.
##

BASE_BODY = ""  # e.g. "pelvis_link"
FOOT_SITES: tuple[str, ...] = ()  # e.g. ("left_foot", "right_foot")
FOOT_GEOMS: tuple[str, ...] = ()  # e.g. ("left_foot_collision", "right_foot_collision")
FOOT_BODY_PATTERN = ""  # e.g. r"^(left|right)_ankle_roll_link$"

# Cadence for the gait clock and alternating-feet reward, when enabled.
GAIT_FREQUENCY = 1.25  # Hz.


def wisco_humanoid_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_wisco_humanoid_robot_cfg()}

  # TODO(wisco-humanoid): simulation limits. Start from the G1 values and re-tune:
  #   cfg.sim.mujoco.ccd_iterations = 500
  #   cfg.sim.contact_sensor_maxmatch = 500
  #   cfg.sim.nconmax = 70

  # TODO(wisco-humanoid): sensors. Full pattern in config/g1/env_cfgs.py.
  #   feet_ground_contact = ContactSensorCfg(
  #     name="feet_ground_contact",
  #     primary=ContactMatch(mode="subtree", pattern=FOOT_BODY_PATTERN, entity="robot"),
  #     secondary=ContactMatch(mode="body", pattern="terrain"),
  #     fields=("found", "force"),
  #     reduce="netforce",
  #     num_slots=1,
  #     track_air_time=True,
  #   )
  #   self_collision = ContactSensorCfg(
  #     name="self_collision",
  #     primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
  #     secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
  #     fields=("found", "force"),
  #     reduce="none",
  #     num_slots=1,
  #     history_length=4,
  #   )
  #   cfg.scene.sensors = (cfg.scene.sensors or ()) + (
  #     feet_ground_contact,
  #     self_collision,
  #   )
  #
  # Point the base sensors at this robot:
  #   terrain_scan.frame.name = BASE_BODY
  #   foot_height_scan.frame = tuple(
  #     ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITES
  #   )
  #   foot_height_scan.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = WISCO_HUMANOID_ACTION_SCALE

  # TODO(wisco-humanoid): body the viewer follows.
  cfg.viewer.body_name = BASE_BODY

  # TODO(wisco-humanoid): velocity ranges this robot can reach. The factory default is
  # +-1 m/s and +-0.5 rad/s; narrow or widen from there.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.0  # Arrow height above the base, e.g. standing height.

  # TODO(wisco-humanoid): observations. The factory assumes a state estimator; a robot
  # without one should drop base_lin_vel:
  #   del cfg.observations["actor"].terms["base_lin_vel"]
  #   del cfg.observations["critic"].terms["base_lin_vel"]
  #
  # Optional phase clock (not part of upstream mjlab):
  #   gait_clock_cfg = ObservationTermCfg(
  #     func=mdp.gait_clock,
  #     params={
  #       "command_name": "twist",
  #       "command_threshold": 0.1,
  #       "gait_frequency": GAIT_FREQUENCY,
  #     },
  #   )
  #   cfg.observations["actor"].terms["gait_clock"] = gait_clock_cfg
  #   cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

  # TODO(wisco-humanoid): events. Friction and centre-of-mass randomization need the
  # foot geoms and the base body:
  #   cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  #   cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)

  # TODO(wisco-humanoid): rewards. Pose tolerances per joint family first, then the
  # body_ang_vel, angular_momentum and air_time weights:
  #   cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  #   cfg.rewards["pose"].params["std_walking"] = {r".*hip_pitch.*": 0.3, ...}
  #   cfg.rewards["pose"].params["std_running"] = {r".*hip_pitch.*": 0.5, ...}
  #   cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  #   cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  #   for name in ("foot_clearance", "foot_slip"):
  #     cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES
  #
  # Optional extra terms (not part of upstream mjlab):
  #   cfg.rewards["self_collisions"] = RewardTermCfg(
  #     func=mdp.self_collision_cost,
  #     weight=-1.0,
  #     params={"sensor_name": "self_collision", "force_threshold": 10.0},
  #   )
  #   cfg.rewards["alternating_feet"] = RewardTermCfg(
  #     func=mdp.alternating_feet_contact,
  #     weight=0.5,
  #     params={
  #       "sensor_name": "feet_ground_contact",
  #       "command_name": "twist",
  #       "command_threshold": 0.1,
  #       "gait_frequency": GAIT_FREQUENCY,
  #     },
  #   )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def wisco_humanoid_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat terrain velocity tracking configuration."""
  cfg = wisco_humanoid_rough_env_cfg(play=play)

  # TODO(wisco-humanoid): flat-terrain simulation limits, e.g. relax nconmax/njmax.

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan on a plane.
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  # TODO(wisco-humanoid): evaluation command ranges when play=True.

  return cfg


def wisco_humanoid_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the balance-only configuration with reactive stepping.

  The robot learns to stay upright in place and take corrective steps when disturbed.
  There is no velocity tracking.
  """
  cfg = wisco_humanoid_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.rewards["track_linear_velocity"].weight = 0.0
  cfg.rewards["track_angular_velocity"].weight = 0.0

  return cfg
