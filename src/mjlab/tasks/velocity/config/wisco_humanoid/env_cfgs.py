"""Wisco Humanoid velocity environment configurations.

Scaffold task configs for a robot whose model is not defined yet. The tuning values are
carried over from the Asimov reference and from the Unitree G1 config; they are
placeholders, not measurements.
"""

from mjlab.asset_zoo.robots import (
  WISCO_HUMANOID_ACTION_SCALE,
  WISCO_HUMANOID_WALKING_REFERENCE,
  get_wisco_humanoid_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

##
# Robot-specific names. Update these when the MJCF naming convention changes.
##

BASE_BODY = "pelvis_link"
FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = (
  "left_ankle_roll_link_collision",
  "right_ankle_roll_link_collision",
)
FOOT_BODY_PATTERN = r"^(left_ankle_roll_link|right_ankle_roll_link)$"

# Gait cadence used by the phase clock, the alternating-feet reward and the imitation
# reward. Keep them in sync, otherwise the phase cues contradict each other.
GAIT_FREQUENCY = 1.25  # Hz.


def wisco_humanoid_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  # TODO(wisco-humanoid): starting point copied from the Unitree G1 config. Re-tune
  # once the real model exists.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_wisco_humanoid_robot_cfg()}

  # Point the raycast sensor at the robot's base body.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY

  # Wire the foot height scan to the per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=FOOT_BODY_PATTERN,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = WISCO_HUMANOID_ACTION_SCALE

  cfg.viewer.body_name = BASE_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8

  # TODO(wisco-humanoid): conservative ranges for a narrow-stance biped. Widen them
  # once the robot's stability envelope is known.
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  # Drop base linear velocity: a real robot needs a state estimator to observe it.
  # Remove these two lines if the robot carries one.
  del cfg.observations["actor"].terms["base_lin_vel"]
  del cfg.observations["critic"].terms["base_lin_vel"]

  # Phase clock for phase-aware locomotion.
  gait_clock_cfg = ObservationTermCfg(
    func=mdp.gait_clock,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "gait_frequency": GAIT_FREQUENCY,
    },
  )
  cfg.observations["actor"].terms["gait_clock"] = gait_clock_cfg
  cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)

  # TODO(wisco-humanoid): placeholder pose tolerances. std values are per joint
  # family: larger means more deviation is allowed.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.2,
    r".*ankle_roll.*": 0.12,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.8,
    r".*hip_roll.*": 0.35,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.8,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["air_time"].weight = 0.5

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Alternating foot contacts for a bipedal gait.
  cfg.rewards["alternating_feet"] = RewardTermCfg(
    func=mdp.alternating_feet_contact,
    weight=0.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
      "gait_frequency": GAIT_FREQUENCY,
    },
  )

  # Imitation reward, only enabled once a reference trajectory has been added to
  # assets/reference/walking_reference.csv.
  if WISCO_HUMANOID_WALKING_REFERENCE.exists():
    cfg.rewards["imitation"] = RewardTermCfg(
      func=mdp.imitation_joint_pos,
      weight=1.5,
      params={
        "data_path": str(WISCO_HUMANOID_WALKING_REFERENCE),
        "gait_frequency": GAIT_FREQUENCY,
        "std": 0.5,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )

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

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

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

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

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
