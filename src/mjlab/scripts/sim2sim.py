"""Sim2sim deployment of a mjlab-trained policy in plain MuJoCo.

Loads an ONNX policy exported by ``mjlab``, reconstructs the actor observation from the
policy metadata and runs it against a robot MJCF using position actuators with the same
gains the policy was trained with.

The observation layout is not hardcoded: it is read from the ONNX metadata written by
``mjlab.rl.exporter_utils`` (``observation_names``, ``observation_terms_scale``,
``observation_terms_clip``). The script therefore keeps working when the observation
set changes, as long as the policy was exported from mjlab.

Only flat-terrain policies are supported. Terrain scans are critic-only in mjlab, so a
flat actor export contains every observation this script can reconstruct.

Usage::

  uv run sim2sim --policy logs/policy.onnx --cmd_vx 0.5

Requires the optional dependencies: ``uv sync --extra sim2sim``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

try:
  import onnx
  import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - optional dependency.
  raise ImportError(
    "sim2sim requires the optional dependencies. Run: uv sync --extra sim2sim"
  ) from exc

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg

##
# Robot configuration.
#
# Kept in one place so a new robot only requires edits here. Armature and effort limits
# are pulled from the robot constants so sim2sim cannot drift from training.
##

ROBOT_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "wisco_humanoid"
  / "xmls"
  / "wisco_humanoid.xml"
)
BASE_BODY = "pelvis_link"
START_HEIGHT = 0.75
CAMERA: str | None = None  # Set to a camera name defined in the MJCF, if any.

# Physics and control, matching the velocity task defaults.
SIMULATION_DT = 0.005
CONTROL_DECIMATION = 4
VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


@dataclass
class RobotSim2SimCfg:
  """Robot-specific settings for sim2sim deployment."""

  xml_path: Path = ROBOT_XML
  base_body: str = BASE_BODY
  start_height: float = START_HEIGHT
  camera: str | None = CAMERA
  # Joint-regex overrides; the robot constants are used for anything not listed here.
  armature: dict[str, float] = field(default_factory=dict)
  effort_limit: dict[str, float] = field(default_factory=dict)


def _actuator_maps() -> tuple[dict[str, float], dict[str, float]]:
  """Build joint-regex -> armature / effort-limit maps from the robot constants."""
  from mjlab.asset_zoo.robots.wisco_humanoid.wisco_humanoid_constants import (
    WISCO_HUMANOID_ARTICULATION,
  )

  arms: dict[str, float] = {}
  efforts: dict[str, float] = {}
  for actuator in WISCO_HUMANOID_ARTICULATION.actuators:
    if not isinstance(actuator, BuiltinPositionActuatorCfg):
      continue
    assert actuator.armature is not None
    assert actuator.effort_limit is not None
    for pattern in actuator.target_names_expr:
      arms[pattern] = actuator.armature
      efforts[pattern] = actuator.effort_limit
  return arms, efforts


def _lookup(mapping: dict[str, float], joint_name: str, default: float) -> float:
  """Resolve the first regex in *mapping* that matches *joint_name*."""
  for pattern, value in mapping.items():
    if re.search(pattern, joint_name):
      return value
  return default


##
# ONNX metadata.
##


def load_metadata(onnx_path: str) -> dict[str, str]:
  model = onnx.load(onnx_path)
  return {prop.key: prop.value for prop in model.metadata_props}


def parse_value(raw: str) -> float | list:
  """Parse a metadata field written by ``list_to_csv_str``.

  Lists are comma separated; nested lists use ``;`` for the inner delimiter.
  """
  parsed: list = []
  for entry in raw.split(","):
    if ";" in entry:
      parsed.append([float(item) for item in entry.split(";")])
    else:
      try:
        parsed.append(float(entry))
      except ValueError:
        parsed.append(entry)
  return parsed[0] if len(parsed) == 1 else parsed


##
# Actuators.
##


def build_position_actuators(
  spec: mujoco.MjSpec,
  joint_names: list[str],
  kps: np.ndarray,
  kds: np.ndarray,
  effort_limits: np.ndarray,
  armatures: np.ndarray,
) -> list[str]:
  """Turn the model's actuators into position servos matching the training gains."""
  actuator_names: list[str] = []
  for index, joint_name in enumerate(joint_names):
    actuator = next(
      (
        candidate
        for candidate in spec.actuators
        if candidate.target is not None
        and candidate.target.split("/")[-1] == joint_name
      ),
      None,
    )
    if actuator is None:
      actuator = spec.add_actuator()
      actuator.target = joint_name
      actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    if not actuator.name:
      # Unnamed actuators cannot be looked up after compilation.
      actuator.name = f"{joint_name}_pos"

    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = kps[index]
    actuator.biasprm[0] = 0.0
    actuator.biasprm[1] = -kps[index]
    actuator.biasprm[2] = -kds[index]
    actuator.forcerange[:] = (-effort_limits[index], effort_limits[index])

    joint = next((item for item in spec.joints if item.name == joint_name), None)
    if joint is not None:
      joint.armature = float(armatures[index])

    actuator_names.append(actuator.name)
  return actuator_names


##
# Observations.
##


def projected_gravity(quat: np.ndarray) -> np.ndarray:
  """Gravity in the body frame, matching mjlab's ``quat_apply_inverse``."""
  w, x, y, z = quat
  return np.array(
    [
      2.0 * (w * y - x * z),
      -2.0 * (w * x + y * z),
      -1.0 + 2.0 * (x * x + y * y),
    ],
    dtype=np.float32,
  )


def body_velocity(
  model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
  """Linear and angular velocity of a body, expressed in its own frame."""
  result = np.zeros(6, dtype=np.float64)
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, result, 1)
  return result[3:].copy(), result[:3].copy()


def sensor_value(
  data: mujoco.MjData, model: mujoco.MjModel, name: str
) -> np.ndarray | None:
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  if sensor_id < 0:
    return None
  address = model.sensor_adr[sensor_id]
  return data.sensordata[address : address + model.sensor_dim[sensor_id]].copy()


_TERM_SIZES = {
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "command": 3,
  "gait_clock": 2,
}


class ObservationBuilder:
  """Assemble the actor observation from the names recorded in the ONNX metadata."""

  def __init__(
    self,
    terms: list[str],
    num_actions: int,
    default_joint_pos: np.ndarray,
    scales: list,
    clips: list,
    gait_frequency: float,
    command_threshold: float = 0.1,
  ) -> None:
    self.terms = terms
    self.num_actions = num_actions
    self.default_joint_pos = default_joint_pos
    self.scales = scales if len(scales) == len(terms) else [1.0] * len(terms)
    self.clips = clips if len(clips) == len(terms) else [None] * len(terms)
    self.gait_frequency = gait_frequency
    self.command_threshold = command_threshold
    self.gait_phase = 0.0
    self._action = np.zeros(num_actions, dtype=np.float32)

  def term_size(self, term: str) -> int:
    if term in ("joint_pos", "joint_vel", "actions"):
      return self.num_actions
    if term not in _TERM_SIZES:
      raise NotImplementedError(
        f"Observation term '{term}' is not reconstructable in sim2sim. Export a "
        "flat-terrain velocity policy, or extend ObservationBuilder."
      )
    return _TERM_SIZES[term]

  def set_action(self, action: np.ndarray) -> None:
    self._action = action.astype(np.float32)

  def build(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_ids: np.ndarray,
    joint_dof_ids: np.ndarray,
    base_body_id: int,
    command: np.ndarray,
    control_dt: float,
  ) -> np.ndarray:
    joint_pos = data.qpos[joint_qpos_ids]
    joint_vel = data.qvel[joint_dof_ids]
    lin_vel, ang_vel = body_velocity(model, data, base_body_id)
    gyro = sensor_value(data, model, "imu_ang_vel")
    if gyro is not None:
      ang_vel = gyro
    quat = data.xquat[base_body_id]

    command_magnitude = float(np.linalg.norm(command[:2]) + abs(command[2]))
    if command_magnitude > self.command_threshold:
      self.gait_phase = (self.gait_phase + control_dt * self.gait_frequency) % 1.0
    phase_2pi = 2.0 * np.pi * self.gait_phase

    values = {
      "base_lin_vel": lin_vel,
      "base_ang_vel": ang_vel,
      "projected_gravity": projected_gravity(quat),
      "joint_pos": joint_pos - self.default_joint_pos,
      "joint_vel": joint_vel,
      "actions": self._action,
      "command": command,
      "gait_clock": np.array([np.cos(phase_2pi), np.sin(phase_2pi)]),
    }

    chunks = []
    for index, term in enumerate(self.terms):
      chunk = np.asarray(values[term], dtype=np.float32)
      scale = self.scales[index]
      if isinstance(scale, list):
        chunk = chunk * np.asarray(scale, dtype=np.float32)
      else:
        chunk = chunk * float(scale)
      clip = self.clips[index]
      if isinstance(clip, list) and len(clip) == 2:
        low, high = clip
        if np.isfinite(low) or np.isfinite(high):
          chunk = np.clip(chunk, low, high)
      chunks.append(chunk)
    return np.concatenate(chunks)


##
# Logging and plots.
##


def write_plots(output_prefix: str, csv_path: Path, start_height: float) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    print(f"[WARNING] matplotlib is not installed; skipping plots. {exc}")
    return

  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  time = data["time"]
  joint_columns = [name for name in data.dtype.names or () if name.startswith("tgt_")]
  short_names = [name.removeprefix("tgt_") for name in joint_columns]

  columns = 3
  rows = max(1, int(np.ceil(len(short_names) / columns)))
  fig, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), squeeze=False)
  fig.suptitle("Target vs Actual Joint Positions", fontweight="bold")
  for axis, short in zip(axes.flatten(), short_names, strict=False):
    axis.plot(time, data[f"tgt_{short}"], label="target")
    axis.plot(time, data[f"act_{short}"], label="actual")
    axis.set_title(short, fontsize=10)
    axis.grid(alpha=0.3)
  for axis in axes.flatten()[len(short_names) :]:
    axis.axis("off")
  if short_names:
    axes.flatten()[0].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(f"{output_prefix}_joints.png", dpi=150, bbox_inches="tight")
  plt.close(fig)

  fig, (ax_height, ax_tilt) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  ax_height.plot(time, data["base_z"], label="base height")
  ax_height.axhline(start_height, color="r", linestyle="--", label="start height")
  ax_height.set_ylabel("Height (m)")
  ax_height.grid(alpha=0.3)
  ax_height.legend()

  ax_tilt.plot(time, data["grav_z"], label="gravity z (body)")
  ax_tilt.axhline(-1.0, color="r", linestyle="--", label="upright")
  ax_tilt.set_xlabel("Time (s)")
  ax_tilt.set_ylabel("Gravity z")
  ax_tilt.grid(alpha=0.3)
  ax_tilt.legend()
  fig.tight_layout()
  fig.savefig(f"{output_prefix}_stability.png", dpi=150, bbox_inches="tight")
  plt.close(fig)

  print(f"Saved plots to {output_prefix}_*.png")


##
# Main.
##


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy", required=True, help="Path to the exported ONNX policy."
  )
  parser.add_argument("--xml", type=Path, default=None, help="Override the robot MJCF.")
  parser.add_argument("--base-body", default=None, help="Override the base body name.")
  parser.add_argument("--camera", default=None, help="Camera name in the MJCF.")
  parser.add_argument("--cmd_vx", type=float, default=0.5)
  parser.add_argument("--cmd_vy", type=float, default=0.0)
  parser.add_argument("--cmd_wz", type=float, default=0.0)
  parser.add_argument("--duration", type=float, default=20.0)
  parser.add_argument("--gait-frequency", type=float, default=1.25)
  parser.add_argument("--push-force", type=float, default=0.0, help="Push force [N].")
  parser.add_argument("--push-time", type=float, default=2.0)
  parser.add_argument("--push-duration", type=float, default=0.1)
  parser.add_argument("--push-dir", default="x", choices=("x", "-x", "y", "-y"))
  parser.add_argument("--no-video", action="store_true")
  parser.add_argument("--output-dir", type=Path, default=None)
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  robot = RobotSim2SimCfg()
  if args.xml is not None:
    robot.xml_path = args.xml
  if args.base_body is not None:
    robot.base_body = args.base_body
  if args.camera is not None:
    robot.camera = args.camera

  armature_map, effort_map = _actuator_maps()
  armature_map |= robot.armature
  effort_map |= robot.effort_limit

  metadata = load_metadata(args.policy)
  joint_names = metadata["joint_names"].split(",")
  kps = np.atleast_1d(
    np.asarray(parse_value(metadata["joint_stiffness"]), dtype=np.float32)
  )
  kds = np.atleast_1d(
    np.asarray(parse_value(metadata["joint_damping"]), dtype=np.float32)
  )
  default_joint_pos = np.atleast_1d(
    np.asarray(parse_value(metadata["default_joint_pos"]), dtype=np.float32)
  )
  action_scale = np.atleast_1d(
    np.asarray(parse_value(metadata["action_scale"]), dtype=np.float32)
  )
  observation_names = metadata["observation_names"].split(",")
  term_scales = parse_value(metadata.get("observation_terms_scale", "1.0"))
  term_clips = parse_value(metadata.get("observation_terms_clip", ""))
  if not isinstance(term_scales, list):
    term_scales = [term_scales] * len(observation_names)
  if not isinstance(term_clips, list):
    term_clips = [term_clips] * len(observation_names)

  num_actions = len(joint_names)
  if len(default_joint_pos) != num_actions:
    raise ValueError(
      f"Metadata lists {num_actions} joints but {len(default_joint_pos)} defaults."
    )

  effort_limits = np.array(
    [_lookup(effort_map, name, 100.0) for name in joint_names], dtype=np.float32
  )
  armatures = np.array(
    [_lookup(armature_map, name, 0.0) for name in joint_names], dtype=np.float32
  )

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  output_dir = args.output_dir or (Path.cwd() / "logs" / timestamp)
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / "sim2sim.csv"
  video_path = output_dir / "sim2sim.mp4"

  print("=" * 60)
  print("mjlab sim2sim")
  print("=" * 60)
  print(f"Policy:       {args.policy}")
  print(f"Robot model:  {robot.xml_path}")
  print(f"Joints:       {joint_names}")
  print(f"Observations: {observation_names}")

  spec = mujoco.MjSpec.from_file(str(robot.xml_path))
  actuator_names = build_position_actuators(
    spec, joint_names, kps, kds, effort_limits, armatures
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  model.opt.timestep = SIMULATION_DT

  base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.base_body)
  if base_body_id < 0:
    raise ValueError(f"Base body '{robot.base_body}' not found in the model.")

  joint_qpos_ids = np.array(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in joint_names
    ]
  )
  joint_dof_ids = np.array(
    [
      model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in joint_names
    ]
  )
  ctrl_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      for name in actuator_names
    ]
  )

  control_dt = SIMULATION_DT * CONTROL_DECIMATION
  builder = ObservationBuilder(
    terms=observation_names,
    num_actions=num_actions,
    default_joint_pos=default_joint_pos,
    scales=term_scales,
    clips=term_clips,
    gait_frequency=args.gait_frequency,
  )

  session = ort.InferenceSession(args.policy)
  input_name = session.get_inputs()[0].name
  expected_obs = int(np.prod(session.get_inputs()[0].shape[1:]))

  command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)

  # Place the robot at the start height using the free joint, if there is one.
  free_joint = next(
    (
      index
      for index in range(model.njnt)
      if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
    ),
    None,
  )
  if free_joint is not None:
    data.qpos[model.jnt_qposadr[free_joint] + 2] = robot.start_height
  data.qpos[joint_qpos_ids] = default_joint_pos
  mujoco.mj_forward(model, data)

  observation = builder.build(
    model, data, joint_qpos_ids, joint_dof_ids, base_body_id, command, control_dt
  )
  if observation.size != expected_obs:
    raise ValueError(
      f"Built a {observation.size}-dim observation but the policy expects "
      f"{expected_obs}. Check observation_names in the ONNX metadata."
    )

  cv2 = None
  video_writer = None
  renderer = None
  if not args.no_video:
    try:
      import importlib

      cv2 = importlib.import_module("cv2")
      renderer = mujoco.Renderer(model, VIDEO_HEIGHT, VIDEO_WIDTH)
      video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (VIDEO_WIDTH, VIDEO_HEIGHT),
      )
    except ImportError as exc:
      print(f"[WARNING] OpenCV is not installed; continuing without video. {exc}")

  csv_file = csv_path.open("w", newline="")
  writer = csv.writer(csv_file)
  short_names = [
    name.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
    for name in joint_names
  ]
  header = ["step", "time", "base_x", "base_y", "base_z", "grav_z"]
  for name in short_names:
    header += [f"tgt_{name}", f"act_{name}"]
  writer.writerow(header)

  action = np.zeros(num_actions, dtype=np.float32)
  target_joint_pos = default_joint_pos.copy()
  steps_per_frame = max(1, int(1.0 / (VIDEO_FPS * SIMULATION_DT)))
  total_steps = int(args.duration / SIMULATION_DT)
  push_start = int(args.push_time / SIMULATION_DT)
  push_end = int((args.push_time + args.push_duration) / SIMULATION_DT)
  push_axis = {"x": 0, "-x": 0, "y": 1, "-y": 1}[args.push_dir]
  push_sign = -1.0 if args.push_dir.startswith("-") else 1.0

  print(f"Running {args.duration:.1f}s of simulation...")
  for step in range(total_steps):
    data.ctrl[ctrl_ids] = target_joint_pos

    if args.push_force > 0 and push_start <= step < push_end:
      data.xfrc_applied[base_body_id, push_axis] = push_sign * args.push_force
    else:
      data.xfrc_applied[base_body_id, :] = 0.0

    mujoco.mj_step(model, data)

    if step % CONTROL_DECIMATION == 0:
      observation = builder.build(
        model, data, joint_qpos_ids, joint_dof_ids, base_body_id, command, control_dt
      )
      action = np.asarray(
        session.run(None, {input_name: observation[None, :]})[0]
      ).squeeze()
      builder.set_action(action)
      target_joint_pos = default_joint_pos + action * action_scale

      row = [
        step,
        f"{step * SIMULATION_DT:.3f}",
        f"{data.xpos[base_body_id, 0]:.4f}",
        f"{data.xpos[base_body_id, 1]:.4f}",
        f"{data.xpos[base_body_id, 2]:.4f}",
        f"{projected_gravity(data.xquat[base_body_id])[2]:.4f}",
      ]
      for _name, target, actual in zip(
        short_names, target_joint_pos, data.qpos[joint_qpos_ids], strict=False
      ):
        row += [f"{target:.4f}", f"{actual:.4f}"]
      writer.writerow(row)

    if (
      cv2 is not None
      and renderer is not None
      and video_writer is not None
      and step % steps_per_frame == 0
    ):
      renderer.update_scene(data, camera=robot.camera)
      frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
      label = (
        f"t={step * SIMULATION_DT:.2f}s  "
        f"cmd=({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f})"
      )
      cv2.putText(
        frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
      )
      video_writer.write(frame)

  csv_file.close()
  if video_writer is not None:
    video_writer.release()
  if renderer is not None:
    renderer.close()

  print(f"Saved rollout to {csv_path}")
  if video_writer is not None:
    print(f"Saved video to {video_path}")
  write_plots(str(output_dir / "sim2sim"), csv_path, robot.start_height)


if __name__ == "__main__":
  main()
