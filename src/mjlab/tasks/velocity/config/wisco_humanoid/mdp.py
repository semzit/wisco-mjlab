"""Robot-local MDP terms for the Wisco Humanoid tasks.

These live next to the robot config instead of in ``mjlab.tasks.velocity.mdp`` so the
scaffold stays self-contained: upstream mjlab source is left untouched, which keeps
future merges conflict-free. Promote a term into the shared ``mdp`` package once it has
proven useful across several robots.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class gait_clock:
  """Phase clock observation for cyclic gaits.

  Returns ``[cos(2*pi*phase), sin(2*pi*phase)]``. The phase only advances while the
  robot is commanded to move and is re-randomized on episode reset, so the policy sees
  a phase that is consistent with the motion it is being asked to produce.

  Configure with ``gait_frequency`` (Hz), ``command_name`` and ``command_threshold``.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.gait_phase = torch.rand(env.num_envs, device=env.device, dtype=torch.float32)
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.gait_phase[env_ids] = torch.rand(
      self.gait_phase[env_ids].shape,
      device=self.gait_phase.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    gait_frequency: float = 1.5,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = ((linear_norm + angular_norm) > command_threshold).float()

    self.gait_phase = torch.fmod(
      self.gait_phase + self.step_dt * gait_frequency * active, 1.0
    )

    phase_2pi = 2.0 * torch.pi * self.gait_phase
    return torch.stack([torch.cos(phase_2pi), torch.sin(phase_2pi)], dim=1)


class alternating_feet_contact:
  """Reward feet that are airborne during their scheduled swing phase.

  Feet are assigned equal slices of the gait cycle in sensor order, so with two feet
  the first swings for the first half of the cycle and the second for the second half.
  The phase only advances while the robot is commanded to move, which keeps a standing
  robot from being rewarded for lifting its feet.

  Configure with ``sensor_name``, ``command_name``, ``command_threshold`` and
  ``gait_frequency`` (Hz).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.gait_phase = torch.rand(env.num_envs, device=env.device, dtype=torch.float32)
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.gait_phase[env_ids] = torch.rand(
      self.gait_phase[env_ids].shape,
      device=self.gait_phase.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    gait_frequency: float = 1.5,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    assert contact_sensor.data.found is not None

    self.gait_phase = torch.fmod(self.gait_phase + self.step_dt * gait_frequency, 1.0)

    in_contact = contact_sensor.data.found > 0  # [B, F]
    num_feet = in_contact.shape[1]
    phase_index = torch.clamp((self.gait_phase * num_feet).long(), max=num_feet - 1)
    foot_index = torch.arange(num_feet, device=env.device).unsqueeze(0)
    should_swing = foot_index == phase_index.unsqueeze(1)

    reward = (should_swing & ~in_contact).float().sum(dim=1)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = ((linear_norm + angular_norm) > command_threshold).float()
    return reward * active


class imitation_joint_pos:
  """Reward tracking of a cyclic reference joint trajectory.

  The reference CSV must have a header row and one column per actuated joint, in robot
  joint order. Pass ``joint_columns`` if the joint columns are not the leading ones.
  The gait phase advances proportionally to the commanded speed, and the reward is only
  applied while the robot is commanded to move.

  Configure with ``data_path``, ``gait_frequency`` (Hz), ``std``, ``command_name`` and
  ``command_threshold``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.step_dt = env.step_dt
    self.gait_phase = torch.rand(env.num_envs, device=env.device, dtype=torch.float32)

    data_path = cfg.params.get("data_path")
    if data_path is None or not os.path.exists(data_path):
      print(f"[WARNING] Imitation data not found at {data_path}, reward disabled.")
      self.enabled = False
      self.ref_joint_pos = None
      return

    data = np.atleast_2d(np.loadtxt(data_path, delimiter=",", skiprows=1))
    joint_columns = cfg.params.get("joint_columns")
    if joint_columns is None:
      num_joints = len(cfg.params["asset_cfg"].joint_ids)
      joint_columns = list(range(num_joints))
    self.enabled = True
    self.ref_joint_pos = torch.tensor(
      data[:, joint_columns], device=env.device, dtype=torch.float32
    )
    self.num_frames = self.ref_joint_pos.shape[0]
    print(f"[INFO] Loaded imitation data: {self.num_frames} frames.")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.gait_phase[env_ids] = torch.rand(
      self.gait_phase[env_ids].shape,
      device=self.gait_phase.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    data_path: str,
    gait_frequency: float,
    std: float,
    command_name: str,
    joint_columns: list[int] | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del data_path, joint_columns  # Used during initialization.

    if not self.enabled:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed
    active = (total_speed > command_threshold).float()

    # Advance the phase faster when the robot is asked to move faster. The reference
    # cadence is expressed at a nominal 0.5 m/s command.
    phase_delta = self.step_dt * gait_frequency * (total_speed / 0.5).clamp(0.5, 2.0)
    self.gait_phase = torch.fmod(self.gait_phase + phase_delta, 1.0)

    frame_index = (self.gait_phase * self.num_frames).long() % self.num_frames
    assert self.ref_joint_pos is not None
    ref_pos = self.ref_joint_pos[frame_index]

    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.sum(torch.square(current_pos - ref_pos), dim=1)
    return torch.exp(-error_squared / (std**2)) * active
