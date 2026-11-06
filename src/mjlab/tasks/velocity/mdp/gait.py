"""Locomotion MDP terms shared by the velocity task.

Holds the phase clock observation and the alternating-feet reward used by the Wisco
Humanoid tasks. They live in ``mjlab.tasks.velocity.mdp`` so any velocity task can enable
them through its env config; the robot config only decides whether to use them and how to
parameterize them.

Neither term is part of upstream mjlab: both are additions carried in this fork and only
run when a robot config opts in. They are ported from the Asimov fork of mjlab by Menlo
Research (https://github.com/menloresearch/asimov-mjlab), where the gait clock follows
Booster's sin/cos phase convention.

Worth knowing before reusing them: Menlo's write-up of the Asimov walking policy takes the
opposite position, listing "No Gait Clock" and arguing the policy should discover its own
gait frequency from the command and body dynamics
(https://menlo.ai/research/teaching-a-humanoid-to-walk). The implementation still ships in
the Asimov fork, and the Wisco config enables it.

Local changes: phase reset goes through the manager ``reset()`` hook instead of reading the
termination buffer, the foot count is taken from the contact sensor, and the frequency and
command threshold are parameters rather than constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


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
