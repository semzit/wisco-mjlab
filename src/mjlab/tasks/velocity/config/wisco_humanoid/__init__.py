"""Wisco Humanoid velocity task configurations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  wisco_humanoid_balance_env_cfg,
  wisco_humanoid_flat_env_cfg,
  wisco_humanoid_rough_env_cfg,
)
from .rl_cfg import wisco_humanoid_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Wisco-Humanoid",
  env_cfg=wisco_humanoid_rough_env_cfg(),
  play_env_cfg=wisco_humanoid_rough_env_cfg(play=True),
  rl_cfg=wisco_humanoid_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Wisco-Humanoid",
  env_cfg=wisco_humanoid_flat_env_cfg(),
  play_env_cfg=wisco_humanoid_flat_env_cfg(play=True),
  rl_cfg=wisco_humanoid_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Balance-Wisco-Humanoid",
  env_cfg=wisco_humanoid_balance_env_cfg(),
  play_env_cfg=wisco_humanoid_balance_env_cfg(play=True),
  rl_cfg=wisco_humanoid_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
