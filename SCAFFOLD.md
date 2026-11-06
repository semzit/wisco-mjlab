# What this repository is

A short orientation for someone new to the project. Three things are worth separating:
mjlab itself, the Asimov fork that inspired this scaffold, and what was added here.

---

## 1. mjlab (upstream)

[mjlab](https://github.com/mujocolab/mjlab) is a reinforcement-learning framework for
legged robots. It keeps the API of NVIDIA's
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) — managers for observations, rewards,
commands, events, terminations, curriculum — but swaps the physics backend for
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). The result is a much
lighter stack that runs on a single GPU and trains thousands of environments in parallel.

The pieces you interact with most:

| Concept | Where it lives | What it does |
| --- | --- | --- |
| Robot | `src/mjlab/asset_zoo/robots/<robot>/` | MJCF path, actuators, PD gains, keyframes, collision policy |
| Task | `src/mjlab/tasks/velocity/config/<robot>/` | Observation/reward/command config for one robot, registered under a task ID |
| Task registry | `mjlab.tasks.registry` | Maps a task ID like `Mjlab-Velocity-Flat-Unitree-G1` to a config |
| MDP terms | `src/mjlab/tasks/velocity/mdp/` | Reusable reward and observation functions |
| RL | `src/mjlab/rl/` | PPO runner, ONNX export, ONNX metadata |

Typical workflow: `uv run train <TASK>` → checkpoints in Weights & Biases → `uv run play
<TASK>` to watch the policy → export ONNX → deploy. Training needs an NVIDIA GPU.

---

## 2. The Asimov fork (Menlo Research)

[`menloresearch/asimov-mjlab`](https://github.com/menloresearch/asimov-mjlab) from
[Menlo Research](https://menlo.ai) is a fork of mjlab for the **Asimov** robot: a 12-DOF biped, six joints per leg (hip pitch/roll/yaw,
knee, ankle pitch/roll), with a canted hip-pitch axis, a narrow 11 cm stance and limited
ankle range.

On top of upstream mjlab it added:

- **A robot package** — `asset_zoo/robots/asimov/` with the MJCF, meshes, actuator
  groups, and physics-based PD gains (`KP = J_reflected · ω²`, `KD` capped at the motor
  maximum for sim2real).
- **Velocity task configs** — rough, flat and balance variants, with conservative
  velocity ranges and tighter pose tolerances around the ankles.
- **Three new MDP terms** — a gait phase clock observation, an alternating-foot-contact
  reward, and an imitation reward that tracks a cyclic walking reference CSV.
- **`deploy.py`** — a standalone sim2sim script that replays a trained policy in plain
  MuJoCo, with video, CSV logging and plots.

The fork's branch is based on an old mjlab commit. Since then upstream moved its API
(observation groups renamed `policy` → `actor`, `joint_names_expr` →
`target_names_expr`, raycast sensors rewired around `ObjRef`, the RL config split into
actor/critic configs), so the Asimov code does **not** run against current mjlab without
translation.

---

## 3. This repository: the Wisco Humanoid scaffold

This repo is **current mjlab `main`** plus a generalized version of the Asimov work,
prepared for a robot whose model does not exist yet. The Asimov fork is attached as a
read-only git remote (`asimov`) for reference.

What changed relative to Asimov:

| Asimov | Here |
| --- | --- |
| `asimov` package and task IDs | `wisco_humanoid` package, `Mjlab-*-Wisco-Humanoid` task IDs |
| Old mjlab API | Current API (translated, not redesigned) |
| Real MJCF, meshes, motor docs, walking CSV | Empty `xmls/` and `xmls/assets/` stubs |
| Hardware numbers inline | All collected in `wisco_humanoid_constants.py`, marked `TODO` |
| `deploy.py` with a 47-dim obs built by hand | `sim2sim` entry point that rebuilds the observation from ONNX metadata |
| `gait_clock` / alternating feet hardcoded to 12 joints, 2 feet | Foot count derived from the contact sensor |
| New MDP terms appended to `observations.py` / `rewards.py` | Added as their own module, `tasks/velocity/mdp/gait.py`, so upstream terms stay untouched |

The result is a template: it imports, registers its tasks and lists them, but any attempt
to actually build the robot fails with a message telling you which file is missing. That
is intentional — it keeps the structure honest until real data exists.

### File map

```
src/mjlab/asset_zoo/robots/wisco_humanoid/
  wisco_humanoid_constants.py   # the only file with hardware numbers
  __init__.py                   # exports
  xmls/.gitkeep                 # put wisco_humanoid.xml here
  xmls/assets/.gitkeep          # put STL/OBJ files here (meshdir="assets")
  README.md                     # naming conventions + fill-in checklist

src/mjlab/tasks/velocity/config/wisco_humanoid/
  __init__.py                   # registers rough / flat / balance tasks
  env_cfgs.py                   # observation, sensor, reward, command wiring
  rl_cfg.py                     # PPO network and training hyperparameters

src/mjlab/tasks/velocity/mdp/
  gait.py                       # gait_clock, alternating_feet_contact

src/mjlab/scripts/sim2sim.py    # uv run sim2sim --policy policy.onnx
```

### How the pieces connect

```
MJCF + meshes
   │
   ▼
wisco_humanoid_constants.py ──► EntityCfg (actuators, gains, collisions, keyframe)
   │
   ▼
env_cfgs.py ──► ManagerBasedRlEnvCfg (observations, rewards, commands, sensors)
   │
   ▼
train ──► play ──► ONNX export (gains + defaults + observation layout in metadata)
   │
   ▼
sim2sim ──► plain MuJoCo rollout with video, CSV and plots
```

### What is still a placeholder

- Every armature, stiffness, damping, effort limit and standing height.
- The MJCF, meshes and reference trajectory.
- Velocity ranges, pose standard deviations and reward weights — carried over from
  Asimov and G1, not measured on this robot.

The package `README.md` lists exactly what to replace, in order. Start there once the
robot model exists.

### Running it

```bash
uv sync                              # core install
uv run list-envs                     # the three Wisco tasks appear
uv run train Mjlab-Velocity-Flat-Wisco-Humanoid --env.scene.num-envs 4096
uv sync --extra sim2sim              # optional: onnx + onnxruntime + opencv + matplotlib
uv run sim2sim --policy logs/policy.onnx --cmd_vx 0.5
```

Training and play will fail until `xmls/wisco_humanoid.xml` exists. `list-envs`,
imports, linting and tests work now.

---

## Known hacks and shortcuts

Things that are deliberately rough, so nobody mistakes them for finished work.

### Hardware numbers are borrowed, not measured

- Every armature, stiffness, damping, effort limit and standing height comes from the
  Asimov datasheets. They are there so the derived formulas and code paths are exercised;
  none of them describe this robot.
- `KNEES_BENT_KEYFRAME` is dead config kept on purpose as a worked example of mirrored
  left/right joint axes.

### Names are hardcoded

- `pelvis_link`, `left_foot` / `right_foot`, the foot collision geoms and the IMU sensor
  names are literals in `wisco_humanoid_constants.py` and `env_cfgs.py`. Nothing
  discovers them from the model. Renaming your links means editing those constants.
- Task IDs are hyphenated (`Wisco-Humanoid`) while the Python package is underscored
  (`wisco_humanoid`). That is unavoidable, but the two have to stay in sync by hand.

### Tuning is inherited, not derived

- Velocity ranges (`0.8 / 0.6 / 0.6`), the pose standard deviations, `air_time = 0.5`,
  `body_ang_vel = -0.08` and the self-collision weight are Asimov values tuned for a
  narrow-stance robot.
- The simulation limits (`ccd_iterations = 500`, `contact_sensor_maxmatch = 500`,
  `nconmax = 70`, flat `njmax = 300`) are Unitree G1 values.

### Small behavioural drift from the Asimov original

- `gait_clock` now re-randomizes through the manager's `reset()` hook instead of inside
  `__call__`. Same effect, not byte-identical.
- `alternating_feet_contact` splits the cycle into equal slices per foot rather than
  hardcoding left/right halves. Identical for two feet; for more feet the behaviour is a
  design choice, not a port.
- The original reset the phase from `termination_manager.terminated` on every step; this
  version relies on the manager calling `reset(env_ids)`.

### `sim2sim` is metadata-driven, but not metadata-complete

- It ignores `observation_terms_history_length`, `observation_terms_flatten_history_dim`
  and the `delay_*` metadata. A policy exported with observation history or latency will
  build the wrong observation size and fail the explicit size check rather than silently
  running.
- It assumes `joint_pos`, `joint_vel` and `actions` are `num_actions` wide and that
  `command` is 3-wide. Any other term raises `NotImplementedError`.
- `joint_pos` is always treated as relative to the default pose (the `biased=True` case).
- Angular velocity prefers an `imu_ang_vel` sensor and falls back to
  `mj_objectVelocity`; small numerical differences from the training sensor are possible.
- Armature and effort lookup takes the first regex match, and silently falls back to
  `100 Nm` / `0 armature` when nothing matches.
- Flat terrain only. `height_scan` and foot contact terms are not reconstructed.
- Video and plots need the `sim2sim` extra; without OpenCV or matplotlib the run degrades
  to a warning instead of failing.

### Validation status

- Verified: package imports, `uv run list-envs` (all three tasks), 18 config and velocity
  tests, `ruff format` / `ruff check`, `py_compile`, `ty check`.
- Not verified: any rollout, training or play. `xmls/wisco_humanoid.xml` does not exist, so
  the first real test of the dynamics path is still ahead.
- `uv.lock` was regenerated to add the `sim2sim` optional extra.
- The new MDP terms are a separate module (`tasks/velocity/mdp/gait.py`), so upstream term
  files are untouched. Two upstream files are edited: a one-line import in
  `tasks/velocity/mdp/__init__.py` and the 9-line robot export in
  `src/mjlab/asset_zoo/robots/__init__.py` (the same file upstream uses for G1/Go1/YAM).
- `README.md` and `docs/source/changelog.rst` were edited on top of upstream, so they will
  conflict if upstream's copies are merged later.
