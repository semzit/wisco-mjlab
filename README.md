# Wisco Humanoid Locomotion

Fork of [mujocolab/mjlab](https://github.com/mujocolab/mjlab), scaffolded for the
**Wisco Humanoid**: a humanoid being built from the ground up at the University of
Wisconsin–Madison by [WiscoHumanoids](https://www.linkedin.com/company/wiscohumanoids).

Inspired by [asimov-mjlab](https://github.com/menloresearch/asimov-mjlab) by
[Menlo Research](https://menlo.ai) — their Asimov fork is the reference this scaffold
follows, from the robot package layout and gait-clock and imitation terms to the
original sim2sim workflow.

<p align="center">
  <img src="docs/static/wiscohumanoids_logo.jpg" alt="Wisco Humanoids" height="180"/>
</p>

> ⚠️ **Scaffold, not a working robot yet.** The task structure, gains and tooling are in
> place, but `xmls/wisco_humanoid.xml` and the meshes do not exist. Every hardware number
> is a placeholder marked `TODO(wisco-humanoid)`. See [SCAFFOLD.md](SCAFFOLD.md) for what
> changed and what is still fake.

---

## Demo

**Sim2Sim in MuJoCo** — velocity tracking with an exported ONNX policy:

_Recording to be added once a policy is trained._

To record one yourself:

```bash
uv sync --extra sim2sim
uv run sim2sim --policy path/to/policy.onnx --cmd_vx 0.5
```

---

## Robot Modeling

**12-DOF bipedal** (placeholder: 6 joints per leg) — `hip_pitch`, `hip_roll`, `hip_yaw`,
`knee`, `ankle_pitch`, `ankle_roll`.

Joint families are defined in
[`wisco_humanoid_constants.py`](src/mjlab/asset_zoo/robots/wisco_humanoid/wisco_humanoid_constants.py)
and matched by regex, so a robot with a different joint layout only needs edits to that
file and the name constants in
[`env_cfgs.py`](src/mjlab/tasks/velocity/config/wisco_humanoid/env_cfgs.py).

Key characteristics (TBD, to be measured on the real machine):

- **Stance width** — affects the lateral velocity limits.
- **Hip axis** — a canted pitch axis couples roll and pitch and needs looser pose tolerances.
- **Ankle range** — the tightest joint in the pose reward.
- **Mass and height** — used to set the standing keyframe and command ranges.

---

## Training

### Reward structure

| Reward | Description |
|--------|-------------|
| **Velocity tracking** | Track the commanded `(vx, vy, ωz)` |
| **Imitation** | Match a cyclic reference trajectory (optional, enabled when a reference CSV exists) |
| **Alternating feet** | Enforce a bipedal gait pattern |
| **Pose** | Speed-dependent joint posture (tighter when standing, looser when running) |
| **Air time** | Encourage dynamic stepping |
| **Foot clearance / swing height** | Lift the foot over terrain and land cleanly |
| **Foot slip** | Penalize sliding while in contact |
| **Soft landing** | Penalize hard impacts |
| **Self-collision** | Penalize inter-link contact |
| **Upright, body ang. vel, angular momentum** | Keep the trunk stable |
| **Joint limits, action rate** | Stay inside the actuators' envelope |

Weights live in
[`env_cfgs.py`](src/mjlab/tasks/velocity/config/wisco_humanoid/env_cfgs.py) and inherit
mjlab's defaults where the robot does not need an override.

### Configuration

- PPO with adaptive learning rate, 30k iterations
- Terrain curriculum on the rough task
- Network: `(256, 256, 128)` — deliberately small for a 12-DOF robot

### Tasks

| Task ID | Terrain | Purpose |
|---------|---------|---------|
| `Mjlab-Velocity-Rough-Wisco-Humanoid` | generated | Full training with terrain curriculum |
| `Mjlab-Velocity-Flat-Wisco-Humanoid` | plane | Flat-ground training and evaluation |
| `Mjlab-Balance-Wisco-Humanoid` | plane | Standing balance with reactive stepping (no velocity command) |

---

## Observations (Policy)

| Observation | Source |
|-------------|--------|
| `base_ang_vel` | IMU |
| `projected_gravity` | IMU |
| `joint_pos` | Relative to the default pose |
| `joint_vel` | Joint encoders |
| `actions` | Action history |
| `command` | `(vx, vy, ωz)` |
| `height_scan` | Terrain raycast (rough task only) |
| `gait_clock` | `[cos(φ), sin(φ)]` at the configured gait frequency |

**Removed:** `base_lin_vel` — a real robot needs a state estimator to observe it. Delete
the two `del` lines in `env_cfgs.py` if the robot carries one.

The critic additionally sees foot height, air time, contact state and contact forces.

---

## Sim2Real

- **Physics-based PD gains:** `KP = J_reflected · ω²` at 10 Hz, `KD` capped at the motor
  maximum so the policy never learns against a stiffer controller than the hardware.
- **Armature:** `J_reflected = J_rotor · gear_ratio²`. A mismatch between the training XML
  and the deployment XML makes a good policy fall over immediately.
- **Default pose:** start from the zero pose that is mechanically stable, then tighten the
  standing tolerance in the pose reward.
- **Action scale:** `0.3 · effort_limit / KP`, so every joint uses a comparable fraction of
  its torque budget.

---

## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Core install
uv sync
uv run list-envs
```

Upstream mjlab documentation: <https://mujocolab.github.io/mjlab/>.

### Train

```bash
uv run train Mjlab-Velocity-Flat-Wisco-Humanoid --env.scene.num-envs 4096
```

Multi-GPU:

```bash
uv run train Mjlab-Velocity-Rough-Wisco-Humanoid --gpu-ids "[0, 1]" --env.scene.num-envs 4096
```

### Evaluate

```bash
uv run play Mjlab-Velocity-Flat-Wisco-Humanoid --wandb-run-path your-org/wisco-mjlab/run-id
```

Sanity-check the MDP before training:

```bash
uv run play Mjlab-Velocity-Flat-Wisco-Humanoid --agent zero
uv run play Mjlab-Velocity-Flat-Wisco-Humanoid --agent random
```

> Training and play fail until the robot model exists. `list-envs` and the task registry
> work today.

---

## Repository layout

```
src/mjlab/asset_zoo/robots/wisco_humanoid/     # robot constants, XML + mesh slots
src/mjlab/tasks/velocity/config/wisco_humanoid/ # task configs + robot-local MDP terms
src/mjlab/scripts/sim2sim.py                    # ONNX policy -> plain MuJoCo rollout
SCAFFOLD.md                                     # orientation + known shortcuts
```

`upstream` tracks `mujocolab/mjlab`, so upstream fixes can be pulled in normally.

---

## Credits

- **[mjlab](https://github.com/mujocolab/mjlab)** by [MuJoCo Lab](https://github.com/mujocolab)
  — the simulation and RL framework this repository forks. The manager API, task
  registry, MDP terms and ONNX export all come from there.
- **[asimov-mjlab](https://github.com/menloresearch/asimov-mjlab)** by
  [Menlo Research](https://menlo.ai) — the inspiration for this scaffold and its
  reference implementation: the Asimov robot package, velocity task configs,
  reflected-inertia and PD-gain approach, the gait clock / alternating-feet / imitation
  rewards, and the original `deploy.py` that `sim2sim` is based on.
- **[WiscoHumanoids](https://www.linkedin.com/company/wiscohumanoids)** — the robot this
  scaffold is being adapted for.

## License

Licensed under the [Apache License, Version 2.0](LICENSE), inherited from mjlab.
