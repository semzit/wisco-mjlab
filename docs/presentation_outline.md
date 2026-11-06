# Presentation outline — mjlab taxonomy and the Wisco-Mjlab scaffold

Working outline for a talk in two halves:

- **Top section — MJLab / Taxonomy:** the system as the [official documentation](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)
  describes it: two layers, the simulation components, the manager layer, and the
  environment lifecycle.
- **Bottom section — Wisco-Mjlab / What we changed:** how this fork adapts mjlab for the
  Wisco Humanoid.

The top section follows mjlab's own docs page by page rather than walking the source tree;
file paths appear once, on the concept-to-code mapping slide (12), so the talk stays
conceptual and the code is only there for backup.

---

## Top section — MJLab / Taxonomy

### 1. What is mjlab — two layers

- The docs describe mjlab as **a simulation layer** (robot and world) plus **a manager
  layer** (the RL problem defined on top).
- The simulation layer is MuJoCo + MuJoCo Warp; the manager layer is Isaac Lab's
  manager-based MDP.
- Keep that split in mind: every later slide belongs to one of the two.
- Doc: [Architecture Overview](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)

### 2. Architecture at a glance

- Show the official diagram: entities composed into an `MjSpec`, compiled to `MjModel`,
  transferred to MuJoCo Warp; `ManagerBasedRlEnv` orchestrates; RSL-RL trains.
- One sentence per box on the diagram; do not explain details yet.
- Doc image: `architecture_diagram.png` from the architecture page.
- This is the slide people photograph.

### 3. Simulation layer — the scene pipeline

- Every entity starts as an MJCF loaded with `MjSpec.from_file()`.
- Python dataclasses can **extend or override** the loaded spec: actuators, collision
  rules, materials, sensors, initial state.
- Entities are composed into one `MjSpec`, compiled to `MjModel` on CPU, then uploaded.
- Takeaway: start from an existing MuJoCo model, layer task config on top, never rewrite
  the XML.
- Doc: [Scene](https://mujocolab.github.io/mjlab/main/source/scene.html)

### 4. Simulation layer — MuJoCo Warp

- GPU backend that keeps MuJoCo's `MjModel` / `MjData` paradigm.
- Adds a leading **world** dimension: one `MjData` holds N independent simulations.
- Model parameters shared by default; fields expand per-world only when domain
  randomization needs it.
- The step is captured as a **CUDA graph** — recorded once, replayed; resets and DR run as
  normal Python between replays.
- Doc: [Architecture Overview → MuJoCo Warp](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html#the-simulation-layer)

### 5. Simulation layer — the four components

- **Entity** — a robot, a manipulated object, or static terrain: MJCF plus optional
  actuator/collision/initial-state config.
  [docs](https://mujocolab.github.io/mjlab/main/source/entity/index.html)
- **Actuators** — how entities are controlled; wrap MJCF actuators or define new ones in
  Python. [docs](https://mujocolab.github.io/mjlab/main/source/actuators.html)
- **Sensors** — how the world is observed: MuJoCo-native sensors plus RGB-D cameras and
  raycasters. [docs](https://mujocolab.github.io/mjlab/main/source/sensors/index.html)
- **Scene** + **Terrain** — composition and placement of entities.
  [scene](https://mujocolab.github.io/mjlab/main/source/scene.html) ·
  [terrain](https://mujocolab.github.io/mjlab/main/source/terrain.html)
- *Wisco touch:* we touch all four — robot package, joint families, contact/ray sensors,
  plane vs generated terrain.

### 6. Manager layer — terms are the unit of composition

- An environment is a set of small, self-contained **terms**: rewards, observations,
  events, commands, and so on.
- A term is a plain function for stateless work, or a **class** deriving from
  `ManagerTermBase` when it must cache setup or hold per-episode state.
- Class terms get a `reset()` hook — that is what makes phase clocks and imitation
  trackers possible.
- *Wisco touch:* all three of our new terms are class-based and live in
  `tasks/velocity/mdp/gait.py`, alongside mjlab's own terms.

### 7. Manager layer — one config object

- Everything is described by `ManagerBasedRlEnvCfg`, a plain dataclass of term
  dictionaries.
- Fields: `scene`, `sim`, `observations`, `actions`, `rewards`, `terminations`, `events`,
  `commands`, `curriculum`, `metrics`, plus `decimation` and `episode_length_s`.
- `decimation = 4` means four physics steps per policy step — the value our tasks inherit.
- Doc: [Environment Configuration](https://mujocolab.github.io/mjlab/main/source/environment_config.html)

### 8. Manager layer — the eight managers

| Manager | Responsibility | Doc |
| --- | --- | --- |
| Observation | Assembles groups; clip, noise, delay, history; asymmetric actor-critic | [observations](https://mujocolab.github.io/mjlab/main/source/observations.html) |
| Action | Routes the policy tensor to actuators; scale and offset | [actions](https://mujocolab.github.io/mjlab/main/source/actions.html) |
| Reward | Weighted sum, scaled by step duration | [rewards](https://mujocolab.github.io/mjlab/main/source/rewards.html) |
| Termination | Stop conditions; terminal vs timeout | [terminations](https://mujocolab.github.io/mjlab/main/source/terminations.html) |
| Event | Fires at startup / reset / interval; implements DR | [events](https://mujocolab.github.io/mjlab/main/source/events.html) |
| Command | Generates and resamples goals | [commands](https://mujocolab.github.io/mjlab/main/source/commands.html) |
| Curriculum | Adjusts training conditions by performance | [curriculum](https://mujocolab.github.io/mjlab/main/source/curriculum.html) |
| Metrics | Logs per-step values as episode averages | [metrics](https://mujocolab.github.io/mjlab/main/source/metrics.html) |

- Plus **Recorders** for dataset capture: [recorders](https://mujocolab.github.io/mjlab/main/source/recorders.html)
- *Wisco touch:* observations, actions, rewards, events and curriculum are the ones we edit.

### 9. Environment lifecycle — the four phases

- **Build:** `Scene` composes MJCFs into `MjSpec`, compiles `MjModel`, `Simulation`
  uploads to the GPU, CUDA graphs captured.
- **Initialize:** managers built, regexes resolved to joint/body/geom indices, history and
  delay buffers allocated, DR fields expanded, startup events fired once.
- **Reset:** reset events randomize the scene, commands resampled, history cleared.
- **Step:** action → physics → termination → reward → events → resets → observations.
- Doc: [Architecture Overview → lifecycle](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html#the-environment-lifecycle)

### 10. The step sequence (deep dive)

- The documented order, worth showing verbatim:
  `process_action` → `for decimation: apply_action / sim.step / scene.update` →
  `termination` → `reward` → `metrics` → step/interval events → reset → `sim.forward()` →
  `command` → `sim.sense()` → `observation`.
- Why it matters: observations are computed **after** the physics and resets, which is why
  a phase clock advances once per control step.
- This slide explains any "why is my reward one step late" question.
- *Wisco touch:* our clock terms advance in `__call__`, reset in `reset()`.

### 11. Training and debugging surface

- **RSL-RL** runner and PPO configs:
  [training/rsl_rl](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)
- Viewers — native, viser, offscreen:
  [viewers](https://mujocolab.github.io/mjlab/main/source/viewers.html)
- Distributed and cloud training:
  [distributed](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) ·
  [cloud](https://mujocolab.github.io/mjlab/main/source/training/cloud.html)
- Debugging: [NaN guard](https://mujocolab.github.io/mjlab/main/source/debugging/nan_guard.html) ·
  [export scene](https://mujocolab.github.io/mjlab/main/source/debugging/export_scene.html)
- *Wisco touch:* we add one entry point here, `sim2sim`, and the ONNX export contract.

### 12. Concept → code map (the only file-path slide)

| Concept | Path |
| --- | --- |
| Entity, actuators, sensors | `src/mjlab/entity/`, `actuator/`, `sensor/` |
| Scene, terrain | `src/mjlab/scene/`, `terrains/` |
| Simulation backend | `src/mjlab/sim/` |
| Managers, MDP terms | `src/mjlab/managers/`, `envs/mdp/` |
| Environment | `src/mjlab/envs/manager_based_rl_env.py` |
| Tasks and registry | `src/mjlab/tasks/` |
| RL, export | `src/mjlab/rl/` |
| Referenced robots | `src/mjlab/asset_zoo/robots/` |
| CLI and viewers | `src/mjlab/scripts/`, `viewer/` |

- Use this only if someone asks "where does that live?"
- Hand-off to the bottom section: "now the parts we actually changed."

---

## Bottom section — Wisco-Mjlab / What we changed

### 13. Repository setup

- Genuine fork of `mujocolab/mjlab`, public; `main` carries the scaffold.
- Remotes: `origin` (our fork), `upstream` (mjlab), `asimov` (Menlo reference).
- No merge of the Asimov branch — the scaffold is a port onto current `main`.

### 14. New robot package

- `src/mjlab/asset_zoo/robots/wisco_humanoid/`.
- `wisco_humanoid_constants.py` is the single place hardware numbers live.
- Empty `xmls/`, `xmls/assets/`, `assets/reference/` stubs.
- Lazy `get_spec()` so tasks register before the model exists.

### 15. Actuator and gain setup

- Five joint families: hip pitch / roll / yaw, knee, ankle.
- Reflected inertia, physics-based KP, hardware-capped KD.
- Action scale derived from effort/stiffness.
- All values marked `TODO(wisco-humanoid)`.

### 16. Task configuration

- Three tasks: Rough (curriculum), Flat (plane), Balance (no velocity command).
- Flat also strips the terrain scan, `height_scan` and the out-of-bounds termination.
- Play mode: infinite episodes, no corruption, no pushes, small terrain grid.

### 17. Sensors and observations

- Feet-ground contact sensor; self-collision sensor with force history.
- Foot-height scan wired to `left_foot` / `right_foot` sites via `ObjRef` +
  `RingPatternCfg`.
- Terrain scan frame pinned to the base body.
- **Removed** `base_lin_vel` (no state estimator).
- **Added** `gait_clock` to actor and critic.

### 18. Rewards

- Inherited mjlab weights for tracking, upright, pose and foot terms.
- Pose standard deviations tuned per joint family.
- Added self-collision, alternating-feet and optional imitation.
- Conservative velocity ranges for a narrow-stance biped.

### 19. New MDP terms (where they live)

- `tasks/velocity/mdp/gait.py`: `gait_clock`, `alternating_feet_contact`,
  `imitation_joint_pos`, exported through the task's `mdp` package.
- **Why a separate module:** Asimov patched `observations.py` and `rewards.py` directly;
  one file of our own keeps the upstream term files untouched and the diff readable.
- Generalized: foot count from the sensor, phase reset via the manager hook, CSV columns
  configurable.

### 20. `sim2sim` tool

- One entry point: `uv run sim2sim --policy policy.onnx`.
- Rebuilds the observation from ONNX metadata instead of a hardcoded 47-dim vector.
- Rebuilds position actuators from KP/KD metadata plus the armature constants.
- Generic plots, optional video, push tests; no terrain support.

### 21. Tooling and docs

- `pyproject.toml`: `sim2sim` extra and console script.
- Nine-line robot export in `asset_zoo/robots/__init__.py` — the only upstream source edit.
- README rewritten in the Asimov style with logo; `SCAFFOLD.md` orientation doc.
- Changelog entry and credits (mjlab, Menlo Research, WiscoHumanoids).

### 22. Validation status (honest slide)

- `uv run list-envs` → all three tasks.
- 18 config tests pass; full fast suite 1122 passed, 13 skipped.
- `ruff` and `ty` clean; the package imports without the model.
- **Not tested:** any rollout, because the MJCF does not exist yet.

### 23. What is still placeholder / next steps

- Add `xmls/wisco_humanoid.xml` and the meshes.
- Fill rotor inertia, gear ratio, torque and standing height.
- Retune pose stds, velocity ranges and reward weights.
- Train flat → rough → balance, then run `sim2sim`.
- `SCAFFOLD.md` "Known hacks" has the shortcut list.

---

## Bridge slide (place between the two sections)

| Documentation concept | What we changed |
| --- | --- |
| Entity + actuators | New robot package, joint families, gains, action scale |
| Sensors | Feet contact, self-collision, foot-height scan, terrain frame |
| Observations | Removed `base_lin_vel`, added `gait_clock` |
| Rewards | Pose stds, self-collision, alternating feet, imitation |
| Environment config / tasks | Three task IDs, rough / flat / balance |
| Training + export | `sim2sim` driven by ONNX metadata |

---

## Timing guide

| Section | Slides | Time |
| --- | --- | --- |
| MJLab / Taxonomy | 1–12 | ~9 min |
| Bridge | — | ~1 min |
| Wisco-Mjlab / Changes | 13–23 | ~6 min |
| Questions | — | rest |

---

## Slide titles (copy list)

**Top section — MJLab / Taxonomy**

1. What is mjlab — two layers
2. Architecture at a glance
3. Simulation layer — the scene pipeline
4. Simulation layer — MuJoCo Warp
5. Simulation layer — the four components
6. Manager layer — terms are the unit of composition
7. Manager layer — one config object
8. Manager layer — the eight managers
9. Environment lifecycle — the four phases
10. The step sequence
11. Training and debugging surface
12. Concept → code map

**Bottom section — Wisco-Mjlab / What we changed**

13. Repository setup
14. New robot package
15. Actuator and gain setup
16. Task configuration
17. Sensors and observations
18. Rewards
19. New MDP terms (where they live)
20. `sim2sim` tool
21. Tooling and docs
22. Validation status (honest slide)
23. What is still placeholder / next steps

**Interstitial**

- Bridge: documentation concept → changes table
- Questions
