# Presentation outline — mjlab taxonomy and the Wisco-Mjlab scaffold

Working outline for a talk in two halves:

- **Top section — MJLab / Taxonomy:** what mjlab is, subsystem by subsystem.
- **Bottom section — Wisco-Mjlab / What we changed:** how this fork adapts mjlab for the
  Wisco Humanoid.

Every slide references real paths in the repository so code can be shown live. Roughly 27
slides; trim to ~18 for a 15-minute slot by collapsing slides 8–10 and 17–19.

---

## Top section — MJLab / Taxonomy

### 1. What is mjlab (one-slide thesis)

- Isaac Lab's API (managers, MDP terms, cfg dataclasses) on a MuJoCo Warp backend.
- Single GPU, thousands of parallel environments, no Omniverse.
- One repo covers simulation + MDP + RL + export + viewers.
- Why we use it: Isaac Lab abstractions, lightweight enough for a student team.

### 2. The data flow (diagram slide)

- MJCF + meshes → `EntityCfg` → `SceneCfg` → `ManagerBasedRlEnv`.
- Managers read and write the MDP on every `step()`.
- `rl/` trains PPO → `play` evaluates → ONNX export → `sim2sim` deploys.
- Put this early; every later slide hangs off it.

### 3. `sim/` — the physics backend

- `SimulationCfg`, `MujocoCfg`, `SimData`.
- Timestep, solver iterations, `nconmax`, `njmax`, `ccd_iterations`, CUDA graphs.
- Warp/CPU backend selection; `randomization.py` for sim-level domain randomization.
- *Wisco:* G1's limits were copied as placeholders.

### 4. `entity/` — what a robot is

- `EntityCfg` (init state, collisions, `spec_fn`, articulation) vs runtime `Entity`
  (`data.py`).
- `EntityData`: joint pos/vel, body poses, root state — everything observations read.
- MJCF loads through `spec_fn` and resolves lazily.
- *Wisco:* `get_wisco_humanoid_robot_cfg()` returns a fresh cfg per call.

### 5. `actuator/` — how joints are driven

- `BuiltinPositionActuatorCfg` (MJCF position servo), DC motor, learned actuator, XML
  actuators.
- Groups: `builtin_group`, `fused_group`.
- Per actuator: stiffness, damping, effort limit, armature, `target_names_expr`.
- *Wisco:* five joint families, `KP = J·ω²`, capped `KD`, action scale `0.3·effort/KP`.

### 6. `sensor/` — how the robot perceives

- Builtin (IMU/gyro), `ContactSensor`, `RayCastSensor`, `CameraSensor`,
  `TerrainHeightSensor`.
- `ObjRef` for frame references (`type`, `name`, `entity`).
- Pattern configs: `GridPatternCfg`, `RingPatternCfg`.
- *Wisco:* feet contact, self-collision, foot-height scan, terrain scan.

### 7. `scene/` + `terrains/` — the world

- `SceneCfg`: entities, sensors, terrain, extent.
- Terrain generator, heightfields, primitives, curriculum hooks.
- `TerrainEntityCfg` with `terrain_type="plane"` vs `"generator"`.
- *Wisco:* rough = generator + curriculum, flat = plane.

### 8. `managers/` — the MDP engine (part 1: the interface)

- One manager per term family; all share `ManagerBase` + `ManagerTermBaseCfg`.
- A term is a **function** or a **class** instantiated as `(cfg, env)`.
- Class terms can implement `reset(env_ids)` — this is why the clock terms are classes.
- *Wisco:* the three new terms are class-based and robot-local.

### 9. `managers/` — the MDP loop (part 2)

- `observation_manager`: groups, noise, clip, scale, delay, history.
- `action_manager`: `JointPositionAction`, scale, offset.
- `command_manager`: `UniformVelocityCommandCfg`.
- `reward_manager`, `termination_manager`.
- *Wisco:* edit the `actor` group, set action scale, constrain the twist command.

### 10. `managers/` — training-time concerns (part 3)

- `event_manager`: `startup` / `reset` / `interval` (push, friction, COM, encoder bias).
- `curriculum_manager`: terrain levels and command ranges.
- `metrics_manager`, `recorder_manager`.
- *Wisco:* added `base_com` body names, kept push/friction events, terrain curriculum.

### 11. `envs/mdp/` — the shared term library

- Actions, observations, rewards, terminations, events, `dr/`.
- Function terms vs class terms; `SceneEntityCfg` resolution.
- *Wisco:* `variable_posture`, `self_collision_cost`, `feet_*` come from here.

### 12. `tasks/` + registry — how a task becomes a command

- `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)`.
- Auto-import of `tasks/**/config/*`.
- Task ID convention: `Mjlab-Velocity-Flat-Unitree-G1`.
- *Wisco:* three IDs, visible in `uv run list-envs`.

### 13. `rl/` — training and export

- `RslRlOnPolicyRunnerCfg`, `RslRlModelCfg`, `RslRlPpoAlgorithmCfg`.
- Vecenv wrapper, `MjlabOnPolicyRunner`.
- `exporter_utils.py`: ONNX metadata (joint names, KP/KD, defaults, action scale,
  observation layout).
- *Wisco:* that metadata is the contract `sim2sim` depends on.

### 14. `asset_zoo/` — reference robots

- G1, Go1, YAM as worked examples.
- Each is a constants module + XML + meshes + factory function.
- *Wisco:* our package copies this shape.

### 15. `viewer/` + `scripts/` — the outer shell

- Native viewer, viser, debug visualizer, offscreen renderer.
- `train`, `play`, `list-envs`, `export-scene`, `nan_viz`, `csv_to_npz`.
- *Wisco:* added the `sim2sim` entry point.

### 16. `utils/` — the glue

- Lab API math (quaternion ops), noise, buffers, spec config, actuator helpers.
- *Wisco:* `reflected_inertia_from_two_stage_planetary` is the G1 armature reference.

---

## Bottom section — Wisco-Mjlab / What we changed

### 17. Repository setup

- Genuine fork of `mujocolab/mjlab`, public; `main` carries the scaffold.
- Remotes: `origin` (our fork), `upstream` (mjlab), `asimov` (Menlo reference).
- No merge of the Asimov branch — the scaffold is a port onto current `main`.

### 18. New robot package

- `src/mjlab/asset_zoo/robots/wisco_humanoid/`.
- `wisco_humanoid_constants.py` is the single place hardware numbers live.
- Empty `xmls/`, `assets/meshes/`, `assets/reference/` stubs.
- Lazy `get_spec()` so tasks register before the model exists.

### 19. Actuator and gain setup

- Five joint families: hip pitch / roll / yaw, knee, ankle.
- Reflected inertia, physics-based KP, hardware-capped KD.
- Action scale derived from effort/stiffness.
- All values marked `TODO(wisco-humanoid)`.

### 20. Task configuration

- Three tasks: Rough (curriculum), Flat (plane), Balance (no velocity command).
- Flat also strips the terrain scan, `height_scan` and the out-of-bounds termination.
- Play mode: infinite episodes, no corruption, no pushes, small terrain grid.

### 21. Sensors and observations

- Feet-ground contact sensor; self-collision sensor with force history.
- Foot-height scan wired to `left_foot` / `right_foot` sites via `ObjRef` +
  `RingPatternCfg`.
- Terrain scan frame pinned to the base body.
- **Removed** `base_lin_vel` (no state estimator).
- **Added** `gait_clock` to actor and critic.

### 22. Rewards

- Inherited mjlab weights for tracking, upright, pose and foot terms.
- Pose standard deviations tuned per joint family.
- Added self-collision, alternating-feet and optional imitation.
- Conservative velocity ranges for a narrow-stance biped.

### 23. Robot-local MDP terms (design decision)

- `config/wisco_humanoid/mdp.py`: `gait_clock`, `alternating_feet_contact`,
  `imitation_joint_pos`.
- **Why:** Asimov patched `velocity/mdp/*.py`; we keep upstream untouched so merges stay
  clean.
- Generalized: foot count from the sensor, phase reset via the manager hook, CSV columns
  configurable.

### 24. `sim2sim` tool

- One entry point: `uv run sim2sim --policy policy.onnx`.
- Rebuilds the observation from ONNX metadata instead of a hardcoded 47-dim vector.
- Rebuilds position actuators from KP/KD metadata plus the armature constants.
- Generic plots, optional video, push tests; no terrain support.

### 25. Tooling and docs

- `pyproject.toml`: `sim2sim` extra and console script.
- Nine-line robot export in `asset_zoo/robots/__init__.py` — the only upstream source edit.
- README rewritten in the Asimov style with logo; `SCAFFOLD.md` orientation doc.
- Changelog entry and credits (mjlab, Menlo Research, WiscoHumanoids).

### 26. Validation status (honest slide)

- `uv run list-envs` → all three tasks.
- 18 config tests pass; full fast suite 1122 passed, 13 skipped.
- `ruff` and `ty` clean; the package imports without the model.
- **Not tested:** any rollout, because the MJCF does not exist yet.

### 27. What is still placeholder / next steps

- Add `xmls/wisco_humanoid.xml` and the meshes.
- Fill rotor inertia, gear ratio, torque and standing height.
- Retune pose stds, velocity ranges and reward weights.
- Train flat → rough → balance, then run `sim2sim`.
- `SCAFFOLD.md` "Known hacks" has the shortcut list.

---

## Bridge slide (place between the two sections)

| Taxonomy area | What we changed |
| --- | --- |
| `entity/`, `actuator/` | New robot package, joint families, gains, action scale |
| `sensor/` | Feet contact, self-collision, foot-height scan, terrain frame |
| `managers/` observations | Removed `base_lin_vel`, added `gait_clock` |
| `managers/` rewards | Pose stds, self-collision, alternating feet, imitation |
| `tasks/` | Three task IDs, rough/flat/balance |
| `rl/` + `scripts/` | `sim2sim` driven by ONNX metadata |

---

## Timing guide

| Section | Slides | Time |
| --- | --- | --- |
| MJLab / Taxonomy | 1–16 | ~8 min |
| Bridge | — | ~1 min |
| Wisco-Mjlab / Changes | 17–27 | ~6 min |
| Questions | — | rest |

---

## Slide titles (copy list)

**Top section — MJLab / Taxonomy**

1. What is mjlab
2. The data flow
3. `sim/` — the physics backend
4. `entity/` — what a robot is
5. `actuator/` — how joints are driven
6. `sensor/` — how the robot perceives
7. `scene/` + `terrains/` — the world
8. `managers/` — the MDP engine (part 1: the interface)
9. `managers/` — the MDP loop (part 2)
10. `managers/` — training-time concerns (part 3)
11. `envs/mdp/` — the shared term library
12. `tasks/` + registry — how a task becomes a command
13. `rl/` — training and export
14. `asset_zoo/` — reference robots
15. `viewer/` + `scripts/` — the outer shell
16. `utils/` — the glue

**Bottom section — Wisco-Mjlab / What we changed**

17. Repository setup
18. New robot package
19. Actuator and gain setup
20. Task configuration
21. Sensors and observations
22. Rewards
23. Robot-local MDP terms (design decision)
24. `sim2sim` tool
25. Tooling and docs
26. Validation status (honest slide)
27. What is still placeholder / next steps

**Interstitial**

- Bridge: taxonomy → changes table
- Questions
