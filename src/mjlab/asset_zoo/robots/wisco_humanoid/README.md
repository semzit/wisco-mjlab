# Wisco Humanoid robot configuration

Scaffold package for a bipedal humanoid whose model is not defined yet. The structure
mirrors the Asimov reference configuration: joint families, physics-based PD gains,
keyframes, collision policy and the action scale all live in
`wisco_humanoid_constants.py`.

Nothing here is hardware-specific yet. Every numeric value is an Asimov placeholder and
is marked `TODO(wisco-humanoid)`.

## What you need to add

```
xmls/wisco_humanoid.xml        # the MuJoCo model
xmls/assets/*.STL              # visual and collision meshes
assets/reference/*.csv         # optional walking reference for the imitation reward
```

Until `xmls/wisco_humanoid.xml` exists, `get_spec()` raises a descriptive error. The
rest of the package still imports, so `list-envs` and `train --help` work.

Meshes live next to the MJCF in `xmls/assets/` because MuJoCo resolves `meshdir`
relative to the model file. `assets/reference/` holds non-MJCF data (the optional walking
reference CSV), not meshes.

## MJCF conventions the task configs rely on

| What | Expected name |
| --- | --- |
| Actuated joints | `left_/right_<joint>_joint` |
| Hip/knee/ankle joints | `*_hip_pitch_joint`, `*_hip_roll_joint`, `*_hip_yaw_joint`, `*_knee_joint`, `*_ankle_pitch_joint`, `*_ankle_roll_joint` |
| Foot sites | `left_foot`, `right_foot` |
| Collision geoms | `*_collision` |
| Foot collision geoms | `{left,right}_ankle_roll_link_collision` |
| Base body | `pelvis_link` |
| IMU sensors | `imu_lin_vel`, `imu_ang_vel` |
| Mesh directory | `xmls/assets`, declared via `<compiler meshdir="assets">` |

If your robot uses different names, update the regex patterns in
`wisco_humanoid_constants.py` and in
`src/mjlab/tasks/velocity/config/wisco_humanoid/env_cfgs.py`.

## Armature (reflected inertia)

MuJoCo's `armature` models the reflected inertia of motor plus gearbox and is critical
for sim2real transfer. For a geared motor:

```
J_reflected = J_rotor * gear_ratio^2
```

The squared term comes from energy conservation: the motor spins `gear_ratio` times
faster than the joint, so `KE = 1/2 * J_rotor * (gear_ratio * qdot)^2`.

For a two-stage planetary gearbox, sum the contribution of each stage instead; see
`mjlab.utils.actuator.reflected_inertia_from_two_stage_planetary` and the Unitree G1
constants for a worked example.

## Physics-based PD gains

```
KP = J_reflected * omega_n^2          # omega_n = 10 Hz * 2*pi
KD = 2 * zeta * J_reflected * omega_n
```

`KD` is capped at the hardware maximum (`KD_MAX`) so the policy is trained against the
same controller it will see on the real robot. A mismatch between the training XML's
armature and the deployment XML will make a policy fall over immediately.

## Checklist

1. Drop the MJCF into `xmls/` and the meshes into `xmls/assets/`.
2. Fill in `J_rotor`, `gear_ratio` and peak torque per joint in
   `wisco_humanoid_constants.py`, then recompute the derived constants.
3. Set `HOME_KEYFRAME.pos` (standing height) to match the real model kinematics.
4. Confirm the regex names in `FULL_COLLISION` match your foot geoms.
5. Run `uv run list-envs` and `uv run play Mjlab-Velocity-Flat-Wisco-Humanoid --agent zero`.
6. Tune the lateral/velocity command ranges, pose standard deviations and reward
   weights in `src/mjlab/tasks/velocity/config/wisco_humanoid/env_cfgs.py`.
