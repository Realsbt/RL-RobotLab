# ToGo_LFs Backflip PPO

This package trains one standing backflip with ordinary PPO. It is isolated
from the locomotion and MoE-CTS tasks and keeps the same 60-D actor, 72-D
critic, and 12-D action contracts in every stage so checkpoints can be moved
forward without partial loading.

## Tasks

| Stage | Task | Default iterations | Purpose |
| --- | --- | ---: | --- |
| Baseline | `RobotLab-ToGo-LFs-Backflip-Genesis-v0` | 1000 | Independent Genesis-style end-to-end flip |
| 1 | `RobotLab-ToGo-LFs-Backflip-Jump-v0` | 5000 | Symmetric vertical jump |
| 2 | `RobotLab-ToGo-LFs-Backflip-Rotate-v0` | 10000 | One complete backward rotation |
| 3 | `RobotLab-ToGo-LFs-Backflip-v0` | 3000 | Unfolding and stable landing |
| 4 | `RobotLab-ToGo-LFs-Backflip-Robust-v0` | 5000 | Actuator latency and narrow randomization |

The first three stages retain the measured 48 N m torque-speed envelope but
temporarily set motor delay and response filtering to zero. Stage 4 restores
the configured 5-20 ms motor delay and 5-15 ms response-time range. All stages
keep self-collision disabled, matching the base LFS asset. Its current collision
meshes generate persistent false torso/leg contacts when blanket self-collision
is enabled; repair and validate those meshes before using self-collision as a
deployment constraint.

## Policy contract

Actor observation (60):

- body angular velocity: 3
- projected gravity: 3
- joint position residuals: 12
- joint velocities: 12
- current and previous actions: 24
- multiscale phase clock: 6

The critic additionally observes root height/velocity, foot contacts, and
simulator-only takeoff/rotation/landing state. The actor never receives these
privileged values.

The action is a joint-position residual with scale 0.5. Processed targets are
clamped to 95% of each URDF joint limit. The episode is 1.8 s at 50 Hz policy
control and 200 Hz physics.

## MuJoCo sim2sim

The selected ordinary-PPO landing policy has a dedicated MuJoCo runner.  It
uses the same ToGo_LFs model selected by the local ``togo-master`` simulation,
the 60-D single-frame observation contract, 50 Hz phase clock, torque-speed
limit, disabled self-collision, and the assisted terminal front-leg transform:

```bash
python \
  deploy/deploy_mujoco/deploy_togo_lfs_backflip.py
```

The robot first holds its normal stance.  After the 0.5 s arming period, press
Xbox/standard controller ``A`` (raw joystick button 0) or ``Space`` in the
MuJoCo window to trigger the skill.  Once a flip completes, it holds the normal
stance and can be triggered again.  ``--auto-start`` restores automatic GUI
playback; ``--stand-time 0`` removes the arming delay.  A startup and dynamics
check without a viewer is available with ``--headless --no-real-time``; headless
mode starts automatically because it has no interactive input.  This runner is separate from
``deploy_togo_lfs.py``, whose observation and inference path belongs to the CTS
locomotion policy.

## Genesis baseline

`RobotLab-ToGo-LFs-Backflip-Genesis-v0` is an additive A/B baseline based on
the public [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
source at commit `6b5a62e6a6bd7f6af038a27788d938269672372a`. The upstream repository is
available locally under `external/Genesis-backflip/` as an ignored, read-only
reference. Its top level has no explicit license file, so the IsaacLab reward
terms here are independent reimplementations rather than copied source.

The baseline preserves the published ordinary-PPO hyperparameters, 2.0 s
episode, half-cycle multiscale clock, 0.5-1.0 s linear rotation reference,
minimal reward set, one-policy-step action latency, and domain-randomization
ranges. It intentionally keeps the ToGo joint convention, default pose,
measured torque-speed envelope, and 60-D actor contract. Self-collision stays
off because the current ToGo collision meshes generate false contacts. The
upstream motor-offset randomization is not yet reproduced.

Train the independent baseline from scratch with:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task RobotLab-ToGo-LFs-Backflip-Genesis-v0 \
  --num_envs 4096 \
  --max_iterations 1000
```

No existing staged checkpoint should be loaded into this baseline because its
phase semantics and reward timing differ. Evaluate a checkpoint with:

```bash
python scripts/rsl_rl/play.py \
  --task RobotLab-ToGo-LFs-Backflip-Genesis-v0 \
  --checkpoint logs/rsl_rl/togo_lfs_backflip_genesis_ppo/RUN/model_999.pt \
  --num_envs 1 \
  --real-time
```

The selected rotation policy leaves the ground near 0.30 s, reaches its apex
near 0.56 s, and first touches down near 0.88 s. Stage 2 therefore uses a
0.30-0.82 s rotation window: half a revolution is scheduled at the apex and
the full revolution is due before touchdown. Its main rotation
signal is whole-body centroidal angular velocity (CAV), not base-link angular
velocity, plus a dense supported-launch pitch signal, takeoff backward-pitch
quality, and persistent milestones at 1.10, 1.15, 1.25, 1.50, 1.60, 1.65, and
1.75 pi. The launch term is active only while a foot is supporting an upward
push, which avoids rewarding a late internal leg/body exchange that does not
establish angular momentum during takeoff. The CAV design follows the framework in
[Kang et al. (CoRL 2025)](https://proceedings.mlr.press/v305/kang25a.html).

## Training sequence

Stage 1:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task RobotLab-ToGo-LFs-Backflip-Jump-v0 \
  --num_envs 4096 \
  --max_iterations 5000
```

Inspect the result before proceeding:

```bash
python scripts/rsl_rl/play.py \
  --task RobotLab-ToGo-LFs-Backflip-Jump-v0 \
  --checkpoint logs/rsl_rl/togo_lfs_backflip_jump_ppo/RUN/model_4999.pt \
  --num_envs 4 \
  --real-time
```

Stage 2 initializes from the selected Stage-1 checkpoint without restoring its
optimizer or iteration counter. Reset the critic when the reward definition
changes substantially, and explicitly choose the initial exploration noise:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task RobotLab-ToGo-LFs-Backflip-Rotate-v0 \
  --num_envs 4096 \
  --max_iterations 10000 \
  --pretrained_checkpoint logs/rsl_rl/togo_lfs_backflip_jump_ppo/RUN/model_4999.pt \
  --pretrained_actor_only \
  --pretrained_action_std 0.6
```

When retiming an already successful rotation policy, keep its critic and use a
smaller exploration standard deviation so PPO does not destroy the full flip
while moving angular momentum toward takeoff:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task RobotLab-ToGo-LFs-Backflip-Rotate-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --pretrained_checkpoint logs/rsl_rl/togo_lfs_backflip_rotate_ppo/RUN/model.pt \
  --pretrained_action_std 0.10
```

At 24 rollout steps per iteration, 1024, 2048, and 4096 environments collect
24576, 49152, and 98304 transitions per PPO iteration respectively. More
environments improve throughput and exploratory coverage when GPU memory is
available, but iteration counts are no longer directly comparable because
each 4096-environment iteration contains four times as much experience as a
1024-environment iteration.

Stage 3 repeats that process with the chosen rotation checkpoint:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task RobotLab-ToGo-LFs-Backflip-v0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --pretrained_checkpoint logs/rsl_rl/togo_lfs_backflip_rotate_ppo/RUN/model_9999.pt
```

Only start Stage 4 after Stage 3 lands consistently:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task RobotLab-ToGo-LFs-Backflip-Robust-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --pretrained_checkpoint logs/rsl_rl/togo_lfs_backflip_land_ppo/RUN/model_2999.pt
```

Replace `RUN` and the checkpoint number with an actual saved run. Use
`--resume` only when continuing the same task/run; use
`--pretrained_checkpoint` when advancing between stages.

For finite deterministic evaluation, add `--headless --max_steps 180
--no-camera-follow` to `play.py`. It reports aggregate `[PLAY_EVAL]` metrics
and exits after two 1.8 s episodes.

## Advancement gates

Do not advance based on mean reward alone. The phase command logs:

- `takeoff_success`
- `rotation_success`
- `touchdown_success`
- `landing_success`
- `max_base_height_m`
- `max_backward_rotation_rad`
- `takeoff_time_s`
- `apex_time_s`
- `backward_rotation_at_apex_rad`
- `first_touchdown_time_s`
- `flight_time_s`
- `backward_pitch_rate_at_takeoff_rad_s`

Recommended gates are:

1. Jump: at least 95% takeoff and enough flight time/height for rotation.
2. Rotate: positive backward pitch rate at takeoff, about pi radians by the
   apex, and at least 80% of episodes reaching 1.75-2.25 pi before touchdown.
3. Land: at least 80% stable landing over several seeds.
4. Robust: retain at least 70% success under the configured randomization.

## Current result (2026-07-20)

The selected ordinary-PPO actor is R9i `model_225.pt`. The R10j assisted task
keeps that actor unchanged and applies a deterministic terminal front-leg
transform followed by a 0.30 s blend to the standing pose after touchdown.
In a 256-environment deterministic evaluation it achieved 100% takeoff,
complete rotation, clean first touchdown, and strict stable landing success.

The publishable TorchScript and ONNX exports are stored under
`deploy/pre_train/togo_lfs/backflip_r10j/`. The detailed experiment sequence,
rejected checkpoints, acceptance metrics, MuJoCo handoff, and native
`togo-master` integration are recorded in
`docs/togo_lfs_backflip_reproduction_log.md`.

Before any physical deployment, verify joint ordering/signs, torque and speed
peaks, landing impulse, self-collision clearance, and the exported policy in the
included MuJoCo model. The task and weights are simulation experiments, not an
authorization to run a flip on untethered hardware.
