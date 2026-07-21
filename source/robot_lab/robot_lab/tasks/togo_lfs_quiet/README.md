# ToGo_LFs MUTE Reproduction

This task is an isolated, fixed-quietness first stage of the MUTE reproduction.
It shares the ToGo_LFs robot asset with the base locomotion task and does not
modify the base task configuration.

## Current scope

- Gym task: `RobotLab-ToGo-LFs-Quiet-v0`
- Physics rate: 200 Hz (`dt=0.005`)
- Policy rate: 50 Hz (`decimation=4`)
- Quiet factor: fixed at experimental `beta=1.4`
- Commands: `vx=[-0.5,0.5]`, `vy=[-0.15,0.15]`, `yaw=[-0.4,0.4]`
- Terrain: flat hard floor
- Observation/action dimensions remain compatible with the base ToGo_LFs
  MoE-CTS checkpoint.

At fixed `beta=1.4`, the MUTE reward is implemented as:

```text
r = 1.4 * r_phase + 0.72 * r_velocity + r_other
r_phase = -0.05 * sum(exp(phase) * drop_velocity^2)
          +0.01 * sum(exp(-phase) * raise_velocity^2)
```

The paper does not specify the exact foot-velocity frame or causal phase-label
construction. This implementation uses world-frame vertical foot velocity and
a contact-timing phase with 0.35 s nominal swing and stance durations. The
paper defines `beta` in `[0,1]`; `beta=1.4` is an out-of-range engineering
experiment rather than the paper's standard reproduction setting.

## Training

Initialize from the existing base locomotion checkpoint without restoring its
optimizer or iteration counter:

```bash
python scripts/rsl_rl/train.py \
  --task RobotLab-ToGo-LFs-Quiet-v0 \
  --headless \
  --pretrained_checkpoint \
  logs/rsl_rl/togo_lfs_moe_cts/2026-07-08_19-22-57_togo_lfs_moe_cts_continue_from_1000_to_150k/model_24000.pt
```

Outputs are written under `logs/rsl_rl/togo_lfs_quiet_moe_cts/`.

## Diagnostics

The `base_velocity` command term logs these impact proxies without adding them
to the reward:

- `touchdown_velocity`: mean downward velocity from the policy frame before a
  first-contact event.
- `phase_weighted_drop_velocity`: mean MUTE drop term before reward weighting.
- `peak_foot_contact_force`: maximum foot contact-force norm over the episode.
- `contact_foot_slip_speed`: mean horizontal foot speed while in contact.

### Isaac Sim visualization

Use the playback-only flag below to draw one sphere above each foot. Green,
amber, and red mean low, medium, and high phase-weighted downward foot speed;
the marker also grows with the proxy magnitude. A short decay keeps touchdown
peaks visible in the viewport.

```bash
python scripts/rsl_rl/play.py \
  --task RobotLab-ToGo-LFs-Quiet-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 2 \
  --quiet-noise-vis
```

The visualized value is the MUTE impact proxy
`exp(phase / 2) * max(-foot_velocity_z, 0)`, in m/s. It is not simulated sound
pressure and cannot be interpreted as dB without microphone calibration data.

## Next stages

1. Add a four-leg learned phase estimator and auxiliary phase loss.
2. Add per-environment `beta` commands in `[0,1]` and extend observations from
   45 to 46 values per frame.
3. Update TorchScript and MuJoCo deployment for the additional command.
4. Compare baseline and MUTE policies at matched actual speeds before real-robot
   acoustic measurements.
