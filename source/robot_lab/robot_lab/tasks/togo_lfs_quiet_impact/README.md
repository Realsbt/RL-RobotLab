# ToGo_LFs Quiet Impact Diagnostics

This task preserves `RobotLab-ToGo-LFs-Quiet-v0` as the trained MUTE baseline
and adds event-based touchdown instrumentation without changing policy,
critic, action, or reward dimensions.

## Task

- Gym task: `RobotLab-ToGo-LFs-Quiet-Impact-v1`
- Policy observation: 450 (unchanged)
- Single observation: 45 (unchanged)
- Critic observation: 275 (unchanged)
- Action: 12 (unchanged)
- Reward: inherited from the fixed-beta quiet baseline

The task can therefore play the existing `model_4999.pt` checkpoint directly.

## Event definition

A touchdown starts when foot normal-force magnitude crosses 1.0 N from below.
Contact ends after force drops below 0.5 N. The hysteresis prevents force chatter
near a single threshold from creating repeated touchdown events.

The dedicated foot sensor stores 20 samples (100 ms) of synchronized contact
force and world-frame foot velocity at the 200 Hz physics rate. Each touchdown
uses the last airborne physics sample for pre-impact speed and collects force
features over the next 50 ms.

## Metrics

- `impact_event_count`: touchdown events per episode.
- `event_preimpact_speed`: mean downward speed immediately before touchdown [m/s].
- `event_peak_force`: mean peak force in the 50 ms event window [N].
- `event_peak_force_rise_rate`: mean maximum positive force slope [N/s].
- `event_impulse`: mean integrated force over the event window [N s].
- `event_force_variation_energy`: mean squared force-increment energy [N^2 s].
- `event_impact_score`: mean dimensionless visualization score.
- `peak_event_impact_score`: largest event score in the episode.
- `contact_slip_power_proxy`: mean `normal force * horizontal foot speed` [W].

The event score is normalized for visualization only. It is not dB and is not
yet used as a reward. Its reference values must eventually be calibrated with
matched real-robot microphone data.

## Playback

```bash
python scripts/rsl_rl/play.py \
  --task RobotLab-ToGo-LFs-Quiet-Impact-v1 \
  --checkpoint logs/rsl_rl/togo_lfs_quiet_moe_cts/2026-07-10_11-42-56_beta_1p4/model_4999.pt \
  --num_envs 2 \
  --quiet-noise-vis \
  --no-camera-follow \
  --real-time
```

The foot marker is invisible outside an event. At touchdown it is displayed
for 0.3 s: green is low, amber is medium, and red is high impact score. Stable
stance does not trigger or update a new touchdown marker.

## Benchmark

`benchmark_quiet_impact.py` compares two checkpoints sequentially in one
environment. Both policies receive the same saved initial scene state and seed.
Observation corruption, pushes, domain randomization, reset randomization, and
terminations are disabled. Each fixed-speed condition has a 2 s warmup followed
by a 10 s measurement window across 128 environments.

```bash
python scripts/rsl_rl/benchmark_quiet_impact.py \
  --headless \
  --num_envs 128 \
  --speeds 0.2 0.4 \
  --warmup_s 2.0 \
  --eval_s 10.0 \
  --baseline_checkpoint logs/rsl_rl/togo_lfs_moe_cts/2026-07-08_19-22-57_togo_lfs_moe_cts_continue_from_1000_to_150k/model_24000.pt \
  --candidate_checkpoint logs/rsl_rl/togo_lfs_quiet_moe_cts/2026-07-10_11-42-56_beta_1p4/model_4999.pt \
  --output logs/benchmarks/quiet_impact_baseline_vs_beta1p4_5000.json
```

The 2026-07-10 benchmark produced these candidate reductions relative to the
pre-quiet-training locomotion baseline. Positive values mean lower impact:

| Metric | 0.2 m/s | 0.4 m/s |
| --- | ---: | ---: |
| Pre-impact downward speed | 65.30% | 65.95% |
| Peak contact force | 35.58% | 37.71% |
| Peak force rise rate | 43.99% | 37.77% |
| Contact impulse | 41.53% | 44.47% |
| Force variation energy | 58.84% | 44.14% |
| Composite impact score | 82.97% | 84.76% |
| Worst event impact score | 79.57% | 78.45% |
| Tracking absolute error | 61.54% | 52.20% |
| Slip power proxy | -9.99% | -16.07% |

The candidate tracked both commands more accurately, so the impact reduction
was not obtained by walking slower. Its slip proxy and detected contact-event
rate increased, however. Those remain explicit regression metrics for the next
reward iteration. The benchmark evaluates simulated mechanical proxies, not
microphone sound-pressure level or dB.

## Implementation record

- `__init__.py`: registers a separate task and leaves `Quiet-v0` untouched.
- `env_cfg.py`: adds the dedicated foot sensor and event command while
  inheriting the baseline observations, actions, rewards, events, and terrain.
- `mdp/impact_sensor.py`: synchronizes 200 Hz contact-force and foot-velocity
  histories so pre-impact velocity is not taken from a previous 50 Hz policy
  frame.
- `mdp/impact_math.py`: contains testable contact hysteresis and normalized
  impact-score equations.
- `mdp/commands.py`: detects touchdown events, accumulates the 50 ms event
  features, logs episode diagnostics, and draws event-only markers.
- `rsl_rl_cfg.py`: keeps the existing PPO/MoE-CTS network definitions and uses
  separate log directories for future experiments.
- `scripts/rsl_rl/play.py`: reports whether continuous MUTE or event-only
  visualization is active.
- `scripts/rsl_rl/benchmark_quiet_impact.py`: runs matched-state, fixed-command
  checkpoint comparisons and writes machine-readable JSON reports.
- `scripts/rsl_rl/quiet_impact_report.py`: computes lower-is-better percentage
  reductions without requiring Isaac Sim.
- `tests/test_togo_lfs_quiet_impact_task.py`: verifies task isolation,
  observation/reward compatibility, hysteresis, and score normalization.
- `tests/test_quiet_impact_benchmark.py`: verifies report arithmetic, benchmark
  protocol settings, and aggregate event-statistics wiring.

Validation performed on 2026-07-10:

- 20 focused quiet-task and benchmark tests passed.
- Full suite: 73 passed and 5 unrelated pre-existing failures remained (two
  deploy `utils` import collisions and three missing `dm_control` failures).
- Isaac Sim loaded the original `model_4999.pt` with policy 450, critic 275,
  single observation 45, and action 12.
- A 64-environment, 60-iteration smoke run produced finite, non-zero values for
  every new event metric.

Current boundary: the new metrics and score are diagnostics only. They do not
alter rewards or policy actions. Matched-speed baseline collection is now
implemented; real-audio calibration is still required before treating these
proxies as acoustic loudness or selecting event reward weights.
