# RoboGauge DMBot Adaptation Log

## Context

- Date: 2026-06-17
- Goal: adapt RoboGauge to evaluate DMBot training checkpoints using the existing OpenDog MuJoCo model from `/home/sbt/Downloads/robot_house/DaMiao/OpenDog`.
- Constraint: keep a markdown record of each modification step while editing.

## Findings

1. `/home/sbt/Downloads/robot_house/DaMiao/OpenDog/mujoco/OpenDog.xml` is a complete MuJoCo model.
2. The model loads with MuJoCo and exposes 12 joints, 12 actuators, and RoboGauge-compatible proprioceptive/IMU sensors.
3. The MuJoCo joint order matches DMBot training `JOINT_NAMES`.
4. The current RoboGauge simulator assumes `base_link`; DMBot uses `L0_torso`.

## Modification Log

- [x] Created implementation plan at `docs/superpowers/plans/2026-06-17-robogauge-dmbot-adaptation.md`.
- [x] Created this running markdown log before code changes.
- [x] Added failing regression tests in `tests/test_robogauge_dmbot_adaptation.py` for DMBot resources, RoboGauge exports, task registration, simulator base body configuration, and training CLI forwarding.
- [x] Ran the focused regression suite and confirmed the expected red baseline: missing DMBot MuJoCo resources, missing `DMBot` exports/registration, missing configurable simulator base body support, and missing `--robogauge_task` CLI forwarding.
- [x] Added RoboGauge-local DMBot MuJoCo resources under `external/RoboGauge/resources/robots/dmbot/`, copied from `/home/sbt/Downloads/robot_house/DaMiao/OpenDog`, with XML adjusted to `model='dmbot'`, local `assets` meshdir, named foot geoms `FL/FR/RL/RR`, and a 0.35 m root spawn offset.
- [x] Extended the DMBot regression tests to assert the 0.35 m MuJoCo root spawn offset and matching RoboGauge asset spawn-height configuration.
- [x] Added `DMBotConfig`, `DMBotTerrainConfig`, and a thin `DMBot` wrapper that reuses the Go2 RobotLab observation/action implementation with DMBot-specific assets, default joint pose, scaling, and PD gains.
- [x] Exported `DMBot`, `DMBotConfig`, and `DMBotTerrainConfig` from `robogauge.tasks.robots`.
- [x] Updated `MujocoSimulator.load()` to accept a configurable `base_body_name` while preserving `base_link` as the default.
- [x] Updated `BasePipeline.load()` to pass `robot_cfg.assets.base_body_name` into the simulator and imported `DMBot` into the pipeline evaluation scope.
- [x] Registered RoboGauge DMBot tasks: `dmbot.flat`, `dmbot.slope_fd`, `dmbot.slope_bd`, `dmbot.wave`, `dmbot.stairs_fd`, `dmbot.stairs_bd`, and `dmbot.obstacle`.
- [x] Added `--robogauge_task` to the RSL-RL CLI with default `go2_lab`.
- [x] Forwarded `args_cli.robogauge_task` through `scripts/rsl_rl/train.py` into `agent_cfg_dict["robogauge"]["task_name"]`.
- [x] Updated `OnPolicyRunnerCTS` to submit RoboGauge tasks using `robogauge_cfg.get("task_name", "go2_lab")` instead of a hard-coded `go2_lab`.
- [x] Replaced RoboGauge simulator runtime construction/step/reset from `dm_control.mjcf.Physics` to the official `mujoco.MjModel/MjData` API after MJCF composition, fixing the current `mujoco 3.9.0` / `dm_control` named-indexer incompatibility that also affected the existing Go2 path.
- [x] Added a no-op fallback logger for direct RoboGauge component tests or interactive loads before `logger.create()` is called.
- [x] Corrected the simulator-load regression test to assert RoboGauge's attached joint prefix (`dmbot/`) while still validating the underlying DMBot joint order.
- [x] Re-ran `tests/test_robogauge_dmbot_adaptation.py`; result: 5 passed.
- [x] Ran simulator smoke checks for both existing Go2 loading and DMBot one-step loading; both completed under the current environment.
- [x] Verified the loaded `dmbot/L0_torso` world height is 0.35 m after simulator load.
- [x] Ran `py_compile` over the modified RoboGauge, RSL-RL, and regression test Python files; result: passed.
- [x] Removed generated DMBot `__pycache__` files created during syntax verification.
