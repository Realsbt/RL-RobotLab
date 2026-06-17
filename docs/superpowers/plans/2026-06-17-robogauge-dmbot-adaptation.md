# RoboGauge DMBot Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DMBot as a first-class RoboGauge robot using the existing OpenDog MuJoCo asset and make training submit DMBot evaluations via `--robogauge_task`.

**Architecture:** Keep the OpenDog MuJoCo model under RoboGauge resources and add a small DMBot robot class/config that mirrors the existing Go2 observation/action contract. Make the simulator base body configurable so Go2 remains unchanged while DMBot can use `L0_torso`.

**Tech Stack:** Python, pytest, MuJoCo Python bindings, RoboGauge editable source, IsaacLab/RSL-RL training scripts.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/test_robogauge_dmbot_adaptation.py`
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Write failing tests for DMBot resource presence, MuJoCo XML loadability, RoboGauge task registration, configurable simulator base body, and training CLI forwarding.
- [ ] Run the focused test file with `/home/sbt/miniforge3/envs/env_isaaclab/bin/python -m pytest tests/test_robogauge_dmbot_adaptation.py -q` and verify failures are due to missing DMBot integration.

### Task 2: DMBot Resources

**Files:**
- Create: `external/RoboGauge/resources/robots/dmbot/dmbot.xml`
- Create: `external/RoboGauge/resources/robots/dmbot/assets/*.stl`
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Copy `/home/sbt/Downloads/robot_house/DaMiao/OpenDog/mujoco/OpenDog.xml` into RoboGauge resources.
- [ ] Copy `/home/sbt/Downloads/robot_house/DaMiao/OpenDog/meshes/*.stl` into RoboGauge DMBot assets.
- [ ] Adjust copied XML to use `model="dmbot"` and `meshdir="assets"`.

### Task 3: RoboGauge DMBot Robot

**Files:**
- Create: `external/RoboGauge/robogauge/tasks/robots/dmbot/dmbot_config.py`
- Create: `external/RoboGauge/robogauge/tasks/robots/dmbot/dmbot.py`
- Create: `external/RoboGauge/robogauge/tasks/robots/dmbot/__init__.py`
- Modify: `external/RoboGauge/robogauge/tasks/robots/__init__.py`
- Modify: `external/RoboGauge/robogauge/tasks/pipeline/base_pipeline.py`
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Add DMBot config with `robot_xml`, `base_body_name`, joint defaults, gains, command scaling, and foot geom names.
- [ ] Add DMBot robot class by reusing Go2 observation/action behavior where the 45D single-observation contract is identical.
- [ ] Export DMBot classes through RoboGauge robot package imports.
- [ ] Update BasePipeline imports so its existing `eval(robot_cfg.robot_class)` can instantiate DMBot.

### Task 4: Configurable Simulator Base Body

**Files:**
- Modify: `external/RoboGauge/robogauge/tasks/simulator/mujoco_simulator.py`
- Modify: `external/RoboGauge/robogauge/tasks/pipeline/base_pipeline.py`
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Extend `MujocoSimulator.load()` with `base_body_name="base_link"`.
- [ ] Use `base_body_name` instead of the hard-coded `base_link` for base relocation, mass randomization, and camera tracking.
- [ ] Pass `robot_cfg.assets.base_body_name` from BasePipeline while keeping Go2 default behavior unchanged.

### Task 5: Task Registration

**Files:**
- Modify: `external/RoboGauge/robogauge/tasks/__init__.py`
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Register `dmbot.flat`, `dmbot.slope_fd`, `dmbot.slope_bd`, `dmbot.wave`, `dmbot.stairs_fd`, `dmbot.stairs_bd`, and `dmbot.obstacle`.

### Task 6: Training CLI Forwarding

**Files:**
- Modify: `scripts/rsl_rl/cli_args.py`
- Modify: `scripts/rsl_rl/train.py`
- Modify: `source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py`
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Add `--robogauge_task` CLI argument with default `go2_lab`.
- [ ] Include task name in the runner `robogauge` config.
- [ ] Submit that configured task name instead of the current hard-coded `go2_lab`.

### Task 7: Verification

**Files:**
- Track: `docs/robogauge-dmbot-adaptation-log.md`

- [ ] Run the focused DMBot adaptation tests.
- [ ] Run existing DMBot/CLI-related tests that cover training arguments and task config.
- [ ] Run a lightweight import/registration smoke test under `env_isaaclab`.
