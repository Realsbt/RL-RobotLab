# ToGo LFS Locomotion

This task trains the Xtellar ToGo LFS quadruped with the MoE-CTS locomotion
policy used by RobotLab. The robot definition includes the corrected joint
convention, 268 rpm joint-speed limit, motor torque response randomization,
control delay, and persistent IMU calibration errors.

## Train

Run from the repository root:

```bash
python scripts/rsl_rl/train.py \
  --task RobotLab-ToGo-LFs-v0 \
  --num_envs 4096 \
  --headless \
  --logger wandb \
  --log_project_name togo_lfs_moe_cts
```

`--max_iterations` is the number of additional iterations when resuming. For
example, to continue a checkpoint at iteration 30500 to approximately 100000:

```bash
python scripts/rsl_rl/train.py \
  --task RobotLab-ToGo-LFs-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run <run-directory> \
  --checkpoint model_30500.pt \
  --max_iterations 69500
```

## Export

```bash
python scripts/rsl_rl/play.py \
  --task RobotLab-ToGo-LFs-v0 \
  --checkpoint <checkpoint-path> \
  --num_envs 4 \
  --headless \
  --export-only \
  --export-name togo_lfs_locomotion
```

## MuJoCo

Pass the exported TorchScript policy explicitly:

```bash
python deploy/deploy_mujoco/deploy_togo_lfs.py \
  --policy-path <exported-policy.pt>
```

Use `--latest-exported-policy` only when the repository contains local training
logs and selecting the newest export is intentional.

## Speed Benchmark

```bash
python scripts/rsl_rl/benchmark_terrain_speed.py \
  --checkpoint <checkpoint-path> \
  --terrain flat \
  --command-speed 2.0 \
  --num_envs 128 \
  --headless
```
