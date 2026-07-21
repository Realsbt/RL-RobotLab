"""Run CTS ToGo_LFs policy deployment in MuJoCo with joystick command input."""

import argparse
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from dmbot_scene import build_scene_with_training_terrain_xml
from utils import (
    build_delay_buffers,
    display_current_command,
    gravity_from_quat,
    init_joystick,
    load_config,
    open_video_writer,
    pd_control,
    read_joystick_command,
    sample_delayed_targets,
    set_initial_state,
    setup_tracking_camera,
    MujocoRenderUtils,
)


CONFIG_NAME = "togo_lfs.yaml"
VIDEO_DIR = Path(__file__).with_name("videos")
ACTUATOR_GROUPS = (
    np.array([0, 3, 6, 9], dtype=np.int64),
    np.array([1, 4, 7, 10], dtype=np.int64),
    np.array([2, 5, 8, 11], dtype=np.int64),
)


def build_features(data, action: np.ndarray, cmd: np.ndarray, cfg):
    """Build the current single-frame feature dictionary for policy inference."""
    joint_pos = (data.qpos[7:] - cfg.default_angles) * cfg.dof_pos_scale
    joint_vel = data.qvel[6:] * cfg.dof_vel_scale
    return {
        "ang_vel": data.qvel[3:6] * cfg.ang_vel_scale,
        "gravity": gravity_from_quat(data.qpos[3:7]),
        "cmd": cmd * cfg.cmd_scale,
        "joint_pos": joint_pos[cfg.idx_mj2model],
        "joint_vel": joint_vel[cfg.idx_mj2model],
        "last_action": action[cfg.idx_mj2model],
    }


def action_to_target(action: np.ndarray, cfg):
    """Convert normalized action output to joint position targets."""
    return cfg.default_angles + action * cfg.action_pos_scale


def build_single_obs(features: dict[str, np.ndarray], layout: list[tuple[str, int]]) -> np.ndarray:
    """Flatten current feature groups into one single observation vector."""
    return np.concatenate([features[name] for name, _ in layout], axis=0).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a ToGo_LFs TorchScript policy in MuJoCo.")
    parser.add_argument(
        "--config",
        default=CONFIG_NAME,
        help="Config file under deploy/deploy_mujoco/configs (default: togo_lfs.yaml).",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Optional TorchScript policy path. Overrides policy_path in configs/togo_lfs.yaml.",
    )
    parser.add_argument(
        "--latest-exported-policy",
        action="store_true",
        help="Use the most recently exported logs/rsl_rl/**/exported/policy.pt policy.",
    )
    return parser.parse_args()


def main() -> None:
    """Run MuJoCo simulation and deploy the ToGo_LFs CTS policy in closed-loop control."""
    args = parse_args()
    if args.policy_path is not None and args.latest_exported_policy:
        raise ValueError("Use either --policy-path or --latest-exported-policy, not both.")
    cfg = load_config(args.config)
    if args.policy_path is not None:
        cfg.policy_path = args.policy_path.expanduser().resolve()
    elif args.latest_exported_policy:
        project_root = Path(__file__).resolve().parents[2]
        candidates = list((project_root / "logs" / "rsl_rl").glob("**/exported/policy.pt"))
        if not candidates:
            raise FileNotFoundError("No exported policy found under logs/rsl_rl.")
        cfg.policy_path = max(candidates, key=lambda path: path.stat().st_mtime)
        print(f"Using latest exported policy: {cfg.policy_path}")
    if cfg.policy_path is None:
        raise ValueError(
            "The default ToGo_LFs asset now uses the new hip-pitch convention. "
            "Pass --policy-path with a policy trained on ToGo_LFs_v0p1_new."
        )
    layout = [
        ("ang_vel", 3),
        ("gravity", 3),
        ("cmd", 3),
        ("joint_pos", cfg.num_actions),
        ("joint_vel", cfg.num_actions),
        ("last_action", cfg.num_actions),
    ]
    joystick = init_joystick()
    cmd = cfg.cmd_init.copy()
    display_current_command(cmd)

    if cfg.terrain_profile == "flat":
        scene_xml = cfg.xml_path
    else:
        scene_xml = Path(tempfile.gettempdir()) / "robotlab_togo_lfs_mujoco_scene.xml"
        build_scene_with_training_terrain_xml(cfg.xml_path, scene_xml, terrain_profile=cfg.terrain_profile)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.dt
    set_initial_state(data, cfg.base_init_pos, cfg.base_init_quat, cfg.default_angles)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=360, width=640)
    policy = torch.jit.load(str(cfg.policy_path))
    writer, frame_skip, video_path = open_video_writer(
        cfg.save_video, policy_path=cfg.policy_path, cmd=cmd, dt=cfg.dt, video_dir=VIDEO_DIR, video_fps=cfg.video_fps
    )

    action = np.zeros(cfg.num_actions, dtype=np.float32)
    target_pos = cfg.default_angles.copy()
    target_vel = np.zeros(cfg.num_actions, dtype=np.float32)
    pos_history = build_delay_buffers(target_pos, delay_max=cfg.delay_max)
    delay_rng = np.random.default_rng(cfg.delay_seed)
    render_substeps = max(1, int((1.0 / cfg.render_fps) / cfg.dt))
    mujoco_render_utils = MujocoRenderUtils()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "F_base")
        setup_tracking_camera(viewer, trackbodyid=base_body_id, distance=3.0)
        start_time = time.time()
        counter = 0

        while viewer.is_running() and time.time() - start_time < cfg.duration:
            step_start = time.time()
            if joystick and counter % cfg.decimation == 0:
                cmd = read_joystick_command(joystick, cfg.max_cmd)

            data.ctrl[:] = pd_control(target_pos, data.qpos[7:], cfg.kps, target_vel, data.qvel[6:], cfg.kds)
            mujoco.mj_step(model, data)
            mujoco_render_utils.update(cmd, data)

            if writer and counter % frame_skip == 0:
                try:
                    renderer.update_scene(data, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype="renderer")
                    writer.append_data(renderer.render())
                except Exception as exc:
                    print(f"Render error: {exc}")

            counter += 1
            if counter % cfg.decimation == 0:
                features = build_features(data, action, cmd, cfg)
                single_obs = build_single_obs(features, layout)
                action_tensor = policy(torch.from_numpy(single_obs).unsqueeze(0))
                action = action_tensor.detach().cpu().numpy().squeeze()[cfg.idx_model2mj]
                pos_history.append(action_to_target(action, cfg).copy())
                target_pos = sample_delayed_targets(pos_history, ACTUATOR_GROUPS, cfg.delay_min, cfg.delay_max, delay_rng)
                display_current_command(cmd)

            if counter % render_substeps == 0:
                mujoco_render_utils.update_external_rendering(viewer, ctype="viewer")
                viewer.sync()
            # Keep the physics loop close to real time.  The previous fixed
            # 0.1 s offset made this negative for the 5 ms simulation step,
            # so the loop never slept and ran as fast as the host allowed.
            sleep_time = cfg.dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print()
    if writer:
        writer.close()
        print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()
