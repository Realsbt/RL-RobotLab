"""Run the ordinary-PPO ToGo_LFs backflip in the togo-master MuJoCo model.

This runner is intentionally separate from ``deploy_togo_lfs.py``: the latter
implements the 45-D/history CTS locomotion contract, while this skill uses one
60-D frame and a deterministic two-second phase clock.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_CONFIG = "togo_lfs_backflip.yaml"


class GamepadTrigger:
    """Detect a rising edge from Xbox A or raw joystick button zero."""

    def __init__(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        from pygame._sdl2 import controller as gamecontroller

        self.pygame = pygame
        self.gamecontroller = gamecontroller
        pygame.init()
        gamecontroller.init()
        self.device = None
        self.standardized = False
        self.was_pressed = False

        if pygame.joystick.get_count() == 0:
            print("No gamepad detected; Space in the MuJoCo window can still trigger the backflip.")
            return
        if gamecontroller.is_controller(0):
            self.device = gamecontroller.Controller(0)
            self.standardized = True
            print(f"Gamepad detected: {self.device.name} (trigger: A)")
        else:
            self.device = pygame.joystick.Joystick(0)
            self.device.init()
            print(f"Joystick detected: {self.device.get_name()} (trigger: button 0)")

    def poll(self) -> bool:
        """Return true once when the trigger changes from released to pressed."""
        self.pygame.event.pump()
        if self.device is None:
            return False
        if self.standardized:
            pressed = bool(
                self.device.get_button(self.pygame.CONTROLLER_BUTTON_A)
            )
        else:
            pressed = bool(self.device.get_button(0))
        rising_edge = pressed and not self.was_pressed
        self.was_pressed = pressed
        return rising_edge

    def read_command(self, maximum: np.ndarray) -> np.ndarray:
        """Read planar velocity and yaw using the existing ToGo stick map."""
        self.pygame.event.pump()
        if self.device is None:
            return np.zeros(3, dtype=np.float32)
        if self.standardized:
            axes = np.array(
                [
                    self.device.get_axis(self.pygame.CONTROLLER_AXIS_LEFTX),
                    self.device.get_axis(self.pygame.CONTROLLER_AXIS_LEFTY),
                    self.device.get_axis(self.pygame.CONTROLLER_AXIS_RIGHTX),
                ],
                dtype=np.float32,
            ) / 32768.0
        else:
            axes = np.array(
                [self.device.get_axis(index) for index in (0, 1, 3)],
                dtype=np.float32,
            )
        axes[np.abs(axes) < 0.1] = 0.0
        return np.array(
            [-axes[1] * maximum[0], -axes[0] * maximum[1], -axes[2] * maximum[2]],
            dtype=np.float32,
        )


def _array(raw: dict, name: str, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(raw[name], dtype=dtype)


def load_backflip_config(config_name: str) -> SimpleNamespace:
    config_path = Path(config_name)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = CONFIG_DIR / config_name
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    def resolved_path(name: str) -> Path:
        value = str(raw[name]).replace("{ROOT_DIR}", str(ROOT_DIR))
        return Path(value).expanduser().resolve()

    cfg = {
        **raw,
        "policy_path": resolved_path("policy_path"),
        "xml_path": resolved_path("xml_path"),
        "dt": float(raw["simulation_dt"]),
        "decimation": int(raw["control_decimation"]),
        "duration": float(raw["skill_duration"]),
        "stand_time": float(raw.get("stand_time", 0.0)),
    }
    if raw.get("locomotion_policy_path") is not None:
        cfg["locomotion_policy_path"] = resolved_path("locomotion_policy_path")
    for name in (
        "base_init_pos",
        "base_init_quat",
        "default_angles",
        "kps",
        "kds",
        "joint_target_lower",
        "joint_target_upper",
        "front_target_offsets",
        "post_touchdown_target_offsets",
    ):
        cfg[name] = _array(raw, name)
    return SimpleNamespace(**cfg)


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return world gravity direction expressed in the floating-base frame."""
    qw, qx, qy, qz = quaternion_wxyz
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def phase_features(
    elapsed_s: float,
    episode_length_s: float = 2.0,
    phase_cycles: float = 0.5,
) -> np.ndarray:
    normalized_time = np.clip(elapsed_s / episode_length_s, 0.0, 1.0)
    phase = 2.0 * math.pi * phase_cycles * normalized_time
    return np.asarray(
        [
            math.sin(phase),
            math.cos(phase),
            math.sin(phase / 2.0),
            math.cos(phase / 2.0),
            math.sin(phase / 4.0),
            math.cos(phase / 4.0),
        ],
        dtype=np.float32,
    )


def torque_speed_pd(
    target_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    cfg: SimpleNamespace,
) -> np.ndarray:
    """Match RobotLab's four-quadrant PhysicalMotor torque clipping."""
    effort = cfg.kps * (target_pos - joint_pos) - cfg.kds * joint_vel
    velocity = np.clip(
        joint_vel,
        -cfg.velocity_limit * (1.0 + cfg.effort_limit / cfg.saturation_effort),
        cfg.velocity_limit * (1.0 + cfg.effort_limit / cfg.saturation_effort),
    )
    maximum = np.minimum(
        cfg.saturation_effort * (1.0 - velocity / cfg.velocity_limit),
        cfg.effort_limit,
    )
    minimum = np.maximum(
        cfg.saturation_effort * (-1.0 - velocity / cfg.velocity_limit),
        -cfg.effort_limit,
    )
    return np.clip(effort, minimum, maximum)


@dataclass
class BackflipState:
    """Deployable subset of the Isaac Lab phase command's landing state."""

    has_support: bool = False
    has_taken_off: bool = False
    was_airborne: bool = False
    has_touched_down: bool = False
    has_landed: bool = False
    invalid_rotation_axis: bool = False
    signed_rotation: float = 0.0
    maximum_rotation: float = 0.0
    wrapped_pitch: float = 0.0
    touchdown_time_s: float = 0.0
    takeoff_time_s: float = 0.0
    stable_support_time_s: float = 0.0
    landing_success: bool = False

    def update(
        self,
        *,
        elapsed_s: float,
        gravity_b: np.ndarray,
        base_height: float,
        upward_speed: float,
        base_linear_speed: float,
        base_angular_speed: float,
        step_dt: float,
        foot_contacts: np.ndarray,
        nonfoot_contact: bool,
        minimum_landing_rotation: float,
    ) -> None:
        contact_count = int(np.count_nonzero(foot_contacts))
        any_foot_contact = contact_count > 0
        any_robot_contact = any_foot_contact or nonfoot_contact
        airborne = not any_robot_contact
        self.has_support = self.has_support or contact_count >= 2

        new_takeoff = (
            airborne
            and self.has_support
            and elapsed_s >= 0.20
            and base_height >= 0.33
            and upward_speed >= 0.25
            and not self.has_taken_off
        )
        if new_takeoff:
            self.has_taken_off = True
            self.takeoff_time_s = elapsed_s

        wrapped_pitch = -math.atan2(float(gravity_b[0]), -float(gravity_b[2]))
        raw_delta = wrapped_pitch - self.wrapped_pitch
        wrapped_delta = math.atan2(math.sin(raw_delta), math.cos(raw_delta))
        xz_norm = math.hypot(float(gravity_b[0]), float(gravity_b[2]))
        rotation_window_open = self.has_support and elapsed_s >= 0.20
        if rotation_window_open and not self.has_touched_down and xz_norm < math.cos(math.radians(30.0)):
            self.invalid_rotation_axis = True
        if rotation_window_open and not self.has_touched_down and not self.invalid_rotation_axis:
            self.signed_rotation += wrapped_delta
            self.maximum_rotation = max(self.maximum_rotation, self.signed_rotation)
        self.wrapped_pitch = wrapped_pitch

        touchdown_now = self.has_taken_off and self.was_airborne and any_robot_contact
        first_touchdown = touchdown_now and not self.has_touched_down
        if first_touchdown:
            self.touchdown_time_s = elapsed_s
            self.has_landed = (
                any_foot_contact
                and not nonfoot_contact
                and not self.invalid_rotation_axis
                and self.maximum_rotation >= minimum_landing_rotation
            )
        self.has_touched_down = self.has_touched_down or touchdown_now
        self.was_airborne = airborne

        stable_now = (
            self.has_landed
            and not nonfoot_contact
            and contact_count >= 3
            and -float(gravity_b[2]) >= math.cos(math.radians(20.0))
            and base_linear_speed <= 0.75
            and base_angular_speed <= 2.0
        )
        self.stable_support_time_s = (
            self.stable_support_time_s + step_dt if stable_now else 0.0
        )
        self.landing_success = self.landing_success or self.stable_support_time_s >= 0.20


class TogoBackflipMujoco:
    def __init__(self, cfg: SimpleNamespace, policy: torch.jit.ScriptModule):
        self.cfg = cfg
        self.policy = policy.eval()
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.dt
        self._resolve_model_contract()
        self._configure_contacts()
        self.reset()

    def _resolve_model_contract(self) -> None:
        if self.model.nu != self.cfg.num_actions:
            raise ValueError(f"Expected {self.cfg.num_actions} actuators, MuJoCo model has {self.model.nu}.")
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(1, self.model.njnt)
        ]
        if joint_names != list(self.cfg.joint_names):
            raise ValueError(f"MuJoCo joint order does not match the policy: {joint_names}")
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "F_base"
        )
        self.foot_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name): index
            for index, name in enumerate(self.cfg.foot_body_names)
        }
        gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        self.gyro_adr = int(self.model.sensor_adr[gyro_id])

        probe = torch.zeros((1, self.cfg.num_observations), dtype=torch.float32)
        with torch.inference_mode():
            output = self.policy(probe)
        if tuple(output.shape) != (1, self.cfg.num_actions):
            raise ValueError(
                f"Expected policy output (1, {self.cfg.num_actions}), got {tuple(output.shape)}."
            )

    def _configure_contacts(self) -> None:
        contact_geoms = self.model.geom_contype != 0
        self.model.geom_friction[contact_geoms, 0] = self.cfg.contact_friction
        if not self.cfg.disable_self_collisions:
            return
        # Separate floor and robot collision bits.  This retains floor-robot
        # contacts while preventing every robot-robot collision, matching the
        # Isaac Lab asset's enabled_self_collisions=False setting.
        self.model.geom_contype[self.floor_geom_id] = 1
        self.model.geom_conaffinity[self.floor_geom_id] = 2
        robot_geom_ids = np.flatnonzero((self.model.geom_bodyid != 0) & contact_geoms)
        self.model.geom_contype[robot_geom_ids] = 2
        self.model.geom_conaffinity[robot_geom_ids] = 1

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.cfg.base_init_pos
        self.data.qpos[3:7] = self.cfg.base_init_quat
        self.data.qpos[7:] = self.cfg.default_angles
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.state = BackflipState()
        self.current_action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.previous_action = np.zeros_like(self.current_action)
        self.target_pos = self.cfg.default_angles.copy()
        self.skill_elapsed_s = 0.0
        self._started = False

    def _contact_state(self) -> tuple[np.ndarray, bool]:
        foot_contacts = np.zeros(4, dtype=bool)
        nonfoot_contact = False
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if self.floor_geom_id not in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_id, contact_force)
            if np.linalg.norm(contact_force[:3]) < self.cfg.contact_force_threshold:
                continue
            robot_geom = contact.geom2 if contact.geom1 == self.floor_geom_id else contact.geom1
            body_id = int(self.model.geom_bodyid[robot_geom])
            foot_index = self.foot_body_ids.get(body_id)
            if foot_index is None:
                nonfoot_contact = True
            else:
                foot_contacts[foot_index] = True
        return foot_contacts, nonfoot_contact

    def _observation(self) -> np.ndarray:
        ang_vel_b = self.data.sensordata[self.gyro_adr : self.gyro_adr + 3]
        joint_pos_rel = self.data.qpos[7:] - self.cfg.default_angles
        joint_vel = self.data.qvel[6:]
        obs = np.concatenate(
            (
                ang_vel_b * self.cfg.angular_velocity_scale,
                projected_gravity(self.data.qpos[3:7]),
                joint_pos_rel,
                joint_vel * self.cfg.joint_velocity_scale,
                self.current_action,
                self.previous_action,
                phase_features(
                    self.skill_elapsed_s,
                    self.cfg.phase_episode_length,
                    self.cfg.phase_cycles,
                ),
            )
        ).astype(np.float32, copy=False)
        if obs.shape != (self.cfg.num_observations,):
            raise RuntimeError(f"Built observation shape {obs.shape}, expected ({self.cfg.num_observations},).")
        return obs

    def _processed_target(self, action: np.ndarray) -> np.ndarray:
        target = np.clip(
            self.cfg.default_angles + self.cfg.action_scale * action,
            self.cfg.joint_target_lower,
            self.cfg.joint_target_upper,
        )
        if not self.cfg.assisted_landing:
            return target

        rotation_span = (
            self.cfg.full_front_blend_rotation
            - self.cfg.minimum_front_blend_rotation
        )
        front_blend = np.clip(
            (self.state.signed_rotation - self.cfg.minimum_front_blend_rotation)
            / rotation_span,
            0.0,
            1.0,
        )
        descending = self.data.qvel[2] < 0.0
        front_blend *= float(self.state.has_taken_off and descending)
        if self.state.has_landed:
            front_blend = 1.0
        front_target = self.cfg.default_angles[:6] + self.cfg.front_target_offsets
        target[:6] = (1.0 - front_blend) * target[:6] + front_blend * front_target

        if self.state.has_landed:
            support_blend = np.clip(
                (self.skill_elapsed_s - self.state.touchdown_time_s)
                / self.cfg.post_touchdown_blend_duration,
                0.0,
                1.0,
            )
            support_target = self.cfg.default_angles + self.cfg.post_touchdown_target_offsets
            target = (1.0 - support_blend) * target + support_blend * support_target
        return target

    def start_skill(self) -> None:
        self.skill_elapsed_s = 0.0
        self.state = BackflipState()
        self.current_action.fill(0.0)
        self.previous_action.fill(0.0)
        self._started = True
        self._infer_action()

    def _infer_action(self) -> None:
        observation = torch.from_numpy(self._observation()).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy(observation).cpu().numpy().reshape(-1)
        self.previous_action = self.current_action.copy()
        self.current_action = action.astype(np.float32, copy=False)
        self.target_pos = self._processed_target(self.current_action)

    def physics_step(self) -> None:
        self.data.ctrl[:] = torque_speed_pd(
            self.target_pos,
            self.data.qpos[7:],
            self.data.qvel[6:],
            self.cfg,
        )
        mujoco.mj_step(self.model, self.data)

    def policy_step(self) -> None:
        self.skill_elapsed_s += self.cfg.dt * self.cfg.decimation
        foot_contacts, nonfoot_contact = self._contact_state()
        gravity_b = projected_gravity(self.data.qpos[3:7])
        self.state.update(
            elapsed_s=self.skill_elapsed_s,
            gravity_b=gravity_b,
            base_height=float(self.data.qpos[2]),
            upward_speed=float(self.data.qvel[2]),
            base_linear_speed=float(np.linalg.norm(self.data.qvel[:3])),
            base_angular_speed=float(np.linalg.norm(self.data.qvel[3:6])),
            step_dt=self.cfg.dt * self.cfg.decimation,
            foot_contacts=foot_contacts,
            nonfoot_contact=nonfoot_contact,
            minimum_landing_rotation=self.cfg.minimum_landing_rotation,
        )
        self._infer_action()

    def diagnostics(self) -> str:
        foot_contacts, nonfoot_contact = self._contact_state()
        upright = -float(projected_gravity(self.data.qpos[3:7])[2])
        return (
            f"t={self.skill_elapsed_s:4.2f}s  z={self.data.qpos[2]:+.3f}m  "
            f"rotation={self.state.maximum_rotation / math.pi:4.2f}pi  "
            f"upright={upright:.3f}  feet={np.count_nonzero(foot_contacts)}  "
            f"nonfoot={int(nonfoot_contact)}  success={int(self.state.landing_success)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the ToGo_LFs ordinary-PPO backflip in the togo-master MuJoCo model."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--xml-path", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--stand-time", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start after --stand-time instead of waiting for controller A or Space.",
    )
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--no-assisted-landing", action="store_true")
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit with an error unless the robot completes and holds a stable landing.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> TogoBackflipMujoco:
    cfg = load_backflip_config(args.config)
    if args.policy_path is not None:
        cfg.policy_path = args.policy_path.expanduser().resolve()
    if args.xml_path is not None:
        cfg.xml_path = args.xml_path.expanduser().resolve()
    if args.duration is not None:
        cfg.duration = args.duration
    if args.stand_time is not None:
        cfg.stand_time = args.stand_time
    if args.no_assisted_landing:
        cfg.assisted_landing = False
    real_time = bool(cfg.real_time) and not args.no_real_time and not args.headless

    if not cfg.policy_path.is_file():
        raise FileNotFoundError(f"TorchScript policy does not exist: {cfg.policy_path}")
    if not cfg.xml_path.is_file():
        raise FileNotFoundError(
            f"togo-master MuJoCo scene does not exist: {cfg.xml_path}\n"
            "Pass --xml-path with ToGo_LFs_v0p1_new/simenv/mujoco/empty.xml."
        )
    policy = torch.jit.load(str(cfg.policy_path), map_location="cpu")
    simulation = TogoBackflipMujoco(cfg, policy)
    counter = 0
    skill_counter = 0
    last_report = -1
    completed_once = False
    auto_start_pending = args.auto_start or args.headless
    keyboard_triggered = False
    trigger_pending = False
    gamepad = None if args.headless else GamepadTrigger()

    def key_callback(keycode: int) -> None:
        nonlocal keyboard_triggered
        if keycode == ord(" "):
            keyboard_triggered = True

    def print_ready() -> None:
        print("Ready: press controller A or Space in the MuJoCo window to backflip.")

    if not auto_start_pending:
        print_ready()

    def advance() -> bool:
        nonlocal counter, skill_counter, last_report
        nonlocal completed_once, auto_start_pending, keyboard_triggered, trigger_pending
        step_start = time.perf_counter()
        gamepad_triggered = gamepad.poll() if gamepad is not None else False
        trigger_requested = keyboard_triggered or gamepad_triggered
        keyboard_triggered = False
        if not simulation._started and trigger_requested:
            trigger_pending = True
        armed = simulation.data.time >= cfg.stand_time
        should_start = (
            not simulation._started
            and armed
            and (trigger_pending or auto_start_pending)
        )
        if should_start:
            simulation.start_skill()
            skill_counter = 0
            auto_start_pending = False
            trigger_pending = False
            last_report = -1
            print("Backflip policy started.")
        simulation.physics_step()
        counter += 1
        policy_updated = False
        if simulation._started:
            skill_counter += 1
        if simulation._started and skill_counter % cfg.decimation == 0:
            simulation.policy_step()
            policy_updated = True
            report_index = int(simulation.skill_elapsed_s * 10.0)
            if report_index != last_report:
                print(f"\r{simulation.diagnostics()}", end="", flush=True)
                last_report = report_index
            if simulation.skill_elapsed_s + 1.0e-9 >= cfg.duration:
                completed_once = True
                simulation._started = False
                simulation.target_pos = cfg.default_angles.copy()
                print(f"\nCompleted: {simulation.diagnostics()}")
                if not args.headless:
                    print_ready()
        if real_time:
            remaining = cfg.dt - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
        # Keep the passive viewer responsive while waiting in the standing
        # pose too; there are no policy updates before a manual trigger.
        return policy_updated or counter % cfg.decimation == 0

    if args.headless:
        while not completed_once:
            advance()
    else:
        with mujoco.viewer.launch_passive(
            simulation.model,
            simulation.data,
            key_callback=key_callback,
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = simulation.base_body_id
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 90.0
            while viewer.is_running():
                if advance():
                    viewer.sync()
    print(f"\nFinal: {simulation.diagnostics()}")
    if args.require_success and not simulation.state.landing_success:
        raise RuntimeError("MuJoCo backflip did not satisfy the stable-landing criterion.")
    return simulation


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
