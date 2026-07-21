"""Compose the ToGo_LFs backflip PPO and CTS locomotion policies in MuJoCo.

The policies remain independent.  A guarded state machine hands absolute joint
targets from the one-shot backflip controller to the continuous locomotion
controller only after a currently-valid, continuously-held stable landing.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import torch

from deploy_togo_lfs_backflip import (
    DEFAULT_CONFIG,
    GamepadTrigger,
    TogoBackflipMujoco,
    load_backflip_config,
    projected_gravity,
)


POLICY_DT_EPSILON = 1.0e-9


class ControlMode(Enum):
    STAND = auto()
    BACKFLIP = auto()
    LOCOMOTION_PRIME = auto()
    LOCOMOTION_BLEND = auto()
    LOCOMOTION_ZERO_HOLD = auto()
    LOCOMOTION = auto()
    PREPARE_BACKFLIP = auto()
    FAILURE_HOLD = auto()


@dataclass
class StabilityGate:
    required_time_s: float
    minimum_contact_feet: int
    minimum_upright: float
    maximum_linear_speed: float
    maximum_angular_speed: float
    maximum_joint_pose_rms: float
    stable_time_s: float = 0.0

    def reset(self) -> None:
        self.stable_time_s = 0.0

    def values(self, simulation: TogoBackflipMujoco) -> dict[str, float | int | bool]:
        feet, nonfoot_contact = simulation._contact_state()
        gravity_b = projected_gravity(simulation.data.qpos[3:7])
        return {
            "feet": int(np.count_nonzero(feet)),
            "nonfoot": bool(nonfoot_contact),
            "upright": -float(gravity_b[2]),
            "linear_speed": float(np.linalg.norm(simulation.data.qvel[:3])),
            "angular_speed": float(np.linalg.norm(simulation.data.qvel[3:6])),
            "joint_pose_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            simulation.data.qpos[7:]
                            - simulation.cfg.default_angles
                        )
                    )
                )
            ),
        }

    def currently_stable(self, simulation: TogoBackflipMujoco) -> bool:
        value = self.values(simulation)
        return bool(
            value["feet"] >= self.minimum_contact_feet
            and not value["nonfoot"]
            and value["upright"] >= self.minimum_upright
            and value["linear_speed"] <= self.maximum_linear_speed
            and value["angular_speed"] <= self.maximum_angular_speed
            and value["joint_pose_rms"] <= self.maximum_joint_pose_rms
        )

    def update(self, simulation: TogoBackflipMujoco, step_dt: float) -> bool:
        if self.currently_stable(simulation):
            self.stable_time_s += step_dt
        else:
            self.stable_time_s = 0.0
        return self.stable_time_s + POLICY_DT_EPSILON >= self.required_time_s


class LocomotionController:
    """45-D CTS TorchScript adapter with guarded history initialization."""

    def __init__(self, cfg: SimpleNamespace, policy: torch.jit.ScriptModule):
        self.cfg = cfg
        self.policy = policy.eval()
        self.action = np.zeros(12, dtype=np.float32)
        self._prime_candidate = self.action.copy()
        if "reset" not in self.policy._c._method_names():
            raise ValueError("The CTS TorchScript policy must export reset().")
        self.reset()
        with torch.inference_mode():
            output = self.policy(
                torch.zeros(
                    (1, cfg.locomotion_num_observations), dtype=torch.float32
                )
            )
        if tuple(output.shape) != (1, 12):
            raise ValueError(
                f"Expected locomotion output (1, 12), got {tuple(output.shape)}."
            )
        self.reset()

    def reset(self) -> None:
        with torch.inference_mode():
            self.policy.reset()
        self.action.fill(0.0)
        self._prime_candidate.fill(0.0)

    def _observation(
        self,
        simulation: TogoBackflipMujoco,
        command: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        ang_vel_b = simulation.data.sensordata[
            simulation.gyro_adr : simulation.gyro_adr + 3
        ]
        observation = np.concatenate(
            (
                ang_vel_b * cfg.angular_velocity_scale,
                projected_gravity(simulation.data.qpos[3:7]),
                command * np.asarray(cfg.locomotion_command_scale),
                simulation.data.qpos[7:] - cfg.default_angles,
                simulation.data.qvel[6:] * cfg.joint_velocity_scale,
                last_action,
            )
        ).astype(np.float32, copy=False)
        expected = (cfg.locomotion_num_observations,)
        if observation.shape != expected:
            raise RuntimeError(
                f"Built locomotion observation {observation.shape}, expected {expected}."
            )
        return observation

    def _infer(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            output = self.policy(torch.from_numpy(observation).unsqueeze(0))
        return output.cpu().numpy().reshape(-1).astype(np.float32, copy=False)

    def prime(self, simulation: TogoBackflipMujoco) -> None:
        """Fill one history slot while keeping the deployed last action zero."""
        observation = self._observation(
            simulation,
            np.zeros(3, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        )
        self._prime_candidate = self._infer(observation)

    def finish_prime(self) -> None:
        self.action = self._prime_candidate.copy()

    def step(
        self,
        simulation: TogoBackflipMujoco,
        command: np.ndarray,
    ) -> np.ndarray:
        observation = self._observation(simulation, command, self.action)
        self.action = self._infer(observation)
        target = (
            self.cfg.default_angles
            + self.cfg.locomotion_action_scale * self.action
        )
        return np.clip(
            target,
            self.cfg.joint_target_lower,
            self.cfg.joint_target_upper,
        )


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def gate_from_config(cfg: SimpleNamespace, prefix: str) -> StabilityGate:
    return StabilityGate(
        required_time_s=float(getattr(cfg, f"{prefix}_required_stable_time")),
        minimum_contact_feet=int(getattr(cfg, f"{prefix}_minimum_contact_feet")),
        minimum_upright=float(getattr(cfg, f"{prefix}_minimum_upright")),
        maximum_linear_speed=float(getattr(cfg, f"{prefix}_maximum_linear_speed")),
        maximum_angular_speed=float(getattr(cfg, f"{prefix}_maximum_angular_speed")),
        maximum_joint_pose_rms=float(
            getattr(cfg, f"{prefix}_maximum_joint_pose_rms")
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose ToGo_LFs backflip PPO and CTS locomotion in MuJoCo."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--backflip-policy-path", type=Path)
    parser.add_argument("--locomotion-policy-path", type=Path)
    parser.add_argument("--xml-path", type=Path)
    parser.add_argument("--stand-time", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--locomotion-test-duration", type=float)
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[TogoBackflipMujoco, ControlMode]:
    cfg = load_backflip_config(args.config)
    if args.backflip_policy_path is not None:
        cfg.policy_path = args.backflip_policy_path.expanduser().resolve()
    if args.locomotion_policy_path is not None:
        cfg.locomotion_policy_path = (
            args.locomotion_policy_path.expanduser().resolve()
        )
    if args.xml_path is not None:
        cfg.xml_path = args.xml_path.expanduser().resolve()
    if args.stand_time is not None:
        cfg.stand_time = args.stand_time
    locomotion_test_duration = (
        float(args.locomotion_test_duration)
        if args.locomotion_test_duration is not None
        else float(cfg.headless_locomotion_test_duration)
    )
    for name, path in (
        ("backflip", cfg.policy_path),
        ("locomotion", cfg.locomotion_policy_path),
        ("MuJoCo scene", cfg.xml_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} path does not exist: {path}")

    backflip_policy = torch.jit.load(str(cfg.policy_path), map_location="cpu")
    locomotion_policy = torch.jit.load(
        str(cfg.locomotion_policy_path), map_location="cpu"
    )
    simulation = TogoBackflipMujoco(cfg, backflip_policy)
    locomotion = LocomotionController(cfg, locomotion_policy)
    handoff_gate = gate_from_config(cfg, "handoff")
    preflip_gate = gate_from_config(cfg, "preflip")

    mode = ControlMode.STAND
    counter = 0
    prime_steps = 0
    blend_elapsed_s = 0.0
    zero_hold_elapsed_s = 0.0
    locomotion_elapsed_s = 0.0
    blend_source = cfg.default_angles.copy()
    command = np.zeros(3, dtype=np.float32)
    trigger_pending = False
    keyboard_triggered = False
    handoff_completed = False
    failure_reason: str | None = None
    headless_complete = False
    auto_start_pending = args.auto_start or args.headless
    gamepad = None if args.headless else GamepadTrigger()
    real_time = bool(cfg.real_time) and not args.no_real_time and not args.headless
    policy_dt = cfg.dt * cfg.decimation
    maximum_command = np.asarray(cfg.locomotion_max_command, dtype=np.float32)

    def key_callback(keycode: int) -> None:
        nonlocal keyboard_triggered
        if keycode == ord(" "):
            keyboard_triggered = True

    def mode_status() -> str:
        return (
            f"mode={mode.name}  sim_t={simulation.data.time:4.2f}s  "
            f"flip_t={simulation.skill_elapsed_s:4.2f}s  "
            f"loco_t={locomotion_elapsed_s:4.2f}s  "
            f"{simulation.diagnostics().split('  ', 1)[1]}  "
            f"cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f})"
        )

    def print_ready() -> None:
        print(
            "Ready: A/Space triggers backflip; after a guarded handoff, "
            "the sticks command locomotion."
        )

    def start_backflip() -> None:
        nonlocal mode, trigger_pending
        handoff_gate.reset()
        simulation.start_skill()
        trigger_pending = False
        mode = ControlMode.BACKFLIP
        print("Backflip started; locomotion command locked to zero.")

    def start_locomotion_prime() -> None:
        nonlocal mode, prime_steps, blend_source, command
        simulation._started = False
        blend_source = simulation.target_pos.copy()
        simulation.target_pos = blend_source.copy()
        command.fill(0.0)
        locomotion.reset()
        prime_steps = 0
        mode = ControlMode.LOCOMOTION_PRIME
        print(
            f"Landing gate held for {handoff_gate.stable_time_s:.2f}s; "
            "priming CTS history."
        )

    if not auto_start_pending:
        print_ready()

    def policy_step() -> None:
        nonlocal mode, prime_steps, blend_elapsed_s, zero_hold_elapsed_s
        nonlocal locomotion_elapsed_s, blend_source, command
        nonlocal handoff_completed, failure_reason, headless_complete
        if mode is ControlMode.BACKFLIP:
            simulation.policy_step()
            if handoff_gate.update(simulation, policy_dt):
                start_locomotion_prime()
            elif simulation.skill_elapsed_s + POLICY_DT_EPSILON >= cfg.duration:
                simulation._started = False
                simulation.target_pos = cfg.default_angles.copy()
                mode = ControlMode.FAILURE_HOLD
                failure_reason = "backflip timed out before the locomotion handoff gate"
                print(f"Handoff failed: {failure_reason}.")
            return

        if mode is ControlMode.LOCOMOTION_PRIME:
            if not handoff_gate.currently_stable(simulation):
                simulation._started = True
                mode = ControlMode.BACKFLIP
                handoff_gate.reset()
                print("CTS priming aborted because the landing became unstable.")
                return
            locomotion.prime(simulation)
            prime_steps += 1
            if prime_steps >= cfg.locomotion_history_prime_steps:
                locomotion.finish_prime()
                blend_source = simulation.target_pos.copy()
                blend_elapsed_s = 0.0
                mode = ControlMode.LOCOMOTION_BLEND
                print("CTS history ready; blending absolute joint targets.")
            return

        if mode is ControlMode.LOCOMOTION_BLEND:
            command.fill(0.0)
            locomotion_target = locomotion.step(simulation, command)
            blend_elapsed_s += policy_dt
            alpha = smoothstep(
                blend_elapsed_s / cfg.locomotion_target_blend_duration
            )
            simulation.target_pos = (
                (1.0 - alpha) * blend_source + alpha * locomotion_target
            )
            if alpha >= 1.0:
                zero_hold_elapsed_s = 0.0
                mode = ControlMode.LOCOMOTION_ZERO_HOLD
                print("Target blend complete; keeping locomotion command at zero.")
            return

        if mode is ControlMode.LOCOMOTION_ZERO_HOLD:
            command.fill(0.0)
            simulation.target_pos = locomotion.step(simulation, command)
            zero_hold_elapsed_s += policy_dt
            if (
                zero_hold_elapsed_s + POLICY_DT_EPSILON
                >= cfg.locomotion_zero_command_hold
            ):
                handoff_completed = True
                locomotion_elapsed_s = 0.0
                mode = ControlMode.LOCOMOTION
                print("Locomotion owns the joints; gamepad velocity command unlocked.")
            return

        if mode is ControlMode.LOCOMOTION:
            if gamepad is None:
                command.fill(0.0)
            else:
                command = gamepad.read_command(maximum_command)
            simulation.target_pos = locomotion.step(simulation, command)
            locomotion_elapsed_s += policy_dt
            if args.headless and locomotion_elapsed_s >= locomotion_test_duration:
                headless_complete = True
            return

        if mode is ControlMode.PREPARE_BACKFLIP:
            command.fill(0.0)
            simulation.target_pos = locomotion.step(simulation, command)
            if preflip_gate.update(simulation, policy_dt):
                start_backflip()

    def advance() -> bool:
        nonlocal counter, mode, trigger_pending
        nonlocal keyboard_triggered, auto_start_pending
        step_start = time.perf_counter()
        gamepad_triggered = gamepad.poll() if gamepad is not None else False
        trigger_requested = keyboard_triggered or gamepad_triggered
        keyboard_triggered = False

        if mode is ControlMode.STAND and trigger_requested:
            trigger_pending = True
        elif mode is ControlMode.LOCOMOTION and trigger_requested:
            trigger_pending = False
            command.fill(0.0)
            preflip_gate.reset()
            mode = ControlMode.PREPARE_BACKFLIP
            print("Backflip requested; waiting for stationary locomotion state.")

        if (
            mode is ControlMode.STAND
            and simulation.data.time >= cfg.stand_time
            and (trigger_pending or auto_start_pending)
        ):
            auto_start_pending = False
            start_backflip()

        simulation.physics_step()
        counter += 1
        render_due = counter % cfg.decimation == 0
        if render_due:
            policy_step()

        if real_time:
            remaining = cfg.dt - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
        return render_due

    if args.headless:
        while not headless_complete and failure_reason is None:
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

    final_gate = handoff_gate.currently_stable(simulation)
    print(f"Final: {mode_status()}  currently_stable={int(final_gate)}")
    if args.require_success:
        if failure_reason is not None:
            raise RuntimeError(failure_reason)
        if not handoff_completed:
            raise RuntimeError("Locomotion handoff was not completed.")
        if not final_gate:
            raise RuntimeError("Locomotion did not retain a currently stable stance.")
    return simulation, mode


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
