from isaaclab.utils import configclass

import robot_lab.tasks.togo_lfs_quiet_impact.mdp as mdp
from robot_lab.tasks.togo_lfs_quiet.env_cfg import (
    ToGoLFsQuietCommandsCfg,
    ToGoLFsQuietEnvCfg,
    ToGoLFsQuietSceneCfg,
)


@configclass
class ToGoLFsQuietImpactSceneCfg(ToGoLFsQuietSceneCfg):
    """Quiet scene with a dedicated 200 Hz foot impact history sensor."""

    quiet_impact_sensor = mdp.FootImpactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=20,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )


@configclass
class ToGoLFsQuietImpactCommandsCfg(ToGoLFsQuietCommandsCfg):
    """Low-speed commands with event-based impact diagnostics."""

    base_velocity = mdp.ImpactEventVelocityCommandCfg(
        asset_name="robot",
        impact_sensor_name="quiet_impact_sensor",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.15,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.ImpactEventVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.4, 0.4),
        ),
    )


@configclass
class ToGoLFsQuietImpactEnvCfg(ToGoLFsQuietEnvCfg):
    """MUTE baseline plus event-only touchdown diagnostics and visualization."""

    scene: ToGoLFsQuietImpactSceneCfg = ToGoLFsQuietImpactSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: ToGoLFsQuietImpactCommandsCfg = ToGoLFsQuietImpactCommandsCfg()
