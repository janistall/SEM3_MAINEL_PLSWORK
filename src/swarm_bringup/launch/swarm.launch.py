from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from swarm_bringup.launch_plan import build_launch_plan, parse_bool


def _package_root() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("swarm_bringup"))
    except Exception:
        return Path(__file__).resolve().parent.parent


def _launch_setup(context, *args, **kwargs):
    package_root = _package_root()

    px4_dir = Path(
        LaunchConfiguration("px4_dir").perform(context)
        or os.environ.get("PX4_DIR", "/home/siddharthg/PX4-Autopilot")
    ).expanduser()
    px4_build = px4_dir / "build" / "px4_sitl_default"
    px4_binary = px4_build / "bin" / "px4"
    if not px4_binary.is_file():
        raise FileNotFoundError(f"PX4 SITL binary not found: {px4_binary}")

    allow_legacy = parse_bool(LaunchConfiguration("allow_legacy_checkpoint").perform(context))
    enable_shield = parse_bool(LaunchConfiguration("enable_shield").perform(context))
    use_supervisor = parse_bool(LaunchConfiguration("use_supervisor").perform(context))
    auto_arm = parse_bool(LaunchConfiguration("auto_arm").perform(context))
    headless = parse_bool(LaunchConfiguration("headless").perform(context))
    rain = parse_bool(LaunchConfiguration("rain").perform(context))
    wind = (
        float(LaunchConfiguration("wind_x").perform(context)),
        float(LaunchConfiguration("wind_y").perform(context)),
        float(LaunchConfiguration("wind_z").perform(context)),
    )

    plan = build_launch_plan(
        package_root,
        scenario_arg=LaunchConfiguration("scenario").perform(context),
        world_arg=LaunchConfiguration("world").perform(context),
        drones_arg=LaunchConfiguration("drones").perform(context),
        mode=LaunchConfiguration("mode").perform(context),
        checkpoint_arg=LaunchConfiguration("checkpoint").perform(context),
        allow_legacy=allow_legacy,
        use_supervisor=use_supervisor,
        quorum_min_peers=int(LaunchConfiguration("quorum_min_peers").perform(context)),
        control_rate=float(LaunchConfiguration("control_rate").perform(context)),
        telemetry_rate=float(LaunchConfiguration("telemetry_rate").perform(context)),
        takeoff_altitude_arg=LaunchConfiguration("takeoff_altitude").perform(context),
        px4_stagger=float(LaunchConfiguration("px4_stagger").perform(context)),
        headless=headless,
        auto_arm=auto_arm,
        seed=int(LaunchConfiguration("scenario_seed").perform(context)),
        obstacles=int(LaunchConfiguration("scenario_obstacles").perform(context)),
        shape=LaunchConfiguration("scenario_shape").perform(context),
        spacing=float(LaunchConfiguration("scenario_spacing").perform(context)),
        wind=wind,
        rain=rain,
    )

    shield_lookahead_tau = float(LaunchConfiguration("shield_lookahead_tau").perform(context))
    pos_noise_std = float(LaunchConfiguration("telemetry_pos_noise_std").perform(context))
    vel_noise_std = float(LaunchConfiguration("telemetry_vel_noise_std").perform(context))
    yaw_noise_std = float(LaunchConfiguration("telemetry_yaw_noise_std").perform(context))
    noise_seed = int(LaunchConfiguration("noise_seed").perform(context))

    fastdds_profile = package_root / "config" / "fastdds_loopback.xml"
    if not fastdds_profile.is_file():
        raise FileNotFoundError(f"Loopback Fast DDS profile not found: {fastdds_profile}")
    fastdds_profile_str = str(fastdds_profile)
    # MicroXRCEAgent's ArgumentParser rejects '-r' values containing hyphens, so also
    # always export FASTDDS_DEFAULT_PROFILES_FILE (Fast DDS 3.x) alongside '-r'.
    xrce_cmd = ["MicroXRCEAgent", "udp4", "-p", "8888"]
    if "-" not in fastdds_profile_str:
        xrce_cmd.extend(["-r", fastdds_profile_str])

    actions = [
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            f"{px4_dir}/Tools/simulation/gz/models:{px4_dir}/Tools/simulation/gz/worlds:{package_root}/worlds",
        ),
        SetEnvironmentVariable(
            "GZ_SIM_SYSTEM_PLUGIN_PATH",
            f"{px4_build}/src/modules/simulation/gz_plugins",
        ),
        SetEnvironmentVariable(
            "GZ_SIM_SERVER_CONFIG_PATH",
            f"{px4_dir}/src/modules/simulation/gz_bridge/server.config",
        ),
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        SetEnvironmentVariable("GZ_PARTITION", "swarm"),
        SetEnvironmentVariable("PX4_GZ_WORLD", "swarm"),
        # Prevent BLAS/OpenMP/PyTorch thread oversubscription across 20 Python ROS nodes.
        SetEnvironmentVariable("OMP_NUM_THREADS", "1"),
        SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1"),
        SetEnvironmentVariable("MKL_NUM_THREADS", "1"),
        SetEnvironmentVariable("NUMEXPR_NUM_THREADS", "1"),
        SetEnvironmentVariable("VECLIB_MAXIMUM_THREADS", "1"),
        SetEnvironmentVariable("TORCH_NUM_THREADS", "1"),
        ExecuteProcess(
            cmd=[
                "bash",
                str(package_root / "scripts" / "install_px4_airframe.sh"),
                str(px4_dir),
            ],
            output="log",
            name="install_px4_airframe",
        ),
        ExecuteProcess(
            cmd=xrce_cmd,
            additional_env={
                "FASTDDS_DEFAULT_PROFILES_FILE": fastdds_profile_str,
            },
            output="log",
            name="micro_xrce_agent",
        ),
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "-s", str(plan.world_path), "--verbose=0"],
            output="screen",
            name="gz_server",
        ),
    ]

    if not plan.headless:
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    ExecuteProcess(
                        cmd=["gz", "sim", "-g", "--verbose=0"],
                        output="log",
                        name="gz_gui",
                    )
                ],
            )
        )

    for drone in plan.drones:
        actions.append(
            TimerAction(
                period=drone.px4_start_delay,
                actions=[
                    ExecuteProcess(
                        cmd=[str(px4_binary), "-i", str(drone.index)],
                        cwd=str(px4_build),
                        additional_env={
                            "PX4_SYS_AUTOSTART": "4002",
                            "PX4_SIM_MODEL": "gz_x500",
                            "PX4_GZ_STANDALONE": "1",
                            "PX4_GZ_WORLD": "swarm",
                            "PX4_GZ_MODEL_POSE": drone.pose_str,
                            "PX4_UXRCE_DDS_NS": f"uav_{drone.index}",
                            "PX4_PARAM_UXRCE_DDS_AG_IP": "2130706433",
                            "PX4_PARAM_UXRCE_DDS_PTCFG": "2",
                            "GZ_IP": "127.0.0.1",
                            "GZ_PARTITION": "swarm",
                        },
                        output="screen",
                        name=f"px4_{drone.index}",
                    )
                ],
            )
        )

    for drone in plan.drones:
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package="px4_adapter",
                        executable="adapter_node",
                        name=f"adapter_{drone.index}",
                        output="screen",
                        parameters=[
                            {
                                "drone_id": drone.drone_id,
                                "px4_namespace": drone.px4_namespace,
                                "system_id": drone.system_id,
                                "spawn_x": float(drone.start[0]),
                                "spawn_y": float(drone.start[1]),
                                "spawn_z": 0.0,
                                "takeoff_altitude": float(drone.takeoff_altitude),
                                "takeoff_velocity": 0.8,
                                "auto_arm": plan.auto_arm,
                                "telemetry_rate": plan.telemetry_rate,
                                "telemetry_pos_noise_std": pos_noise_std,
                                "telemetry_vel_noise_std": vel_noise_std,
                                "telemetry_yaw_noise_std": yaw_noise_std,
                                "noise_seed": noise_seed,
                            }
                        ],
                    )
                ],
            )
        )

    for drone in plan.drones:
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package="swarm_control",
                        executable="controller",
                        name=f"control_{drone.index}",
                        output="screen",
                        parameters=[
                            {
                                "drone_id": drone.drone_id,
                                "agent_index": drone.index,
                                "agent_ids": plan.agent_ids,
                                "slot": list(drone.goal),
                                "start": list(drone.start),
                                "scenario": str(plan.scenario_path),
                                "mode": plan.mode,
                                "checkpoint": plan.checkpoint,
                                "allow_legacy_checkpoint": allow_legacy,
                                "actor_index": drone.index,
                                "control_rate": plan.control_rate,
                                "enable_shield": enable_shield,
                                "use_supervisor": plan.use_supervisor,
                                "quorum_min_peers": plan.quorum_min_peers,
                                "altitude": float(drone.goal[2]),
                                "speed": plan.speed,
                                "accel": plan.accel,
                                "yaw_rate": plan.yaw_rate,
                                "shield_lookahead_tau": shield_lookahead_tau,
                            }
                        ],
                    )
                ],
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario",
                default_value="",
                description="Scenario JSON; empty uses the packaged 10-drone seed-7 scenario.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="",
                description="Gazebo world SDF; empty uses the packaged 10-drone swarm world.",
            ),
            DeclareLaunchArgument(
                "drones",
                default_value="",
                description="Optional drone count override (e.g. 3, 6, 10); empty uses scenario count (10).",
            ),
            DeclareLaunchArgument(
                "mode",
                default_value="learned",
                description="Controller mode: learned, classical, or zero.",
            ),
            DeclareLaunchArgument(
                "checkpoint",
                default_value="",
                description="Contract v2 actor checkpoint for mode:=learned (unused in zero/classical).",
            ),
            DeclareLaunchArgument(
                "allow_legacy_checkpoint",
                default_value="false",
                description="Allow a state_dict without Contract v2 metadata.",
            ),
            DeclareLaunchArgument(
                "enable_shield",
                default_value="true",
                description="Enable the per-drone 3-D prioritized-slack QP CBF shield.",
            ),
            DeclareLaunchArgument(
                "use_supervisor",
                default_value="false",
                description="Classical supervisor toggle (always bypassed in mode:=learned).",
            ),
            DeclareLaunchArgument(
                "quorum_min_peers",
                default_value="3",
                description="Minimum fresh active peers required by controller quorum gating.",
            ),
            DeclareLaunchArgument(
                "auto_arm",
                default_value="false",
                description="Automatically arm and enter offboard (keep false for grounded validation).",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run Gazebo server headless without gz sim -g GUI.",
            ),
            DeclareLaunchArgument(
                "px4_dir",
                default_value="",
                description="PX4-Autopilot checkout; empty uses $PX4_DIR or default path.",
            ),
            DeclareLaunchArgument(
                "px4_stagger",
                default_value="2.0",
                description="Stagger interval in seconds between PX4 SITL instance launches (6 + stagger*i).",
            ),
            DeclareLaunchArgument(
                "takeoff_altitude",
                default_value="",
                description="Optional ENU takeoff altitude override; empty syncs with each scenario goal z.",
            ),
            DeclareLaunchArgument(
                "control_rate",
                default_value="10.0",
                description="Controller rate in Hz (use 5.0 to reduce CPU load in 10-drone SITL).",
            ),
            DeclareLaunchArgument(
                "telemetry_rate",
                default_value="10.0",
                description="Adapter ENU telemetry publish rate in Hz (PX4 offboard setpoints remain 20 Hz).",
            ),
            DeclareLaunchArgument(
                "shield_lookahead_tau",
                default_value="0.0",
                description="Optional dynamic obstacle lookahead time in seconds (e.g. 0.3 for birds).",
            ),
            DeclareLaunchArgument(
                "scenario_seed",
                default_value="7",
                description="Seed used when dynamically generating a 3/6-drone scenario.",
            ),
            DeclareLaunchArgument(
                "scenario_obstacles",
                default_value="7",
                description="Static obstacle count when dynamically generating a 3/6-drone scenario.",
            ),
            DeclareLaunchArgument(
                "scenario_shape",
                default_value="",
                description="Formation shape (ring, grid, v, triangle) for generated scenarios.",
            ),
            DeclareLaunchArgument(
                "scenario_spacing",
                default_value="1.6",
                description="Slot spacing in metres for generated scenarios.",
            ),
            DeclareLaunchArgument(
                "telemetry_pos_noise_std",
                default_value="0.0",
                description="Optional adapter ENU position Gaussian noise std (m) for Phase 8 disturbance tests.",
            ),
            DeclareLaunchArgument(
                "telemetry_vel_noise_std",
                default_value="0.0",
                description="Optional adapter ENU velocity Gaussian noise std (m/s) for Phase 8 disturbance tests.",
            ),
            DeclareLaunchArgument(
                "telemetry_yaw_noise_std",
                default_value="0.0",
                description="Optional adapter ENU yaw Gaussian noise std (rad) for Phase 8 disturbance tests.",
            ),
            DeclareLaunchArgument(
                "noise_seed",
                default_value="-1",
                description="Deterministic adapter noise RNG seed (>= 0); -1 uses non-deterministic RNG.",
            ),
            DeclareLaunchArgument(
                "wind_x",
                default_value="0.0",
                description="Optional Gazebo world ENU wind X velocity (m/s).",
            ),
            DeclareLaunchArgument(
                "wind_y",
                default_value="0.0",
                description="Optional Gazebo world ENU wind Y velocity (m/s).",
            ),
            DeclareLaunchArgument(
                "wind_z",
                default_value="0.0",
                description="Optional Gazebo world ENU wind Z velocity (m/s).",
            ),
            DeclareLaunchArgument(
                "rain",
                default_value="false",
                description="Enable cosmetic Gazebo Harmonic rain particle emitter in runtime world SDF.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
