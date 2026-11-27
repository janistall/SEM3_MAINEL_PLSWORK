from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool, Trigger
from swarm_interfaces.msg import AgentTelemetry

from .noise import inject_telemetry_noise
from .rate import TelemetryRateLimiter


def _px4_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _telemetry_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class PX4AdapterNode(Node):
    def __init__(self):
        super().__init__("px4_adapter")

        self.declare_parameter("drone_id", "drone_0")
        self.declare_parameter("px4_namespace", "/uav_0")
        self.declare_parameter("system_id", 1)
        self.declare_parameter("spawn_x", 0.0)
        self.declare_parameter("spawn_y", 0.0)
        self.declare_parameter("spawn_z", 0.1)
        self.declare_parameter("takeoff_altitude", 2.5)
        self.declare_parameter("takeoff_velocity", 0.8)
        self.declare_parameter("auto_arm", False)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("odom_timeout", 0.5)
        self.declare_parameter("status_timeout", 1.0)
        self.declare_parameter("telemetry_rate", 10.0)
        self.declare_parameter("telemetry_pos_noise_std", 0.0)
        self.declare_parameter("telemetry_vel_noise_std", 0.0)
        self.declare_parameter("telemetry_yaw_noise_std", 0.0)
        self.declare_parameter("noise_seed", -1)

        self.drone_id = str(self.get_parameter("drone_id").value)
        namespace = str(self.get_parameter("px4_namespace").value).rstrip("/")
        self.namespace = namespace
        self.system_id = int(self.get_parameter("system_id").value)
        self.spawn = np.array(
            [
                float(self.get_parameter("spawn_x").value),
                float(self.get_parameter("spawn_y").value),
                float(self.get_parameter("spawn_z").value),
            ]
        )
        self.takeoff_altitude = float(self.get_parameter("takeoff_altitude").value)
        self.takeoff_velocity = float(self.get_parameter("takeoff_velocity").value)
        self.auto_arm = bool(self.get_parameter("auto_arm").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.status_timeout = float(self.get_parameter("status_timeout").value)
        self.telemetry_rate = float(self.get_parameter("telemetry_rate").value)
        self._telemetry_limiter = TelemetryRateLimiter(self.telemetry_rate)
        self.telemetry_pos_noise_std = max(
            0.0, float(self.get_parameter("telemetry_pos_noise_std").value)
        )
        self.telemetry_vel_noise_std = max(
            0.0, float(self.get_parameter("telemetry_vel_noise_std").value)
        )
        self.telemetry_yaw_noise_std = max(
            0.0, float(self.get_parameter("telemetry_yaw_noise_std").value)
        )
        noise_seed = int(self.get_parameter("noise_seed").value)
        self._noise_rng = (
            np.random.default_rng(noise_seed + self.system_id)
            if noise_seed >= 0
            else np.random.default_rng()
        )

        qos = _px4_qos()
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            f"{self.namespace}/fmu/out/vehicle_odometry",
            self._on_odometry,
            qos,
        )
        self.status_subs = []
        for topic in ("vehicle_status", "vehicle_status_v1", "vehicle_status_v4"):
            self.status_subs.append(
                self.create_subscription(
                    VehicleStatus,
                    f"{self.namespace}/fmu/out/{topic}",
                    self._on_status,
                    qos,
                )
            )
        self.ack_subs = []
        for topic in ("vehicle_command_ack", "vehicle_command_ack_v1"):
            self.ack_subs.append(
                self.create_subscription(
                    VehicleCommandAck,
                    f"{self.namespace}/fmu/out/{topic}",
                    self._on_ack,
                    qos,
                )
            )
        self.cmd_sub = self.create_subscription(
            Twist, f"/{self.drone_id}/cmd_vel", self._on_cmd, 10
        )

        self.telemetry_pub = self.create_publisher(
            AgentTelemetry, f"/{self.drone_id}/telemetry", _telemetry_qos()
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, f"{self.namespace}/fmu/in/trajectory_setpoint", qos
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, f"{self.namespace}/fmu/in/offboard_control_mode", qos
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, f"{self.namespace}/fmu/in/vehicle_command", qos
        )

        self._telemetry_msg = AgentTelemetry()
        self._telemetry_msg.header.frame_id = "world_enu"
        self._telemetry_msg.drone_id = self.drone_id

        self._offboard_msg = OffboardControlMode()
        self._offboard_msg.position = False
        self._offboard_msg.velocity = True
        self._offboard_msg.acceleration = False
        self._offboard_msg.attitude = False
        self._offboard_msg.body_rate = False

        self._traj_msg = TrajectorySetpoint()
        self._traj_msg.position = [math.nan, math.nan, math.nan]
        self._traj_msg.acceleration = [math.nan, math.nan, math.nan]
        self._traj_msg.jerk = [math.nan, math.nan, math.nan]
        self._traj_msg.yaw = math.nan
        self._zero_vel = np.zeros(3, dtype=np.float64)

        self.arm_srv = self.create_service(SetBool, f"/{self.drone_id}/arm", self._on_arm)
        self.offboard_srv = self.create_service(
            SetBool, f"/{self.drone_id}/offboard", self._on_offboard
        )
        self.land_srv = self.create_service(Trigger, f"/{self.drone_id}/land", self._on_land)

        self.odom: VehicleOdometry | None = None
        self.odom_time = 0.0
        self.status: VehicleStatus | None = None
        self.status_time = 0.0
        self.last_ack = None
        self.command = np.zeros(3)
        self.yaw_rate = 0.0
        self.cmd_time = 0.0

        self.phase = "WAIT"
        self.phase_time = 0.0
        self.flying = False
        self.last_arm_request = 0.0
        self.last_offboard_request = 0.0

        self.timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"{self.drone_id}: PX4 namespace={self.namespace or '/'} "
            f"sysid={self.system_id} spawn={self.spawn.tolist()} "
            f"takeoff={self.takeoff_altitude} auto_arm={self.auto_arm}"
        )

    def _on_odometry(self, msg: VehicleOdometry) -> None:
        self.odom = msg
        self.odom_time = time.monotonic()

    def _on_status(self, msg: VehicleStatus) -> None:
        self.status = msg
        self.status_time = time.monotonic()

    def _on_ack(self, msg: VehicleCommandAck) -> None:
        self.last_ack = msg

    def _on_cmd(self, msg: Twist) -> None:
        self.command = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        self.yaw_rate = float(msg.angular.z)
        self.cmd_time = time.monotonic()

    def _on_arm(self, request, response):
        self._send_arm(request.data)
        response.success = True
        response.message = f"{self.drone_id}: arm={request.data}"
        return response

    def _on_offboard(self, request, response):
        if request.data:
            self._send_mode_offboard()
        response.success = True
        response.message = f"{self.drone_id}: offboard={request.data}"
        return response

    def _on_land(self, request, response):
        self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        response.success = True
        response.message = f"{self.drone_id}: land requested"
        return response

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _send_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self._now_us()
        message.command = int(command)
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.target_system = int(self.system_id)
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_pub.publish(message)

    def _send_arm(self, arm: bool) -> None:
        self._send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0 if arm else 0.0)

    def _send_mode_offboard(self) -> None:
        self._send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def _publish_offboard(self) -> None:
        self._offboard_msg.timestamp = self._now_us()
        self.offboard_pub.publish(self._offboard_msg)

    def _publish_setpoint(self, velocity_enu: np.ndarray, yaw_rate_enu: float) -> None:
        message = self._traj_msg
        message.timestamp = self._now_us()
        message.velocity = [
            float(velocity_enu[1]),
            float(velocity_enu[0]),
            float(-velocity_enu[2]),
        ]
        message.yawspeed = float(-yaw_rate_enu)
        self.traj_pub.publish(message)

    def _enu_state(self):
        if self.odom is None:
            return None
        position = np.array(
            [
                float(self.odom.position[1]),
                float(self.odom.position[0]),
                -float(self.odom.position[2]),
            ]
        ) + self.spawn
        velocity = np.array(
            [
                float(self.odom.velocity[1]),
                float(self.odom.velocity[0]),
                -float(self.odom.velocity[2]),
            ]
        )
        w, x, y, z = [float(value) for value in self.odom.q]
        yaw_ned = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_enu = math.pi / 2.0 - yaw_ned
        yaw_enu = (yaw_enu + math.pi) % (2.0 * math.pi) - math.pi
        return position, velocity, yaw_enu

    def _publish_telemetry(self, state=None, now: float | None = None) -> None:
        if state is None:
            state = self._enu_state()
        now_f = time.monotonic() if now is None else float(now)
        message = self._telemetry_msg
        message.header.stamp = self.get_clock().now().to_msg()
        if state is None:
            message.active_status = False
        else:
            position, velocity, yaw = inject_telemetry_noise(
                *state,
                pos_std=self.telemetry_pos_noise_std,
                vel_std=self.telemetry_vel_noise_std,
                yaw_std=self.telemetry_yaw_noise_std,
                rng=self._noise_rng,
            )
            message.position.x = float(position[0])
            message.position.y = float(position[1])
            message.position.z = float(position[2])
            message.velocity.x = float(velocity[0])
            message.velocity.y = float(velocity[1])
            message.velocity.z = float(velocity[2])
            message.yaw = float(yaw)
            fresh = now_f - self.odom_time <= self.odom_timeout
            message.active_status = bool(self.flying and fresh)
        self.telemetry_pub.publish(message)

    def _status_fresh(self) -> bool:
        return (
            self.status is not None
            and time.monotonic() - self.status_time <= self.status_timeout
        )

    def _offboard_active(self) -> bool:
        return (
            self._status_fresh()
            and self.status is not None
            and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )

    def _armed(self) -> bool:
        return (
            self._status_fresh()
            and self.status is not None
            and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        )

    def _tick(self) -> None:
        now = time.monotonic()
        self._publish_offboard()
        state = self._enu_state()
        if self._telemetry_limiter.should_publish(now):
            self._publish_telemetry(state, now)

        if state is None:
            return
        position, _, _ = state

        if self.phase == "WAIT":
            if now - self.odom_time <= self.odom_timeout:
                self.phase = "STREAM"
                self.phase_time = now
                self.get_logger().info(f"{self.drone_id}: streaming offboard setpoints")
            self._publish_setpoint(self._zero_vel, 0.0)
            return

        if self.phase == "STREAM":
            self._publish_setpoint(self._zero_vel, 0.0)
            if now - self.phase_time >= 1.5:
                if self.auto_arm:
                    self._send_arm(True)
                    self.last_arm_request = now
                self.phase = "ARM"
                self.phase_time = now
            return

        if self.phase == "ARM":
            self._publish_setpoint(self._zero_vel, 0.0)
            if self._armed():
                self._send_mode_offboard()
                self.last_offboard_request = now
                self.phase = "OFFBOARD"
                self.phase_time = now
                self.get_logger().info(f"{self.drone_id}: armed, requesting offboard")
            elif self.auto_arm and now - self.last_arm_request > 1.0:
                self._send_arm(True)
                self.last_arm_request = now
            return

        if self.phase == "OFFBOARD":
            self._publish_setpoint(self._zero_vel, 0.0)
            if self._offboard_active():
                self.phase = "TAKEOFF"
                self.phase_time = now
                self.get_logger().info(f"{self.drone_id}: offboard active, taking off")
            elif now - self.last_offboard_request > 1.0:
                self._send_mode_offboard()
                self.last_offboard_request = now
            return

        if self.phase == "TAKEOFF":
            climb = np.array([0.0, 0.0, self.takeoff_velocity])
            self._publish_setpoint(climb, 0.0)
            if position[2] >= self.takeoff_altitude - 0.15:
                self.flying = True
                self.phase = "FLYING"
                self.phase_time = now
                self.get_logger().info(
                    f"{self.drone_id}: reached {position[2]:.2f} m, accepting cmd_vel"
                )
            elif now - self.phase_time > 20.0:
                self.get_logger().error(f"{self.drone_id}: takeoff timeout")
                self.phase = "OFFBOARD"
                self.phase_time = now
            return

        if self.phase == "FLYING":
            if not self._offboard_active():
                self.flying = False
                self.phase = "OFFBOARD"
                self.phase_time = now
                self._publish_setpoint(np.zeros(3), 0.0)
                return
            if now - self.cmd_time > self.cmd_timeout:
                self._publish_setpoint(np.zeros(3), 0.0)
            else:
                self._publish_setpoint(self.command, self.yaw_rate)
            return

        self._publish_setpoint(np.zeros(3), 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = PX4AdapterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish_setpoint(np.zeros(3), 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
