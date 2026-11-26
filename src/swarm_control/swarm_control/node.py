from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String
from swarm_interfaces.msg import AgentTelemetry, DynamicObstacleArray

from .contract import DEFAULT_CONTRACT
from .controller_core import ControllerConfig, SwarmControllerCore
from .observation import DynamicObstacle, Obstacle


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SwarmControlNode(Node):
    def __init__(self):
        super().__init__("swarm_control")

        default_agents = [f"drone_{i}" for i in range(DEFAULT_CONTRACT.agents)]

        self.declare_parameter("drone_id", "drone_0")
        self.declare_parameter("agent_index", 0)
        self.declare_parameter("agent_ids", default_agents)
        self.declare_parameter("slot", [0.0, 0.0, DEFAULT_CONTRACT.alt])
        self.declare_parameter("start", [0.0, 0.0, 0.45])
        self.declare_parameter("scenario", "")
        self.declare_parameter("mode", "learned")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("allow_legacy_checkpoint", False)
        self.declare_parameter("actor_index", 0)
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("enable_shield", True)
        self.declare_parameter("use_supervisor", False)
        self.declare_parameter("quorum_min_peers", 3)
        self.declare_parameter("altitude", DEFAULT_CONTRACT.alt)
        self.declare_parameter("speed", DEFAULT_CONTRACT.speed)
        self.declare_parameter("accel", DEFAULT_CONTRACT.accel)
        self.declare_parameter("yaw_rate", DEFAULT_CONTRACT.yaw_rate)
        self.declare_parameter("docking_timeout", 15.0)
        self.declare_parameter("hold_dwell", 1.5)
        self.declare_parameter("telemetry_timeout", 0.5)
        self.declare_parameter("dynamic_obstacle_timeout", 1.0)
        self.declare_parameter("shield_lookahead_tau", 0.0)
        self.declare_parameter("stall_recovery", True)
        self.declare_parameter("stall_time", 1.0)
        self.declare_parameter("stall_recover_time", 1.5)
        self.declare_parameter("stall_clearance", 0.6)

        self.scenario_path = str(self.get_parameter("scenario").value)
        obstacles = self._load_obstacles()

        cfg = ControllerConfig(
            drone_id=str(self.get_parameter("drone_id").value),
            agent_index=int(self.get_parameter("agent_index").value),
            agent_ids=list(self.get_parameter("agent_ids").value),
            slot=list(self.get_parameter("slot").value),
            start=list(self.get_parameter("start").value),
            mode=str(self.get_parameter("mode").value),
            checkpoint=str(self.get_parameter("checkpoint").value),
            allow_legacy_checkpoint=bool(self.get_parameter("allow_legacy_checkpoint").value),
            actor_index=int(self.get_parameter("actor_index").value),
            control_rate=float(self.get_parameter("control_rate").value),
            enable_shield=bool(self.get_parameter("enable_shield").value),
            use_supervisor=bool(self.get_parameter("use_supervisor").value),
            quorum_min_peers=int(self.get_parameter("quorum_min_peers").value),
            altitude=float(self.get_parameter("altitude").value),
            speed=float(self.get_parameter("speed").value),
            accel=float(self.get_parameter("accel").value),
            yaw_rate=float(self.get_parameter("yaw_rate").value),
            docking_timeout=float(self.get_parameter("docking_timeout").value),
            hold_dwell=float(self.get_parameter("hold_dwell").value),
            telemetry_timeout=float(self.get_parameter("telemetry_timeout").value),
            dynamic_obstacle_timeout=float(
                self.get_parameter("dynamic_obstacle_timeout").value
            ),
            shield_lookahead_tau=float(self.get_parameter("shield_lookahead_tau").value),
            stall_recovery=bool(self.get_parameter("stall_recovery").value),
            stall_time=float(self.get_parameter("stall_time").value),
            stall_recover_time=float(self.get_parameter("stall_recover_time").value),
            stall_clearance=float(self.get_parameter("stall_clearance").value),
        )

        self.core = SwarmControllerCore(cfg, obstacles=obstacles)
        self.drone_id = self.core.drone_id
        self.agent_index = self.core.agent_index
        self.agent_ids = self.core.agent_ids
        self.contract = self.core.contract

        if self.core.supervisor_override_warning:
            self.get_logger().warn(self.core.supervisor_override_warning)
        if self.core.mode_name == "learned":
            self.get_logger().info(
                f"loaded actor {self.core.actor_index} from {self.core.checkpoint} "
                f"(schema={self.contract.schema}, obs_dim={self.contract.obs_dim}, "
                f"speed={self.contract.speed}, accel={self.contract.accel}, "
                f"yaw_rate={self.contract.yaw_rate})"
            )

        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.telemetry_subs = []
        for agent_id in self.agent_ids:
            self.telemetry_subs.append(
                self.create_subscription(
                    AgentTelemetry,
                    f"/{agent_id}/telemetry",
                    lambda msg, aid=agent_id: self._on_telemetry(aid, msg),
                    telemetry_qos,
                )
            )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            f"/{self.drone_id}/goal",
            self._on_goal,
            goal_qos,
        )
        self.dynamic_obs_sub = self.create_subscription(
            DynamicObstacleArray,
            "/obstacles/dynamic",
            self._on_dynamic_obstacles,
            telemetry_qos,
        )

        self.cmd_pub = self.create_publisher(Twist, f"/{self.drone_id}/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, f"/{self.drone_id}/status", 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, f"/{self.drone_id}/policy_debug", 10
        )
        self._cmd_msg = Twist()
        self._debug_msg = Float32MultiArray()
        self._status_msg = String()

        self.dt = self.core.dt
        self.last_log = 0.0
        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            f"{self.drone_id} index={self.agent_index}/{len(self.agent_ids)} "
            f"mode={self.core.mode_name} supervisor={self.core.use_supervisor} "
            f"quorum_min_peers={self.core.quorum_min_peers} "
            f"slot={self.core.slot.tolist()} shield={self.core.enable_shield}"
        )

    def _load_obstacles(self) -> list[Obstacle]:
        if not self.scenario_path:
            return []
        payload = json.loads(Path(self.scenario_path).read_text(encoding="utf-8"))
        return [
            Obstacle(float(x), float(y), float(radius), float(height))
            for x, y, radius, height in payload.get("obstacles", [])
        ]

    def _on_telemetry(self, agent_id: str, msg: AgentTelemetry) -> None:
        self.core.update_telemetry(
            agent_id=agent_id,
            position=(msg.position.x, msg.position.y, msg.position.z),
            velocity=(msg.velocity.x, msg.velocity.y, msg.velocity.z),
            yaw=float(msg.yaw),
            stamp=_stamp_seconds(msg.header.stamp),
            active=bool(msg.active_status),
            now=time.monotonic(),
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        accepted, reason = self.core.update_goal(
            (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        )
        if accepted:
            self.get_logger().info(f"{self.drone_id}: {reason}")
        else:
            self.get_logger().warn(f"{self.drone_id}: rejected /{self.drone_id}/goal — {reason}")

    def _on_dynamic_obstacles(self, msg: DynamicObstacleArray) -> None:
        obstacles = [
            DynamicObstacle(
                position=np.array(
                    [obs.position.x, obs.position.y, obs.position.z], dtype=np.float64
                ),
                velocity=np.array(
                    [obs.velocity.x, obs.velocity.y, obs.velocity.z], dtype=np.float64
                ),
                radius=float(obs.radius),
                active=bool(obs.active),
            )
            for obs in msg.obstacles
        ]
        self.core.update_dynamic_obstacles(obstacles, now=time.monotonic())

    def _publish(self, velocity: np.ndarray, yaw_rate: float) -> None:
        command = self._cmd_msg
        command.linear.x = float(velocity[0])
        command.linear.y = float(velocity[1])
        command.linear.z = float(velocity[2])
        command.angular.z = float(yaw_rate)
        self.cmd_pub.publish(command)

    def tick(self) -> None:
        now = time.monotonic()
        output = self.core.step(now)
        self._publish(output.velocity, output.yaw_rate)

        if not output.quorum_ok:
            return

        if output.debug_vector is not None:
            self._debug_msg.data = [float(x) for x in output.debug_vector.tolist()]
            self.debug_pub.publish(self._debug_msg)

        if output.status_dict is not None and now - self.last_log >= 1.0:
            self.last_log = now
            self._status_msg.data = json.dumps(output.status_dict)
            self.status_pub.publish(self._status_msg)
            self.get_logger().info(self._status_msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = SwarmControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish(np.zeros(3), 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
