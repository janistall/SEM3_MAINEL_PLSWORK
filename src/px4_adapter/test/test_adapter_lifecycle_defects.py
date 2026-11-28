"""Offline mocked regression tests preserving reproducible defect evidence for PX4AdapterNode.

Imports the production `px4_adapter.adapter_node.PX4AdapterNode` inside an isolated,
hermetic `sys.modules` mock of `rclpy`, `geometry_msgs`, `px4_msgs`, `std_srvs`, and
`swarm_interfaces` (with a lightweight `numpy` fallback only when `numpy` is unavailable).
No ROS 2 daemon, DDS traffic, PX4 SITL, Gazebo, or flight commands are executed.

Scope:
  - Defects 1–4 assert the currently confirmed lifecycle safety defects in
    `src/px4_adapter/px4_adapter/adapter_node.py` as reproducible evidence while
    active-flight acceptance is blocked pending safety review.
  - Test 5 serves as a control check confirming that `cmd_timeout` (0.5 s) properly
    zeroes velocity while remaining in `OFFBOARD`/`FLYING`.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_ADAPTER_PKG_ROOT = Path(__file__).resolve().parents[1]


def _build_ros_and_optional_numpy_stubs() -> dict[str, types.ModuleType]:
    """Construct minimal in-memory stubs for ROS 2 / px4_msgs (and numpy if absent)."""
    stubs: dict[str, types.ModuleType] = {}

    if importlib.util.find_spec("numpy") is None:
        class _Vec(list):
            def __add__(self, other):
                return _Vec(float(a) + float(b) for a, b in zip(self, other))

            def tolist(self):
                return [float(x) for x in self]

            def copy(self):
                return _Vec(self)

            def reshape(self, n):
                if len(self) != n:
                    raise ValueError(f"cannot reshape vector of length {len(self)} to ({n},)")
                return _Vec(self)

        class _FakeRng:
            def normal(self, loc=0.0, scale=1.0, size=None):
                if size is None:
                    return float(loc)
                return _Vec(float(loc) for _ in range(int(size)))

        np_mod = types.ModuleType("numpy")
        np_mod.float64 = float
        np_mod.ndarray = _Vec
        np_mod.array = lambda seq, dtype=None: _Vec(float(x) for x in seq)
        np_mod.asarray = lambda seq, dtype=None: _Vec(float(x) for x in seq)
        np_mod.zeros = lambda n, dtype=None: _Vec(0.0 for _ in range(int(n)))
        np_mod.random = types.SimpleNamespace(
            Generator=_FakeRng,
            default_rng=lambda seed=None: _FakeRng(),
        )
        stubs["numpy"] = np_mod

    # rclpy & submodules
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.ok = lambda: True
    rclpy_mod.init = lambda args=None: None
    rclpy_mod.spin = lambda node: None
    rclpy_mod.shutdown = lambda: None

    class _Param:
        def __init__(self, value):
            self.value = value

    class _Stamp:
        def __init__(self, ns: int):
            self.sec = ns // 1_000_000_000
            self.nanosec = ns % 1_000_000_000

    class _Now:
        def __init__(self, ns: int):
            self.nanoseconds = ns

        def to_msg(self):
            return _Stamp(self.nanoseconds)

    class _Clock:
        def __init__(self):
            self.ns = 1_000_000_000

        def now(self):
            return _Now(self.ns)

    class _Logger:
        def __init__(self):
            self.records: list[tuple[str, str]] = []

        def info(self, msg: str):
            self.records.append(("INFO", str(msg)))

        def warn(self, msg: str):
            self.records.append(("WARN", str(msg)))

        def error(self, msg: str):
            self.records.append(("ERROR", str(msg)))

    class _Publisher:
        def __init__(self, msg_type, topic: str):
            self.msg_type = msg_type
            self.topic = topic
            self.messages: list[object] = []

        def publish(self, msg):
            # Snapshot mutable reused message instances so full command history is preserved.
            if hasattr(msg, "velocity") and hasattr(msg, "yawspeed"):
                snap = types.SimpleNamespace(
                    timestamp=msg.timestamp,
                    velocity=list(msg.velocity),
                    yawspeed=float(msg.yawspeed),
                )
            elif hasattr(msg, "command") and hasattr(msg, "param1"):
                snap = types.SimpleNamespace(
                    timestamp=msg.timestamp,
                    command=int(msg.command),
                    param1=float(msg.param1),
                    param2=float(msg.param2),
                    target_system=int(msg.target_system),
                )
            elif hasattr(msg, "active_status"):
                snap = types.SimpleNamespace(
                    drone_id=msg.drone_id,
                    position=(msg.position.x, msg.position.y, msg.position.z),
                    velocity=(msg.velocity.x, msg.velocity.y, msg.velocity.z),
                    yaw=float(msg.yaw),
                    active_status=bool(msg.active_status),
                )
            elif hasattr(msg, "velocity") and hasattr(msg, "position"):
                snap = types.SimpleNamespace(
                    timestamp=msg.timestamp,
                    position=bool(msg.position),
                    velocity=bool(msg.velocity),
                )
            else:
                snap = msg
            self.messages.append(snap)

    class _FakeNode:
        _override_params: dict[str, object] = {}

        def __init__(self, name: str):
            self._name = name
            self._params: dict[str, object] = {}
            self._clock = _Clock()
            self._logger = _Logger()
            self._timers: list[tuple[float, object]] = []
            self._services: dict[str, object] = {}

        def declare_parameter(self, name: str, default):
            self._params[name] = self._override_params.get(name, default)

        def get_parameter(self, name: str):
            return _Param(self._params[name])

        def create_subscription(self, msg_type, topic, cb, qos):
            return types.SimpleNamespace(msg_type=msg_type, topic=topic, cb=cb, qos=qos)

        def create_publisher(self, msg_type, topic, qos):
            return _Publisher(msg_type, topic)

        def create_service(self, srv_type, name, cb):
            srv = types.SimpleNamespace(srv_type=srv_type, name=name, cb=cb)
            self._services[name] = srv
            return srv

        def create_timer(self, period, cb):
            timer = (period, cb)
            self._timers.append(timer)
            return timer

        def get_clock(self):
            return self._clock

        def get_logger(self):
            return self._logger

        def destroy_node(self):
            pass

    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _FakeNode

    exec_mod = types.ModuleType("rclpy.executors")
    exec_mod.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})

    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.DurabilityPolicy = types.SimpleNamespace(VOLATILE=1, TRANSIENT_LOCAL=2)
    qos_mod.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    qos_mod.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1, RELIABLE=2)
    qos_mod.QoSProfile = lambda **kwargs: types.SimpleNamespace(**kwargs)

    # geometry_msgs.msg
    class _Vec3:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _Twist:
        def __init__(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
            self.linear = _Vec3(*linear)
            self.angular = _Vec3(*angular)

    geom_pkg = types.ModuleType("geometry_msgs")
    geom_msg = types.ModuleType("geometry_msgs.msg")
    geom_msg.Twist = _Twist

    # px4_msgs.msg
    class _OffboardControlMode:
        def __init__(self):
            self.timestamp = 0
            self.position = False
            self.velocity = False
            self.acceleration = False
            self.attitude = False
            self.body_rate = False

    class _TrajectorySetpoint:
        def __init__(self):
            self.timestamp = 0
            self.position = [math.nan, math.nan, math.nan]
            self.velocity = [0.0, 0.0, 0.0]
            self.acceleration = [math.nan, math.nan, math.nan]
            self.jerk = [math.nan, math.nan, math.nan]
            self.yaw = math.nan
            self.yawspeed = 0.0

    class _VehicleCommand:
        VEHICLE_CMD_NAV_LAND = 21
        VEHICLE_CMD_DO_SET_MODE = 176
        VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

        def __init__(self):
            self.timestamp = 0
            self.command = 0
            self.param1 = 0.0
            self.param2 = 0.0
            self.target_system = 0
            self.target_component = 0
            self.source_system = 0
            self.source_component = 0
            self.from_external = False

    class _VehicleCommandAck:
        pass

    class _VehicleOdometry:
        def __init__(
            self,
            pos_ned=(0.0, 0.0, 0.0),
            vel_ned=(0.0, 0.0, 0.0),
            q=(1.0, 0.0, 0.0, 0.0),
        ):
            self.position = list(pos_ned)
            self.velocity = list(vel_ned)
            self.q = list(q)

    class _VehicleStatus:
        NAVIGATION_STATE_OFFBOARD = 14
        NAVIGATION_STATE_AUTO_LAND = 18
        NAVIGATION_STATE_AUTO_LOITER = 4
        ARMING_STATE_DISARMED = 1
        ARMING_STATE_ARMED = 2

        def __init__(self, nav_state=4, arming_state=1):
            self.nav_state = nav_state
            self.arming_state = arming_state

    px4_pkg = types.ModuleType("px4_msgs")
    px4_msg = types.ModuleType("px4_msgs.msg")
    px4_msg.OffboardControlMode = _OffboardControlMode
    px4_msg.TrajectorySetpoint = _TrajectorySetpoint
    px4_msg.VehicleCommand = _VehicleCommand
    px4_msg.VehicleCommandAck = _VehicleCommandAck
    px4_msg.VehicleOdometry = _VehicleOdometry
    px4_msg.VehicleStatus = _VehicleStatus

    # std_srvs.srv
    std_srvs_pkg = types.ModuleType("std_srvs")
    std_srvs_srv = types.ModuleType("std_srvs.srv")
    std_srvs_srv.SetBool = types.SimpleNamespace(
        Request=lambda data=False: types.SimpleNamespace(data=data),
        Response=lambda: types.SimpleNamespace(success=False, message=""),
    )
    std_srvs_srv.Trigger = types.SimpleNamespace(
        Request=lambda: types.SimpleNamespace(),
        Response=lambda: types.SimpleNamespace(success=False, message=""),
    )

    # swarm_interfaces.msg
    class _AgentTelemetry:
        def __init__(self):
            self.header = types.SimpleNamespace(frame_id="", stamp=None)
            self.drone_id = ""
            self.position = _Vec3()
            self.velocity = _Vec3()
            self.yaw = 0.0
            self.active_status = False

    si_pkg = types.ModuleType("swarm_interfaces")
    si_msg = types.ModuleType("swarm_interfaces.msg")
    si_msg.AgentTelemetry = _AgentTelemetry

    stubs.update(
        {
            "rclpy": rclpy_mod,
            "rclpy.node": node_mod,
            "rclpy.executors": exec_mod,
            "rclpy.qos": qos_mod,
            "geometry_msgs": geom_pkg,
            "geometry_msgs.msg": geom_msg,
            "px4_msgs": px4_pkg,
            "px4_msgs.msg": px4_msg,
            "std_srvs": std_srvs_pkg,
            "std_srvs.srv": std_srvs_srv,
            "swarm_interfaces": si_pkg,
            "swarm_interfaces.msg": si_msg,
        }
    )
    return stubs


class _FakeClock:
    def __init__(self, start: float = 100.0):
        self.t = float(start)

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


class TestAdapterLifecycleDefects(unittest.TestCase):
    """Reproducible offline defect evidence for PX4AdapterNode lifecycle safety."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._added_sys_path = False
        if str(_ADAPTER_PKG_ROOT) not in sys.path:
            sys.path.insert(0, str(_ADAPTER_PKG_ROOT))
            cls._added_sys_path = True

        cls._stubs = _build_ros_and_optional_numpy_stubs()
        cls._modules_patch = patch.dict(sys.modules, cls._stubs)
        cls._modules_patch.start()

        # Force fresh import of adapter_node against the isolated stubs.
        for mod_name in ("px4_adapter.adapter_node", "px4_adapter.noise"):
            sys.modules.pop(mod_name, None)

        cls.adapter_mod = importlib.import_module("px4_adapter.adapter_node")
        cls.PX4AdapterNode = cls.adapter_mod.PX4AdapterNode
        cls.NodeBase = cls._stubs["rclpy.node"].Node
        cls.VehicleCommand = cls._stubs["px4_msgs.msg"].VehicleCommand
        cls.VehicleOdometry = cls._stubs["px4_msgs.msg"].VehicleOdometry
        cls.VehicleStatus = cls._stubs["px4_msgs.msg"].VehicleStatus
        cls.Twist = cls._stubs["geometry_msgs.msg"].Twist
        cls.SetBool = cls._stubs["std_srvs.srv"].SetBool
        cls.Trigger = cls._stubs["std_srvs.srv"].Trigger

    @classmethod
    def tearDownClass(cls) -> None:
        for mod_name in ("px4_adapter.adapter_node", "px4_adapter.noise"):
            sys.modules.pop(mod_name, None)
        cls._modules_patch.stop()
        if cls._added_sys_path and str(_ADAPTER_PKG_ROOT) in sys.path:
            sys.path.remove(str(_ADAPTER_PKG_ROOT))

    def _make_node(self, **params):
        old = dict(self.NodeBase._override_params)
        self.NodeBase._override_params = dict(params)
        try:
            return self.PX4AdapterNode()
        finally:
            self.NodeBase._override_params = old

    def _feed_px4(self, node, *, z_enu: float, nav_state: int, arming_state: int) -> None:
        # ENU altitude z_enu maps to NED z = -z_enu when spawn_z = 0.0.
        node._on_odometry(self.VehicleOdometry(pos_ned=(0.0, 0.0, -z_enu)))
        node._on_status(self.VehicleStatus(nav_state=nav_state, arming_state=arming_state))

    def _bring_to_flying(self, node, clock: _FakeClock) -> None:
        # WAIT -> STREAM
        self._feed_px4(
            node,
            z_enu=0.0,
            nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
            arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
        )
        node._tick()
        self.assertEqual(node.phase, "STREAM")

        # STREAM -> ARM after >= 1.5 s
        clock.advance(1.6)
        self._feed_px4(
            node,
            z_enu=0.0,
            nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
            arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
        )
        node._tick()
        self.assertEqual(node.phase, "ARM")

        # PX4 arms -> OFFBOARD
        clock.advance(0.1)
        self._feed_px4(
            node,
            z_enu=0.0,
            nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
            arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
        )
        node._tick()
        self.assertEqual(node.phase, "OFFBOARD")

        # PX4 enters OFFBOARD -> TAKEOFF
        clock.advance(0.1)
        self._feed_px4(
            node,
            z_enu=0.0,
            nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
            arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
        )
        node._tick()
        self.assertEqual(node.phase, "TAKEOFF")

        # Drone climbs to takeoff_altitude (2.5 m) -> FLYING
        clock.advance(3.5)
        self._feed_px4(
            node,
            z_enu=2.5,
            nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
            arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
        )
        node._tick()
        self.assertEqual(node.phase, "FLYING")
        self.assertTrue(node.flying)

    def test_defect_1_land_from_flying_re_requests_offboard_and_resumes_takeoff_climb(self) -> None:
        clock = _FakeClock(100.0)
        with patch.object(self.adapter_mod.time, "monotonic", clock.monotonic):
            node = self._make_node(
                spawn_z=0.0, takeoff_altitude=2.5, takeoff_velocity=0.8, auto_arm=True
            )
            self._bring_to_flying(node, clock)
            node.command_pub.messages.clear()
            node.traj_pub.messages.clear()

            # 1. Request landing via /{drone}/land
            resp = node._on_land(self.Trigger.Request(), self.Trigger.Response())
            self.assertTrue(resp.success)
            self.assertEqual(
                node.command_pub.messages[-1].command, self.VehicleCommand.VEHICLE_CMD_NAV_LAND
            )
            # Defect evidence A: phase and flying remain unchanged after _on_land()
            self.assertEqual(node.phase, "FLYING")
            self.assertTrue(node.flying)

            # 2. PX4 honors NAV_LAND: enters AUTO_LAND and descends to z = 1.8 m
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=1.8,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            # Defect evidence B: FLYING drops back to OFFBOARD instead of a LANDING phase
            self.assertEqual(node.phase, "OFFBOARD")
            self.assertFalse(node.flying)

            # 3. On the next 20 Hz tick (+0.05 s), last_offboard_request is >1 s old,
            #    so the adapter immediately sends VEHICLE_CMD_DO_SET_MODE(OFFBOARD)
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=1.75,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            offboard_cmds = [
                m
                for m in node.command_pub.messages
                if m.command == self.VehicleCommand.VEHICLE_CMD_DO_SET_MODE
                and m.param1 == 1.0
                and m.param2 == 6.0
            ]
            self.assertEqual(len(offboard_cmds), 1)

            # 4. PX4 obeys DO_SET_MODE -> re-enters OFFBOARD at z = 1.75 m
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=1.75,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            # Defect evidence C: adapter transitions back into TAKEOFF
            self.assertEqual(node.phase, "TAKEOFF")

            # 5. Next tick commands active upward climb (+0.8 m/s ENU == -0.8 m/s NED vz)
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=1.75,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertAlmostEqual(node.traj_pub.messages[-1].velocity[2], -0.8, places=6)

    def test_defect_2_land_during_takeoff_continues_climb_setpoints_then_retries_offboard(
        self,
    ) -> None:
        clock = _FakeClock(200.0)
        with patch.object(self.adapter_mod.time, "monotonic", clock.monotonic):
            node = self._make_node(
                spawn_z=0.0, takeoff_altitude=2.5, takeoff_velocity=0.8, auto_arm=True
            )
            # Advance to TAKEOFF at z = 1.0 m
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
                arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
            )
            node._tick()
            clock.advance(1.6)
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "ARM")
            clock.advance(0.1)
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "OFFBOARD")
            clock.advance(0.1)
            self._feed_px4(
                node,
                z_enu=1.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "TAKEOFF")

            # Request landing mid-takeoff
            node.command_pub.messages.clear()
            node.traj_pub.messages.clear()
            node._on_land(self.Trigger.Request(), self.Trigger.Response())
            self.assertEqual(
                node.command_pub.messages[-1].command, self.VehicleCommand.VEHICLE_CMD_NAV_LAND
            )

            # Even after PX4 enters AUTO_LAND, TAKEOFF keeps publishing +0.8 m/s ENU climb setpoints
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=0.9,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "TAKEOFF")
            self.assertAlmostEqual(node.traj_pub.messages[-1].velocity[2], -0.8, places=6)

            # After 20 s takeoff timeout, adapter drops to OFFBOARD and re-requests OFFBOARD mode
            clock.advance(20.1)
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
                arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "OFFBOARD")
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
                arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
            )
            node._tick()
            self.assertEqual(
                node.command_pub.messages[-1].command, self.VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            )

    def test_defect_3_offboard_false_is_noop_and_continues_streaming_cmd_vel(self) -> None:
        clock = _FakeClock(300.0)
        with patch.object(self.adapter_mod.time, "monotonic", clock.monotonic):
            node = self._make_node(
                spawn_z=0.0, takeoff_altitude=2.5, takeoff_velocity=0.8, auto_arm=True
            )
            self._bring_to_flying(node, clock)
            node._on_cmd(self.Twist(linear=(1.0, 0.0, 0.0)))
            node.command_pub.messages.clear()
            node.traj_pub.messages.clear()

            # Request offboard=False
            resp = node._on_offboard(self.SetBool.Request(data=False), self.SetBool.Response())
            self.assertTrue(resp.success)
            # Zero VehicleCommand emitted; phase stays FLYING; active velocity keeps streaming
            self.assertEqual(len(node.command_pub.messages), 0)
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=2.5,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "FLYING")
            # ENU vx=1.0 maps to NED vy=1.0
            self.assertAlmostEqual(node.traj_pub.messages[-1].velocity[1], 1.0, places=6)

    def test_defect_4_disarm_service_rearms_within_1s_when_auto_arm_true(self) -> None:
        clock = _FakeClock(400.0)
        with patch.object(self.adapter_mod.time, "monotonic", clock.monotonic):
            node = self._make_node(
                spawn_z=0.0, takeoff_altitude=2.5, takeoff_velocity=0.8, auto_arm=True
            )
            # Advance WAIT -> STREAM -> ARM
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
                arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
            )
            node._tick()
            clock.advance(1.6)
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
                arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
            )
            node._tick()
            self.assertEqual(node.phase, "ARM")
            node.command_pub.messages.clear()

            # Explicitly request /{drone}/arm(False) to stop/disarm
            resp = node._on_arm(self.SetBool.Request(data=False), self.SetBool.Response())
            self.assertTrue(resp.success)
            self.assertEqual(
                node.command_pub.messages[-1].command,
                self.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            )
            self.assertEqual(node.command_pub.messages[-1].param1, 0.0)

            # After >1.0 s in ARM, auto_arm loop re-sends ARM(True)
            clock.advance(1.1)
            self._feed_px4(
                node,
                z_enu=0.0,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_AUTO_LOITER,
                arming_state=self.VehicleStatus.ARMING_STATE_DISARMED,
            )
            node._tick()
            self.assertEqual(
                node.command_pub.messages[-1].command,
                self.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            )
            self.assertEqual(node.command_pub.messages[-1].param1, 1.0)

    def test_control_check_cmd_timeout_zeros_velocity_while_in_offboard_flying(self) -> None:
        clock = _FakeClock(500.0)
        with patch.object(self.adapter_mod.time, "monotonic", clock.monotonic):
            node = self._make_node(
                spawn_z=0.0, takeoff_altitude=2.5, takeoff_velocity=0.8, auto_arm=True
            )
            self._bring_to_flying(node, clock)
            node._on_cmd(self.Twist(linear=(1.2, -0.5, 0.0), angular=(0.0, 0.0, 0.3)))
            clock.advance(0.05)
            self._feed_px4(
                node,
                z_enu=2.5,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            self.assertAlmostEqual(node.traj_pub.messages[-1].velocity[1], 1.2, places=6)

            # After cmd_timeout (0.5 s) with no new cmd_vel, adapter zeroes velocity & yawspeed
            clock.advance(0.55)
            self._feed_px4(
                node,
                z_enu=2.5,
                nav_state=self.VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                arming_state=self.VehicleStatus.ARMING_STATE_ARMED,
            )
            node._tick()
            last_sp = node.traj_pub.messages[-1]
            self.assertEqual(last_sp.velocity, [0.0, 0.0, 0.0])
            self.assertEqual(last_sp.yawspeed, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
