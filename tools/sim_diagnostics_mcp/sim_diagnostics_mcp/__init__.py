"""Read-only Streamable HTTP MCP diagnostic bridge for ROS 2, Gazebo Harmonic, and PX4 SITL."""

from .config import ALLOWED_HOST, ALLOWED_PORT, BridgeConfig

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_HOST",
    "ALLOWED_PORT",
    "BridgeConfig",
    "__version__",
]
