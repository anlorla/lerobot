"""Piper leader arm teleoperator for HIL-SERL."""

import logging
import threading
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)

# Lazy import ROS
rospy = None
JointState = None


def _import_ros():
    global rospy, JointState
    if rospy is None:
        import rospy as _rospy
        from sensor_msgs.msg import JointState as _JointState
        rospy = _rospy
        JointState = _JointState


class PiperLeader(Teleoperator):
    """
    Piper leader arm(s) for teleoperation.

    Used for human intervention in HIL-SERL training.
    Detects when human moves the leader arms and provides
    the corresponding joint positions as actions.
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        _import_ros()

        self.config = config
        self._connected = False
        self._lock = threading.Lock()

        # Store latest joint states
        self._latest_joints = {
            "left": None,
            "right": None,
        }
        self._prev_joints = {
            "left": None,
            "right": None,
        }

        # Intervention detection
        self._is_intervening = False

        # ROS subscribers
        self._subs = []

    @property
    def action_features(self) -> dict:
        """Action features for the teleoperator."""
        n_joints = self.config.joints_per_arm
        if self.config.mode == "bimanual":
            shape = (n_joints * 2,)
        else:
            shape = (n_joints,)

        return {
            "dtype": "float32",
            "shape": shape,
            "names": None,
        }

    @property
    def feedback_features(self) -> dict:
        """No feedback for leader arms."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed

    def _cb_left_joints(self, msg: Any) -> None:
        """Callback for left leader arm."""
        with self._lock:
            joints = np.array(msg.position, dtype=np.float32)
            if self._latest_joints["left"] is not None:
                self._prev_joints["left"] = self._latest_joints["left"].copy()
            self._latest_joints["left"] = joints

    def _cb_right_joints(self, msg: Any) -> None:
        """Callback for right leader arm."""
        with self._lock:
            joints = np.array(msg.position, dtype=np.float32)
            if self._latest_joints["right"] is not None:
                self._prev_joints["right"] = self._latest_joints["right"].copy()
            self._latest_joints["right"] = joints

    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS and subscribe to leader arm topics."""
        if self._connected:
            logger.warning("Already connected")
            return

        # Initialize ROS if needed
        if not rospy.core.is_initialized():
            rospy.init_node("piper_leader_teleop", anonymous=True)

        # Subscribe to leader arm topics
        if self.config.mode in ["bimanual", "single_left"]:
            sub = rospy.Subscriber(
                self.config.left_leader_topic,
                JointState,
                self._cb_left_joints,
                queue_size=1,
            )
            self._subs.append(sub)

        if self.config.mode in ["bimanual", "single_right"]:
            sub = rospy.Subscriber(
                self.config.right_leader_topic,
                JointState,
                self._cb_right_joints,
                queue_size=1,
            )
            self._subs.append(sub)

        self._connected = True
        logger.info(f"PiperLeader connected in {self.config.mode} mode")

    def calibrate(self) -> None:
        """No calibration needed."""
        pass

    def configure(self) -> None:
        """No configuration needed."""
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Get current action from leader arms.

        Returns:
            Dictionary with joint positions from leader arms.
        """
        with self._lock:
            positions = []
            n = self.config.joints_per_arm

            if self.config.mode in ["bimanual", "single_left"]:
                if self._latest_joints["left"] is not None:
                    positions.extend(self._latest_joints["left"][:n].tolist())
                else:
                    positions.extend([0.0] * n)

            if self.config.mode in ["bimanual", "single_right"]:
                if self._latest_joints["right"] is not None:
                    positions.extend(self._latest_joints["right"][:n].tolist())
                else:
                    positions.extend([0.0] * n)

        return {
            "positions": np.array(positions, dtype=np.float32),
        }

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get teleoperation events.

        Detects intervention by checking if joint positions changed
        significantly since last check.
        """
        is_intervention = False
        threshold = self.config.intervention_threshold

        with self._lock:
            # Check left arm
            if (self._latest_joints["left"] is not None and
                self._prev_joints["left"] is not None):
                delta = np.abs(self._latest_joints["left"] - self._prev_joints["left"])
                if np.any(delta > threshold):
                    is_intervention = True

            # Check right arm
            if (self._latest_joints["right"] is not None and
                self._prev_joints["right"] is not None):
                delta = np.abs(self._latest_joints["right"] - self._prev_joints["right"])
                if np.any(delta > threshold):
                    is_intervention = True

        self._is_intervening = is_intervention

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Leader arms don't receive feedback."""
        pass

    def disconnect(self) -> None:
        """Disconnect from ROS."""
        for sub in self._subs:
            sub.unregister()
        self._subs.clear()
        self._connected = False
        logger.info("PiperLeader disconnected")