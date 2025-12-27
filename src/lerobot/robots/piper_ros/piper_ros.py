"""Piper dual-arm robot implementation using ROS topics.

This module provides a Robot interface for the Piper dual-arm robot
that communicates via ROS topics. It's designed to work with:
- piper_ros driver nodes (from NPM-Ros/piper_ros)
- HIL-SERL training pipeline
- VLA inference scripts (from NPM-VLA)

Architecture:
    PiperRos
    ├── PiperRosBus (joint state/command via ROS)
    └── PiperRosCameras (image via ROS CompressedImage)

ROS Topics:
    Subscribed (observation):
        /robot/arm_left/joint_states_single - Left arm joint states
        /robot/arm_right/joint_states_single - Right arm joint states
        /robot/arm_left/end_pose - Left arm end effector pose
        /robot/arm_right/end_pose - Right arm end effector pose
        /realsense_*/color/image_raw/compressed - Camera images

    Published (action):
        /robot/arm_left/vla_joint_cmd - Left arm joint commands
        /robot/arm_right/vla_joint_cmd - Right arm joint commands
"""

import logging
import threading
import time
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from ..robot import Robot
from .config_piper_ros import PiperRosConfig

logger = logging.getLogger(__name__)

# Lazy import ROS to avoid issues when ROS is not available
rospy = None
CompressedImage = None
JointState = None
PoseStamped = None
CvBridge = None


def _import_ros():
    """Lazy import ROS modules."""
    global rospy, CompressedImage, JointState, PoseStamped, CvBridge
    if rospy is None:
        import rospy as _rospy
        from sensor_msgs.msg import CompressedImage as _CompressedImage
        from sensor_msgs.msg import JointState as _JointState
        from geometry_msgs.msg import PoseStamped as _PoseStamped
        from cv_bridge import CvBridge as _CvBridge

        rospy = _rospy
        CompressedImage = _CompressedImage
        JointState = _JointState
        PoseStamped = _PoseStamped
        CvBridge = _CvBridge


class PiperRosBus:
    """
    ROS-based motor bus abstraction for Piper robot.

    Provides the same interface as FeetechMotorsBus for compatibility
    with gym_manipulator.py, but communicates via ROS topics.
    """

    def __init__(self, robot: "PiperRos"):
        self._robot = robot
        self._is_connected = False

    @property
    def motors(self) -> dict[str, Any]:
        """Motor names as dict keys (compatible with FeetechMotorsBus)."""
        return {name: None for name in self._robot.motor_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Mark as connected."""
        self._is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        """Mark as disconnected."""
        self._is_connected = False

    def sync_read(self, register: str) -> dict[str, float]:
        """
        Read current motor positions.

        Args:
            register: Register name (only 'Present_Position' supported)

        Returns:
            Dictionary mapping motor names to positions
        """
        if register == "Present_Position":
            obs = self._robot._get_raw_joint_positions()
            return {name: obs.get(name, 0.0) for name in self._robot.motor_names}
        return {}

    def sync_write(self, register: str, values: dict[str, float]) -> None:
        """
        Write goal positions to motors.

        Args:
            register: Register name (only 'Goal_Position' supported)
            values: Dictionary mapping motor names to goal positions
        """
        if register == "Goal_Position":
            action = {f"{name}.pos": val for name, val in values.items()}
            self._robot.send_action(action)


class PiperRosCamera:
    """
    ROS-based camera abstraction for a single camera.

    Subscribes to a ROS CompressedImage topic and provides the same
    interface as OpenCV cameras for compatibility.
    """

    def __init__(self, name: str, topic: str, image_size: tuple[int, int]):
        self.name = name
        self.topic = topic
        self.image_size = image_size
        self._is_connected = False
        self._latest_image = None
        self._lock = threading.Lock()
        self._sub = None
        self._bridge = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Subscribe to camera topic."""
        _import_ros()
        self._bridge = CvBridge()

        self._sub = rospy.Subscriber(
            self.topic,
            CompressedImage,
            self._callback,
            queue_size=1,
        )
        self._is_connected = True
        logger.info(f"Camera {self.name} connected to {self.topic}")

    def _callback(self, msg) -> None:
        """ROS callback for incoming images."""
        with self._lock:
            try:
                img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
                h, w = self.image_size
                img = cv2.resize(img, (w, h))
                self._latest_image = img
            except Exception as e:
                logger.warning(f"Failed to decode image: {e}")

    def async_read(self) -> np.ndarray:
        """Read the latest image (non-blocking)."""
        with self._lock:
            if self._latest_image is not None:
                return self._latest_image.copy()
            else:
                h, w = self.image_size
                return np.zeros((h, w, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        """Unsubscribe from camera topic."""
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None
        self._is_connected = False
        logger.info(f"Camera {self.name} disconnected")


class PiperRos(Robot):
    """
    Piper dual-arm robot via ROS topics.

    Requires piper_ros driver nodes to be running. Communicates via:
    - Joint state topics for observation
    - End effector pose topics for observation
    - Compressed image topics for camera observation
    - Joint command topics for action execution
    """

    config_class = PiperRosConfig
    name = "piper_ros"

    def __init__(self, config: PiperRosConfig):
        super().__init__(config)
        _import_ros()

        self.config = config
        self._connected = False
        self._lock = threading.Lock()

        # Latest sensor data storage
        self._latest_joints = {
            "left": None,
            "right": None,
        }
        self._latest_velocities = {
            "left": None,
            "right": None,
        }
        self._latest_efforts = {
            "left": None,
            "right": None,
        }
        self._latest_ee_pose = {
            "left": None,
            "right": None,
        }

        # ROS subscribers (initialized on connect)
        self._subs = []
        self._pubs = {}

        # Create ROS-based bus for compatibility with gym_manipulator
        self.bus = PiperRosBus(self)

        # Create ROS-based cameras
        self.cameras = {}
        for cam_name, topic in config.camera_topics.items():
            self.cameras[cam_name] = PiperRosCamera(
                name=cam_name,
                topic=topic,
                image_size=config.image_size,
            )

    @cached_property
    def motor_names(self) -> list[str]:
        """Joint names for each arm."""
        names = []
        if self.config.mode in ["bimanual", "single_left"]:
            for i in range(6):
                names.append(f"left_joint_{i+1}")
            names.append("left_gripper")
        if self.config.mode in ["bimanual", "single_right"]:
            for i in range(6):
                names.append(f"right_joint_{i+1}")
            names.append("right_gripper")
        return names

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation features."""
        features = {}
        # Joint positions
        for name in self.motor_names:
            features[f"{name}.pos"] = float
        # Camera images
        h, w = self.config.image_size
        for cam_name in self.config.camera_topics:
            features[cam_name] = (h, w, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action features (joint positions)."""
        return {f"{name}.pos": float for name in self.motor_names}

    @property
    def is_connected(self) -> bool:
        """Check if connected to ROS."""
        return self._connected and not rospy.is_shutdown()

    @property
    def is_calibrated(self) -> bool:
        """ROS robot doesn't need calibration."""
        return True

    def calibrate(self) -> None:
        """No-op for ROS robot."""
        pass

    def configure(self) -> None:
        """No-op for ROS robot."""
        pass

    # ========== ROS Callbacks ==========

    def _cb_left_joints(self, msg: Any) -> None:
        """Callback for left arm joint states."""
        with self._lock:
            self._latest_joints["left"] = np.array(msg.position, dtype=np.float32)
            if msg.velocity:
                self._latest_velocities["left"] = np.array(msg.velocity, dtype=np.float32)
            if msg.effort:
                self._latest_efforts["left"] = np.array(msg.effort, dtype=np.float32)

    def _cb_right_joints(self, msg: Any) -> None:
        """Callback for right arm joint states."""
        with self._lock:
            self._latest_joints["right"] = np.array(msg.position, dtype=np.float32)
            if msg.velocity:
                self._latest_velocities["right"] = np.array(msg.velocity, dtype=np.float32)
            if msg.effort:
                self._latest_efforts["right"] = np.array(msg.effort, dtype=np.float32)

    def _cb_left_ee_pose(self, msg: Any) -> None:
        """Callback for left arm end effector pose."""
        with self._lock:
            pose = msg.pose
            self._latest_ee_pose["left"] = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ], dtype=np.float32)

    def _cb_right_ee_pose(self, msg: Any) -> None:
        """Callback for right arm end effector pose."""
        with self._lock:
            pose = msg.pose
            self._latest_ee_pose["right"] = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ], dtype=np.float32)

    # ========== Robot Interface Methods ==========

    def connect(self, calibrate: bool = False) -> None:
        """
        Connect to ROS and subscribe to topics.

        Args:
            calibrate: Not used (kept for API compatibility).
        """
        _ = calibrate  # Not used for ROS-based robot

        if self._connected:
            logger.warning(f"{self} already connected")
            return

        # Initialize ROS node if not already done
        if not rospy.core.is_initialized():
            rospy.init_node("piper_ros_lerobot", anonymous=True)

        # Subscribe to joint state topics
        if self.config.mode in ["bimanual", "single_left"]:
            sub = rospy.Subscriber(
                self.config.left_joint_state_topic,
                JointState,
                self._cb_left_joints,
                queue_size=1,
            )
            self._subs.append(sub)

            # Subscribe to end effector pose if topic is configured
            if self.config.left_ee_pose_topic:
                sub = rospy.Subscriber(
                    self.config.left_ee_pose_topic,
                    PoseStamped,
                    self._cb_left_ee_pose,
                    queue_size=1,
                )
                self._subs.append(sub)

        if self.config.mode in ["bimanual", "single_right"]:
            sub = rospy.Subscriber(
                self.config.right_joint_state_topic,
                JointState,
                self._cb_right_joints,
                queue_size=1,
            )
            self._subs.append(sub)

            # Subscribe to end effector pose if topic is configured
            if self.config.right_ee_pose_topic:
                sub = rospy.Subscriber(
                    self.config.right_ee_pose_topic,
                    PoseStamped,
                    self._cb_right_ee_pose,
                    queue_size=1,
                )
                self._subs.append(sub)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Create publishers for joint commands
        if self.config.mode in ["bimanual", "single_left"]:
            self._pubs["left"] = rospy.Publisher(
                self.config.left_joint_cmd_topic,
                JointState,
                queue_size=1,
            )

        if self.config.mode in ["bimanual", "single_right"]:
            self._pubs["right"] = rospy.Publisher(
                self.config.right_joint_cmd_topic,
                JointState,
                queue_size=1,
            )

        self._connected = True
        self.bus.connect()
        logger.info(f"{self} connected to ROS successfully")

        # Wait for initial data
        logger.info("Waiting for sensor data...")
        timeout = 10.0
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                has_joints = all(
                    v is not None for k, v in self._latest_joints.items()
                    if (k == "left" and self.config.mode in ["bimanual", "single_left"]) or
                       (k == "right" and self.config.mode in ["bimanual", "single_right"])
                )
                has_images = all(
                    cam.async_read().any() for cam in self.cameras.values()
                ) if self.cameras else True
            if has_joints and has_images:
                logger.info("Sensor data received successfully")
                return
            time.sleep(0.1)

        logger.warning(f"Timeout waiting for sensor data after {timeout}s")

    def _get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions as dict (for bus compatibility)."""
        positions = {}
        with self._lock:
            if self.config.mode in ["bimanual", "single_left"]:
                left_q = self._latest_joints["left"]
                if left_q is not None:
                    for i in range(min(6, len(left_q))):
                        positions[f"left_joint_{i+1}"] = float(left_q[i])
                    if len(left_q) > 6:
                        positions["left_gripper"] = float(left_q[6])
                    else:
                        positions["left_gripper"] = 0.0

            if self.config.mode in ["bimanual", "single_right"]:
                right_q = self._latest_joints["right"]
                if right_q is not None:
                    for i in range(min(6, len(right_q))):
                        positions[f"right_joint_{i+1}"] = float(right_q[i])
                    if len(right_q) > 6:
                        positions["right_gripper"] = float(right_q[6])
                    else:
                        positions["right_gripper"] = 0.0

        return positions

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from robot."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")

        obs = {}

        # Joint positions (with .pos suffix)
        raw_joints = self._get_raw_joint_positions()
        for name, val in raw_joints.items():
            obs[f"{name}.pos"] = val

        # Camera images
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """
        Send action to robot.

        Args:
            action: Dictionary with joint position targets.

        Returns:
            The action actually sent.
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")

        # Extract positions from action dict
        left_positions = []
        right_positions = []

        for i in range(6):
            key = f"left_joint_{i+1}.pos"
            if key in action:
                left_positions.append(action[key])
        if "left_gripper.pos" in action:
            left_positions.append(action["left_gripper.pos"])

        for i in range(6):
            key = f"right_joint_{i+1}.pos"
            if key in action:
                right_positions.append(action[key])
        if "right_gripper.pos" in action:
            right_positions.append(action["right_gripper.pos"])

        # Apply safety clipping if configured
        if self.config.max_delta_pos is not None:
            raw_joints = self._get_raw_joint_positions()
            left_positions = self._clip_positions("left", left_positions, raw_joints)
            right_positions = self._clip_positions("right", right_positions, raw_joints)

        # Publish commands
        if left_positions and "left" in self._pubs:
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.position = left_positions
            self._pubs["left"].publish(msg)

        if right_positions and "right" in self._pubs:
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.position = right_positions
            self._pubs["right"].publish(msg)

        # Return actual sent action
        sent_action = {}
        for i, pos in enumerate(left_positions[:6]):
            sent_action[f"left_joint_{i+1}.pos"] = pos
        if len(left_positions) > 6:
            sent_action["left_gripper.pos"] = left_positions[6]

        for i, pos in enumerate(right_positions[:6]):
            sent_action[f"right_joint_{i+1}.pos"] = pos
        if len(right_positions) > 6:
            sent_action["right_gripper.pos"] = right_positions[6]

        return sent_action

    def _clip_positions(
        self, arm: str, positions: list[float], raw_joints: dict[str, float]
    ) -> list[float]:
        """Clip positions to max delta from current position."""
        clipped = []
        prefix = f"{arm}_joint_"
        for i, pos in enumerate(positions[:6]):
            key = f"{prefix}{i+1}"
            if key in raw_joints:
                current = raw_joints[key]
                delta = pos - current
                if abs(delta) > self.config.max_delta_pos:
                    pos = current + np.sign(delta) * self.config.max_delta_pos
            clipped.append(pos)

        # Gripper (no clipping)
        if len(positions) > 6:
            clipped.append(positions[6])

        return clipped

    def disconnect(self) -> None:
        """Disconnect from ROS."""
        if not self._connected:
            logger.warning(f"{self} not connected")
            return

        # Unsubscribe
        for sub in self._subs:
            sub.unregister()
        self._subs.clear()

        # Shutdown publishers
        for pub in self._pubs.values():
            pub.unregister()
        self._pubs.clear()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        self._connected = False
        self.bus.disconnect()
        logger.info(f"{self} disconnected")

    # ========== Extended Methods for RL ==========

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        """Get current end effector poses."""
        with self._lock:
            result = {}
            if self._latest_ee_pose["left"] is not None:
                result["left"] = self._latest_ee_pose["left"].copy()
            if self._latest_ee_pose["right"] is not None:
                result["right"] = self._latest_ee_pose["right"].copy()
            return result

    def get_joint_velocities(self) -> dict[str, np.ndarray]:
        """Get current joint velocities."""
        with self._lock:
            result = {}
            if self._latest_velocities["left"] is not None:
                result["left"] = self._latest_velocities["left"].copy()
            if self._latest_velocities["right"] is not None:
                result["right"] = self._latest_velocities["right"].copy()
            return result

    def get_joint_efforts(self) -> dict[str, np.ndarray]:
        """Get current joint efforts (torques)."""
        with self._lock:
            result = {}
            if self._latest_efforts["left"] is not None:
                result["left"] = self._latest_efforts["left"].copy()
            if self._latest_efforts["right"] is not None:
                result["right"] = self._latest_efforts["right"].copy()
            return result

    def get_state_vector(self) -> np.ndarray:
        """
        Get full state vector for RL (joints concatenated).

        Returns:
            Array of shape (14,) for bimanual: [left_j1..j6, left_gripper, right_j1..j6, right_gripper]
        """
        with self._lock:
            parts = []
            if self.config.mode in ["bimanual", "single_left"]:
                if self._latest_joints["left"] is not None:
                    parts.append(self._latest_joints["left"][:7])
                else:
                    parts.append(np.zeros(7, dtype=np.float32))

            if self.config.mode in ["bimanual", "single_right"]:
                if self._latest_joints["right"] is not None:
                    parts.append(self._latest_joints["right"][:7])
                else:
                    parts.append(np.zeros(7, dtype=np.float32))

            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    def send_action_array(self, action: np.ndarray) -> np.ndarray:
        """
        Send action as array (for RL compatibility).

        Args:
            action: Array of joint positions [left_j1..j6, left_gripper, right_j1..j6, right_gripper]

        Returns:
            The action actually sent.
        """
        action_dict = {}
        idx = 0

        if self.config.mode in ["bimanual", "single_left"]:
            for i in range(6):
                action_dict[f"left_joint_{i+1}.pos"] = float(action[idx])
                idx += 1
            action_dict["left_gripper.pos"] = float(action[idx])
            idx += 1

        if self.config.mode in ["bimanual", "single_right"]:
            for i in range(6):
                action_dict[f"right_joint_{i+1}.pos"] = float(action[idx])
                idx += 1
            action_dict["right_gripper.pos"] = float(action[idx])

        sent = self.send_action(action_dict)

        # Convert back to array
        result = []
        if self.config.mode in ["bimanual", "single_left"]:
            for i in range(6):
                result.append(sent.get(f"left_joint_{i+1}.pos", 0.0))
            result.append(sent.get("left_gripper.pos", 0.0))

        if self.config.mode in ["bimanual", "single_right"]:
            for i in range(6):
                result.append(sent.get(f"right_joint_{i+1}.pos", 0.0))
            result.append(sent.get("right_gripper.pos", 0.0))

        return np.array(result, dtype=np.float32)

    def __repr__(self) -> str:
        return f"PiperRos(mode={self.config.mode})"