"""Configuration for Piper dual-arm robot via ROS."""

from dataclasses import dataclass, field
from typing import Literal

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_ros")
@dataclass
class PiperRosConfig(RobotConfig):
    """
    Configuration for Piper dual-arm robot using ROS topics.

    This robot class communicates via ROS topics, requiring the piper_ros
    driver nodes to be running. Much simpler than direct CAN communication.

    ROS Topics (default):
        Subscribed:
            /robot/arm_left/joint_states_single - Left arm joint states
            /robot/arm_right/joint_states_single - Right arm joint states
            /robot/arm_left/end_pose - Left arm end effector pose
            /robot/arm_right/end_pose - Right arm end effector pose
            /realsense_*/color/image_raw/compressed - Camera images

        Published:
            /robot/arm_left/policy_joint_cmd - Left arm joint commands
            /robot/arm_right/policy_joint_cmd - Right arm joint commands
    """

    # Robot mode: "bimanual" for dual-arm, "single_left", "single_right"
    mode: Literal["bimanual", "single_left", "single_right"] = "bimanual"

    # Number of joints per arm (Piper has 6 DOF + gripper)
    joints_per_arm: int = 7  # 6 joints + 1 gripper

    # ========== ROS Topic Configuration ==========

    # Joint state topics (input - observation)
    left_joint_state_topic: str = "/robot/arm_left/joint_states_single"
    right_joint_state_topic: str = "/robot/arm_right/joint_states_single"

    # Joint command topics (output - action)
    left_joint_cmd_topic: str = "/robot/arm_left/policy_joint_cmd"
    right_joint_cmd_topic: str = "/robot/arm_right/policy_joint_cmd"

    # End effector pose topics (optional, for RL with EE control)
    left_ee_pose_topic: str = "/robot/arm_left/end_pose"
    right_ee_pose_topic: str = "/robot/arm_right/end_pose"

    # Camera topics (compressed images)
    camera_topics: dict[str, str] = field(default_factory=lambda: {
        "cam_top": "/realsense_top/color/image_raw/compressed",
        "cam_left_wrist": "/realsense_left/color/image_raw/compressed",
        "cam_right_wrist": "/realsense_right/color/image_raw/compressed",
    })

    # ========== Image Configuration ==========

    # Image size for resizing (policy input size)
    image_size: tuple[int, int] = (224, 224)

    # ========== Control Configuration ==========

    # Control frequency in Hz
    control_freq: int = 10

    # Safety: max joint position change per step (radians)
    # Set to None to disable safety clipping
    max_delta_pos: float | None = 0.1  # ~5.7 degrees

    # Disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # ========== Timeout Configuration ==========

    # Timeout for waiting for initial sensor data (seconds)
    sensor_timeout: float = 10.0