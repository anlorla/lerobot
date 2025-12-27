import logging
import sys
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)

# Add piper_sdk to Python path if available
try:
    # Try to import from NPM-Ros project structure
    piper_sdk_path = Path("/home/zeno-yifan/NPM-Project/NPM-Ros/piper_sdk")
    if piper_sdk_path.exists():
        sys.path.insert(0, str(piper_sdk_path))
    from piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2
    PIPER_SDK_AVAILABLE = True
except ImportError:
    logger.warning("Piper SDK not found. Please install piper_sdk or check the path.")
    PIPER_SDK_AVAILABLE = False
    C_PiperInterface_V2 = None


class PiperFollower(Robot):
    """
    Piper robotic arm follower implementation for LeRobot.

    This class provides an interface to control the Piper 6-DOF robotic arm
    via CAN bus communication using the piper_sdk library.

    Features:
    - 6-DOF joint control with gripper
    - CAN bus communication
    - Joint angle and gripper position control
    - Camera integration support
    """

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)

        if not PIPER_SDK_AVAILABLE:
            raise ImportError(
                "Piper SDK is not available. Please install piper_sdk from NPM-Ros project."
            )

        self.config = config

        # Initialize Piper interface
        self.interface = C_PiperInterface_V2(
            can_name=config.can_name,
            judge_flag=True,
            can_auto_init=True,
            dh_is_offset=config.dh_is_offset,
            start_sdk_joint_limit=config.enable_sdk_joint_limit,
            start_sdk_gripper_limit=config.enable_sdk_gripper_limit,
        )

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Store motor names for joint motors
        self.motor_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation features including motor positions and camera images."""
        motors_ft = {f"{motor}.pos": float for motor in self.motor_names}
        cameras_ft = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        return {**motors_ft, **cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action features (motor positions)."""
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def is_connected(self) -> bool:
        """Check if the robot and cameras are connected."""
        return (
            self.interface.get_connect_status()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = False) -> None:
        """
        Connect to the Piper arm via CAN bus and initialize cameras.

        Args:
            calibrate: Not used for Piper (kept for API compatibility).
                      Piper arm uses SDK-based joint limits instead of calibration files.
        """
        # Note: calibrate parameter kept for API compatibility with Robot base class
        # Piper uses SDK joint limits configured via enable_sdk_joint_limit
        _ = calibrate  # Mark as intentionally unused

        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN interface
        self.interface.ConnectPort(can_init=True, piper_init=True, start_thread=True)

        # Wait for connection to stabilize
        time.sleep(0.5)

        # Enable the arm
        self.interface.EnableArm(7)  # 7 means all motors

        # Set control mode
        self.interface.ModeCtrl(
            ctrl_mode=0x01,  # CAN command control mode
            move_mode=self.config.move_mode,
            move_spd_rate_ctrl=self.config.move_speed_rate,
        )

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected successfully.")

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from the robot.

        Returns:
            Dictionary containing joint positions, gripper position, and camera images.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read joint positions
        start = time.perf_counter()
        joint_msgs = self.interface.GetArmJointMsgs()
        gripper_msgs = self.interface.GetArmGripperMsgs()

        # Convert joint angles from 0.001 degrees to normalized values
        # Piper SDK returns angles in 0.001 degrees (milli-degrees)
        obs_dict = {
            "joint_1.pos": joint_msgs.joint_state.joint_1 / 1000.0 / 180.0 * 3.14159,  # Convert to radians
            "joint_2.pos": joint_msgs.joint_state.joint_2 / 1000.0 / 180.0 * 3.14159,
            "joint_3.pos": joint_msgs.joint_state.joint_3 / 1000.0 / 180.0 * 3.14159,
            "joint_4.pos": joint_msgs.joint_state.joint_4 / 1000.0 / 180.0 * 3.14159,
            "joint_5.pos": joint_msgs.joint_state.joint_5 / 1000.0 / 180.0 * 3.14159,
            "joint_6.pos": joint_msgs.joint_state.joint_6 / 1000.0 / 180.0 * 3.14159,
            "gripper.pos": gripper_msgs.gripper_state.grippers_angle / 1000.0 / 1000.0,  # Convert from 0.001mm to m
        }

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """
        Send action commands to the robot.

        Args:
            action: Dictionary containing target joint and gripper positions

        Returns:
            The action actually sent (potentially clipped for safety)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position for safety
        if self.config.max_relative_target is not None:
            present_obs = self.get_observation()
            present_pos = {key.removesuffix(".pos"): val for key, val in present_obs.items() if key.endswith(".pos")}
            goal_present_pos = {key: (goal_pos[key], present_pos[key]) for key in goal_pos.keys()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Convert joint angles from radians to 0.001 degrees for Piper SDK
        # Piper SDK expects angles in 0.001 degrees (milli-degrees)
        joint_1 = int(goal_pos.get("joint_1", 0) / 3.14159 * 180.0 * 1000.0)
        joint_2 = int(goal_pos.get("joint_2", 0) / 3.14159 * 180.0 * 1000.0)
        joint_3 = int(goal_pos.get("joint_3", 0) / 3.14159 * 180.0 * 1000.0)
        joint_4 = int(goal_pos.get("joint_4", 0) / 3.14159 * 180.0 * 1000.0)
        joint_5 = int(goal_pos.get("joint_5", 0) / 3.14159 * 180.0 * 1000.0)
        joint_6 = int(goal_pos.get("joint_6", 0) / 3.14159 * 180.0 * 1000.0)

        # Send joint control command
        self.interface.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)

        # Send gripper command if present
        if "gripper" in goal_pos:
            # Convert gripper position from meters to 0.001mm for Piper SDK
            gripper_angle = int(goal_pos["gripper"] * 1000.0 * 1000.0)
            self.interface.GripperCtrl(
                gripper_angle=gripper_angle,
                gripper_effort=1000,  # Default effort (1 N·m)
                gripper_code=0x01,    # Enable gripper
                set_zero=0x00         # Don't set zero
            )

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        """Disconnect from the robot and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disable arm if configured
        if self.config.disable_torque_on_disconnect:
            self.interface.DisableArm(7)  # 7 means all motors

        # Disconnect CAN interface
        self.interface.DisconnectPort()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected successfully.")
