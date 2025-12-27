from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    # CAN interface name to connect to the arm
    can_name: str = "can0"

    # Disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Control mode: "joint" for joint control, "end" for end-effector control
    control_mode: str = "joint"

    # Movement mode: 0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L, 0x03=MOVE_C
    move_mode: int = 0x01

    # Movement speed percentage (0-100)
    move_speed_rate: int = 50

    # Enable SDK joint limit checking
    enable_sdk_joint_limit: bool = True

    # Enable SDK gripper limit checking
    enable_sdk_gripper_limit: bool = True

    # DH parameter offset flag (0x00 or 0x01)
    dh_is_offset: int = 0x01
