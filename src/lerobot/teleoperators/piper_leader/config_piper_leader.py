"""Configuration for Piper leader arm teleoperator."""

from dataclasses import dataclass, field
from typing import Literal

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    """
    Configuration for Piper leader arm(s) as teleoperator.

    The leader arms are used for teleoperation/intervention in HIL-SERL.
    Communication is via ROS topics.
    """

    # Mode: which arms to use for teleoperation
    mode: Literal["bimanual", "single_left", "single_right"] = "bimanual"

    # ROS topics for leader arm joint states
    left_leader_topic: str = "/robot/leader_left/joint_states"
    right_leader_topic: str = "/robot/leader_right/joint_states"

    # Intervention detection threshold
    # If any joint moves more than this (rad), consider it intervention
    intervention_threshold: float = 0.02

    # Number of joints per arm
    joints_per_arm: int = 7

    # Deadzone for joystick-like control
    deadzone: float = 0.01