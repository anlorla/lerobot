"""
Piper dual-arm robot Gym environment for HIL-SERL.

This module provides a Gym-compatible environment for the Piper robot
that integrates with the HIL-SERL training pipeline.

Architecture:
    PiperRosEnv (gym.Env)
    └── PiperRos (Robot)
        ├── PiperRosBus (joint state/command via ROS)
        └── PiperRosCameras (image via ROS CompressedImage)

Usage:
    # Create environment
    env = make_piper_env(mode="bimanual", fps=10)

    # Run episode
    obs, info = env.reset()
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from lerobot.robots.piper_ros import PiperRos, PiperRosConfig
from lerobot.teleoperators.utils import TeleopEvents

logger = logging.getLogger(__name__)


# ============================================================================
# Safety Configuration for Piper Robot
# ============================================================================

@dataclass
class JointLimits:
    """Joint position limits from URDF (radians)."""
    lower: float
    upper: float


# Piper robot joint limits (from piper_x_description.urdf)
# Format: (lower_limit, upper_limit) in radians
PIPER_JOINT_LIMITS: dict[str, JointLimits] = {
    # Joint 1: base rotation
    "joint1": JointLimits(lower=-2.618, upper=2.618),    # ~±150°
    # Joint 2: shoulder
    "joint2": JointLimits(lower=0.0, upper=3.14),        # 0° ~ 180°
    # Joint 3: elbow
    "joint3": JointLimits(lower=-2.9671, upper=0.0),     # ~-170° ~ 0°
    # Joint 4: wrist pitch
    "joint4": JointLimits(lower=-1.57, upper=1.57),      # ~±90°
    # Joint 5: wrist roll
    "joint5": JointLimits(lower=-1.57, upper=1.57),      # ~±90°
    # Joint 6: wrist yaw
    "joint6": JointLimits(lower=-3.14, upper=3.14),      # ~±180°
    # Joint 7: gripper
    "joint7": JointLimits(lower=0.0, upper=0.05),        # 0 ~ 5cm
}


@dataclass
class SafetyConfig:
    """Safety configuration for robot control."""

    # Joint position limits (will be populated from PIPER_JOINT_LIMITS)
    joint_limits: dict[str, JointLimits] = field(default_factory=dict)

    # Maximum joint velocity (rad/step) - max change per control step
    max_delta_position: float = 0.1  # ~5.7° per step

    # Safety margin from joint limits (radians)
    limit_margin: float = 0.05  # ~3° margin

    # Enable/disable safety features
    enable_position_limits: bool = True
    enable_velocity_limits: bool = True
    enable_emergency_stop: bool = True

    # Emergency stop threshold (if joint moves unexpectedly fast)
    emergency_velocity_threshold: float = 0.5  # rad/step

    def __post_init__(self):
        if not self.joint_limits:
            self.joint_limits = PIPER_JOINT_LIMITS.copy()


class SafetyChecker:
    """
    Safety checker for Piper robot actions.

    Provides:
    - Joint position limit checking and clamping
    - Velocity (delta position) limiting
    - Emergency stop detection
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._last_positions: dict[str, float] = {}
        self._emergency_stop = False
        self._violation_count = 0

    def reset(self) -> None:
        """Reset safety checker state."""
        self._last_positions = {}
        self._emergency_stop = False
        self._violation_count = 0

    def check_and_clip_action(
        self,
        target_positions: dict[str, float],
        current_positions: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Check action safety and clip to safe ranges.

        Args:
            target_positions: Desired joint positions (joint_name.pos -> value)
            current_positions: Current joint positions

        Returns:
            Tuple of (safe_positions, safety_info)
        """
        safe_positions = {}
        safety_info = {
            "position_violations": [],
            "velocity_violations": [],
            "clipped_joints": [],
            "emergency_stop": False,
        }

        for key, target in target_positions.items():
            # Extract joint name (e.g., "left_joint1.pos" -> "joint1")
            joint_name = self._extract_joint_name(key)
            current = current_positions.get(key, target)

            # Get limits for this joint
            limits = self._get_joint_limits(joint_name)

            safe_target = target
            was_clipped = False

            # Check velocity limits (delta position)
            if self.config.enable_velocity_limits:
                delta = target - current
                if abs(delta) > self.config.max_delta_position:
                    # Clip delta to max velocity
                    clipped_delta = np.clip(
                        delta,
                        -self.config.max_delta_position,
                        self.config.max_delta_position
                    )
                    safe_target = current + clipped_delta
                    was_clipped = True
                    safety_info["velocity_violations"].append({
                        "joint": key,
                        "requested_delta": delta,
                        "actual_delta": clipped_delta,
                    })

                # Check for emergency stop condition
                if self.config.enable_emergency_stop:
                    if abs(delta) > self.config.emergency_velocity_threshold:
                        logger.warning(
                            f"Emergency velocity detected on {key}: delta={delta:.4f}"
                        )
                        self._violation_count += 1
                        if self._violation_count > 3:
                            self._emergency_stop = True
                            safety_info["emergency_stop"] = True

            # Check position limits
            if self.config.enable_position_limits and limits is not None:
                lower = limits.lower + self.config.limit_margin
                upper = limits.upper - self.config.limit_margin

                if safe_target < lower:
                    safety_info["position_violations"].append({
                        "joint": key,
                        "requested": safe_target,
                        "limit": "lower",
                        "limit_value": lower,
                    })
                    safe_target = lower
                    was_clipped = True
                elif safe_target > upper:
                    safety_info["position_violations"].append({
                        "joint": key,
                        "requested": safe_target,
                        "limit": "upper",
                        "limit_value": upper,
                    })
                    safe_target = upper
                    was_clipped = True

            if was_clipped:
                safety_info["clipped_joints"].append(key)

            safe_positions[key] = safe_target

        # Update last positions for next check
        self._last_positions = current_positions.copy()

        return safe_positions, safety_info

    def _extract_joint_name(self, key: str) -> str:
        """Extract joint name from full key (e.g., 'left_joint1.pos' -> 'joint1')."""
        # Remove arm prefix and .pos suffix
        name = key.replace(".pos", "")
        for prefix in ["left_", "right_"]:
            if prefix in name:
                name = name.replace(prefix, "")
        return name

    def _get_joint_limits(self, joint_name: str) -> JointLimits | None:
        """Get limits for a joint by name."""
        return self.config.joint_limits.get(joint_name)

    @property
    def is_emergency_stop(self) -> bool:
        """Check if emergency stop is active."""
        return self._emergency_stop

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop (requires manual acknowledgment)."""
        self._emergency_stop = False
        self._violation_count = 0
        logger.info("Emergency stop cleared")


class PiperRosEnv(gym.Env):
    """
    Gym environment for Piper dual-arm robot via ROS.

    Compatible with HIL-SERL training pipeline and gym_manipulator.py.

    Features:
    - Joint position control (absolute positions)
    - Camera observation (RGB images)
    - Compatible with gym_manipulator.py RobotEnv interface
    - Support for teleoperation intervention
    """

    def __init__(
        self,
        robot_config: PiperRosConfig | None = None,
        use_gripper: bool = True,
        display_cameras: bool = False,
        reset_pose: dict[str, float] | None = None,
        reset_time_s: float = 3.0,
        control_time_s: float = 30.0,
        fps: int = 10,
        safety_config: SafetyConfig | None = None,
        enable_safety: bool = True,
    ):
        """
        Initialize Piper robot environment.

        Args:
            robot_config: Configuration for PiperRos robot.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to display camera feeds.
            reset_pose: Joint positions for reset (dict with joint names).
            reset_time_s: Time to wait during reset.
            control_time_s: Max episode time in seconds.
            fps: Control frequency.
            safety_config: Safety configuration for joint limits and velocity.
            enable_safety: Whether to enable safety checking (default: True).
        """
        super().__init__()

        # Create robot
        if robot_config is None:
            robot_config = PiperRosConfig(
                mode="bimanual",
                image_size=(224, 224),
                control_freq=fps,
            )
        self.robot = PiperRos(robot_config)

        self.use_gripper = use_gripper
        self.display_cameras = display_cameras
        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s
        self.control_time_s = control_time_s
        self.fps = fps
        self.max_episode_steps = int(control_time_s * fps)

        # Safety checker
        self.enable_safety = enable_safety
        self.safety_checker = SafetyChecker(safety_config) if enable_safety else None

        # Episode tracking
        self.current_step = 0
        self._raw_joint_positions = None

        # Joint names (from robot)
        self._joint_names = None
        self._image_keys = list(robot_config.camera_topics.keys())

        # Connect robot
        if not self.robot.is_connected:
            self.robot.connect()

        # Setup motor names after connection
        self._joint_names = self.robot.motor_names

        # Setup spaces after connection
        self._setup_spaces()

        if self.enable_safety:
            logger.info("Safety checker enabled with joint limits and velocity limits")

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces."""
        obs = self._get_observation()

        # Observation space
        observation_spaces = {}

        # Images
        for key in self._image_keys:
            if "pixels" in obs and key in obs["pixels"]:
                img = obs["pixels"][key]
                observation_spaces[f"observation.images.{key}"] = gym.spaces.Box(
                    low=0, high=255, shape=img.shape, dtype=np.uint8
                )

        # State (joint positions)
        agent_pos = obs["agent_pos"]
        observation_spaces["observation.state"] = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=agent_pos.shape, dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Action space: absolute joint positions
        # For HIL-SERL, actions are absolute joint positions
        action_dim = len(self._joint_names)
        self.action_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(action_dim,), dtype=np.float32
        )

    def _get_observation(self) -> dict[str, Any]:
        """Get current observation from robot."""
        obs_dict = self.robot.get_observation()

        # Extract joint positions
        joint_positions = []
        raw_positions = {}
        for name in self._joint_names:
            key = f"{name}.pos"
            val = obs_dict.get(key, 0.0)
            joint_positions.append(val)
            raw_positions[key] = val

        self._raw_joint_positions = raw_positions

        # Extract images
        pixels = {}
        for key in self._image_keys:
            if key in obs_dict:
                pixels[key] = obs_dict[key]

        return {
            "agent_pos": np.array(joint_positions, dtype=np.float32),
            "pixels": pixels,
            **raw_positions,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed, options=options)

        # Reset safety checker
        if self.safety_checker is not None:
            self.safety_checker.reset()

        # Move to reset pose if specified
        if self.reset_pose is not None:
            logger.info("Moving to reset pose...")
            self._move_to_pose(self.reset_pose)

        # Wait for reset
        time.sleep(self.reset_time_s)

        self.current_step = 0
        obs = self._get_observation()

        info = {
            TeleopEvents.IS_INTERVENTION: False,
            "raw_joint_positions": self._raw_joint_positions,
            "safety_info": None,
        }

        return obs, info

    def _move_to_pose(self, target_pose: dict[str, float], steps: int = 50) -> None:
        """Smoothly move to target pose."""
        current_obs = self._get_observation()
        current_pos = {k: current_obs[k] for k in target_pose.keys() if k in current_obs}

        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = {}
            for key in target_pose:
                if key in current_pos:
                    interpolated[key] = (1 - alpha) * current_pos[key] + alpha * target_pose[key]
                else:
                    interpolated[key] = target_pose[key]

            self.robot.send_action(interpolated)
            time.sleep(0.02)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Array of absolute joint positions

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Check for emergency stop
        if self.safety_checker is not None and self.safety_checker.is_emergency_stop:
            logger.error("Emergency stop active! Cannot execute action.")
            obs = self._get_observation()
            return obs, 0.0, True, False, {
                TeleopEvents.IS_INTERVENTION: False,
                "safety_info": {"emergency_stop": True},
                "terminated_reason": "emergency_stop",
            }

        # Get current observation for safety checking
        current_obs = self._get_observation()
        current_positions = {
            f"{name}.pos": float(current_obs["agent_pos"][i])
            for i, name in enumerate(self._joint_names)
        }

        # Convert action array to dict
        if len(action) == len(self._joint_names):
            joint_targets = {
                f"{name}.pos": float(action[i])
                for i, name in enumerate(self._joint_names)
            }
        else:
            # Fallback: use current + delta
            current_joints = current_obs["agent_pos"]
            joint_targets = {}
            for i, name in enumerate(self._joint_names):
                if i < len(action):
                    # Small delta scale for safety
                    delta = action[i] * 0.05
                    joint_targets[f"{name}.pos"] = float(current_joints[i] + delta)
                else:
                    joint_targets[f"{name}.pos"] = float(current_joints[i])

        # Apply safety checking and clipping
        safety_info = None
        if self.safety_checker is not None:
            joint_targets, safety_info = self.safety_checker.check_and_clip_action(
                joint_targets, current_positions
            )

            # Log safety violations
            if safety_info["clipped_joints"]:
                logger.debug(f"Safety clipped joints: {safety_info['clipped_joints']}")

            if safety_info["position_violations"]:
                logger.warning(
                    f"Position limit violations: {len(safety_info['position_violations'])}"
                )

            if safety_info["velocity_violations"]:
                logger.debug(
                    f"Velocity limit violations: {len(safety_info['velocity_violations'])}"
                )

            # Check for emergency stop triggered during this step
            if safety_info["emergency_stop"]:
                logger.error("Emergency stop triggered!")
                obs = self._get_observation()
                return obs, 0.0, True, False, {
                    TeleopEvents.IS_INTERVENTION: False,
                    "safety_info": safety_info,
                    "terminated_reason": "emergency_stop",
                }

        # Send action to robot
        self.robot.send_action(joint_targets)

        # Get new observation
        obs = self._get_observation()
        self.current_step += 1

        # Display if requested
        if self.display_cameras:
            self.render()

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # Reward is computed externally (by reward classifier in HIL-SERL)
        reward = 0.0

        info = {
            TeleopEvents.IS_INTERVENTION: False,
            "safety_info": safety_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Display camera feeds."""
        obs = self._get_observation()
        if "pixels" in obs:
            for key, img in obs["pixels"].items():
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(key, img_bgr)
            cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()
        cv2.destroyAllWindows()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions (for processor compatibility)."""
        return self._raw_joint_positions or {}

    # Properties for compatibility with HIL-SERL and gym_manipulator.py
    @property
    def motor_names(self) -> list[str]:
        """Motor names for compatibility."""
        return self._joint_names


def make_piper_env(
    mode: str = "bimanual",
    use_gripper: bool = True,
    fps: int = 10,
    image_size: tuple[int, int] = (224, 224),
    reset_pose: dict[str, float] | None = None,
    reset_time_s: float = 3.0,
    control_time_s: float = 30.0,
    display_cameras: bool = False,
    enable_safety: bool = True,
    max_delta_position: float = 0.1,
    **kwargs,
) -> PiperRosEnv:
    """
    Factory function to create Piper environment.

    Args:
        mode: Robot mode ("bimanual", "single_left", "single_right").
        use_gripper: Whether to include gripper.
        fps: Control frequency.
        image_size: Image size for cameras.
        reset_pose: Joint positions for reset.
        reset_time_s: Time to wait during reset.
        control_time_s: Max episode time.
        display_cameras: Whether to display camera feeds.
        enable_safety: Whether to enable safety checking (default: True).
        max_delta_position: Maximum joint position change per step (radians).
        **kwargs: Additional arguments for PiperRosConfig.

    Returns:
        Configured PiperRosEnv instance.
    """
    config = PiperRosConfig(
        mode=mode,
        image_size=image_size,
        control_freq=fps,
        **kwargs,
    )

    # Create safety config
    safety_config = SafetyConfig(
        max_delta_position=max_delta_position,
    ) if enable_safety else None

    return PiperRosEnv(
        robot_config=config,
        use_gripper=use_gripper,
        fps=fps,
        reset_pose=reset_pose,
        reset_time_s=reset_time_s,
        control_time_s=control_time_s,
        display_cameras=display_cameras,
        safety_config=safety_config,
        enable_safety=enable_safety,
    )


def make_piper_robot_env(cfg) -> tuple[gym.Env, Any]:
    """
    Create Piper robot environment from HIL-SERL config.

    This function matches the interface of make_robot_env in gym_manipulator.py.

    Args:
        cfg: HILSerlRobotEnvConfig configuration

    Returns:
        Tuple of (gym environment, teleoperator device)
    """
    from lerobot.teleoperators import make_teleoperator_from_config

    # Create robot config from env config
    robot_config = PiperRosConfig(
        mode="bimanual",
        image_size=(224, 224),
        control_freq=cfg.fps,
    )

    # Create teleoperator if configured
    teleop_device = None
    if cfg.teleop is not None:
        teleop_device = make_teleoperator_from_config(cfg.teleop)
        teleop_device.connect()

    # Get reset configuration
    reset_pose = None
    reset_time_s = 5.0
    if cfg.processor.reset is not None:
        reset_pose = cfg.processor.reset.fixed_reset_joint_positions
        reset_time_s = cfg.processor.reset.reset_time_s if hasattr(cfg.processor.reset, 'reset_time_s') else 5.0

    # Get display configuration
    display_cameras = False
    if cfg.processor.observation is not None:
        display_cameras = cfg.processor.observation.display_cameras

    # Create environment
    use_gripper = True
    if cfg.processor.gripper is not None:
        use_gripper = cfg.processor.gripper.use_gripper

    env = PiperRosEnv(
        robot_config=robot_config,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
        reset_time_s=reset_time_s,
        fps=cfg.fps,
    )

    return env, teleop_device