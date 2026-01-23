"""
Real Robot Environment Interface

Provides a unified interface for real robot control and sensing,
similar to PyBullet environment structure.

Responsibilities:
- Robot arm control (move, get pose)
- Gripper control (open, close, get state)
- Camera image acquisition (head, hand-eye)
- TF transformations
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Callable
from scipy.spatial.transform import Rotation as R

from airbot_sdk import MMK2RealRobot
from tf_client import TFClient
from mmk2_types.types import MMK2Components, ImageTypes
from mmk2_types.grpc_msgs import Pose, Position, Orientation, JointState, TrajectoryParams, GoalStatus


# ==================== Configuration ====================
CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")
CONFIG_DIR = Path("/home/ypf/qiuzhiarm_LLM/config")


class RealRobotEnv:
    """
    Real Robot Environment Interface.
    
    Provides a clean interface for:
    - Robot state reading (arm pose, gripper state)
    - Robot control (arm movement, gripper control)
    - Sensor data acquisition (cameras, TF)
    
    Similar to PyBullet environment but for real hardware.
    """
    
    def __init__(
        self,
        ip: str = "192.168.11.200",
        tf_host: str = "127.0.0.1",
        tf_port: int = 9999,
        use_left: bool = True,
        config_path: Optional[str] = None,
    ):
        """
        Initialize robot environment.
        
        Args:
            ip: Robot IP address
            tf_host: TF server host
            tf_port: TF server port
            use_left: Use left arm (True) or right arm (False)
            config_path: Path to configuration file
        """
        self.use_left = use_left
        self.arm_key = 'left_arm' if use_left else 'right_arm'
        self.arm_component = MMK2Components.LEFT_ARM if use_left else MMK2Components.RIGHT_ARM
        self.eef_component = MMK2Components.LEFT_ARM_EEF if use_left else MMK2Components.RIGHT_ARM_EEF
        self.camera_component = MMK2Components.LEFT_CAMERA if use_left else MMK2Components.RIGHT_CAMERA
        
        # Load config
        if config_path is None:
            config_path = str(CONFIG_DIR / "llm_config.json")
        self.config = self._load_config(config_path)
        
        # Gripper parameters
        gripper_config = self.config.get("gripper", {})
        self.gripper_max_opening = gripper_config.get("max_opening", 0.08)
        
        # Initialize robot
        print(f"[RobotEnv] Connecting to robot {ip}...")
        self.robot = MMK2RealRobot(ip=ip)
        
        # Initialize TF client
        self.tf_client = TFClient(host=tf_host, port=tf_port, auto_connect=True)
        
        # Set initial pose
        self._init_robot_pose()
        
        print(f"[RobotEnv] Initialized (use_left={use_left})")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(config_path)
        if not path.exists():
            print(f"[RobotEnv] Warning: Config not found at {path}")
            return {}
        with open(path) as f:
            return json.load(f)
    
    def _init_robot_pose(self):
        """Initialize robot to default pose."""
        self.robot.set_robot_head_pose(0, -1.08)
        self.robot.set_spine(0.05)
    
    # ==================== Arm State ====================
    
    def get_arm_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get current arm end-effector pose.
        
        Returns:
            Tuple of (rotation_matrix, position) or None if failed
            - rotation_matrix: 3x3 rotation matrix
            - position: [x, y, z] position in meters
        """
        try:
            poses = self.robot.get_arm_end_poses()
            if poses and self.arm_key in poses:
                p = poses[self.arm_key]
                rotation = R.from_quat(p['orientation']).as_matrix()
                position = np.array(p['position'])
                return rotation, position
        except Exception as e:
            print(f"[RobotEnv] Error getting arm pose: {e}")
        return None
    
    def get_tcp_position(self) -> Optional[np.ndarray]:
        """Get current TCP position [x, y, z]."""
        pose = self.get_arm_pose()
        return pose[1] if pose else None
    
    def get_tcp_pose_dict(self) -> Optional[Dict[str, float]]:
        """Get TCP pose as dictionary with x, y, z, roll, pitch, yaw."""
        pose = self.get_arm_pose()
        if pose is None:
            return None
        
        R_mat, position = pose
        euler = R.from_matrix(R_mat).as_euler('xyz')
        
        return {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'roll': float(euler[0]),
            'pitch': float(euler[1]),
            'yaw': float(euler[2]),
        }
    
    # ==================== Arm Control ====================
    
    def move_arm(
        self, 
        position: List[float], 
        yaw: float, 
        wait: float = 2.0,
        check_abort: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """
        Move arm to target position with specified yaw.
        
        Args:
            position: [x, y, z] target position in meters
            yaw: target yaw angle in radians
            wait: time to wait after movement (seconds)
            check_abort: optional function to check if abort requested
            
        Returns:
            True if successful, False otherwise
        """
        if check_abort and check_abort():
            return False
        
        # Compute orientation quaternion
        ox, oy, oz, ow = self._compute_grasp_orientation(yaw)
        
        target_pose = {
            self.arm_component: Pose(
                position=Position(
                    x=float(position[0]), 
                    y=float(position[1]), 
                    z=float(position[2])
                ),
                orientation=Orientation(x=ox, y=oy, z=oz, w=ow),
            ),
        }
        
        try:
            self.robot.control_arm_poses(target_pose)
            return self._wait_with_check(wait, check_abort)
        except Exception as e:
            print(f"[RobotEnv] Move arm failed: {e}")
            return False
    
    def _compute_grasp_orientation(self, yaw: float) -> Tuple[float, float, float, float]:
        """Compute quaternion for grasp orientation (gripper pointing down)."""
        r_base = R.from_euler('y', np.pi / 2)
        r_yaw = R.from_euler('z', yaw)
        r_final = r_yaw * r_base
        quat = r_final.as_quat()
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def reset_position(self) -> bool:
        """Reset arm to initial position."""
        print("[RobotEnv] Resetting to initial position...")
        
        arm_action = {
            MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
            MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.SPINE: JointState(position=[0.05]),
        }
        
        try:
            result = self.robot.mmk2.set_goal(arm_action, TrajectoryParams())
            success = result.value == GoalStatus.Status.SUCCESS
            if success:
                print("[RobotEnv] Reset complete")
            return success
        except Exception as e:
            print(f"[RobotEnv] Reset failed: {e}")
            return False
    
    # ==================== Gripper Control ====================
    
    def open_gripper(self, gap: Optional[float] = None) -> bool:
        """
        Open gripper to specified gap or fully open.
        
        Args:
            gap: target gap in meters (None for fully open)
            
        Returns:
            True if successful
        """
        if gap is None:
            position = 1.0
        else:
            position = min(1.0, gap / self.gripper_max_opening)
        
        try:
            action = {self.eef_component: JointState(position=[position])}
            self.robot.mmk2.set_goal(action, TrajectoryParams())
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"[RobotEnv] Gripper open failed: {e}")
            return False
    
    def close_gripper(self) -> bool:
        """Close gripper."""
        try:
            self.robot.close_gripper(left=self.use_left, right=not self.use_left)
            time.sleep(0.8)
            return True
        except Exception as e:
            print(f"[RobotEnv] Gripper close failed: {e}")
            return False
    
    def get_gripper_state(self) -> Dict[str, Optional[float]]:
        """
        Get gripper state including effort and gap.
        
        Returns:
            Dictionary with 'effort' (A) and 'gap' (m) keys
        """
        effort = None
        gap = None
        
        try:
            # Get effort
            effort = self.robot.get_gripper_effort(left=self.use_left)
            
            # Get gap from joint position
            result = self.robot.get_joint_positions()
            if result is not None:
                names, positions = result
                target_keyword = 'left' if self.use_left else 'right'
                for name, pos in zip(names, positions):
                    if target_keyword in name.lower() and 'eef' in name.lower():
                        gap = pos * self.gripper_max_opening
                        break
        except Exception as e:
            print(f"[RobotEnv] Get gripper state failed: {e}")
        
        return {'effort': effort, 'gap': gap}

    def get_arm_total_effort(self) -> Optional[float]:
        """
        Get total effort (current) of arm joints (excluding gripper).
        
        Returns:
            Sum of absolute joint efforts in Amps, or None if failed
        """
        try:
            efforts = self.robot.get_joint_efforts()
            if efforts is None:
                return None
            
            total = 0.0
            for name, effort in efforts.items():
                name_lower = name.lower()
                # Exclude non-arm joints
                if any(kw in name_lower for kw in ['eef', 'base', 'head', 'spine']):
                    continue
                # Include arm joints
                if 'arm' in name_lower or 'joint' in name_lower:
                    total += abs(effort)
            
            return total
        except Exception as e:
            return None
        
    # ==================== Camera ====================
    
    def get_head_camera_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get head camera RGB and depth images.
        
        Returns:
            Tuple of (rgb, depth) or (None, None) if failed
            - rgb: BGR image (H, W, 3)
            - depth: depth image in mm (H, W)
        """
        try:
            for rgb, depth, _, _ in self.robot.camera:
                if rgb is not None and rgb.size > 0:
                    return rgb, depth
                break
        except Exception as e:
            print(f"[RobotEnv] Head camera failed: {e}")
        return None, None
    
    def get_handeye_camera_frame(self) -> Optional[np.ndarray]:
        """
        Get hand-eye camera RGB image.
        
        Returns:
            BGR image (480, 640, 3) or None if failed
        """
        try:
            goal = {self.camera_component: [ImageTypes.COLOR]}
            for c, imgs in self.robot.mmk2.get_image(goal).items():
                if c == self.camera_component:
                    for _, img in imgs.data.items():
                        if img is not None and img.shape[0] > 1:
                            return cv2.resize(img, (640, 480))
        except Exception as e:
            print(f"[RobotEnv] Hand-eye camera failed: {e}")
        return None
    
    @property
    def camera_iterator(self):
        """Get camera iterator for continuous frame capture."""
        return self.robot.camera
    
    # ==================== TF ====================
    
    def get_transform(self, parent: str, child: str) -> Optional[Dict]:
        """
        Get transform between two frames.
        
        Args:
            parent: parent frame name
            child: child frame name
            
        Returns:
            Dictionary with 'matrix' (4x4) key or None if failed
        """
        return self.tf_client.get_transform(parent, child)
    
    def get_head_camera_transform(self) -> Optional[np.ndarray]:
        """Get 4x4 transform matrix from base_link to head_camera_link."""
        tf_data = self.get_transform('base_link', 'head_camera_link')
        return tf_data['matrix'] if tf_data else None
    
    # ==================== Utility ====================
    
    def _wait_with_check(
        self, 
        duration: float, 
        check_abort: Optional[Callable[[], bool]] = None
    ) -> bool:
        """Wait for duration with optional abort check."""
        interval = 0.05
        elapsed = 0.0
        while elapsed < duration:
            if check_abort and check_abort():
                return False
            time.sleep(interval)
            elapsed += interval
        return True
    
    def disconnect(self):
        """Disconnect from robot and TF server."""
        self.tf_client.disconnect()
        print("[RobotEnv] Disconnected")