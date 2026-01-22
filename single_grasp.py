"""
Brick Auto Grasp Program
Flow: Head Detection -> Move Above -> Hand-Eye Fine Positioning -> Descend Grasp -> Lift -> Move Place -> Release

Keys:
  Space - Start grasp task
  r     - Return to initial position
  q/Esc - Exit
"""

import sys
sys.path.insert(0, "/home/ypf/sam3-main")

import cv2
import argparse
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import gc
import threading
from queue import Queue
from typing import Optional, Tuple, Dict, List
from scipy.spatial.transform import Rotation as R

from airbot_sdk import MMK2RealRobot
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tf_client import TFClient
from mmk2_types.types import MMK2Components, ImageTypes
from mmk2_types.grpc_msgs import Pose, Position, Orientation, JointState, TrajectoryParams, GoalStatus

# ==================== Configuration Constants ====================
HEAD_INTRINSICS = {'fx': 607.15, 'fy': 607.02, 'cx': 324.25, 'cy': 248.46}
HOVER_HEIGHT = 0.15  # Hover height
GRASP_EFFORT_THRESHOLD = 2.0  # Grasp success current threshold (A)
PLACE_Y_OFFSET = -0.08  # Place position Y-axis offset (towards robot)
CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")


# ==================== Dynamic Z Compensator ====================
class DynamicZCompensator:
    """Dynamic Z-axis Compensator"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.model_type = "linear"
        self.coefficients = {
            'intercept': 0.0,
            'x_coeff': 0.0,
            'y_coeff': 0.0,
            'xy_coeff': 0.0,
            'x2_coeff': 0.0,
            'y2_coeff': 0.0,
        }
        self.enabled = False
        self._load_config()
    
    def _load_config(self):
        """Load from config file"""
        if not self.config_path.exists():
            print("[Z-Comp] Config file not found, using defaults")
            return
        
        try:
            with open(self.config_path) as f:
                data = json.load(f)
            
            dz = data.get('dynamic_z_compensation', {})
            self.enabled = dz.get('enabled', False)
            self.model_type = dz.get('model', 'linear')
            self.coefficients = dz.get('coefficients', self.coefficients)
            
            if self.enabled:
                print(f"[Z-Comp] Loaded dynamic compensation model: {self.model_type}")
        except Exception as e:
            print(f"[Z-Comp] Failed to load config: {e}")
    
    def compute_compensation(self, x: float, y: float) -> float:
        """Compute Z compensation value based on X, Y coordinates"""
        if not self.enabled:
            return 0.0
        
        c = self.coefficients
        
        if self.model_type == "linear":
            return c['intercept'] + c.get('x_coeff', 0.0) * x + c.get('y_coeff', 0.0) * y
        
        elif self.model_type == "quadratic":
            return (c['intercept'] + 
                    c.get('x_coeff', 0.0) * x + 
                    c.get('y_coeff', 0.0) * y +
                    c.get('xy_coeff', 0.0) * x * y +
                    c.get('x2_coeff', 0.0) * x * x +
                    c.get('y2_coeff', 0.0) * y * y)
        
        return 0.0


# ==================== SAM3 Segmenter ====================
class SAM3Segmenter:
    """SAM3 Segmenter (Singleton, Thread-safe)"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, checkpoint_path: str, confidence: float = 0.5):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    print("[SAM3] Loading model...")
                    cls._instance.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
                    cls._instance.processor = Sam3Processor(
                        cls._instance.model, resolution=1008, confidence_threshold=confidence
                    )
                    cls._instance._segment_lock = threading.Lock()
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("[SAM3] Model loaded")
        return cls._instance
    
    def segment(self, img_bgr: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """Segment image, return mask (thread-safe)"""
        with self._segment_lock:
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            state = self.processor.set_image(pil_img)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            masks = out["masks"].cpu().numpy() if out["masks"] is not None else None
            torch.cuda.empty_cache()
            return masks


# ==================== Position Calculators ====================
def estimate_orientation(mask: np.ndarray) -> float:
    """Estimate orientation angle (yaw) from mask"""
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    
    rect = cv2.minAreaRect(largest)
    (_, _), (w, h), angle = rect
    long_angle = angle + 90 if w < h else angle
    yaw = -np.radians(long_angle)
    
    # Normalize to [-pi/2, pi/2]
    while yaw > np.pi / 2: yaw -= np.pi
    while yaw < -np.pi / 2: yaw += np.pi
    
    return float(yaw)


class HeadCameraCalculator:
    """Head Camera Position Calculator"""
    
    def __init__(self, z_compensator: Optional[DynamicZCompensator] = None):
        self.fx, self.fy = HEAD_INTRINSICS['fx'], HEAD_INTRINSICS['fy']
        self.cx, self.cy = HEAD_INTRINSICS['cx'], HEAD_INTRINSICS['cy']
        self.z_compensator = z_compensator
        
        # Load extrinsic offset compensation
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            with open(offset_path) as f:
                data = json.load(f)
                self.offset = np.array(data['offset_xyz'])
                print(f"[Head Camera] Static offset: {self.offset}")
    
    def compute(self, mask: np.ndarray, depth: np.ndarray, tf_matrix: np.ndarray) -> Optional[Dict]:
        """Compute brick position in base_link frame"""
        h, w = depth.shape
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        # Pixel center
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        yaw_cam = estimate_orientation(mask_bool)
        
        # Depth calculation (median after erosion)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
        valid = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
        valid = valid[valid > 0]
        if len(valid) == 0:
            return None
        
        z = np.median(valid) / 1000.0
        pos_cam = np.array([(px - self.cx) * z / self.fx, (py - self.cy) * z / self.fy, z])
        
        # Transform to base_link
        pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3] + self.offset
        yaw_base = yaw_cam + np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]) + np.pi
        
        # Apply dynamic Z compensation
        z_compensation = 0.0
        if self.z_compensator and self.z_compensator.enabled:
            z_compensation = self.z_compensator.compute_compensation(pos_base[0], pos_base[1])
            pos_base[2] += z_compensation
            print(f"[Z-Comp] Position({pos_base[0]:.3f}, {pos_base[1]:.3f}) -> Compensation: {z_compensation:+.4f}m")
        
        # Normalize yaw
        while yaw_base > np.pi: yaw_base -= 2 * np.pi
        while yaw_base < -np.pi: yaw_base += 2 * np.pi
        
        return {'position': pos_base, 'yaw': yaw_base, 'z_compensation': z_compensation}


class HandEyeCalculator:
    """Hand-Eye Camera Position Calculator"""
    
    def __init__(self, intrinsics: dict, extrinsics: dict):
        self.fx, self.fy = intrinsics['fx'], intrinsics['fy']
        self.cx, self.cy = intrinsics['cx'], intrinsics['cy']
        
        self.T_cam2gripper = np.eye(4)
        self.T_cam2gripper[:3, :3] = np.array(extrinsics['rotation_matrix'])
        self.T_cam2gripper[:3, 3] = np.array(extrinsics['translation'])
    
    def compute(self, mask: np.ndarray, shape: Tuple[int, int], 
                R_g2b: np.ndarray, t_g2b: np.ndarray,
                reference_z: float, reference_yaw: float) -> Optional[Dict]:
        """Compute brick position (using head camera Z as reference)"""
        h, w = shape
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        # Pixel center
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        yaw_cam = estimate_orientation(mask_bool)
        
        # Build transformation matrix
        T_g2b = np.eye(4)
        T_g2b[:3, :3], T_g2b[:3, 3] = R_g2b, t_g2b.flatten()
        T_cam2base = T_g2b @ self.T_cam2gripper
        
        # Compute camera depth from reference Z
        R_mat = T_cam2base[:3, :3]
        t_vec = T_cam2base[:3, 3]
        nx = (px - self.cx) / self.fx
        ny = (py - self.cy) / self.fy
        coeff = R_mat[2, 0] * nx + R_mat[2, 1] * ny + R_mat[2, 2]
        z_cam = (reference_z - t_vec[2]) / coeff if abs(coeff) > 1e-6 else 0.3
        z_cam = max(0.05, min(1.0, z_cam))
        
        # Compute position in base_link frame
        pos_cam = np.array([(px - self.cx) * z_cam / self.fx, (py - self.cy) * z_cam / self.fy, z_cam])
        pos_base = (T_cam2base @ np.append(pos_cam, 1))[:3]
        yaw_base = yaw_cam + np.arctan2(T_cam2base[1, 0], T_cam2base[0, 0])
        
        # Normalize yaw
        while yaw_base > np.pi: yaw_base -= 2 * np.pi
        while yaw_base < -np.pi: yaw_base += 2 * np.pi
        
        # Fix yaw jump (ensure continuity with head detection yaw)
        diff = yaw_base - reference_yaw
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        if abs(diff) > np.pi / 2:
            yaw_base = yaw_base - np.pi if yaw_base > 0 else yaw_base + np.pi
        
        return {'position': pos_base, 'yaw': yaw_base}


# ==================== Auto Grasp Controller ====================
class BrickGraspController:
    """Brick Auto Grasp Controller (Multi-threaded Version)"""
    
    def __init__(self, ip: str, checkpoint: str, prompt: str, tf_host: str, tf_port: int, use_left: bool):
        self.prompt = prompt
        self.use_left = use_left
        self.arm_key = 'left_arm' if use_left else 'right_arm'
        self.arm_component = MMK2Components.LEFT_ARM if use_left else MMK2Components.RIGHT_ARM
        self.camera_component = MMK2Components.LEFT_CAMERA if use_left else MMK2Components.RIGHT_CAMERA
        
        # Initialize robot
        print(f"[Init] Connecting to robot {ip}...")
        self.robot = MMK2RealRobot(ip=ip)
        self.robot.set_robot_head_pose(0, -1.08)
        self.robot.set_spine(0.08)
        
        # Initialize segmenter
        self.segmenter = SAM3Segmenter(checkpoint)
        
        # Initialize dynamic Z compensator
        self.z_compensator = DynamicZCompensator(CALIB_DIR / "head_camera_offset.json")
        
        # Initialize head camera calculator (with dynamic compensation)
        self.head_calc = HeadCameraCalculator(z_compensator=self.z_compensator)
        
        self.tf_client = TFClient(host=tf_host, port=tf_port, auto_connect=True)
        
        # Load hand-eye calibration
        side = "left" if use_left else "right"
        intr_path = CALIB_DIR / f"hand_eye_intrinsics_{side}.json"
        extr_path = CALIB_DIR / f"hand_eye_extrinsics_{side}.json"
        with open(intr_path) as f: intr = json.load(f)
        with open(extr_path) as f: extr = json.load(f)
        self.handeye_calc = HandEyeCalculator(intr, extr)
        
        # Thread control
        self._task_thread: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._task_running = threading.Event()
        
        # Shared state (thread-safe)
        self._state_lock = threading.Lock()
        self._current_step = ""  # Current execution step
        self._task_success = False
        self._last_head_result: Optional[Dict] = None
        self._last_handeye_result: Optional[Dict] = None
        
        # Latest frame cache (for task thread)
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        
        print("[Init] Initialization complete")
        print("=" * 50)
        print("Key Instructions:")
        print("  Space - Start grasp task")
        print("  r     - Abort task and return to initial position")
        print("  q/Esc - Exit program")
        if self.z_compensator.enabled:
            print(f"  [Z-Comp enabled: {self.z_compensator.model_type}]")
        print("=" * 50)

    def _set_step(self, step: str):
        """Set current step (thread-safe)"""
        with self._state_lock:
            self._current_step = step
        print(step)
    
    def _get_step(self) -> str:
        """Get current step (thread-safe)"""
        with self._state_lock:
            return self._current_step
    
    def _set_head_result(self, result: Optional[Dict]):
        """Set head detection result (thread-safe)"""
        with self._state_lock:
            self._last_head_result = result
    
    def _get_head_result(self) -> Optional[Dict]:
        """Get head detection result (thread-safe)"""
        with self._state_lock:
            return self._last_head_result.copy() if self._last_head_result else None
    
    def _set_handeye_result(self, result: Optional[Dict]):
        """Set hand-eye detection result (thread-safe)"""
        with self._state_lock:
            self._last_handeye_result = result
    
    def _get_handeye_result(self) -> Optional[Dict]:
        """Get hand-eye detection result (thread-safe)"""
        with self._state_lock:
            return self._last_handeye_result.copy() if self._last_handeye_result else None
    
    def _get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get latest frame (thread-safe)"""
        with self._frame_lock:
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            return rgb, depth
    
    def _update_frame(self, rgb: np.ndarray, depth: np.ndarray):
        """Update latest frame (thread-safe)"""
        with self._frame_lock:
            self._latest_rgb = rgb.copy()
            self._latest_depth = depth.copy() if depth is not None else None
    
    def _check_abort(self) -> bool:
        """Check if task should be aborted"""
        return self._abort.is_set()
    
    def _wait_with_abort_check(self, duration: float) -> bool:
        """Wait for specified duration, checking abort signal. Returns False if aborted"""
        interval = 0.05
        elapsed = 0.0
        while elapsed < duration:
            if self._check_abort():
                return False
            time.sleep(interval)
            elapsed += interval
        return True
        
    def _compute_grasp_orientation(self, yaw: float) -> Tuple[float, float, float, float]:
        """Compute grasp pose quaternion (X-axis pointing down)"""
        r_base = R.from_euler('y', np.pi / 2)
        r_yaw = R.from_euler('z', yaw)
        r_final = r_yaw * r_base
        quat = r_final.as_quat()
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def _normalize_yaw_to_target(self, yaw: float) -> float:
        """Normalize yaw to 0 or pi"""
        while yaw > np.pi: yaw -= 2 * np.pi
        while yaw < -np.pi: yaw += 2 * np.pi
        if abs(yaw) <= np.pi / 2:
            return 0.0
        return np.pi if yaw > 0 else -np.pi
    
    def _get_handeye_image(self) -> Optional[np.ndarray]:
        """Get hand-eye camera image"""
        goal = {self.camera_component: [ImageTypes.COLOR]}
        for c, imgs in self.robot.mmk2.get_image(goal).items():
            if c == self.camera_component:
                for _, img in imgs.data.items():
                    if img is not None and img.shape[0] > 1:
                        return cv2.resize(img, (640, 480))
        return None
    
    def _get_arm_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get arm end effector pose"""
        poses = self.robot.get_arm_end_poses()
        if poses and self.arm_key in poses:
            p = poses[self.arm_key]
            return R.from_quat(p['orientation']).as_matrix(), np.array(p['position'])
        return None
    
    def _move_arm(self, position: np.ndarray, yaw: float, wait: float = 1.5) -> bool:
        """Move arm to specified position"""
        if self._check_abort():
            return False
            
        ox, oy, oz, ow = self._compute_grasp_orientation(yaw)
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                orientation=Orientation(x=ox, y=oy, z=oz, w=ow),
            ),
        }
        try:
            self.robot.control_arm_poses(target_pose)
            return self._wait_with_abort_check(wait)
        except Exception as e:
            print(f"[Error] Move failed: {e}")
            return False
    
    def _reset_position(self):
        """Return to initial position"""
        self._set_step("[Reset] Returning to initial position...")
        arm_action = {
            MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
            MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.SPINE: JointState(position=[0.08]),
        }
        try:
            result = self.robot.mmk2.set_goal(arm_action, TrajectoryParams())
            if result.value == GoalStatus.Status.SUCCESS:
                print("[Reset] Complete")
            else:
                print(f"[Reset] Status: {result}")
        except Exception as e:
            print(f"[Reset] Exception: {e}")
    
    def _execute_grasp_task(self):
        """Execute single grasp task (runs in worker thread)"""
        self._task_success = False
        
        print("\n" + "=" * 60)
        print("[Task Started] Press 'r' to abort anytime")
        print("=" * 60)
        
        # 1. Get current frame
        rgb, depth = self._get_latest_frame()
        if rgb is None or depth is None:
            self._set_step("[Failed] Cannot get camera frame")
            return
        
        # 2. Head camera detection
        self._set_step("\n[Step 1] Head camera detecting brick...")
        if self._check_abort():
            return
            
        tf_data = self.tf_client.get_transform('base_link', 'head_camera_link')
        if not tf_data:
            self._set_step("[Failed] TF acquisition failed")
            return
        
        masks = self.segmenter.segment(rgb, self.prompt)
        if self._check_abort():
            return
        if masks is None or len(masks) == 0:
            self._set_step("[Failed] No brick detected")
            return
        
        head_result = self.head_calc.compute(masks[0], depth, tf_data['matrix'])
        if head_result is None:
            self._set_step("[Failed] Cannot compute brick position")
            return
        
        self._set_head_result(head_result)
        
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        print(f"  Position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}]m")
        print(f"  Yaw: {np.degrees(brick_yaw):.1f} deg")
        
        # 3. Move above brick
        self._set_step("\n[Step 2] Moving above brick...")
        hover_pos = np.array([brick_pos[0], brick_pos[1], brick_pos[2] + HOVER_HEIGHT])
        if not self._move_arm(hover_pos, brick_yaw):
            return
        print("  Reached hover position")
        
        # 4. Hand-eye camera fine positioning
        self._set_step("\n[Step 3] Hand-eye camera fine positioning...")
        if self._check_abort():
            return
        handeye_img = self._get_handeye_image()
        arm_pose = self._get_arm_pose()
        if handeye_img is None or arm_pose is None:
            self._set_step("[Failed] Cannot get hand-eye image or arm pose")
            return
        
        masks = self.segmenter.segment(handeye_img, self.prompt)
        if self._check_abort():
            return
        if masks is None or len(masks) == 0:
            self._set_step("[Failed] Hand-eye camera did not detect brick")
            return
        
        handeye_result = self.handeye_calc.compute(
            masks[0], handeye_img.shape[:2], arm_pose[0], arm_pose[1],
            reference_z=brick_pos[2], reference_yaw=brick_yaw
        )
        if handeye_result is None:
            self._set_step("[Failed] Hand-eye calculation failed")
            return
        
        self._set_handeye_result(handeye_result)
        
        fine_pos = handeye_result['position']
        fine_yaw = handeye_result['yaw']
        print(f"  Fine position: [{fine_pos[0]:.4f}, {fine_pos[1]:.4f}, {fine_pos[2]:.4f}]m")
        print(f"  Fine Yaw: {np.degrees(fine_yaw):.1f} deg")
        
        # 5. Fine align XY (maintain current height)
        self._set_step("\n[Step 4] Fine alignment...")
        current_pose = self._get_arm_pose()
        if current_pose is None:
            return
        current_z = current_pose[1][2]
        align_pos = np.array([fine_pos[0], fine_pos[1], current_z])
        if not self._move_arm(align_pos, fine_yaw):
            return
        print("  XY alignment complete")
        
        # 6. Descend to brick
        self._set_step("\n[Step 5] Descending to brick...")
        grasp_pos = np.array([fine_pos[0], fine_pos[1], fine_pos[2]])
        if not self._move_arm(grasp_pos, fine_yaw):
            return
        print("  Reached grasp position")
        
        # 7. Close gripper
        self._set_step("\n[Step 6] Closing gripper...")
        if self._check_abort():
            return
        if self.use_left:
            self.robot.close_gripper(left=True, right=False)
        else:
            self.robot.close_gripper(left=False, right=True)
        
        if not self._wait_with_abort_check(0.8):
            return
        
        # Check current
        effort = self.robot.get_gripper_effort(left=self.use_left)
        if effort is not None:
            print(f"  Gripper current: {effort:.2f}A")
            if effort < GRASP_EFFORT_THRESHOLD:
                self._set_step("[Failed] Did not grasp brick, current too low")
                return
        else:
            print("  Cannot read current, continuing")
        
        print("  Grasp successful!")
        
        # 8. Lift
        self._set_step("\n[Step 7] Lifting...")
        lift_pos = np.array([fine_pos[0], fine_pos[1], fine_pos[2] + HOVER_HEIGHT])
        if not self._move_arm(lift_pos, fine_yaw):
            return
        print("  Lift complete")
        
        # 9. Calculate place position and move
        self._set_step("\n[Step 8] Moving to place position...")
        place_y = fine_pos[1] + PLACE_Y_OFFSET
        place_yaw = self._normalize_yaw_to_target(fine_yaw)
        place_hover_pos = np.array([fine_pos[0], place_y, fine_pos[2] + HOVER_HEIGHT])
        print(f"  Place position: Y={place_y:.4f}m, Yaw={np.degrees(place_yaw):.1f} deg")
        
        if not self._move_arm(place_hover_pos, place_yaw):
            return
        print("  Reached above place position")
        
        # 10. Descend to place
        self._set_step("\n[Step 9] Descending to place...")
        place_pos = np.array([fine_pos[0], place_y, fine_pos[2]])
        if not self._move_arm(place_pos, place_yaw):
            return
        print("  Reached place height")
        
        # 11. Release gripper
        self._set_step("\n[Step 10] Releasing gripper...")
        if self._check_abort():
            return
        if self.use_left:
            self.robot.open_gripper(left=True, right=False)
        else:
            self.robot.open_gripper(left=False, right=True)
        
        if not self._wait_with_abort_check(0.5):
            return
        print("  Gripper released")
        
        # 12. Lift away
        self._set_step("\n[Step 11] Lifting away...")
        if not self._move_arm(place_hover_pos, place_yaw):
            return
        
        self._set_step("\n[Task Complete] Brick successfully placed")
        print("=" * 60)
        self._task_success = True
    
    def _task_worker(self):
        """Task worker thread"""
        try:
            self._execute_grasp_task()
        except Exception as e:
            print(f"[Error] Task execution exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # If task failed or aborted, reset position
            if not self._task_success or self._check_abort():
                self._reset_position()
                if self._check_abort():
                    print("[Aborted] Task has been aborted")
                else:
                    print("[Hint] Task failed, please adjust brick position and retry")
            
            self._task_running.clear()
            self._set_step("")
    
    def _start_task(self):
        """Start grasp task"""
        if self._task_running.is_set():
            print("[Warning] Task already in progress")
            return
        
        self._abort.clear()
        self._task_running.set()
        self._set_head_result(None)
        self._set_handeye_result(None)
        
        self._task_thread = threading.Thread(target=self._task_worker, daemon=True)
        self._task_thread.start()
    
    def _abort_task(self):
        """Abort current task"""
        if self._task_running.is_set():
            print("\n[Abort] Received abort signal, stopping...")
            self._abort.set()
        else:
            # Not executing task, reset directly
            self._reset_position()
    
    def _draw_detection(self, frame: np.ndarray, result: Optional[Dict], label: str = "") -> np.ndarray:
        """Draw detection result on image"""
        out = frame.copy()
        if result is None:
            return out
        
        pos = result['position']
        yaw_deg = np.degrees(result['yaw'])
        
        text = f"{label} X:{pos[0]:.3f} Y:{pos[1]:.3f} Z:{pos[2]:.3f} Yaw:{yaw_deg:.1f}"
        cv2.putText(out, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return out

    def run(self):
        """Main loop (display thread)"""
        cv2.namedWindow("Head Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HandEye Camera", cv2.WINDOW_NORMAL)
        
        try:
            for rgb, depth, _, _ in self.robot.camera:
                if rgb is None or rgb.size == 0:
                    continue
                
                # Update latest frame (for task thread)
                self._update_frame(rgb, depth)
                
                # Display head camera image
                disp = rgb.copy()
                
                # Status display
                is_running = self._task_running.is_set()
                current_step = self._get_step()
                
                if is_running:
                    status = f"[Running] {current_step[:40]}..." if len(current_step) > 40 else f"[Running] {current_step}"
                    color = (0, 165, 255)  # Orange
                else:
                    status = "[Standby] Space=Start, r=Reset, q=Quit"
                    color = (0, 255, 0)  # Green
                
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw head detection result
                head_result = self._get_head_result()
                if head_result:
                    disp = self._draw_detection(disp, head_result, "Head:")
                
                cv2.imshow("Head Camera", disp)
                
                # Display hand-eye camera image
                handeye_img = self._get_handeye_image()
                if handeye_img is not None:
                    he_disp = handeye_img.copy()
                    cv2.putText(he_disp, "HandEye", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw hand-eye detection result
                    handeye_result = self._get_handeye_result()
                    if handeye_result:
                        he_disp = self._draw_detection(he_disp, handeye_result, "Fine:")
                    
                    cv2.imshow("HandEye Camera", he_disp)
                
                # Key handling
                k = cv2.waitKey(1) & 0xFF
                
                if k == ord(' ') and not self._task_running.is_set():
                    self._start_task()
                
                elif k == ord('r'):
                    self._abort_task()
                
                elif k in (ord('q'), 27):
                    # Abort task before exit
                    if self._task_running.is_set():
                        self._abort.set()
                        self._task_thread.join(timeout=3.0)
                    break
        
        finally:
            cv2.destroyAllWindows()
            self.tf_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Brick Auto Grasp Program")
    parser.add_argument("--ip", default="192.168.11.200", help="Robot IP")
    parser.add_argument("--prompt", default="block, brick, rectangular object", help="Detection prompt")
    parser.add_argument("--checkpoint", default="/home/ypf/sam3-main/checkpoint/sam3.pt", help="SAM3 model path")
    parser.add_argument("--tf-host", default="127.0.0.1", help="TF server address")
    parser.add_argument("--tf-port", type=int, default=9999, help="TF server port")
    parser.add_argument("--right-arm", action="store_true", help="Use right arm (default left)")
    args = parser.parse_args()
    
    controller = BrickGraspController(
        ip=args.ip,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        tf_host=args.tf_host,
        tf_port=args.tf_port,
        use_left=not args.right_arm
    )
    controller.run()


if __name__ == '__main__':
    main()