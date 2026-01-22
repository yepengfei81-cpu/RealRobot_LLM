"""
LLM-based Brick Grasp Program (Step-by-Step Version)

Current implementation:
1. Head camera detection (SAM3) → Get brick position
2. LLM pre-grasp planning → Move to hover position
3. Hand-eye camera fine positioning → Refine brick position (geometric)
4. LLM descend planning → Get descent position and gripper gap
5. Open gripper and descend to grasp position

NOT YET IMPLEMENTED:
- Close gripper and grasp
- Lift and place

Keys:
  Space - Start task
  r     - Abort and return to initial position
  q/Esc - Exit
"""

import sys
sys.path.insert(0, "/home/ypf/sam3-main")

# Suppress verbose logging from OpenAI SDK and HTTP libraries
import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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
from typing import Optional, Tuple, Dict, List
from scipy.spatial.transform import Rotation as R

from airbot_sdk import MMK2RealRobot
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tf_client import TFClient
from mmk2_types.types import MMK2Components, ImageTypes
from mmk2_types.grpc_msgs import Pose, Position, Orientation, JointState, TrajectoryParams, GoalStatus

# Import LLM modules
from llm import create_grasp_planner

# ==================== Configuration Constants ====================
HEAD_INTRINSICS = {'fx': 607.15, 'fy': 607.02, 'cx': 324.25, 'cy': 248.46}
CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")
CONFIG_DIR = Path("/home/ypf/qiuzhiarm_LLM/config")


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
    """Estimate orientation angle (yaw) from mask."""
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
    
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    
    return float(np.clip(yaw, -np.pi / 2, np.pi / 2))


class HeadCameraCalculator:
    """Head Camera Position Calculator"""
    
    def __init__(self):
        self.fx, self.fy = HEAD_INTRINSICS['fx'], HEAD_INTRINSICS['fy']
        self.cx, self.cy = HEAD_INTRINSICS['cx'], HEAD_INTRINSICS['cy']
        
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            with open(offset_path) as f:
                data = json.load(f)
                self.offset = np.array(data.get('offset_xyz', [0, 0, 0]))
                print(f"[Head Camera] Static offset: {self.offset}")
    
    def compute(self, mask: np.ndarray, depth: np.ndarray, tf_matrix: np.ndarray) -> Optional[Dict]:
        """Compute brick position in base_link frame."""
        h, w = depth.shape
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        yaw_cam = estimate_orientation(mask_bool)
        
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
        valid = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
        valid = valid[valid > 0]
        if len(valid) == 0:
            return None
        
        z = np.median(valid) / 1000.0
        z = np.clip(z, 0.1, 2.0)
        
        pos_cam = np.array([
            (px - self.cx) * z / self.fx,
            (py - self.cy) * z / self.fy,
            z
        ])
        
        pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3] + self.offset
        yaw_base = yaw_cam + np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]) + np.pi
        
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        return {'position': pos_base, 'yaw': yaw_base}


class HandEyeCalculator:
    """Hand-Eye Camera Position Calculator (Geometric)"""
    
    def __init__(self, intrinsics: dict, extrinsics: dict):
        self.fx, self.fy = intrinsics['fx'], intrinsics['fy']
        self.cx, self.cy = intrinsics['cx'], intrinsics['cy']
        
        self.T_cam2gripper = np.eye(4)
        self.T_cam2gripper[:3, :3] = np.array(extrinsics['rotation_matrix'])
        self.T_cam2gripper[:3, 3] = np.array(extrinsics['translation'])
    
    def compute(self, mask: np.ndarray, shape: Tuple[int, int], 
                R_g2b: np.ndarray, t_g2b: np.ndarray,
                reference_z: float, reference_yaw: float) -> Optional[Dict]:
        """Compute brick position using geometric calculation (no LLM)."""
        h, w = shape
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        yaw_cam = estimate_orientation(mask_bool)
        
        T_g2b = np.eye(4)
        T_g2b[:3, :3], T_g2b[:3, 3] = R_g2b, t_g2b.flatten()
        T_cam2base = T_g2b @ self.T_cam2gripper
        
        R_mat = T_cam2base[:3, :3]
        t_vec = T_cam2base[:3, 3]
        nx = (px - self.cx) / self.fx
        ny = (py - self.cy) / self.fy
        coeff = R_mat[2, 0] * nx + R_mat[2, 1] * ny + R_mat[2, 2]
        z_cam = (reference_z - t_vec[2]) / coeff if abs(coeff) > 1e-6 else 0.3
        z_cam = max(0.05, min(1.0, z_cam))
        
        pos_cam = np.array([(px - self.cx) * z_cam / self.fx, (py - self.cy) * z_cam / self.fy, z_cam])
        pos_base = (T_cam2base @ np.append(pos_cam, 1))[:3]
        yaw_base = yaw_cam + np.arctan2(T_cam2base[1, 0], T_cam2base[0, 0])
        
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        # Fix yaw jump
        diff = yaw_base - reference_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        if abs(diff) > np.pi / 2:
            yaw_base = yaw_base - np.pi if yaw_base > 0 else yaw_base + np.pi
        
        return {'position': pos_base, 'yaw': yaw_base}


# ==================== LLM Grasp Controller ====================
class LLMGraspController:
    """
    LLM-based Brick Grasp Controller (Step-by-Step Version)
    
    Current implementation stops at descend position.
    """
    
    def __init__(
        self,
        ip: str,
        checkpoint: str,
        prompt: str,
        tf_host: str,
        tf_port: int,
        use_left: bool,
        config_path: Optional[str] = None,
    ):
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
        
        # Initialize calculators
        self.head_calc = HeadCameraCalculator()
        
        # Load hand-eye calibration
        side = "left" if use_left else "right"
        intr_path = CALIB_DIR / f"hand_eye_intrinsics_{side}.json"
        extr_path = CALIB_DIR / f"hand_eye_extrinsics_{side}.json"
        with open(intr_path) as f:
            intr = json.load(f)
        with open(extr_path) as f:
            extr = json.load(f)
        self.handeye_calc = HandEyeCalculator(intr, extr)
        
        # Initialize TF client
        self.tf_client = TFClient(host=tf_host, port=tf_port, auto_connect=True)
        
        # Initialize LLM planner
        if config_path is None:
            config_path = str(CONFIG_DIR / "llm_config.json")
        self.llm_planner = create_grasp_planner(config_path)
        
        # Load config for gripper conversion
        with open(config_path) as f:
            config = json.load(f)
        self.gripper_max_opening = config.get("gripper", {}).get("max_opening", 0.08)
        
        # Thread control
        self._task_thread: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._task_running = threading.Event()
        
        # Shared state
        self._state_lock = threading.Lock()
        self._current_step = ""
        self._task_success = False
        self._last_head_result: Optional[Dict] = None
        self._last_handeye_result: Optional[Dict] = None
        self._last_llm_result: Optional[Dict] = None
        
        # Frame cache
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        
        print("[Init] Complete")
        print("=" * 60)
        print("LLM Grasp Controller (Step-by-Step)")
        print("=" * 60)
        print("  Space - Start task (hover → fine → descend)")
        print("  r     - Abort / Reset")
        print("  q/Esc - Exit")
        print("=" * 60)
    
    # ==================== State Management ====================
    
    def _set_step(self, step: str):
        with self._state_lock:
            self._current_step = step
        print(step)
    
    def _get_step(self) -> str:
        with self._state_lock:
            return self._current_step
    
    def _set_head_result(self, result: Optional[Dict]):
        with self._state_lock:
            self._last_head_result = result
    
    def _get_head_result(self) -> Optional[Dict]:
        with self._state_lock:
            if self._last_head_result is None:
                return None
            return {'position': self._last_head_result['position'].copy(), 'yaw': self._last_head_result['yaw']}
    
    def _set_handeye_result(self, result: Optional[Dict]):
        with self._state_lock:
            self._last_handeye_result = result
    
    def _get_handeye_result(self) -> Optional[Dict]:
        with self._state_lock:
            if self._last_handeye_result is None:
                return None
            return {'position': self._last_handeye_result['position'].copy(), 'yaw': self._last_handeye_result['yaw']}
    
    def _set_llm_result(self, result: Optional[Dict]):
        with self._state_lock:
            self._last_llm_result = result
    
    def _get_llm_result(self) -> Optional[Dict]:
        with self._state_lock:
            return self._last_llm_result.copy() if self._last_llm_result else None
    
    def _get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._frame_lock:
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            return rgb, depth
    
    def _update_frame(self, rgb: np.ndarray, depth: np.ndarray):
        with self._frame_lock:
            self._latest_rgb = rgb.copy()
            self._latest_depth = depth.copy() if depth is not None else None
    
    def _check_abort(self) -> bool:
        return self._abort.is_set()
    
    def _wait_with_abort_check(self, duration: float) -> bool:
        interval = 0.05
        elapsed = 0.0
        while elapsed < duration:
            if self._check_abort():
                return False
            time.sleep(interval)
            elapsed += interval
        return True
    
    # ==================== Robot Control ====================
    
    def _compute_grasp_orientation(self, yaw: float) -> Tuple[float, float, float, float]:
        r_base = R.from_euler('y', np.pi / 2)
        r_yaw = R.from_euler('z', yaw)
        r_final = r_yaw * r_base
        quat = r_final.as_quat()
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def _move_arm(self, position: List[float], yaw: float, wait: float = 2.0) -> bool:
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
    
    def _open_gripper_to_gap(self, gap: float) -> bool:
        """Open gripper to specified gap width."""
        if self._check_abort():
            return False
        
        position = min(1.0, gap / self.gripper_max_opening)
        
        try:
            eef_component = MMK2Components.LEFT_ARM_EEF if self.use_left else MMK2Components.RIGHT_ARM_EEF
            action = {eef_component: JointState(position=[position])}
            self.robot.mmk2.set_goal(action, TrajectoryParams())
            return self._wait_with_abort_check(0.5)
        except Exception as e:
            print(f"[Error] Gripper open failed: {e}")
            return False
    
    def _reset_position(self):
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
        except Exception as e:
            print(f"[Reset] Exception: {e}")
    
    def _get_handeye_image(self) -> Optional[np.ndarray]:
        goal = {self.camera_component: [ImageTypes.COLOR]}
        for c, imgs in self.robot.mmk2.get_image(goal).items():
            if c == self.camera_component:
                for _, img in imgs.data.items():
                    if img is not None and img.shape[0] > 1:
                        return cv2.resize(img, (640, 480))
        return None
    
    def _get_arm_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        poses = self.robot.get_arm_end_poses()
        if poses and self.arm_key in poses:
            p = poses[self.arm_key]
            return R.from_quat(p['orientation']).as_matrix(), np.array(p['position'])
        return None

    def _close_gripper(self) -> bool:
        """Close gripper and return success."""
        if self._check_abort():
            return False
        try:
            self.robot.close_gripper(left=self.use_left, right=not self.use_left)
            return self._wait_with_abort_check(0.8)
        except Exception as e:
            print(f"[Error] Gripper close failed: {e}")
            return False
    
    def _get_gripper_state(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get gripper effort and gap.
        
        Returns:
            (effort, gap) or (None, None) if failed
        """
        try:
            # Get effort
            effort = self.robot.get_gripper_effort(left=self.use_left)
            
            # Get gap from joint position
            result = self.robot.get_joint_positions()
            if result is None:
                return effort, None
            
            names, positions = result
            gap = None
            target_keyword = 'left' if self.use_left else 'right'
            for name, pos in zip(names, positions):
                if target_keyword in name.lower() and 'eef' in name.lower():
                    # Convert position (0-1) to gap (0 - max_opening)
                    gap = pos * self.gripper_max_opening
                    break
            
            return effort, gap
        except Exception as e:
            print(f"[Error] Get gripper state failed: {e}")
            return None, None    
    # ==================== Task Execution ====================
    
    def _execute_task(self):
        """
        Execute LLM-based grasp task with closed-loop feedback.
        
        Flow:
        1. Head camera detection → brick position
        2. LLM pre-grasp planning → hover position
        3. Hand-eye fine positioning → refined XY
        4. LLM descend planning → initial grasp position
        5. Closed-loop grasp with LLM feedback:
           - Descend → Close gripper → Check sensors → LLM analysis
           - If failed, adjust Z and retry (max 3 attempts)
        """
        self._task_success = False
        max_grasp_attempts = 3
        
        print("\n" + "=" * 60)
        print("[Task Started] LLM Closed-Loop Grasp")
        print("=" * 60)
        
        # ===== Step 1: Head camera detection =====
        rgb, depth = self._get_latest_frame()
        if rgb is None or depth is None:
            self._set_step("[Failed] Cannot get camera frame")
            return
        
        self._set_step("\n[Step 1/6] Head camera detection...")
        if self._check_abort():
            return
        
        tf_data = self.tf_client.get_transform('base_link', 'head_camera_link')
        if not tf_data:
            self._set_step("[Failed] TF failed")
            return
        
        masks = self.segmenter.segment(rgb, self.prompt)
        if self._check_abort():
            return
        if masks is None or len(masks) == 0:
            self._set_step("[Failed] No brick detected")
            return
        
        head_result = self.head_calc.compute(masks[0], depth, tf_data['matrix'])
        if head_result is None:
            self._set_step("[Failed] Position calculation failed")
            return
        
        self._set_head_result(head_result)
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        print(f"  Brick: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m, yaw={np.degrees(brick_yaw):.1f}°")
        
        # ===== Step 2: LLM pre-grasp planning =====
        self._set_step("\n[Step 2/6] LLM pre-grasp planning...")
        if self._check_abort():
            return
        
        success, llm_result, error = self.llm_planner.plan_pre_grasp(
            brick_position=brick_pos.tolist(),
            brick_yaw=float(brick_yaw),
        )
        if not success:
            self._set_step(f"[Failed] LLM planning: {error}")
            return
        
        self._set_llm_result(llm_result)
        hover_pos = llm_result['target_position']
        hover_yaw = llm_result['target_yaw']
        
        # ===== Step 3: Move to pre-grasp =====
        self._set_step("\n[Step 3/6] Moving to hover position...")
        if not self._move_arm(hover_pos, hover_yaw, wait=2.5):
            return
        print("  Reached hover position")
        
        # ===== Step 4: Hand-eye fine positioning =====
        self._set_step("\n[Step 4/6] Hand-eye fine positioning...")
        if self._check_abort():
            return
        
        handeye_img = self._get_handeye_image()
        arm_pose = self._get_arm_pose()
        if handeye_img is None or arm_pose is None:
            self._set_step("[Failed] Cannot get hand-eye data")
            return
        
        masks = self.segmenter.segment(handeye_img, self.prompt)
        if self._check_abort():
            return
        if masks is None or len(masks) == 0:
            self._set_step("[Failed] Hand-eye detection failed")
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
        print(f"  Refined: [{fine_pos[0]:.4f}, {fine_pos[1]:.4f}, {fine_pos[2]:.4f}] m, yaw={np.degrees(fine_yaw):.1f}°")
        
        # Fine XY alignment (keep current Z)
        current_pose = self._get_arm_pose()
        if current_pose is None:
            return
        align_pos = [fine_pos[0], fine_pos[1], current_pose[1][2]]
        if not self._move_arm(align_pos, fine_yaw, wait=1.5):
            return
        print("  XY aligned")
        
        # ===== Step 5: Get initial grasp parameters from LLM =====
        self._set_step("\n[Step 5/6] LLM descend planning...")
        if self._check_abort():
            return
        
        success, descend_result, error = self.llm_planner.plan_descend(
            brick_position=fine_pos.tolist(),
            brick_yaw=float(fine_yaw),
        )
        if not success:
            self._set_step(f"[Failed] LLM descend: {error}")
            return
        
        grasp_pos = list(descend_result['target_position'])
        grasp_yaw = descend_result['target_yaw']
        gripper_gap = descend_result['gripper_gap']
        print(f"  Initial grasp pos: [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}] m")
        print(f"  Gripper gap: {gripper_gap:.4f} m")
        
        # ===== Step 6: Closed-loop grasp with LLM feedback =====
        self._set_step("\n[Step 6/6] Closed-loop grasp...")
        
        for attempt in range(1, max_grasp_attempts + 1):
            if self._check_abort():
                return
            
            print(f"\n  --- Grasp Attempt {attempt}/{max_grasp_attempts} ---")
            print(f"  Target Z: {grasp_pos[2]:.4f} m")
            
            # Open gripper
            if not self._open_gripper_to_gap(gripper_gap):
                return
            
            # Descend to grasp position
            if not self._move_arm(grasp_pos, grasp_yaw, wait=2.0):
                return
            print(f"  Descended to grasp position")
            
            # Close gripper
            if not self._close_gripper():
                return
            time.sleep(0.3)  # Wait for stable reading
            
            # Get sensor feedback
            effort, gap = self._get_gripper_state()
            if effort is None:
                effort = 0.0
            if gap is None:
                gap = 0.0
            
            print(f"  Gripper effort: {effort:.3f} A")
            print(f"  Gripper gap: {gap:.4f} m")
            
            # Get current TCP position
            arm_pose = self._get_arm_pose()
            tcp_pos = arm_pose[1].tolist() if arm_pose else grasp_pos
            
            # LLM analyze feedback
            print(f"  Analyzing with LLM...")
            success, feedback_result, error = self.llm_planner.analyze_grasp_feedback(
                brick_position=grasp_pos,
                brick_yaw=grasp_yaw,
                tcp_position=tcp_pos,
                gripper_effort=effort,
                gripper_gap_after_close=gap,
                attempt_number=attempt,
                max_attempts=max_grasp_attempts,
            )
            
            if not success:
                print(f"  [Warning] LLM analysis failed: {error}")
                # Fallback: simple threshold check
                if effort > 2.0 and gap > 0.03:
                    print("  [Fallback] Effort and gap indicate success")
                    self._task_success = True
                    break
                continue
            
            # Check if grasp succeeded
            if feedback_result['grasp_success']:
                print(f"\n  ✓ GRASP SUCCESSFUL (confidence: {feedback_result['confidence']:.2f})")
                self._task_success = True
                break
            
            # Grasp failed - check if adjustment needed
            adjustment = feedback_result['adjustment']
            if not adjustment['needed']:
                print(f"  ✗ Grasp failed but no adjustment suggested")
                break
            
            if attempt >= max_grasp_attempts:
                print(f"  ✗ Max attempts reached")
                break
            
            # Apply Z adjustment
            delta_z = adjustment['delta_z']
            print(f"  Adjusting Z by {delta_z*1000:+.1f} mm ({adjustment['reason']})")
            grasp_pos[2] += delta_z
            
            # Open gripper and move up before retry
            if not self._open_gripper_to_gap(gripper_gap):
                return
            lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.05]
            if not self._move_arm(lift_pos, grasp_yaw, wait=1.0):
                return
        
        # ===== Task Complete =====
        if self._task_success:
            self._set_step("\n[Task Complete] Brick grasped successfully!")
            print("=" * 60)
            print("Brick is now grasped. Ready for lift and place.")
            print("=" * 60)
        else:
            self._set_step("\n[Task Failed] Could not grasp brick after retries")
    
    def _task_worker(self):
        try:
            self._execute_task()
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 任务成功时不自动复位，让机械臂保持在抓取位置
            if self._check_abort():
                self._reset_position()
            self._task_running.clear()
            self._set_step("")
    
    def _start_task(self):
        if self._task_running.is_set():
            print("[Warning] Task running")
            return
        
        self._abort.clear()
        self._task_running.set()
        self._set_head_result(None)
        self._set_handeye_result(None)
        self._set_llm_result(None)
        
        self._task_thread = threading.Thread(target=self._task_worker, daemon=True)
        self._task_thread.start()
    
    def _abort_task(self):
        if self._task_running.is_set():
            print("\n[Abort] Stopping...")
            self._abort.set()
        else:
            self._reset_position()
    
    # ==================== Display ====================
    
    def _draw_info(self, frame: np.ndarray, result: Optional[Dict], label: str, y_offset: int, color: Tuple) -> np.ndarray:
        if result is None:
            return frame
        pos = result.get('position', result.get('target_position', [0, 0, 0]))
        yaw = result.get('yaw', result.get('target_yaw', 0))
        text = f"{label} [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] yaw={np.degrees(yaw):.1f}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def run(self):
        cv2.namedWindow("Head Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HandEye Camera", cv2.WINDOW_NORMAL)
        
        try:
            for rgb, depth, _, _ in self.robot.camera:
                if rgb is None or rgb.size == 0:
                    continue
                
                self._update_frame(rgb, depth)
                disp = rgb.copy()
                
                is_running = self._task_running.is_set()
                step = self._get_step()
                
                status = f"[Running] {step[:45]}..." if is_running else "[Ready] Space=Start, r=Reset, q=Quit"
                color = (0, 165, 255) if is_running else (0, 255, 0)
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                disp = self._draw_info(disp, self._get_head_result(), "Head:", 60, (0, 255, 255))
                disp = self._draw_info(disp, self._get_handeye_result(), "Fine:", 85, (255, 255, 0))
                disp = self._draw_info(disp, self._get_llm_result(), "LLM:", 110, (255, 0, 255))
                
                cv2.imshow("Head Camera", disp)
                
                he_img = self._get_handeye_image()
                if he_img is not None:
                    cv2.putText(he_img, "HandEye", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("HandEye Camera", he_img)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' ') and not is_running:
                    self._start_task()
                elif k == ord('r'):
                    self._abort_task()
                elif k in (ord('q'), 27):
                    if is_running:
                        self._abort.set()
                        self._task_thread.join(timeout=3.0)
                    break
        finally:
            cv2.destroyAllWindows()
            self.tf_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="LLM Grasp Controller")
    parser.add_argument("--ip", default="192.168.11.200")
    parser.add_argument("--prompt", default="block, brick, rectangular object")
    parser.add_argument("--checkpoint", default="/home/ypf/sam3-main/checkpoint/sam3.pt")
    parser.add_argument("--tf-host", default="127.0.0.1")
    parser.add_argument("--tf-port", type=int, default=9999)
    parser.add_argument("--right-arm", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    
    controller = LLMGraspController(
        ip=args.ip,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        tf_host=args.tf_host,
        tf_port=args.tf_port,
        use_left=not args.right_arm,
        config_path=args.config,
    )
    controller.run()


if __name__ == '__main__':
    main()