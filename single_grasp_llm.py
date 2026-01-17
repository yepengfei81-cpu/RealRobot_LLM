"""
LLM-based Brick Grasp Program (Simplified Version)

This is a simplified version that only implements:
1. Head camera detection (SAM3)
2. LLM planning for pre-grasp position
3. Move arm to hover above brick

Flow: Head Detection -> LLM Planning -> Move Above Brick

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


# ==================== Position Calculator ====================
def estimate_orientation(mask: np.ndarray) -> float:
    """
    Estimate orientation angle (yaw) from mask.
    
    Returns yaw in radians, normalized to [-pi/2, pi/2] with safety checks.
    """
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
    
    # Determine long axis angle
    long_angle = angle + 90 if w < h else angle
    yaw = -np.radians(long_angle)
    
    # Normalize to [-pi/2, pi/2] with protection
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    
    # Safety clamp
    yaw = np.clip(yaw, -np.pi / 2, np.pi / 2)
    
    return float(yaw)


class HeadCameraCalculator:
    """Head Camera Position Calculator"""
    
    def __init__(self):
        self.fx, self.fy = HEAD_INTRINSICS['fx'], HEAD_INTRINSICS['fy']
        self.cx, self.cy = HEAD_INTRINSICS['cx'], HEAD_INTRINSICS['cy']
        
        # Load extrinsic offset compensation
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            with open(offset_path) as f:
                data = json.load(f)
                self.offset = np.array(data.get('offset_xyz', [0, 0, 0]))
                print(f"[Head Camera] Static offset: {self.offset}")
    
    def compute(self, mask: np.ndarray, depth: np.ndarray, tf_matrix: np.ndarray) -> Optional[Dict]:
        """
        Compute brick position in base_link frame.
        
        Returns dict with 'position' (np.ndarray) and 'yaw' (float), or None on failure.
        """
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
        
        # Depth calculation (median after erosion for robustness)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
        valid = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
        valid = valid[valid > 0]
        if len(valid) == 0:
            return None
        
        z = np.median(valid) / 1000.0  # Convert mm to m
        
        # Sanity check on depth
        if z < 0.1 or z > 2.0:
            print(f"[Head Camera] Warning: Unusual depth {z:.3f}m, clamping")
            z = np.clip(z, 0.1, 2.0)
        
        pos_cam = np.array([
            (px - self.cx) * z / self.fx,
            (py - self.cy) * z / self.fy,
            z
        ])
        
        # Transform to base_link
        pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3] + self.offset
        yaw_base = yaw_cam + np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]) + np.pi
        
        # Normalize yaw to [-pi, pi]
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        return {'position': pos_base, 'yaw': yaw_base}


# ==================== LLM Grasp Controller ====================
class LLMGraspController:
    """
    LLM-based Brick Grasp Controller (Simplified Version)
    
    Only implements:
    1. Head camera detection
    2. LLM pre-grasp planning
    3. Move to hover position
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
        self.robot.set_spine(0.15)
        
        # Initialize segmenter
        self.segmenter = SAM3Segmenter(checkpoint)
        
        # Initialize head camera calculator
        self.head_calc = HeadCameraCalculator()
        
        # Initialize TF client
        self.tf_client = TFClient(host=tf_host, port=tf_port, auto_connect=True)
        
        # Initialize LLM planner
        if config_path is None:
            config_path = str(CONFIG_DIR / "llm_config.json")
        self.llm_planner = create_grasp_planner(config_path)
        
        # Thread control
        self._task_thread: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._task_running = threading.Event()
        
        # Shared state (thread-safe)
        self._state_lock = threading.Lock()
        self._current_step = ""
        self._task_success = False
        self._last_head_result: Optional[Dict] = None
        self._last_llm_result: Optional[Dict] = None
        
        # Latest frame cache
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        
        print("[Init] Initialization complete")
        print("=" * 60)
        print("LLM Grasp Controller - Simplified Version")
        print("=" * 60)
        print("Key Instructions:")
        print("  Space - Start LLM grasp task")
        print("  r     - Abort and return to initial position")
        print("  q/Esc - Exit program")
        print("=" * 60)
    
    # ==================== Thread-safe State Management ====================
    
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
            if self._last_head_result is None:
                return None
            return {
                'position': self._last_head_result['position'].copy(),
                'yaw': self._last_head_result['yaw']
            }
    
    def _set_llm_result(self, result: Optional[Dict]):
        """Set LLM planning result (thread-safe)"""
        with self._state_lock:
            self._last_llm_result = result
    
    def _get_llm_result(self) -> Optional[Dict]:
        """Get LLM planning result (thread-safe)"""
        with self._state_lock:
            return self._last_llm_result.copy() if self._last_llm_result else None
    
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
    
    # ==================== Robot Control ====================
    
    def _compute_grasp_orientation(self, yaw: float) -> Tuple[float, float, float, float]:
        """Compute grasp pose quaternion (gripper pointing down)"""
        r_base = R.from_euler('y', np.pi / 2)
        r_yaw = R.from_euler('z', yaw)
        r_final = r_yaw * r_base
        quat = r_final.as_quat()
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def _move_arm(self, position: List[float], yaw: float, wait: float = 2.0) -> bool:
        """
        Move arm to specified position.
        
        Returns False if aborted or failed.
        """
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
            MMK2Components.SPINE: JointState(position=[0.15]),
        }
        try:
            result = self.robot.mmk2.set_goal(arm_action, TrajectoryParams())
            if result.value == GoalStatus.Status.SUCCESS:
                print("[Reset] Complete")
            else:
                print(f"[Reset] Status: {result}")
        except Exception as e:
            print(f"[Reset] Exception: {e}")
    
    def _get_handeye_image(self) -> Optional[np.ndarray]:
        """Get hand-eye camera image for display"""
        goal = {self.camera_component: [ImageTypes.COLOR]}
        for c, imgs in self.robot.mmk2.get_image(goal).items():
            if c == self.camera_component:
                for _, img in imgs.data.items():
                    if img is not None and img.shape[0] > 1:
                        return cv2.resize(img, (640, 480))
        return None
    
    # ==================== Task Execution ====================
    
    def _execute_llm_grasp_task(self):
        """
        Execute LLM-based grasp task (runs in worker thread).
        
        Simplified flow:
        1. Head camera detection
        2. LLM pre-grasp planning
        3. Move to hover position
        """
        self._task_success = False
        
        print("\n" + "=" * 60)
        print("[LLM Task Started] Press 'r' to abort anytime")
        print("=" * 60)
        
        # ===== Step 1: Get current frame =====
        rgb, depth = self._get_latest_frame()
        if rgb is None or depth is None:
            self._set_step("[Failed] Cannot get camera frame")
            return
        
        # ===== Step 2: Head camera detection =====
        self._set_step("\n[Step 1/3] Head camera detecting brick...")
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
            self._set_step("[Failed] No brick detected by SAM3")
            return
        
        head_result = self.head_calc.compute(masks[0], depth, tf_data['matrix'])
        if head_result is None:
            self._set_step("[Failed] Cannot compute brick position from depth")
            return
        
        self._set_head_result(head_result)
        
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        print(f"  Detected brick position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m")
        print(f"  Detected brick yaw: {np.degrees(brick_yaw):.1f}°")
        
        # ===== Step 3: LLM pre-grasp planning =====
        self._set_step("\n[Step 2/3] LLM planning pre-grasp position...")
        if self._check_abort():
            return
        
        # Convert numpy array to list for LLM planner
        brick_position_list = brick_pos.tolist() if hasattr(brick_pos, 'tolist') else list(brick_pos)
        
        success, llm_result, error = self.llm_planner.plan_pre_grasp(
            brick_position=brick_position_list,
            brick_yaw=float(brick_yaw),
        )
        
        if self._check_abort():
            return
        
        if not success:
            self._set_step(f"[Failed] LLM planning failed: {error}")
            return
        
        self._set_llm_result(llm_result)
        
        target_pos = llm_result['target_position']
        target_yaw = llm_result['target_yaw']
        reasoning = llm_result.get('reasoning', 'N/A')
        
        print(f"  LLM target position: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}] m")
        print(f"  LLM target yaw: {np.degrees(target_yaw):.1f}°")
        print(f"  LLM reasoning: {reasoning}")
        
        # ===== Step 4: Move to pre-grasp position =====
        self._set_step("\n[Step 3/3] Moving to pre-grasp position...")
        if self._check_abort():
            return
        
        if not self._move_arm(target_pos, target_yaw, wait=2.5):
            self._set_step("[Failed] Move to pre-grasp position failed or aborted")
            return
        
        self._set_step("\n[Task Complete] Arm positioned above brick!")
        print("=" * 60)
        print("LLM-based pre-grasp positioning successful.")
        print("Next steps (not implemented yet):")
        print("  - Hand-eye fine positioning")
        print("  - Descend and grasp")
        print("  - Lift and place")
        print("=" * 60)
        self._task_success = True
    
    def _task_worker(self):
        """Task worker thread"""
        try:
            self._execute_llm_grasp_task()
        except Exception as e:
            print(f"[Error] Task execution exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset position if failed or aborted
            if not self._task_success or self._check_abort():
                self._reset_position()
                if self._check_abort():
                    print("[Aborted] Task has been aborted by user")
                else:
                    print("[Hint] Task failed. Check brick visibility and try again.")
            
            self._task_running.clear()
            self._set_step("")
    
    def _start_task(self):
        """Start LLM grasp task"""
        if self._task_running.is_set():
            print("[Warning] Task already in progress")
            return
        
        self._abort.clear()
        self._task_running.set()
        self._set_head_result(None)
        self._set_llm_result(None)
        
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
    
    # ==================== Display ====================
    
    def _draw_detection(self, frame: np.ndarray, result: Optional[Dict], label: str, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw detection/planning result on image"""
        out = frame.copy()
        if result is None:
            return out
        
        pos = result.get('position', result.get('target_position', [0, 0, 0]))
        yaw = result.get('yaw', result.get('target_yaw', 0))
        yaw_deg = np.degrees(yaw)
        
        text = f"{label} X:{pos[0]:.3f} Y:{pos[1]:.3f} Z:{pos[2]:.3f} Yaw:{yaw_deg:.1f}"
        cv2.putText(out, text, (10, 60 if 'Head' in label else 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
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
                    status = f"[Running] {current_step[:50]}..." if len(current_step) > 50 else f"[Running] {current_step}"
                    status_color = (0, 165, 255)  # Orange
                else:
                    status = "[Standby] Space=Start LLM Task, r=Reset, q=Quit"
                    status_color = (0, 255, 0)  # Green
                
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Draw head detection result
                head_result = self._get_head_result()
                if head_result:
                    disp = self._draw_detection(disp, head_result, "Head:", (0, 255, 255))
                
                # Draw LLM planning result
                llm_result = self._get_llm_result()
                if llm_result:
                    disp = self._draw_detection(disp, llm_result, "LLM:", (255, 0, 255))
                
                cv2.imshow("Head Camera", disp)
                
                # Display hand-eye camera image
                handeye_img = self._get_handeye_image()
                if handeye_img is not None:
                    he_disp = handeye_img.copy()
                    cv2.putText(he_disp, "HandEye Camera", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("HandEye Camera", he_disp)
                
                # Key handling (non-blocking)
                k = cv2.waitKey(1) & 0xFF
                
                if k == ord(' ') and not self._task_running.is_set():
                    self._start_task()
                
                elif k == ord('r'):
                    self._abort_task()
                
                elif k in (ord('q'), 27):  # q or ESC
                    # Abort task before exit
                    if self._task_running.is_set():
                        self._abort.set()
                        self._task_thread.join(timeout=3.0)
                    break
        
        finally:
            cv2.destroyAllWindows()
            self.tf_client.disconnect()


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="LLM-based Brick Grasp Program")
    parser.add_argument("--ip", default="192.168.11.200", help="Robot IP")
    parser.add_argument("--prompt", default="block, brick, rectangular object", help="SAM3 detection prompt")
    parser.add_argument("--checkpoint", default="/home/ypf/sam3-main/checkpoint/sam3.pt", help="SAM3 model path")
    parser.add_argument("--tf-host", default="127.0.0.1", help="TF server address")
    parser.add_argument("--tf-port", type=int, default=9999, help="TF server port")
    parser.add_argument("--right-arm", action="store_true", help="Use right arm (default: left)")
    parser.add_argument("--config", default=None, help="LLM config path (default: config/llm_config.json)")
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