"""
LLM-based Brick Grasp Program (Refactored)

Clean architecture with separated concerns:
- robot/: Robot control and sensing
- perception/: SAM3 segmentation + position calculation
- llm/: Grasp planning with closed-loop feedback

Keys:
  Space - Start task
  r     - Abort / Reset
  q/Esc - Exit
"""

# Suppress verbose logging
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

import cv2
import argparse
import time
import numpy as np
from pathlib import Path
import threading
from typing import Optional, Dict, Tuple, List

# Local modules - clean imports
from robot import RobotEnv
from perception import create_segmenter, create_head_calculator, create_handeye_calculator
from llm import create_grasp_planner


# ==================== Configuration ====================
CONFIG_DIR = Path("/home/ypf/qiuzhiarm_LLM/config")


# ==================== LLM Grasp Controller ====================
class LLMGraspController:
    """
    LLM-based Brick Grasp Controller
    
    Orchestrates:
    - Robot environment (control & sensing)
    - Perception (segmentation & position calculation)
    - LLM planning (pre-grasp, descend, feedback)
    """
    
    def __init__(
        self,
        ip: str = "192.168.11.200",
        checkpoint: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
        prompt: str = "block, brick, rectangular object",
        tf_host: str = "127.0.0.1",
        tf_port: int = 9999,
        use_left: bool = True,
        config_path: Optional[str] = None,
    ):
        self.prompt = prompt
        self.use_left = use_left
        
        if config_path is None:
            config_path = str(CONFIG_DIR / "llm_config.json")
        
        # ===== Initialize modules =====
        print("[Init] Setting up robot environment...")
        self.robot_env = RobotEnv(
            ip=ip,
            tf_host=tf_host,
            tf_port=tf_port,
            use_left=use_left,
            config_path=config_path,
        )
        
        print("[Init] Loading perception modules...")
        self.segmenter = create_segmenter(checkpoint)
        self.head_calc = create_head_calculator()
        self.handeye_calc = create_handeye_calculator(
            side="left" if use_left else "right"
        )
        
        print("[Init] Initializing LLM planner...")
        self.llm_planner = create_grasp_planner(config_path)
        
        # ===== Thread control =====
        self._task_thread: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._task_running = threading.Event()
        
        # ===== State =====
        self._state_lock = threading.Lock()
        self._current_step = ""
        self._task_success = False
        self._last_head_result: Optional[Dict] = None
        self._last_handeye_result: Optional[Dict] = None
        self._last_llm_result: Optional[Dict] = None
        
        # ===== Frame cache =====
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_mask: Optional[np.ndarray] = None
        self._head_mask_time: float = 0 
        self._handeye_mask: Optional[np.ndarray] = None
        self._handeye_mask_time: float = 0 
        
        print("[Init] Complete")
        self._print_help()
    
    def _print_help(self):
        print("=" * 60)
        print("LLM Grasp Controller (Modular Architecture)")
        print("=" * 60)
        print("  Space - Start grasp task")
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
    
    def _set_result(self, key: str, result: Optional[Dict]):
        with self._state_lock:
            if key == 'head':
                self._last_head_result = result
            elif key == 'handeye':
                self._last_handeye_result = result
            elif key == 'llm':
                self._last_llm_result = result
    
    def _get_result(self, key: str) -> Optional[Dict]:
        with self._state_lock:
            if key == 'head':
                r = self._last_head_result
            elif key == 'handeye':
                r = self._last_handeye_result
            elif key == 'llm':
                r = self._last_llm_result
            else:
                return None
            
            if r is None:
                return None
            
            # Deep copy for thread safety
            if 'position' in r:
                pos = r['position']
                return {
                    'position': pos.copy() if hasattr(pos, 'copy') else list(pos),
                    'yaw': r.get('yaw', 0)
                }
            return r.copy() if hasattr(r, 'copy') else dict(r)
    
    def _update_frame(self, rgb: np.ndarray, depth: np.ndarray):
        with self._frame_lock:
            self._latest_rgb = rgb.copy()
            self._latest_depth = depth.copy() if depth is not None else None
    
    def _get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._frame_lock:
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            return rgb, depth
    
    def _check_abort(self) -> bool:
        return self._abort.is_set()

    # ==================== Task Execution ====================
    
    def _execute_task(self):
        """Execute closed-loop grasp task."""
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
        
        self._set_step("\n[Step 1/10] Head camera detection...")
        if self._check_abort():
            return
        
        tf_matrix = self.robot_env.get_head_camera_transform()
        if tf_matrix is None:
            self._set_step("[Failed] TF failed")
            return
        
        masks = self.segmenter.segment(rgb, self.prompt)
        if self._check_abort():
            return
        if masks is None or len(masks) == 0:
            self._set_step("[Failed] No brick detected")
            return
        
        head_result = self.head_calc.compute(masks[0], depth, tf_matrix)
        self._latest_mask = masks[0]
        self._head_mask_time = time.time()
        if head_result is None:
            self._set_step("[Failed] Position calculation failed")
            return
        
        self._set_result('head', head_result)
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        print(f"  Brick: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m, yaw={np.degrees(brick_yaw):.1f}°")
        
        # ===== Step 2: LLM pre-grasp planning =====
        self._set_step("\n[Step 2/10] LLM pre-grasp planning...")
        if self._check_abort():
            return
        
        success, llm_result, error = self.llm_planner.plan_pre_grasp(
            brick_position=brick_pos.tolist(),
            brick_yaw=float(brick_yaw),
        )
        if not success:
            self._set_step(f"[Failed] LLM planning: {error}")
            return
        
        self._set_result('llm', llm_result)
        hover_pos = llm_result['target_position']
        hover_yaw = llm_result['target_yaw']
        
        # ===== Step 3: Move to hover position =====
        self._set_step("\n[Step 3/10] Moving to hover position...")
        if not self.robot_env.move_arm(hover_pos, hover_yaw, wait=2.5, check_abort=self._check_abort):
            return
        print("  Reached hover position")
        
        # ===== Step 4: Hand-eye fine positioning =====
        self._set_step("\n[Step 4/10] Hand-eye fine positioning...")
        if self._check_abort():
            return
        
        handeye_img = self.robot_env.get_handeye_camera_frame()
        arm_pose = self.robot_env.get_arm_pose()
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
        self._handeye_mask = masks[0]
        self._handeye_mask_time = time.time()        
        if handeye_result is None:
            self._set_step("[Failed] Hand-eye calculation failed")
            return
        
        self._set_result('handeye', handeye_result)
        fine_pos = handeye_result['position']
        fine_yaw = handeye_result['yaw']
        print(f"  Refined: [{fine_pos[0]:.4f}, {fine_pos[1]:.4f}, {fine_pos[2]:.4f}] m, yaw={np.degrees(fine_yaw):.1f}°")
        
        # Fine XY alignment
        current_pos = self.robot_env.get_tcp_position()
        if current_pos is None:
            return
        align_pos = [fine_pos[0], fine_pos[1], current_pos[2]]
        if not self.robot_env.move_arm(align_pos, fine_yaw, wait=1.5, check_abort=self._check_abort):
            return
        print("  XY aligned")
        
        # ===== Step 5: LLM descend planning =====
        self._set_step("\n[Step 5/10] LLM descend planning...")
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
        
        # ===== Step 6: Closed-loop grasp =====
        self._set_step("\n[Step 6/10] Closed-loop grasp...")
        
        for attempt in range(1, max_grasp_attempts + 1):
            if self._check_abort():
                return
            
            print(f"\n  --- Grasp Attempt {attempt}/{max_grasp_attempts} ---")
            print(f"  Target Z: {grasp_pos[2]:.4f} m")
            
            # Open gripper
            if not self.robot_env.open_gripper(gripper_gap):
                return
            
            # Descend
            if not self.robot_env.move_arm(grasp_pos, grasp_yaw, wait=2.0, check_abort=self._check_abort):
                return
            print("  Descended to grasp position")
            
            # Close gripper
            if not self.robot_env.close_gripper():
                return
            time.sleep(0.3)
            
            # Get sensor feedback
            gripper_state = self.robot_env.get_gripper_state()
            effort = gripper_state['effort'] or 0.0
            gap = gripper_state['gap'] or 0.0
            
            print(f"  Gripper effort: {effort:.3f} A")
            print(f"  Gripper gap: {gap:.4f} m")
            
            # Get TCP position
            tcp_pos = self.robot_env.get_tcp_position()
            tcp_pos_list = tcp_pos.tolist() if tcp_pos is not None else grasp_pos
            
            # LLM analyze feedback
            print("  Analyzing with LLM...")
            success, feedback_result, error = self.llm_planner.analyze_grasp_feedback(
                brick_position=grasp_pos,
                brick_yaw=grasp_yaw,
                tcp_position=tcp_pos_list,
                gripper_effort=effort,
                gripper_gap_after_close=gap,
                attempt_number=attempt,
                max_attempts=max_grasp_attempts,
            )
            
            if not success:
                print(f"  [Warning] LLM analysis failed: {error}")
                # Fallback check
                if effort > 2.0 and gap > 0.03:
                    print("  [Fallback] Effort and gap indicate success")
                    self._task_success = True
                    break
                continue
            
            if feedback_result['grasp_success']:
                print(f"\n  ✓ GRASP SUCCESSFUL (confidence: {feedback_result['confidence']:.2f})")
                self._task_success = True
                break
            
            # Check adjustment
            adjustment = feedback_result['adjustment']
            if not adjustment['needed'] or attempt >= max_grasp_attempts:
                print("  ✗ Grasp failed, no more retries")
                break
            
            # Apply adjustment
            delta_z = adjustment['delta_z']
            print(f"  Adjusting Z by {delta_z*1000:+.1f} mm ({adjustment['reason']})")
            grasp_pos[2] += delta_z
            
            # Open gripper before retry (no lift, directly retry at new position)
            self.robot_env.open_gripper(gripper_gap)
        
        # ===== Step 7: Lift after successful grasp =====
        if self._task_success:
            self._set_step("\n[Step 7/10] Lifting brick...")
            if self._check_abort():
                return
            
            # Get current TCP pose for lift planning
            tcp_pos = self.robot_env.get_tcp_position()
            if tcp_pos is None:
                tcp_pos_list = grasp_pos
            else:
                tcp_pos_list = tcp_pos.tolist()
            
            # Plan lift with LLM
            success, lift_result, error = self.llm_planner.plan_lift(
                tcp_position=tcp_pos_list,
                tcp_yaw=grasp_yaw,
                brick_position=fine_pos.tolist(),
                brick_yaw=fine_yaw,
            )
            
            if not success:
                print(f"  [Warning] LLM lift planning failed: {error}")
                # Fallback: simple lift by 0.10m
                lift_pos = [tcp_pos_list[0], tcp_pos_list[1], tcp_pos_list[2] + 0.10]
            else:
                lift_pos = lift_result['target_position']
                print(f"  Lift target: [{lift_pos[0]:.4f}, {lift_pos[1]:.4f}, {lift_pos[2]:.4f}] m")
            
            # Execute lift
            if not self.robot_env.move_arm(lift_pos, grasp_yaw, wait=2.0, check_abort=self._check_abort):
                self._set_step("\n[Task Failed] Lift movement failed")
                return
            
            print("  Brick lifted successfully!")

            # ===== Step 8: Move to place position =====
            self._set_step("\n[Step 8/10] Moving to place position...")
            if self._check_abort():
                return
            
            # Get current TCP position after lift
            tcp_pos = self.robot_env.get_tcp_position()
            if tcp_pos is None:
                tcp_pos_list = lift_pos
            else:
                tcp_pos_list = tcp_pos.tolist()
            
            # Plan move to place with LLM
            success, move_result, error = self.llm_planner.plan_move_to_place(
                tcp_position=tcp_pos_list,
                tcp_yaw=grasp_yaw,
            )
            
            if not success:
                print(f"  [Warning] LLM move_to_place failed: {error}")
                # Fallback: use config target position directly
                place_config = self.llm_planner.config.get("place", {})
                place_pos = place_config.get("target_position", [0.54, -0.04, tcp_pos_list[2]])
                place_yaw = place_config.get("target_yaw", 0.0)
                # Keep current Z height
                place_pos[2] = tcp_pos_list[2]
            else:
                place_pos = move_result['target_position']
                place_yaw = move_result['target_yaw']
                print(f"  Move target: [{place_pos[0]:.4f}, {place_pos[1]:.4f}, {place_pos[2]:.4f}] m, yaw={np.degrees(place_yaw):.1f}°")
            
            # Execute move to place
            if not self.robot_env.move_arm(place_pos, place_yaw, wait=3.0, check_abort=self._check_abort):
                self._set_step("\n[Task Failed] Move to place failed")
                return
            
            print("  Reached place position!")
            
            # ===== Step 9: Descend to place brick =====
            self._set_step("\n[Step 9/10] Descending to place...")
            if self._check_abort():
                return
            
            # Get current TCP position
            tcp_pos = self.robot_env.get_tcp_position()
            if tcp_pos is None:
                tcp_pos_list = place_pos
            else:
                tcp_pos_list = tcp_pos.tolist()
            
            # Get place target Z from config (this is the SURFACE height to place brick on)
            place_config = self.llm_planner.config.get("place", {})
            place_surface_z = place_config.get("surface_z", 0.88)  # 放置表面的高度
            
            # Calculate descend target: current XY, descend to surface_z
            # Note: We descend FROM current height TO surface_z (should be lower)
            final_place_pos = [tcp_pos_list[0], tcp_pos_list[1], place_surface_z]
            
            print(f"  Current Z: {tcp_pos_list[2]:.4f} m")
            print(f"  Target surface Z: {place_surface_z:.4f} m")
            print(f"  Descend distance: {tcp_pos_list[2] - place_surface_z:.4f} m")
            
            # Safety check: ensure we are descending, not ascending
            if place_surface_z > tcp_pos_list[2]:
                print(f"  [Warning] Target Z ({place_surface_z:.4f}) is HIGHER than current ({tcp_pos_list[2]:.4f})")
                print(f"  [Warning] This would move UP, not down. Check config!")
            
            # Execute descend (gripper stays closed)
            if not self.robot_env.move_arm(final_place_pos, place_yaw, wait=2.5, check_abort=self._check_abort):
                self._set_step("\n[Task Failed] Place descend failed")
                return
            
            print("  Descended to place position!")
            
            # ===== Step 10: Closed-loop release =====
            self._set_step("\n[Step 10/10] Closed-loop release...")
            if self._check_abort():
                return
            
            print("\n  === Closed-Loop Placement ===")
            print("  Detecting surface contact via Z position error...")
            
            max_release_attempts = 10
            contact_threshold_mm = 0.3  # 0.3mm threshold
            descend_step = 0.005  # 5mm per step
            lift_step = 0.01  # 10mm lift before release
            
            current_place_pos = list(final_place_pos)
            
            for attempt in range(1, max_release_attempts + 1):
                if self._check_abort():
                    return
                
                # Get current TCP Z
                tcp_pos = self.robot_env.get_tcp_position()
                if tcp_pos is None:
                    print("  [Warning] Cannot get TCP position")
                    continue
                
                actual_z = tcp_pos[2]
                target_z = current_place_pos[2]
                z_error = actual_z - target_z
                z_error_mm = z_error * 1000
                
                print(f"\n  --- Release Attempt {attempt}/{max_release_attempts} ---")
                print(f"  Actual Z: {actual_z:.4f} m, Target Z: {target_z:.4f} m, Error: {z_error_mm:+.1f} mm")
                
                # LLM analyze release feedback
                success, release_result, error = self.llm_planner.analyze_release_feedback(
                    actual_z=actual_z,
                    target_z=target_z,
                    contact_threshold_mm=contact_threshold_mm,
                    descend_step=descend_step,
                    lift_step=lift_step,
                    attempt_number=attempt,
                    max_attempts=max_release_attempts,
                )
                
                if not success:
                    print(f"  [Warning] LLM release analysis failed: {error}")
                    # Fallback: simple threshold check
                    if z_error_mm > contact_threshold_mm:
                        print("  [Fallback] Contact detected, releasing...")
                        break
                    else:
                        current_place_pos[2] -= descend_step
                        self.robot_env.move_arm(current_place_pos, place_yaw, wait=1.0, check_abort=self._check_abort)
                        continue
                
                action = release_result['action']
                action_type = action['type']
                delta_z = action['delta_z']
                
                if action_type == 'release':
                    print("  ✓ Contact detected, releasing immediately!")
                    break
                    
                elif action_type == 'lift_then_release':
                    print(f"  ✓ Pressing detected, lifting {delta_z*1000:.1f} mm before release...")
                    current_place_pos[2] += delta_z
                    self.robot_env.move_arm(current_place_pos, place_yaw, wait=1.0, check_abort=self._check_abort)
                    break
                    
                elif action_type == 'descend':
                    print(f"  → No contact, descending {abs(delta_z)*1000:.1f} mm...")
                    current_place_pos[2] += delta_z  # delta_z is negative
                    if not self.robot_env.move_arm(current_place_pos, place_yaw, wait=1.0, check_abort=self._check_abort):
                        print("  [Warning] Descend move failed")
                        break
            
            # Release gripper
            print("\n  Opening gripper to release brick...")
            if not self.robot_env.open_gripper(0.08):
                self._set_step("\n[Task Failed] Gripper open failed")
                return
            
            time.sleep(0.3)
            
            # Retreat upward
            print("  Retreating...")
            retreat_pos = [current_place_pos[0], current_place_pos[1], current_place_pos[2] + 0.10]
            self.robot_env.move_arm(retreat_pos, place_yaw, wait=1.5, check_abort=self._check_abort)
            
            # Return to initial position
            print("  Returning to initial position...")
            self.robot_env.reset_position()
            
            print("\n  ✓ Brick placed successfully!")
            self._set_step("\n[Task Complete] Brick placed!")
    
    def _task_worker(self):
        try:
            self._execute_task()
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self._check_abort():
                self.robot_env.reset_position()
            self._task_running.clear()
            self._set_step("")
    
    def _start_task(self):
        if self._task_running.is_set():
            print("[Warning] Task already running")
            return
        
        self._abort.clear()
        self._task_running.set()
        self._set_result('head', None)
        self._set_result('handeye', None)
        self._set_result('llm', None)
        
        self._task_thread = threading.Thread(target=self._task_worker, daemon=True)
        self._task_thread.start()
    
    def _abort_task(self):
        if self._task_running.is_set():
            print("\n[Abort] Stopping...")
            self._abort.set()
        else:
            self.robot_env.reset_position()
    
    # ==================== Display ====================
    
    def _draw_info(self, frame: np.ndarray, result: Optional[Dict], 
                   label: str, y_offset: int, color: tuple) -> np.ndarray:
        if result is None:
            return frame
        pos = result.get('position', result.get('target_position', [0, 0, 0]))
        yaw = result.get('yaw', result.get('target_yaw', 0))
        text = f"{label} [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] yaw={np.degrees(yaw):.1f}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def run(self):
        """Main loop."""
        cv2.namedWindow("Head Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HandEye Camera", cv2.WINDOW_NORMAL)
        last_effort_print_time = 0
        try:
            for rgb, depth, _, _ in self.robot_env.camera_iterator:
                if rgb is None or rgb.size == 0:
                    continue
                
                self._update_frame(rgb, depth)
                disp = rgb.copy()
                
                is_running = self._task_running.is_set()
                step = self._get_step()
                # ===== Real-Time TCP Z Monitoring =====
                current_time = time.time()
                if current_time - last_effort_print_time > 0.5:
                    last_effort_print_time = current_time
                    tcp_pos = self.robot_env.get_tcp_position()
                    if tcp_pos is not None:
                        print(f"\r[TCP] Z: {tcp_pos[2]:.4f} m", end="", flush=True)
                # ===== Real-Time TCP Z Monitoring End =====           
                status = f"[Running] {step[:45]}..." if is_running else "[Ready] Space=Start, r=Reset, q=Quit"
                color = (0, 165, 255) if is_running else (0, 255, 0)
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                disp = self._draw_info(disp, self._get_result('head'), "Head:", 60, (0, 255, 255))
                disp = self._draw_info(disp, self._get_result('handeye'), "Fine:", 85, (255, 255, 0))
                disp = self._draw_info(disp, self._get_result('llm'), "LLM:", 110, (255, 0, 255))
                # Draw detection box (head camera, show for 3 seconds)
                if self._latest_mask is not None and (time.time() - self._head_mask_time) < 3.0:
                    disp = self.segmenter.draw_detection(disp, self._latest_mask)
                cv2.imshow("Head Camera", disp)

                he_img = self.robot_env.get_handeye_camera_frame()
                if he_img is not None:
                    cv2.putText(he_img, "HandEye", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Draw detection box (handeye camera, show for 3 seconds)
                    if self._handeye_mask is not None and (time.time() - self._handeye_mask_time) < 3.0:
                        he_img = self.segmenter.draw_detection(he_img, self._handeye_mask)
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
            self.robot_env.disconnect()


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