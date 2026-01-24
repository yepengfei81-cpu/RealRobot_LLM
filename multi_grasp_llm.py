"""
LLM-based Multi-Brick Stacking Program

Extends single_grasp_llm to handle multiple bricks:
- Detect all bricks using head camera
- Use LLM to analyze scene and select next brick
- Stack bricks at target location with incrementing Z height
- Repeat until all bricks are stacked

Keys:
  Space - Start multi-brick stacking task
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

# Local modules
from robot import RobotEnv, create_executor
from perception import create_segmenter, create_head_calculator, create_handeye_calculator
from llm import create_grasp_planner


# ==================== Configuration ====================
CONFIG_DIR = Path("/home/ypf/qiuzhiarm_LLM/config")

# Multi-brick stacking parameters
BRICK_HEIGHT = 0.025          # 2.5cm per brick
SAFETY_MARGIN = 0.003         # 0.3cm safety margin
STACK_INCREMENT = BRICK_HEIGHT + SAFETY_MARGIN  # 2.8cm per layer
PROXIMITY_THRESHOLD = 0.08    # 8cm - bricks within this distance are "at target"
MAX_TOTAL_ATTEMPTS = 20       # Maximum total grasp attempts
MAX_BRICKS = 10               # Maximum number of bricks to handle


# ==================== Multi-Brick Grasp Controller ====================
class MultiGraspController:
    """
    LLM-based Multi-Brick Stacking Controller
    
    Fully LLM-driven approach:
    1. Detect all bricks with head camera + SAM3
    2. LLM analyzes scene to determine task state and select next brick
    3. Execute grasp-place with incrementing stack height
    4. Repeat until LLM determines task is complete
    
    No fallback logic - completely relies on LLM decision making.
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
        self.config_path = config_path
        
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
        
        print("[Init] Creating motion executor...")
        self.executor = create_executor(
            robot_env=self.robot_env,
            segmenter=self.segmenter,
            head_calc=self.head_calc,
            handeye_calc=self.handeye_calc,
            llm_planner=self.llm_planner,
            prompt=self.prompt,
        )
        
        # ===== Load place config =====
        self._load_place_config()
        
        # ===== Thread control =====
        self._task_thread: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._task_running = threading.Event()
        
        # ===== State =====
        self._state_lock = threading.Lock()
        self._current_step = ""
        self._task_success = False
        self._placed_count = 0
        self._initial_brick_count = 0
        self._total_attempts = 0
        
        # ===== Result cache =====
        self._last_head_result: Optional[Dict] = None
        self._last_handeye_result: Optional[Dict] = None
        self._last_llm_result: Optional[Dict] = None
        self._last_scene_result: Optional[Dict] = None
        
        # ===== Frame cache =====
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_mask: Optional[np.ndarray] = None  # All masks from detection
        self._head_mask_time: float = 0 
        self._handeye_mask: Optional[np.ndarray] = None
        self._handeye_mask_time: float = 0 
        self._selected_brick_index: Optional[int] = None  # LLM selected brick index
        
        # ===== Setup executor callbacks =====
        self.executor.set_callbacks(
            step_callback=self._set_step,
            result_callback=self._set_result,
            mask_callback=self._set_mask,
            check_abort=self._check_abort,
        )
        
        print("[Init] Complete")
        self._print_help()
    
    def _load_place_config(self):
        """Load placement configuration from config file."""
        import json
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            
            place_config = config.get("place", {})
            self.target_position = place_config.get("target_position", [0.54, -0.04, 1.0])
            self.base_surface_z = place_config.get("surface_z", 0.89)
            self.target_yaw = place_config.get("target_yaw", 0.0)
            
            print(f"[Config] Target position: {self.target_position}")
            print(f"[Config] Base surface Z: {self.base_surface_z}")
            print(f"[Config] Stack increment: {STACK_INCREMENT*100:.1f} cm")
            
        except Exception as e:
            print(f"[Warning] Failed to load place config: {e}")
            self.target_position = [0.54, -0.04, 1.0]
            self.base_surface_z = 0.89
            self.target_yaw = 0.0
    
    def _print_help(self):
        print("=" * 60)
        print("Multi-Brick Stacking Controller (Pure LLM-driven)")
        print("=" * 60)
        print("  Space - Start multi-brick stacking task")
        print("  r     - Abort / Reset")
        print("  q/Esc - Exit")
        print("=" * 60)
        print(f"  Brick height: {BRICK_HEIGHT*100:.1f} cm")
        print(f"  Safety margin: {SAFETY_MARGIN*100:.1f} cm")
        print(f"  Stack increment: {STACK_INCREMENT*100:.1f} cm per layer")
        print("=" * 60)
    
    # ==================== State Management ====================
    
    def _set_step(self, step: str):
        with self._state_lock:
            self._current_step = step
    
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
            elif key == 'scene':
                self._last_scene_result = result
    
    def _set_mask(self, key: str, mask: np.ndarray):
        """Callback for mask updates from executor."""
        if key == 'head':
            self._latest_mask = mask
            self._head_mask_time = time.time()
            # Reset selection when new detection happens
            self._selected_brick_index = None
        elif key == 'handeye':
            self._handeye_mask = mask
            self._handeye_mask_time = time.time()

    def _set_selected_brick(self, index: int):
        """Set the LLM-selected brick index for visualization."""
        self._selected_brick_index = index

    def _get_result(self, key: str) -> Optional[Dict]:
        with self._state_lock:
            if key == 'head':
                r = self._last_head_result
            elif key == 'handeye':
                r = self._last_handeye_result
            elif key == 'llm':
                r = self._last_llm_result
            elif key == 'scene':
                r = self._last_scene_result
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
    
    # ==================== Scene Detection ====================
    
    def _detect_all_bricks(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Detect all bricks in the scene using head camera + SAM3.
        
        Returns:
            Tuple of (success, list_of_bricks, error_message)
            Each brick dict contains: position, yaw, mask_index
        """
        print("\n[Scene] Detecting all bricks with SAM3...")
        
        if self._check_abort():
            return False, [], "Aborted"
        
        # Get TF matrix for coordinate transformation
        tf_matrix = self.robot_env.get_head_camera_transform()
        if tf_matrix is None:
            return False, [], "TF failed"
        
        # Segment all bricks using SAM3
        masks = self.segmenter.segment(rgb, self.prompt)
        
        if self._check_abort():
            return False, [], "Aborted"
        
        if masks is None or len(masks) == 0:
            return False, [], "No bricks detected by SAM3"
        
        # Save ALL masks for display (not just first one)
        self._latest_mask = masks
        self._head_mask_time = time.time()
        self._selected_brick_index = None  # Reset selection

        # Calculate position for each detected brick
        all_bricks = []
        for i, mask in enumerate(masks):
            result = self.head_calc.compute(mask, depth, tf_matrix)
            if result is not None:
                pos = result['position']
                yaw = result['yaw']
                
                # Convert to list if numpy array
                pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
                
                all_bricks.append({
                    'position': pos_list,
                    'yaw': float(yaw),
                    'mask_index': i,
                })
                
                print(f"  Brick {i}: [{pos_list[0]:.4f}, {pos_list[1]:.4f}, {pos_list[2]:.4f}] m, yaw={np.degrees(yaw):.1f}°")
        
        print(f"[Scene] SAM3 detected {len(all_bricks)} brick(s)")
        
        return True, all_bricks, None
        
    # ==================== Stack Height Calculation ====================
    
    def _calculate_target_z(self, placed_count: int, verbose: bool = True) -> float:
        """
        Calculate target Z height for the next brick placement.
        
        Formula: target_z = base_surface_z + (placed_count × stack_increment)
        
        Args:
            placed_count: Number of bricks already placed
            verbose: Whether to print calculation details
            
        Returns:
            Target Z height in meters
        """
        target_z = self.base_surface_z + (placed_count * STACK_INCREMENT)
        
        if verbose:
            print(f"[Stack] Calculating target Z:")
            print(f"  - Placed count: {placed_count}")
            print(f"  - Base Z: {self.base_surface_z:.4f} m")
            print(f"  - Increment: {STACK_INCREMENT*100:.1f} cm × {placed_count} = {placed_count * STACK_INCREMENT * 100:.1f} cm")
            print(f"  - Target Z: {target_z:.4f} m")
        
        return target_z
    
    # ==================== Main Task Execution ====================
    
    def _execute_multi_grasp_task(self):
        """
        Execute multi-brick stacking task with pure LLM decision making.
        
        Flow:
        1. Return to initial position
        2. Detect all bricks (SAM3)
        3. LLM analyzes scene → decides next action
        4. If complete → exit
        5. If grasp → execute grasp-place cycle
        6. Repeat from step 1
        """
        self._task_success = False
        self._placed_count = 0
        self._total_attempts = 0
        self._initial_brick_count = 0
        self._selected_brick_index = None  # Reset selection
        
        print("\n" + "=" * 60)
        print("[Multi-Brick Stacking Task] Pure LLM-driven Mode")
        print("=" * 60)
        
        # ===== Initial Detection =====
        self._set_step("Initial scene detection...")
        
        print("\n[Init] Returning to initial position...")
        self.robot_env.reset_position()
        time.sleep(1.0)
        
        # Get initial frame
        rgb, depth = self._get_latest_frame()
        if rgb is None or depth is None:
            self._set_step("[Failed] Cannot get camera frame")
            return
        
        # Initial detection to count bricks
        success, all_bricks, error = self._detect_all_bricks(rgb, depth)
        if not success or len(all_bricks) == 0:
            self._set_step(f"[Failed] Initial detection: {error or 'No bricks found'}")
            return
        
        self._initial_brick_count = len(all_bricks)
        print(f"\n[Task] Initial brick count: {self._initial_brick_count}")
        
        # ===== Main Loop =====
        while not self._check_abort():
            self._total_attempts += 1
            
            if self._total_attempts > MAX_TOTAL_ATTEMPTS:
                self._set_step(f"[Failed] Exceeded max attempts ({MAX_TOTAL_ATTEMPTS})")
                break
            
            print(f"\n{'='*60}")
            print(f"[Cycle {self._total_attempts}] Placed: {self._placed_count}/{self._initial_brick_count}")
            print(f"{'='*60}")
            
            # ===== Step 1: Return to initial position =====
            self._set_step(f"[Cycle {self._total_attempts}] Returning to initial position...")
            print("\n[Step 1] Returning to initial position...")
            self.robot_env.reset_position()
            time.sleep(0.5)
            
            if self._check_abort():
                break
            
            # ===== Step 2: Detect current scene =====
            self._set_step(f"[Cycle {self._total_attempts}] Detecting scene with SAM3...")
            
            rgb, depth = self._get_latest_frame()
            if rgb is None or depth is None:
                print("[Error] Cannot get camera frame")
                self._set_step("[Failed] Camera frame unavailable")
                break
            
            success, detected_bricks, error = self._detect_all_bricks(rgb, depth)
            if not success:
                print(f"[Error] Detection failed: {error}")
                self._set_step(f"[Failed] Detection: {error}")
                break
            
            if len(detected_bricks) == 0:
                # No bricks detected - check if we placed any
                if self._placed_count > 0:
                    print(f"\n[Complete] No more bricks detected. Placed {self._placed_count} bricks.")
                    self._task_success = True
                    self._set_step(f"[Complete] Stacked {self._placed_count} bricks!")
                else:
                    print("[Error] No bricks detected and none placed")
                    self._set_step("[Failed] No bricks in scene")
                break
            
            # ===== Step 3: LLM Scene Analysis =====
            self._set_step(f"[Cycle {self._total_attempts}] LLM analyzing scene...")
            print("\n[Step 3] LLM scene analysis...")
            print(f"  Input: {len(detected_bricks)} brick(s) detected")
            print(f"  Target position: {self.target_position[:2]}")
            print(f"  Proximity threshold: {PROXIMITY_THRESHOLD} m")
            
            success, scene_result, error = self.llm_planner.analyze_scene(
                detected_bricks=detected_bricks,
                target_position=self.target_position,
                placed_count=self._placed_count,
                initial_brick_count=self._initial_brick_count,
                proximity_threshold=PROXIMITY_THRESHOLD,
            )
            
            if not success:
                print(f"[Error] LLM scene analysis failed: {error}")
                self._set_step(f"[Failed] LLM analysis: {error}")
                # Return to initial position and exit
                self.robot_env.reset_position()
                break
            
            self._set_result('scene', scene_result)
            
            # ===== Step 4: Process LLM Decision =====
            task_complete = scene_result['task_complete']
            next_action = scene_result['next_action']
            action_type = next_action['type']
            
            print(f"\n[LLM Decision]")
            print(f"  Task complete: {task_complete}")
            print(f"  Action type: {action_type}")
            print(f"  Reason: {next_action.get('reason', 'N/A')}")
            
            # Check completion
            if task_complete or action_type == 'complete':
                print(f"\n{'='*60}")
                print(f"[Task Complete] LLM determined all bricks are stacked!")
                print(f"  Total placed: {self._placed_count}")
                print(f"  Confidence: {scene_result['confidence']:.2f}")
                print(f"{'='*60}")
                self._task_success = True
                self._set_step(f"[Complete] Stacked {self._placed_count} bricks!")
                break
            
            # Check for error
            if action_type == 'error':
                print(f"[Error] LLM reported error: {next_action['reason']}")
                self._set_step(f"[Failed] LLM error: {next_action['reason']}")
                self.robot_env.reset_position()
                break
            
            # Get target brick index
            target_brick_idx = next_action['target_brick_index']
            
            if target_brick_idx is None or target_brick_idx >= len(detected_bricks):
                print(f"[Error] Invalid brick index from LLM: {target_brick_idx}")
                self._set_step(f"[Failed] Invalid brick index: {target_brick_idx}")
                self.robot_env.reset_position()
                break
            
            # ===== Set selected brick for visualization =====
            self._selected_brick_index = target_brick_idx
            
            target_brick = detected_bricks[target_brick_idx]
            print(f"\n[Step 4] LLM selected brick #{target_brick_idx}")
            print(f"  Position: [{target_brick['position'][0]:.4f}, {target_brick['position'][1]:.4f}, {target_brick['position'][2]:.4f}] m")

            # ===== Step 5: Calculate target Z for stacking =====
            target_z = self._calculate_target_z(self._placed_count)
            
            # ===== Step 6: Execute grasp-place cycle =====
            self._set_step(f"[Cycle {self._total_attempts}] Executing grasp-place...")
            print(f"\n[Step 6] Executing grasp-place cycle")
            print(f"  Target brick: #{target_brick_idx}")
            print(f"  Target stack Z: {target_z:.4f} m")
            
            # Get fresh frame for executor
            rgb, depth = self._get_latest_frame()
            if rgb is None or depth is None:
                print("[Error] Cannot get camera frame for execution")
                self._set_step("[Failed] Camera frame unavailable")
                break
            
            # Execute grasp-place with custom target Z
            success, error = self.executor.execute_single_grasp_place(
                rgb=rgb,
                depth=depth,
                target_surface_z=target_z,
                brick_index=target_brick_idx,
            )
            
            if success:
                self._placed_count += 1
                print(f"\n[Success] Brick #{target_brick_idx} placed!")
                print(f"  Total placed: {self._placed_count}/{self._initial_brick_count}")
                self._set_step(f"[Success] Placed {self._placed_count}/{self._initial_brick_count}")
            else:
                print(f"\n[Failed] Grasp-place failed: {error}")
                self._set_step(f"[Failed] {error}")
                # Don't break - let LLM decide next action in next cycle
        
        # ===== Final cleanup =====
        print("\n[Final] Returning to initial position...")
        self.robot_env.reset_position()
        
        if self._task_success:
            print(f"\n{'='*60}")
            print(f"[Task Complete] Successfully stacked {self._placed_count} bricks!")
            print(f"{'='*60}")
            self._set_step(f"[Complete] Stacked {self._placed_count} bricks!")
        else:
            print(f"\n[Task Ended] Placed {self._placed_count}/{self._initial_brick_count} bricks")
            self._set_step(f"[Ended] Placed {self._placed_count}/{self._initial_brick_count}")
    
    def _task_worker(self):
        """Worker thread for task execution."""
        try:
            self._execute_multi_grasp_task()
        except Exception as e:
            print(f"[Error] Exception in task: {e}")
            import traceback
            traceback.print_exc()
            self._set_step(f"[Error] {str(e)[:40]}")
        finally:
            if self._check_abort():
                print("[Abort] Returning to initial position...")
                self.robot_env.reset_position()
            self._task_running.clear()
    
    def _start_task(self):
        """Start the multi-brick stacking task."""
        if self._task_running.is_set():
            print("[Warning] Task already running")
            return
        
        self._abort.clear()
        self._task_running.set()
        self._set_result('head', None)
        self._set_result('handeye', None)
        self._set_result('llm', None)
        self._set_result('scene', None)
        
        self._task_thread = threading.Thread(target=self._task_worker, daemon=True)
        self._task_thread.start()
    
    def _abort_task(self):
        """Abort the current task."""
        if self._task_running.is_set():
            print("\n[Abort] Stopping task...")
            self._abort.set()
        else:
            print("[Reset] Returning to initial position...")
            self.robot_env.reset_position()
    
    # ==================== Display ====================
    
    def _draw_info(self, frame: np.ndarray, result: Optional[Dict], 
                   label: str, y_offset: int, color: tuple) -> np.ndarray:
        """Draw result info on frame."""
        if result is None:
            return frame
        pos = result.get('position', result.get('target_position', [0, 0, 0]))
        yaw = result.get('yaw', result.get('target_yaw', 0))
        text = f"{label} [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] yaw={np.degrees(yaw):.1f}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def _draw_stack_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw stacking progress on frame."""
        text = f"Stack: {self._placed_count}/{self._initial_brick_count} | Cycle: {self._total_attempts}"
        cv2.putText(frame, text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        if self._task_running.is_set() or self._placed_count > 0:
            # Use verbose=False to avoid repeated printing
            next_z = self._calculate_target_z(self._placed_count, verbose=False) if self._initial_brick_count > 0 else self.base_surface_z
            text2 = f"Next Z: {next_z:.4f} m"
            cv2.putText(frame, text2, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        return frame
    
    def run(self):
        """Main loop."""
        cv2.namedWindow("Head Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HandEye Camera", cv2.WINDOW_NORMAL)
        last_print_time = 0
        
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
                if current_time - last_print_time > 0.5:
                    last_print_time = current_time
                    tcp_pos = self.robot_env.get_tcp_position()
                    arm_effort = self.robot_env.get_arm_total_effort()
                    if tcp_pos is not None and not is_running:
                        effort_str = f", Arm Effort: {arm_effort:.2f}A" if arm_effort is not None else ""
                        print(f"\r[TCP] Z: {tcp_pos[2]:.4f} m{effort_str}", end="", flush=True)
                
                # Display status
                if is_running:
                    status = f"[Running] {step[:45]}..."
                else:
                    status = "[Ready] Space=Start Multi-Stack, r=Reset, q=Quit"
                
                color = (0, 165, 255) if is_running else (0, 255, 0)
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw results
                disp = self._draw_info(disp, self._get_result('head'), "Head:", 60, (0, 255, 255))
                disp = self._draw_info(disp, self._get_result('handeye'), "Fine:", 85, (255, 255, 0))
                disp = self._draw_info(disp, self._get_result('llm'), "LLM:", 110, (255, 0, 255))
                disp = self._draw_stack_info(disp)
                
                # Draw detection box (head camera) with selection highlighting
                if self._latest_mask is not None and (time.time() - self._head_mask_time) < 3.0:
                    disp = self.segmenter.draw_detection(
                        disp, 
                        self._latest_mask,
                        selected_index=self._selected_brick_index
                    )
                cv2.imshow("Head Camera", disp)

                # HandEye camera display
                he_img = self.robot_env.get_handeye_camera_frame()
                if he_img is not None:
                    cv2.putText(he_img, "HandEye", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if self._handeye_mask is not None and (time.time() - self._handeye_mask_time) < 3.0:
                        he_img = self.segmenter.draw_detection(he_img, self._handeye_mask)
                    cv2.imshow("HandEye Camera", he_img)
                
                # Key handling
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
    parser = argparse.ArgumentParser(description="LLM Multi-Brick Stacking Controller (Pure LLM-driven)")
    parser.add_argument("--ip", default="192.168.11.200")
    parser.add_argument("--prompt", default="block, brick, rectangular object")
    parser.add_argument("--checkpoint", default="/home/ypf/sam3-main/checkpoint/sam3.pt")
    parser.add_argument("--tf-host", default="127.0.0.1")
    parser.add_argument("--tf-port", type=int, default=9999)
    parser.add_argument("--right-arm", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    
    controller = MultiGraspController(
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