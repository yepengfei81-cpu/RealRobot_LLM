"""
LLM-based Brick Grasp Program (Refactored)

Clean architecture with separated concerns:
- robot/: Robot control, sensing, and motion execution
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
from typing import Optional, Dict, Tuple

# Local modules - clean imports
from robot import RobotEnv, create_executor
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
    - Motion executor (grasp-place workflow)
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
        
        print("[Init] Creating motion executor...")
        self.executor = create_executor(
            robot_env=self.robot_env,
            segmenter=self.segmenter,
            head_calc=self.head_calc,
            handeye_calc=self.handeye_calc,
            llm_planner=self.llm_planner,
            prompt=self.prompt,
        )
        
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
        
        # ===== Setup executor callbacks =====
        self.executor.set_callbacks(
            step_callback=self._set_step,
            result_callback=self._set_result,
            mask_callback=self._set_mask,
            check_abort=self._check_abort,
        )
        
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
    
    def _set_mask(self, key: str, mask: np.ndarray):
        """Callback for mask updates from executor."""
        if key == 'head':
            self._latest_mask = mask
            self._head_mask_time = time.time()
        elif key == 'handeye':
            self._handeye_mask = mask
            self._handeye_mask_time = time.time()
    
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
        """Execute closed-loop grasp task using executor."""
        self._task_success = False
        
        # Get current frame
        rgb, depth = self._get_latest_frame()
        if rgb is None or depth is None:
            self._set_step("[Failed] Cannot get camera frame")
            return
        
        # Execute complete grasp-place cycle
        success, error = self.executor.execute_single_grasp_place(rgb, depth)
        
        if success:
            self._task_success = True
            # Return to initial position
            print("  Returning to initial position...")
            self.robot_env.reset_position()
            self._set_step("\n[Task Complete] Brick placed!")
        else:
            self._set_step(f"\n[Task Failed] {error}")
    
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
                    if tcp_pos is not None:
                        print(f"\r[TCP] Z: {tcp_pos[2]:.4f} m", end="", flush=True)
                
                # Display status
                status = f"[Running] {step[:45]}..." if is_running else "[Ready] Space=Start, r=Reset, q=Quit"
                color = (0, 165, 255) if is_running else (0, 255, 0)
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                disp = self._draw_info(disp, self._get_result('head'), "Head:", 60, (0, 255, 255))
                disp = self._draw_info(disp, self._get_result('handeye'), "Fine:", 85, (255, 255, 0))
                disp = self._draw_info(disp, self._get_result('llm'), "LLM:", 110, (255, 0, 255))
                
                # Draw detection box (head camera)
                if self._latest_mask is not None and (time.time() - self._head_mask_time) < 3.0:
                    disp = self.segmenter.draw_detection(disp, self._latest_mask)
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