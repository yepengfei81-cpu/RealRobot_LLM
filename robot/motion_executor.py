"""
Motion Executor for Grasp-Place Tasks

Abstracts the complete grasp-place workflow into reusable components.
Can be used by both single_grasp_llm and multi_grasp_llm.
"""

import time
import numpy as np
from typing import Optional, Dict, Tuple, List, Callable, Any


class GraspPlaceExecutor:
    """
    Executes grasp-place motion sequences.
    
    Provides:
    - Individual step methods (detect, hover, grasp, lift, place, release)
    - Complete single grasp-place cycle
    - Configurable target Z for multi-brick stacking
    """
    
    def __init__(
        self,
        robot_env,
        segmenter,
        head_calc,
        handeye_calc,
        llm_planner,
        prompt: str = "block, brick, rectangular object",
    ):
        """
        Initialize executor with required components.
        
        Args:
            robot_env: RobotEnv instance for robot control
            segmenter: SAM3 segmenter for brick detection
            head_calc: Head camera position calculator (with DynamicZCompensator)
            handeye_calc: Hand-eye camera position calculator
            llm_planner: LLM grasp planner
            prompt: Segmentation prompt for brick detection
        """
        self.robot_env = robot_env
        self.segmenter = segmenter
        self.head_calc = head_calc
        self.handeye_calc = handeye_calc
        self.llm_planner = llm_planner
        self.prompt = prompt
        
        # Callbacks for UI updates (optional)
        self._step_callback: Optional[Callable[[str], None]] = None
        self._result_callback: Optional[Callable[[str, Dict], None]] = None
        self._mask_callback: Optional[Callable[[str, np.ndarray], None]] = None

        # Abort check function
        self._check_abort: Callable[[], bool] = lambda: False
    
    def set_callbacks(
        self,
        step_callback: Optional[Callable[[str], None]] = None,
        result_callback: Optional[Callable[[str, Dict], None]] = None,
        mask_callback: Optional[Callable[[str, np.ndarray], None]] = None,
        check_abort: Optional[Callable[[], bool]] = None,
    ):
        """Set optional callbacks for UI integration."""
        if step_callback:
            self._step_callback = step_callback
        if result_callback:
            self._result_callback = result_callback
        if mask_callback:
            self._mask_callback = mask_callback
        if check_abort:
            self._check_abort = check_abort
    
    def _log_step(self, message: str):
        """Log step message."""
        print(message)
        if self._step_callback:
            self._step_callback(message)
    
    def _save_result(self, key: str, result: Dict):
        """Save result via callback."""
        if self._result_callback:
            self._result_callback(key, result)
    
    def _save_mask(self, key: str, mask: np.ndarray):
        """Save mask via callback."""
        if self._mask_callback:
            self._mask_callback(key, mask)

    # ==================== Step 1: Head Camera Detection ====================
    
    def detect_with_head_camera(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        select_index: int = 0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Detect brick(s) using head camera.
        
        The head_calc already includes DynamicZCompensator for Z correction.
        
        Args:
            rgb: RGB image from head camera
            depth: Depth image from head camera
            select_index: Which brick to select if multiple detected (default: 0)
            
        Returns:
            Tuple of (success, result_dict, error_message)
            result_dict contains: position (Z-compensated), yaw, z_compensation, all_masks (if multiple)
        """
        self._log_step("\n[Step 1/10] Head camera detection...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        # Get TF matrix
        tf_matrix = self.robot_env.get_head_camera_transform()
        if tf_matrix is None:
            return False, None, "TF failed"
        
        # Segment
        masks = self.segmenter.segment(rgb, self.prompt)
        if self._check_abort():
            return False, None, "Aborted"
        if masks is None or len(masks) == 0:
            return False, None, "No brick detected"
        
        # Select target brick
        if select_index >= len(masks):
            select_index = 0
        
        # Calculate position (head_calc includes DynamicZCompensator)
        head_result = self.head_calc.compute(masks[select_index], depth, tf_matrix)
        if head_result is None:
            return False, None, "Position calculation failed"
        
        # Save mask
        self._save_mask('head', masks[select_index])
        
        # Build result
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        z_compensation = head_result.get('z_compensation', 0.0)
        
        result = {
            'position': brick_pos,
            'yaw': brick_yaw,
            'z_compensation': z_compensation,
            'selected_index': select_index,
            'total_detected': len(masks),
        }
        
        # If multiple bricks, calculate all positions for scene analysis
        if len(masks) > 1:
            all_bricks = []
            for i, mask in enumerate(masks):
                r = self.head_calc.compute(mask, depth, tf_matrix)
                if r is not None:
                    all_bricks.append({
                        'position': r['position'].tolist() if hasattr(r['position'], 'tolist') else list(r['position']),
                        'yaw': float(r['yaw']),
                        'z_compensation': r.get('z_compensation', 0.0),
                    })
            result['all_bricks'] = all_bricks
        
        self._save_result('head', result)
        print(f"  Brick: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m, yaw={np.degrees(brick_yaw):.1f}°")
        if abs(z_compensation) > 0.001:
            print(f"  Z compensation applied: {z_compensation*1000:+.1f} mm")
        print(f"  Total detected: {len(masks)}")
        
        return True, result, None
    
    # ==================== Step 2-3: Pre-grasp Planning & Move ====================
    
    def plan_and_move_to_hover(
        self,
        brick_position: List[float],
        brick_yaw: float,
        z_compensation: float = 0.0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan pre-grasp position with LLM and move to hover.
        
        Args:
            brick_position: [x, y, z] brick position (already Z-compensated)
            brick_yaw: brick yaw angle
            z_compensation: Z compensation value (for LLM context)
        
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        # Step 2: LLM planning
        self._log_step("\n[Step 2/10] LLM pre-grasp planning...")
        if self._check_abort():
            return False, None, "Aborted"
        
        success, llm_result, error = self.llm_planner.plan_pre_grasp(
            brick_position=brick_position,
            brick_yaw=brick_yaw,
            z_compensation=z_compensation,
        )
        if not success:
            return False, None, f"LLM planning failed: {error}"
        
        self._save_result('llm', llm_result)
        hover_pos = llm_result['target_position']
        hover_yaw = llm_result['target_yaw']
        
        # Step 3: Move to hover
        self._log_step("\n[Step 3/10] Moving to hover position...")
        if not self.robot_env.move_arm(hover_pos, hover_yaw, wait=2.5, check_abort=self._check_abort):
            return False, None, "Move to hover failed"
        
        print("  Reached hover position")
        return True, llm_result, None
    
    # ==================== Step 4: Hand-eye Fine Positioning ====================
    
    def fine_position_with_handeye(
        self,
        reference_z: float,
        reference_yaw: float,
        reference_xy: Optional[List[float]] = None,
        z_compensation: float = 0.0,  # 添加此参数
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Fine positioning using hand-eye camera.
        
        Args:
            reference_z: Z position from head camera (already compensated, used as reference)
            reference_yaw: Yaw from head camera (used as reference)
            reference_xy: [x, y] position from head camera (used to select closest brick)
            z_compensation: Z compensation value from head camera (to pass through)
            
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        self._log_step("\n[Step 4/10] Hand-eye fine positioning...")
        if self._check_abort():
            return False, None, "Aborted"
        
        handeye_img = self.robot_env.get_handeye_camera_frame()
        arm_pose = self.robot_env.get_arm_pose()
        if handeye_img is None or arm_pose is None:
            return False, None, "Cannot get hand-eye data"
        
        masks = self.segmenter.segment(handeye_img, self.prompt)
        if self._check_abort():
            return False, None, "Aborted"
        if masks is None or len(masks) == 0:
            return False, None, "Hand-eye detection failed"
        
        # If multiple masks detected, select the one closest to reference position
        selected_mask_idx = 0
        if len(masks) > 1 and reference_xy is not None:
            print(f"  Hand-eye detected {len(masks)} objects, selecting closest to head camera target...")
            
            min_dist = float('inf')
            for i, mask in enumerate(masks):
                result = self.handeye_calc.compute(
                    mask, handeye_img.shape[:2], arm_pose[0], arm_pose[1],
                    reference_z=reference_z, reference_yaw=reference_yaw
                )
                if result is not None:
                    pos = result['position']
                    dx = pos[0] - reference_xy[0]
                    dy = pos[1] - reference_xy[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    print(f"    Mask {i}: pos=[{pos[0]:.4f}, {pos[1]:.4f}], dist={dist:.4f} m")
                    
                    if dist < min_dist:
                        min_dist = dist
                        selected_mask_idx = i
            
            print(f"  Selected mask #{selected_mask_idx} (closest, dist={min_dist:.4f} m)")
        
        # Use selected mask
        handeye_result = self.handeye_calc.compute(
            masks[selected_mask_idx], handeye_img.shape[:2], arm_pose[0], arm_pose[1],
            reference_z=reference_z, reference_yaw=reference_yaw
        )
        
        self._save_mask('handeye', masks[selected_mask_idx])
        
        if handeye_result is None:
            return False, None, "Hand-eye calculation failed"
        
        self._save_result('handeye', handeye_result)
        fine_pos = handeye_result['position']
        fine_yaw = handeye_result['yaw']
        fine_pos_corrected = np.array([fine_pos[0], fine_pos[1], reference_z])
        
        print(f"  Hand-eye XY: [{fine_pos[0]:.4f}, {fine_pos[1]:.4f}]")
        print(f"  Using head camera Z (with compensation): {reference_z:.4f} m")
        print(f"  Final refined: [{fine_pos_corrected[0]:.4f}, {fine_pos_corrected[1]:.4f}, {fine_pos_corrected[2]:.4f}] m, yaw={np.degrees(fine_yaw):.1f}°")
        
        # Validate: refined position should be close to reference
        if reference_xy is not None:
            dx = fine_pos[0] - reference_xy[0]
            dy = fine_pos[1] - reference_xy[1]
            xy_error = np.sqrt(dx*dx + dy*dy)
            max_allowed_error = 0.05  # 5cm max deviation
            
            if xy_error > max_allowed_error:
                print(f"  [Warning] Refined position deviates {xy_error*100:.1f} cm from head camera target")
        
        # Fine XY alignment
        current_pos = self.robot_env.get_tcp_position()
        if current_pos is None:
            return False, None, "Cannot get TCP position"
        
        align_pos = [fine_pos_corrected[0], fine_pos_corrected[1], current_pos[2]]
        if not self.robot_env.move_arm(align_pos, fine_yaw, wait=1.5, check_abort=self._check_abort):
            return False, None, "XY alignment failed"
        
        print("  XY aligned")
        
        return True, {
            'position': fine_pos_corrected,  # 使用修正后的位置
            'yaw': fine_yaw,
            'z_compensation': z_compensation,
        }, None
    
    # ==================== Step 5-6: Descend & Grasp (Simplified) ====================
    
    def execute_grasp(
        self,
        brick_position: List[float],
        brick_yaw: float,
        z_compensation: float = 0.0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan descend and execute grasp using DynamicZCompensator.
        
        The brick_position already includes Z compensation from head camera.
        No LLM-driven adaptive learning - uses static/dynamic compensation from config.
        
        Args:
            brick_position: [x, y, z] brick position (Z already compensated by DynamicZCompensator)
            brick_yaw: brick yaw angle
            z_compensation: Z compensation value applied (for logging/LLM context)
            
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        # Step 5: LLM descend planning
        self._log_step("\n[Step 5/10] LLM descend planning...")
        if self._check_abort():
            return False, None, "Aborted"
        
        success, descend_result, error = self.llm_planner.plan_descend(
            brick_position=brick_position,
            brick_yaw=brick_yaw,
            z_compensation=z_compensation,
        )
        if not success:
            return False, None, f"LLM descend failed: {error}"
        
        grasp_pos = list(descend_result['target_position'])
        grasp_yaw = descend_result['target_yaw']
        gripper_gap = descend_result['gripper_gap']
        
        print(f"  Grasp position: [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}] m")
        if abs(z_compensation) > 0.001:
            print(f"  (Z compensation from DynamicZCompensator: {z_compensation*1000:+.1f} mm)")
        print(f"  Gripper gap: {gripper_gap:.4f} m")
        
        # Step 6: Execute grasp (single attempt, no retry loop)
        self._log_step("\n[Step 6/10] Executing grasp...")
        
        if self._check_abort():
            return False, None, "Aborted"
        
        # Open gripper
        if not self.robot_env.open_gripper(gripper_gap):
            return False, None, "Open gripper failed"
        
        # Descend to grasp position
        if not self.robot_env.move_arm(grasp_pos, grasp_yaw, wait=2.0, check_abort=self._check_abort):
            return False, None, "Descend failed"
        print("  Descended to grasp position")
        
        # Close gripper
        if not self.robot_env.close_gripper():
            return False, None, "Close gripper failed"
        time.sleep(0.3)
        
        # Check grasp success via sensor feedback
        gripper_state = self.robot_env.get_gripper_state()
        effort = gripper_state['effort'] or 0.0
        gap = gripper_state['gap'] or 0.0
        
        print(f"  Gripper effort: {effort:.3f} A")
        print(f"  Gripper gap: {gap:.4f} m")
        
        # Simple success check (same as single_grasp.py)
        grasp_effort_threshold = 2.0  # Same as GRASP_EFFORT_THRESHOLD in single_grasp.py
        brick_width = 0.05  # Default brick width
        
        grasp_success = (effort > grasp_effort_threshold) and (gap > brick_width * 0.5)
        
        if grasp_success:
            print(f"\n  ✓ GRASP SUCCESSFUL!")
        else:
            if effort < grasp_effort_threshold:
                print(f"\n  ✗ Grasp failed: effort too low ({effort:.3f} A < {grasp_effort_threshold} A)")
            else:
                print(f"\n  ✗ Grasp failed: gap too small ({gap:.4f} m)")
            return False, None, "Grasp failed - check Z compensation calibration"
        
        return True, {
            'grasp_position': grasp_pos,
            'grasp_yaw': grasp_yaw,
            'z_compensation': z_compensation,
            'effort': effort,
            'gap': gap,
        }, None
    
    # ==================== Step 7: Lift ====================
    
    def execute_lift(
        self,
        brick_position: List[float],
        brick_yaw: float,
        grasp_yaw: float,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Lift the grasped brick.
        
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        self._log_step("\n[Step 7/10] Lifting brick...")
        if self._check_abort():
            return False, None, "Aborted"
        
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"
        tcp_pos_list = tcp_pos.tolist()
        
        # Plan lift with LLM
        success, lift_result, error = self.llm_planner.plan_lift(
            tcp_position=tcp_pos_list,
            tcp_yaw=grasp_yaw,
            brick_position=brick_position,
            brick_yaw=brick_yaw,
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
            return False, None, "Lift movement failed"
        
        print("  Brick lifted successfully!")
        
        return True, {
            'lift_position': lift_pos,
        }, None
    
    # ==================== Step 8: Move to Place ====================
    
    def move_to_place(
        self,
        grasp_yaw: float,
        custom_place_position: Optional[List[float]] = None,
        custom_place_yaw: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Move to place position (XY only, keep Z).
        
        Args:
            grasp_yaw: Current TCP yaw angle
            custom_place_position: Custom [x, y, z] position (for anomaly removal)
            custom_place_yaw: Custom yaw angle (optional)
        
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        self._log_step("\n[Step 8/10] Moving to place position...")
        if self._check_abort():
            return False, None, "Aborted"
        
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"
        tcp_pos_list = tcp_pos.tolist()
        
        # Use custom position if provided (for anomaly removal)
        if custom_place_position is not None:
            place_pos = list(custom_place_position)
            place_pos[2] = tcp_pos_list[2]  # Keep current Z (lifted height)
            place_yaw = custom_place_yaw if custom_place_yaw is not None else grasp_yaw
            print(f"  Using custom place position: [{place_pos[0]:.4f}, {place_pos[1]:.4f}, {place_pos[2]:.4f}] m")
        else:
            # Plan move to place with LLM (normal stacking)
            success, move_result, error = self.llm_planner.plan_move_to_place(
                tcp_position=tcp_pos_list,
                tcp_yaw=grasp_yaw,
            )
            
            if not success:
                print(f"  [Warning] LLM move_to_place failed: {error}")
                # Fallback: use config target position directly
                place_config = self.llm_planner.config.get("place", {})
                place_pos = list(place_config.get("target_position", [0.54, -0.04, tcp_pos_list[2]]))
                place_yaw = place_config.get("target_yaw", 0.0)
                place_pos[2] = tcp_pos_list[2]  # Keep current Z
            else:
                place_pos = move_result['target_position']
                place_yaw = move_result['target_yaw']
                print(f"  Move target: [{place_pos[0]:.4f}, {place_pos[1]:.4f}, {place_pos[2]:.4f}] m, yaw={np.degrees(place_yaw):.1f}°")
        
        # Execute move to place
        if not self.robot_env.move_arm(place_pos, place_yaw, wait=3.0, check_abort=self._check_abort):
            return False, None, "Move to place failed"
        
        print("  Reached place position!")
        
        return True, {
            'place_position': place_pos,
            'place_yaw': place_yaw,
        }, None
    
    # ==================== Step 9-10: Descend & Release ====================
    
    def execute_place_and_release(
        self,
        place_yaw: float,
        target_surface_z: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Descend to place surface and release with closed-loop control.
        
        Uses two contact detection methods:
        1. Z position error (z_error > threshold_mm)
        2. Arm effort (total arm current > effort_threshold)
        
        IMPORTANT: Real-time effort protection is implemented BEFORE LLM call
        to prevent damage from LLM response delay.
        
        Args:
            place_yaw: Yaw angle for placement
            target_surface_z: Target Z height for placement (if None, use config)
            
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        # Step 9: Descend to place
        self._log_step("\n[Step 9/10] Descending to place...")
        if self._check_abort():
            return False, None, "Aborted"
        
        tcp_pos = self.robot_env.get_tcp_position()
        if tcp_pos is None:
            return False, None, "Cannot get TCP position"
        tcp_pos_list = tcp_pos.tolist()
        
        # Get target Z
        if target_surface_z is None:
            place_config = self.llm_planner.config.get("place", {})
            place_surface_z = place_config.get("surface_z", 0.88)
        else:
            place_surface_z = target_surface_z
        
        final_place_pos = [tcp_pos_list[0], tcp_pos_list[1], place_surface_z]
        
        print(f"  Current Z: {tcp_pos_list[2]:.4f} m")
        print(f"  Target surface Z: {place_surface_z:.4f} m")
        print(f"  Descend distance: {tcp_pos_list[2] - place_surface_z:.4f} m")
        
        # Safety check
        if place_surface_z > tcp_pos_list[2]:
            print(f"  [Warning] Target Z ({place_surface_z:.4f}) is HIGHER than current ({tcp_pos_list[2]:.4f})")
        
        # Execute descend
        if not self.robot_env.move_arm(final_place_pos, place_yaw, wait=2.5, check_abort=self._check_abort):
            return False, None, "Place descend failed"
        
        print("  Descended to place position!")
        
        # Step 10: Closed-loop release
        self._log_step("\n[Step 10/10] Closed-loop release...")
        if self._check_abort():
            return False, None, "Aborted"
        
        print("\n  === Closed-Loop Placement ===")
        print("  Detecting surface contact via Z position error AND arm effort...")
        
        release_config = self.llm_planner.config.get("release", {})
        max_release_attempts = release_config.get("max_attempts", 10)
        contact_threshold_mm = release_config.get("contact_threshold_mm", 0.8)
        descend_step = release_config.get("descend_step", 0.005)
        lift_step = release_config.get("lift_step", 0.01)
        arm_effort_threshold = release_config.get("arm_effort_threshold", 12.0)
        
        print(f"  Config: z_threshold={contact_threshold_mm:.1f}mm, effort_threshold={arm_effort_threshold:.1f}A")
        print(f"          descend={descend_step*1000:.1f}mm, lift={lift_step*1000:.1f}mm")
        
        current_place_pos = list(final_place_pos)
        
        for attempt in range(1, max_release_attempts + 1):
            if self._check_abort():
                return False, None, "Aborted"
            
            tcp_pos = self.robot_env.get_tcp_position()
            if tcp_pos is None:
                print("  [Warning] Cannot get TCP position")
                continue
            
            actual_z = tcp_pos[2]
            target_z = current_place_pos[2]
            z_error = actual_z - target_z
            z_error_mm = z_error * 1000
    
            arm_effort = self.robot_env.get_arm_total_effort()
            arm_effort_str = f"{arm_effort:.2f}A" if arm_effort is not None else "N/A"
            
            print(f"\n  --- Release Attempt {attempt}/{max_release_attempts} ---")
            print(f"  Actual Z: {actual_z:.4f} m, Target Z: {target_z:.4f} m, Error: {z_error_mm:+.1f} mm")
            print(f"  Arm total effort: {arm_effort_str}")
            
            # ========== (BEFORE LLM) ==========
            if arm_effort is not None and arm_effort > arm_effort_threshold:
                print(f"  ⚠️ REAL-TIME EFFORT PROTECTION TRIGGERED!")
                print(f"     Effort {arm_effort:.2f}A > threshold {arm_effort_threshold:.1f}A")
                print(f"     Lifting {lift_step*1000:.1f} mm immediately before release...")
                
                current_place_pos[2] += lift_step
                self.robot_env.move_arm(current_place_pos, place_yaw, wait=0.5, check_abort=self._check_abort)
                break
            
            success, release_result, error = self.llm_planner.analyze_release_feedback(
                actual_z=actual_z,
                target_z=target_z,
                contact_threshold_mm=contact_threshold_mm,
                descend_step=descend_step,
                lift_step=lift_step,
                attempt_number=attempt,
                max_attempts=max_release_attempts,
                arm_effort=arm_effort,
                arm_effort_threshold=arm_effort_threshold,
            )
            
            if not success:
                print(f"  [Warning] LLM release analysis failed: {error}")
                current_place_pos[2] -= descend_step
                self.robot_env.move_arm(current_place_pos, place_yaw, wait=1.0, check_abort=self._check_abort)
                continue
            
            action = release_result['action']
            action_type = action['type']
            delta_z = action['delta_z']
            
            if action_type == 'release':
                print("  ✓ LLM: Contact detected, releasing immediately!")
                break
            elif action_type == 'lift_then_release':
                print(f"  ✓ LLM: Pressing detected, lifting {delta_z*1000:.1f} mm before release...")
                current_place_pos[2] += delta_z
                self.robot_env.move_arm(current_place_pos, place_yaw, wait=1.0, check_abort=self._check_abort)
                break
            elif action_type == 'descend':
                print(f"  → LLM: No contact, descending {abs(delta_z)*1000:.1f} mm...")
                current_place_pos[2] += delta_z
                if not self.robot_env.move_arm(current_place_pos, place_yaw, wait=1.0, check_abort=self._check_abort):
                    print("  [Warning] Descend move failed")
                    break
        
        # Release gripper
        print("\n  Opening gripper to release brick...")
        if not self.robot_env.open_gripper(0.08):
            return False, None, "Gripper open failed"
        
        time.sleep(0.3)
        
        # Retreat upward
        print("  Retreating...")
        retreat_pos = [current_place_pos[0], current_place_pos[1], current_place_pos[2] + 0.10]
        self.robot_env.move_arm(retreat_pos, place_yaw, wait=1.5, check_abort=self._check_abort)
        
        print("\n  ✓ Brick placed successfully!")
        
        return True, {
            'final_place_position': current_place_pos,
            'target_surface_z': place_surface_z,
        }, None
    
    # ==================== Complete Cycle ====================
    
    def execute_single_grasp_place(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        target_surface_z: Optional[float] = None,
        brick_index: int = 0,
        custom_place_position: Optional[List[float]] = None,
        custom_place_yaw: Optional[float] = None,
        is_anomaly_removal: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute a complete grasp-place cycle.
        
        Uses DynamicZCompensator from head_calc for Z correction.
        
        Args:
            rgb: RGB image from head camera
            depth: Depth image from head camera
            target_surface_z: Target Z height for placement (if None, use config)
            brick_index: Which brick to grasp (if multiple detected)
            custom_place_position: Custom [x, y, z] for placement (for anomaly removal)
            custom_place_yaw: Custom yaw for placement (optional)
            is_anomaly_removal: If True, prints anomaly removal messages
            
        Returns:
            Tuple of (success, error_message)
        """
        task_type = "Anomaly Removal" if is_anomaly_removal else "LLM Grasp-Place"
        print("\n" + "=" * 60)
        print(f"[Task Started] {task_type} with DynamicZCompensator")
        print("=" * 60)
        
        # Step 1: Head camera detection (includes DynamicZCompensator)
        success, head_result, error = self.detect_with_head_camera(rgb, depth, brick_index)
        if not success:
            return False, f"Detection failed: {error}"
        
        brick_pos = head_result['position']
        brick_yaw = head_result['yaw']
        z_compensation = head_result.get('z_compensation', 0.0)
        brick_pos_list = brick_pos.tolist() if hasattr(brick_pos, 'tolist') else list(brick_pos)
        
        print(f"\n[Debug] Head camera result:")
        print(f"  Position: {brick_pos_list}")
        print(f"  Z compensation applied: {z_compensation*1000:+.1f} mm")
        
        # Step 2-3: Plan and move to hover
        success, _, error = self.plan_and_move_to_hover(
            brick_pos_list, float(brick_yaw), z_compensation
        )
        if not success:
            return False, error
        
        # Step 4: Hand-eye fine positioning
        reference_xy = [brick_pos_list[0], brick_pos_list[1]]
        success, fine_result, error = self.fine_position_with_handeye(
            reference_z=brick_pos_list[2],
            reference_yaw=brick_yaw,
            reference_xy=reference_xy,
            z_compensation=z_compensation,
        )
        if not success:
            return False, error
        
        fine_pos = fine_result['position']
        fine_yaw = fine_result['yaw']
        fine_pos_list = fine_pos.tolist() if hasattr(fine_pos, 'tolist') else list(fine_pos)
        
        print(f"\n[Debug] Fine positioning result:")
        print(f"  Position: {fine_pos_list}")
        print(f"  Z (from head camera with compensation): {fine_pos_list[2]:.4f} m")
        
        # Step 5-6: Grasp
        success, grasp_result, error = self.execute_grasp(
            fine_pos_list, float(fine_yaw), z_compensation
        )
        if not success:
            return False, error
        
        grasp_yaw = grasp_result['grasp_yaw']
        
        # Step 7: Lift
        success, _, error = self.execute_lift(fine_pos_list, fine_yaw, grasp_yaw)
        if not success:
            return False, error
        
        # Step 8: Move to place (use custom position if provided)
        success, move_result, error = self.move_to_place(
            grasp_yaw,
            custom_place_position=custom_place_position,
            custom_place_yaw=custom_place_yaw,
        )
        if not success:
            return False, error
        
        place_yaw = move_result['place_yaw']
        
        # Step 9-10: Place and release
        # For anomaly removal, use current brick's Z as target surface
        if is_anomaly_removal and target_surface_z is None:
            # Use a safe height for placing removed bricks
            target_surface_z = brick_pos_list[2]  # Same height as where we picked it
        
        success, _, error = self.execute_place_and_release(place_yaw, target_surface_z)
        if not success:
            return False, error
        
        if is_anomaly_removal:
            print("\n" + "=" * 60)
            print("[Anomaly Removal Complete] Brick moved to safe location")
            print("=" * 60)
        
        return True, None

def create_executor(
    robot_env,
    segmenter,
    head_calc,
    handeye_calc,
    llm_planner,
    prompt: str = "block, brick, rectangular object",
) -> GraspPlaceExecutor:
    """Factory function to create executor."""
    return GraspPlaceExecutor(
        robot_env=robot_env,
        segmenter=segmenter,
        head_calc=head_calc,
        handeye_calc=handeye_calc,
        llm_planner=llm_planner,
        prompt=prompt,
    )