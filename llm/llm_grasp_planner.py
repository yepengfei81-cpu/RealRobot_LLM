"""
LLM Grasp Planner for Real Robot

This module serves as the bridge between:
- Sensor data (brick position, orientation from SAM3)
- LLM planning (via prompts)
- Robot control commands

It handles:
- Building context from sensor data
- Calling appropriate prompts for each phase
- Parsing LLM responses
- Converting to robot control commands
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from .llm_client import LLMClient, create_llm_client
from .prompts import get_pre_grasp_prompt, build_pre_grasp_context
from .prompts import get_descend_prompt, build_descend_context
from .prompts import get_grasp_feedback_prompt, build_grasp_feedback_context
from .prompts import get_lift_prompt, build_lift_context
from .prompts import get_move_to_place_prompt, build_move_to_place_context

class LLMGraspPlanner:
    """
    LLM-based Grasp Planner for real robot.
    
    This planner uses LLM to compute grasp poses based on sensor observations.
    Currently supports:
    - Pre-grasp planning: Move to hover position above brick
    - Descend planning: Compute descent position and gripper opening
    
    Future phases (can be added later):
    - Place planning: Compute placement position
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        verbose: bool = True,
    ):
        """
        Initialize the LLM Grasp Planner.
        
        Args:
            config_path: Path to llm_config.json (default: config/llm_config.json)
            llm_client: Pre-configured LLM client (optional, will create from config if None)
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize LLM client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = create_llm_client(str(self.config_path))
        
        # Extract config sections
        self.brick_config = self.config.get("brick", {})
        self.grasp_config = self.config.get("grasp", {})
        self.gripper_config = self.config.get("gripper", {})
        self.debug_config = self.config.get("debug", {})
        
        # Default brick size from config
        self.default_brick_size = self.brick_config.get("size_LWH", [0.11, 0.05, 0.025])
        self.hover_height = self.grasp_config.get("hover_height", 0.15)
        
        # Gripper parameters from config
        self.gripper_tip_length = self.gripper_config.get("tip_length", 0.04)
        self.gripper_clearance = self.gripper_config.get("clearance", 0.01)
        self.gripper_max_opening = self.gripper_config.get("max_opening", 0.08)
        
        if self.verbose:
            print(f"[LLM Planner] Initialized")
            print(f"  - hover_height: {self.hover_height}m")
            print(f"  - gripper_tip_length: {self.gripper_tip_length}m")
            print(f"  - gripper_clearance: {self.gripper_clearance}m")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            print(f"[LLM Planner] Warning: Config not found at {self.config_path}, using defaults")
            return {}
        
        with open(self.config_path) as f:
            return json.load(f)
    
    def _log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[LLM Planner] {message}")
    
    def _print_llm_response_summary(self, response: Dict, phase: str = ""):
        """Print a simplified summary of LLM response"""
        target_pose = response.get("target_pose", {})
        xyz = target_pose.get("xyz", [0, 0, 0])
        yaw = target_pose.get("yaw", 0)
        reasoning = response.get("reasoning", "N/A")
        
        print(f"  [LLM {phase}] Target: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m, yaw={np.degrees(yaw):.1f}°")
        
        # Print gripper info if present
        gripper = response.get("gripper", {})
        if gripper:
            gap = gripper.get("gap", 0)
            print(f"  [LLM {phase}] Gripper gap: {gap:.4f} m")
        
        print(f"  [LLM {phase}] Reasoning: {reasoning}")
    
    # ==================== Pre-Grasp Planning ====================
    
    def plan_pre_grasp(
        self,
        brick_position: List[float],
        brick_yaw: float,
        tcp_position: Optional[List[float]] = None,
        tcp_orientation: Optional[List[float]] = None,
        brick_size: Optional[List[float]] = None,
        attempt_idx: int = 0,
        feedback: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan pre-grasp position using LLM.
        
        This computes the hover position directly above the brick center.
        
        Args:
            brick_position: [x, y, z] brick center position in base_link frame (meters)
            brick_yaw: brick yaw angle in radians
            tcp_position: [x, y, z] current TCP position (optional)
            tcp_orientation: [roll, pitch, yaw] current TCP orientation (optional)
            brick_size: [L, W, H] brick dimensions (optional, uses config default)
            attempt_idx: Retry attempt number
            feedback: Error feedback from previous attempt
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - target_position: [x, y, z] target TCP position
            - target_yaw: target yaw angle in radians
            - reasoning: LLM's explanation
        """
        # Use default brick size if not provided
        if brick_size is None:
            brick_size = self.default_brick_size
        
        # Build context for prompt
        context = build_pre_grasp_context(
            brick_position=brick_position,
            brick_yaw=brick_yaw,
            tcp_position=tcp_position,
            tcp_orientation=tcp_orientation,
            brick_size=brick_size,
            hover_height=self.hover_height,
            gripper_max_opening=self.gripper_max_opening,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_pre_grasp_prompt(context, attempt_idx, feedback)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Prompt - System]")
            print(system_prompt)
            print("\n[LLM Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM call failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_llm_response_summary(response, "PreGrasp")
        
        # Parse response
        result = self._parse_pre_grasp_response(response, brick_position, brick_size)
        
        if result is None:
            return False, None, "Failed to parse LLM response"
        
        return True, result, None
    
    def _parse_pre_grasp_response(
        self, 
        response: Dict, 
        brick_position: List[float],
        brick_size: List[float],
    ) -> Optional[Dict]:
        """Parse LLM response for pre-grasp planning."""
        try:
            target_pose = response.get("target_pose", {})
            xyz = target_pose.get("xyz", [])
            yaw = target_pose.get("yaw", 0.0)
            reasoning = response.get("reasoning", "")
            verification = response.get("verification", {})
            
            if len(xyz) != 3:
                self._log(f"Invalid xyz format: {xyz}")
                return None
            
            # Basic sanity checks
            expected_z = brick_position[2] + brick_size[2] / 2 + self.hover_height
            z_tolerance = 0.05
            
            if abs(xyz[2] - expected_z) > z_tolerance:
                self._log(f"Warning: Z deviation large. Expected ~{expected_z:.4f}, got {xyz[2]:.4f}")
            
            return {
                "target_position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "target_yaw": float(yaw),
                "reasoning": reasoning,
                "verification": verification,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing response: {e}")
            return None
    
    # ==================== Descend Planning ====================

    def plan_descend(
        self,
        brick_position: List[float],
        brick_yaw: float,
        tcp_position: Optional[List[float]] = None,
        brick_size: Optional[List[float]] = None,
        attempt_idx: int = 0,
        feedback: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan descent from hover to grasp position.
        
        Args:
            brick_position: [x, y, z] - brick TOP SURFACE position from hand-eye camera
            brick_yaw: brick yaw angle in radians
            tcp_position: current TCP position (optional)
            brick_size: [L, W, H] brick dimensions (optional, uses config default)
            attempt_idx: Retry attempt number
            feedback: Error feedback from previous attempt
            
        Returns:
            Tuple of (success, result_dict, error_message)
            result_dict contains: target_position, target_yaw, gripper_gap
        """
        # Use default brick size if not provided
        if brick_size is None:
            brick_size = self.default_brick_size
        
        # Build context for descend prompt
        context = build_descend_context(
            brick_position=brick_position,
            brick_yaw=brick_yaw,
            tcp_position=tcp_position,
            brick_size=brick_size,
            gripper_max_opening=self.gripper_max_opening,
            gripper_clearance=self.gripper_clearance,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_descend_prompt(context, attempt_idx, feedback)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Descend Prompt - System]")
            print(system_prompt)
            print("\n[LLM Descend Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM call failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_llm_response_summary(response, "Descend")
        
        # Parse response
        result = self._parse_descend_response(response, brick_position, brick_size)
        
        if result is None:
            return False, None, "Failed to parse LLM descend response"
        
        return True, result, None
    
    def _parse_descend_response(
        self, 
        response: Dict, 
        brick_position: List[float],
        brick_size: List[float],
    ) -> Optional[Dict]:
        """Parse LLM response for descend planning."""
        try:
            target_pose = response.get("target_pose", {})
            xyz = target_pose.get("xyz", [])
            yaw = target_pose.get("yaw", 0.0)
            
            gripper = response.get("gripper", {})
            gap = gripper.get("gap", 0.0)
            
            reasoning = response.get("reasoning", "")
            verification = response.get("verification", {})
            
            if len(xyz) != 3:
                self._log(f"Invalid xyz format: {xyz}")
                return None
            
            # Validate gripper gap
            brick_width = brick_size[1]
            expected_gap = brick_width + self.gripper_clearance
            gap_tolerance = 0.01  # 1cm tolerance
            
            if abs(gap - expected_gap) > gap_tolerance:
                self._log(f"Warning: Gripper gap deviation. Expected ~{expected_gap:.4f}, got {gap:.4f}")
            
            # Validate Z height - should match input brick_position[2] directly
            expected_z = brick_position[2]  # 直接使用输入的 Z 坐标
            z_tolerance = 0.02  # 2cm tolerance
            
            if abs(xyz[2] - expected_z) > z_tolerance:
                self._log(f"Warning: Z deviation. Expected {expected_z:.4f}, got {xyz[2]:.4f}")
                # 如果 LLM 输出的 Z 偏差太大，强制使用输入值
                self._log(f"Correcting Z to input value: {expected_z:.4f}")
                xyz[2] = expected_z
            
            # Ensure gap is within gripper limits
            if gap > self.gripper_max_opening:
                self._log(f"Warning: Gap {gap:.4f} exceeds max {self.gripper_max_opening:.4f}, clamping")
                gap = self.gripper_max_opening
            
            if gap < brick_width:
                self._log(f"Warning: Gap {gap:.4f} less than brick width {brick_width:.4f}, adjusting")
                gap = brick_width + 0.005  # Minimum clearance
            
            return {
                "target_position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "target_yaw": float(yaw),
                "gripper_gap": float(gap),
                "reasoning": reasoning,
                "verification": verification,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing descend response: {e}")
            return None

    # ==================== Grasp Feedback Analysis ====================
    
    def analyze_grasp_feedback(
        self,
        brick_position: List[float],
        brick_yaw: float,
        tcp_position: List[float],
        gripper_effort: float,
        gripper_gap_after_close: float,
        brick_size: Optional[List[float]] = None,
        attempt_number: int = 1,
        max_attempts: int = 3,
        compensation_history: Optional[List[float]] = None,
        current_compensation: float = 0.0,
        applied_compensation: float = 0.0,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Analyze grasp attempt using sensor feedback with adaptive learning.
        
        Args:
            brick_position: [x, y, z] target grasp position
            brick_yaw: brick yaw angle
            tcp_position: [x, y, z] actual TCP position after grasp
            gripper_effort: gripper motor current after closing (A)
            gripper_gap_after_close: gripper gap after closing (m)
            brick_size: [L, W, H] brick dimensions
            attempt_number: current attempt number
            max_attempts: maximum retry attempts
            compensation_history: list of past successful compensation values (meters)
            current_compensation: current learned compensation value (meters)
            applied_compensation: compensation applied to this grasp attempt (meters)
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - grasp_success: bool
            - confidence: float
            - analysis: {effort_indicates_contact, position_at_brick_center, failure_mode}
            - adjustment: {needed: bool, delta_z: float, reason: str}
            - learning: {recommended_compensation: float, compensation_confidence: float}
            - reasoning: str
        """
        if brick_size is None:
            brick_size = self.default_brick_size
        
        effort_threshold = self.grasp_config.get("grasp_effort_threshold", 2.0)
        
        # Build context with compensation data
        context = build_grasp_feedback_context(
            brick_position=brick_position,
            brick_yaw=brick_yaw,
            brick_size=brick_size,
            tcp_position=tcp_position,
            gripper_effort=gripper_effort,
            gripper_gap_after_close=gripper_gap_after_close,
            effort_threshold=effort_threshold,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            compensation_history=compensation_history,
            current_compensation=current_compensation,
            applied_compensation=applied_compensation,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_grasp_feedback_prompt(context)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Grasp Feedback Prompt - System]")
            print(system_prompt)
            print("\n[LLM Grasp Feedback Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM feedback analysis failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_grasp_feedback_summary(response, applied_compensation)
        
        # Parse response
        result = self._parse_grasp_feedback_response(response, applied_compensation)
        
        if result is None:
            return False, None, "Failed to parse LLM feedback response"
        
        return True, result, None
    
    def _print_grasp_feedback_summary(self, response: Dict, applied_compensation: float = 0.0):
        """Print grasp feedback analysis summary."""
        grasp_success = response.get("grasp_success", False)
        confidence = response.get("confidence", 0.0)
        analysis = response.get("analysis", {})
        adjustment = response.get("adjustment", {})
        learning = response.get("learning", {})
        reasoning = response.get("reasoning", "N/A")
        
        status = "✓ SUCCESS" if grasp_success else "✗ FAILED"
        failure_mode = analysis.get("failure_mode", "unknown")
        
        print(f"  [LLM Feedback] {status} (confidence: {confidence:.2f})")
        if not grasp_success:
            print(f"  [LLM Feedback] Failure mode: {failure_mode}")
        
        if adjustment.get("needed", False):
            delta_z = adjustment.get("delta_z", 0)
            direction = "UP" if delta_z > 0 else "DOWN"
            print(f"  [LLM Feedback] Adjustment: {direction} {abs(delta_z)*1000:.1f} mm")
            print(f"  [LLM Feedback] Reason: {adjustment.get('reason', 'N/A')}")
        
        # Show learning output
        recommended = learning.get("recommended_compensation", applied_compensation)
        comp_confidence = learning.get("compensation_confidence", 0.0)
        print(f"  [LLM Learning] Recommended compensation: {recommended*1000:+.1f} mm (confidence: {comp_confidence:.2f})")
        
        print(f"  [LLM Feedback] Analysis: {reasoning}")
    
    def _parse_grasp_feedback_response(self, response: Dict, applied_compensation: float = 0.0) -> Optional[Dict]:
        """Parse LLM response for grasp feedback analysis."""
        try:
            grasp_success = response.get("grasp_success", False)
            confidence = response.get("confidence", 0.0)
            analysis = response.get("analysis", {})
            adjustment = response.get("adjustment", {})
            learning = response.get("learning", {})
            reasoning = response.get("reasoning", "")
            
            # Validate and clamp delta_z
            delta_z = adjustment.get("delta_z", 0.0)
            max_adjustment = 0.05  # 5cm max
            delta_z = float(np.clip(delta_z, -max_adjustment, max_adjustment))
            
            # If grasp successful, no adjustment needed
            if grasp_success:
                delta_z = 0.0
                adjustment["needed"] = False
            
            # Parse learning output
            recommended_compensation = learning.get("recommended_compensation", applied_compensation)
            compensation_confidence = learning.get("compensation_confidence", 0.5)
            
            # Clamp compensation to reasonable range
            max_compensation = 0.05  # 5cm max
            recommended_compensation = float(np.clip(recommended_compensation, -max_compensation, max_compensation))
            
            return {
                "grasp_success": bool(grasp_success),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "analysis": {
                    "effort_indicates_contact": analysis.get("effort_indicates_contact", False),
                    "position_at_brick_center": analysis.get("position_at_brick_center", False),
                    "failure_mode": analysis.get("failure_mode", "unknown"),
                },
                "adjustment": {
                    "needed": bool(adjustment.get("needed", False)),
                    "delta_z": delta_z,
                    "reason": adjustment.get("reason", ""),
                },
                "learning": {
                    "recommended_compensation": recommended_compensation,
                    "compensation_confidence": float(np.clip(compensation_confidence, 0.0, 1.0)),
                },
                "reasoning": reasoning,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing grasp feedback response: {e}")
            return None
    # ==================== Lift Planning ====================      
  
    def plan_lift(
        self,
        tcp_position: List[float],
        tcp_yaw: float,
        brick_position: Optional[List[float]] = None,
        brick_yaw: Optional[float] = None,
        brick_size: Optional[List[float]] = None,
        lift_height: Optional[float] = None,
        attempt_idx: int = 0,
        feedback: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan lift trajectory after successful grasp.
        
        Args:
            tcp_position: [x, y, z] current TCP position (at grasp height)
            tcp_yaw: current TCP yaw angle in radians
            brick_position: [x, y, z] brick position (optional)
            brick_yaw: brick yaw angle (optional)
            brick_size: [L, W, H] brick dimensions (optional)
            lift_height: height to lift (optional, uses config default)
            attempt_idx: Retry attempt number
            feedback: Error feedback from previous attempt
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - target_position: [x, y, z] target TCP position after lift
            - target_yaw: target yaw angle (unchanged)
            - lift_height: actual lift height used
            - reasoning: LLM's explanation
        """
        # Use defaults from config
        if brick_size is None:
            brick_size = self.default_brick_size
        
        if lift_height is None:
            lift_height = self.grasp_config.get("lift_height", 0.10)
        
        # Build context
        context = build_lift_context(
            tcp_position=tcp_position,
            tcp_yaw=tcp_yaw,
            brick_position=brick_position,
            brick_yaw=brick_yaw,
            brick_size=brick_size,
            lift_height=lift_height,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_lift_prompt(context, attempt_idx, feedback)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Lift Prompt - System]")
            print(system_prompt)
            print("\n[LLM Lift Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM lift call failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_llm_response_summary(response, "Lift")
        
        # Parse response
        result = self._parse_lift_response(response, tcp_position, tcp_yaw, lift_height)
        
        if result is None:
            return False, None, "Failed to parse LLM lift response"
        
        return True, result, None
    
    def _parse_lift_response(
        self,
        response: Dict,
        tcp_position: List[float],
        tcp_yaw: float,
        expected_lift_height: float,
    ) -> Optional[Dict]:
        """Parse LLM response for lift planning."""
        try:
            target_pose = response.get("target_pose", {})
            xyz = target_pose.get("xyz", [])
            yaw = target_pose.get("yaw", 0.0)
            
            lift_params = response.get("lift_params", {})
            lift_height = lift_params.get("lift_height", expected_lift_height)
            target_z = lift_params.get("target_z", tcp_position[2] + expected_lift_height)
            
            reasoning = response.get("reasoning", "")
            verification = response.get("verification", {})
            
            if len(xyz) != 3:
                self._log(f"Invalid xyz format: {xyz}")
                return None
            
            # Validate XY unchanged
            xy_tolerance = 0.005  # 5mm
            if abs(xyz[0] - tcp_position[0]) > xy_tolerance:
                self._log(f"Warning: X changed. Expected {tcp_position[0]:.4f}, got {xyz[0]:.4f}, correcting")
                xyz[0] = tcp_position[0]
            
            if abs(xyz[1] - tcp_position[1]) > xy_tolerance:
                self._log(f"Warning: Y changed. Expected {tcp_position[1]:.4f}, got {xyz[1]:.4f}, correcting")
                xyz[1] = tcp_position[1]
            
            # Validate Z is higher
            expected_z = tcp_position[2] + expected_lift_height
            z_tolerance = 0.02  # 2cm
            if abs(xyz[2] - expected_z) > z_tolerance:
                self._log(f"Warning: Z deviation. Expected {expected_z:.4f}, got {xyz[2]:.4f}, correcting")
                xyz[2] = expected_z
            
            # Validate yaw unchanged
            yaw_tolerance = 0.05  # ~3 degrees
            if abs(yaw - tcp_yaw) > yaw_tolerance:
                self._log(f"Warning: Yaw changed. Expected {tcp_yaw:.4f}, got {yaw:.4f}, correcting")
                yaw = tcp_yaw
            
            return {
                "target_position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "target_yaw": float(yaw),
                "lift_height": float(lift_height),
                "reasoning": reasoning,
                "verification": verification,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing lift response: {e}")
            return None

    # ==================== Move to Place Planning ====================
    
    def plan_move_to_place(
        self,
        tcp_position: List[float],
        tcp_yaw: float,
        target_position: Optional[List[float]] = None,
        target_yaw: Optional[float] = None,
        brick_size: Optional[List[float]] = None,
        attempt_idx: int = 0,
        feedback: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan horizontal movement from lift position to above placement target.
        
        Args:
            tcp_position: [x, y, z] current TCP position (after lift)
            tcp_yaw: current TCP yaw angle in radians
            target_position: [x, y, z] target placement position (optional, uses config)
            target_yaw: target yaw for placement (optional, uses config)
            brick_size: [L, W, H] brick dimensions (optional)
            attempt_idx: Retry attempt number
            feedback: Error feedback from previous attempt
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - target_position: [x, y, z] target TCP position (Z unchanged)
            - target_yaw: target yaw angle
            - reasoning: LLM's explanation
        """
        # Use defaults from config
        if brick_size is None:
            brick_size = self.default_brick_size
        
        # Get place config
        place_config = self.config.get("place", {})
        
        if target_position is None:
            target_position = place_config.get("target_position", [0.54, -0.04, 1.00])
        
        if target_yaw is None:
            target_yaw = place_config.get("target_yaw", 0.0)
        
        # Build context
        context = build_move_to_place_context(
            tcp_position=tcp_position,
            tcp_yaw=tcp_yaw,
            target_position=target_position,
            target_yaw=target_yaw,
            brick_size=brick_size,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_move_to_place_prompt(context, attempt_idx, feedback)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Move to Place Prompt - System]")
            print(system_prompt)
            print("\n[LLM Move to Place Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM move_to_place call failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_llm_response_summary(response, "MoveToPlace")
        
        # Parse response
        result = self._parse_move_to_place_response(
            response, tcp_position, target_position, target_yaw
        )
        
        if result is None:
            return False, None, "Failed to parse LLM move_to_place response"
        
        return True, result, None
    
    def _parse_move_to_place_response(
        self,
        response: Dict,
        tcp_position: List[float],
        target_position: List[float],
        target_yaw: float,
    ) -> Optional[Dict]:
        """Parse LLM response for move-to-place planning."""
        try:
            target_pose = response.get("target_pose", {})
            xyz = target_pose.get("xyz", [])
            yaw = target_pose.get("yaw", 0.0)
            
            motion_params = response.get("motion_params", {})
            reasoning = response.get("reasoning", "")
            verification = response.get("verification", {})
            
            if len(xyz) != 3:
                self._log(f"Invalid xyz format: {xyz}")
                return None
            
            # Validate XY at target
            xy_tolerance = 0.01  # 1cm
            if abs(xyz[0] - target_position[0]) > xy_tolerance:
                self._log(f"Warning: X deviation. Expected {target_position[0]:.4f}, got {xyz[0]:.4f}, correcting")
                xyz[0] = target_position[0]
            
            if abs(xyz[1] - target_position[1]) > xy_tolerance:
                self._log(f"Warning: Y deviation. Expected {target_position[1]:.4f}, got {xyz[1]:.4f}, correcting")
                xyz[1] = target_position[1]
            
            # Validate Z unchanged (should stay at lift height)
            z_tolerance = 0.02  # 2cm
            if abs(xyz[2] - tcp_position[2]) > z_tolerance:
                self._log(f"Warning: Z should be unchanged. Expected {tcp_position[2]:.4f}, got {xyz[2]:.4f}, correcting")
                xyz[2] = tcp_position[2]
            
            # Validate yaw at target
            yaw_tolerance = 0.1  # ~6 degrees
            if abs(yaw - target_yaw) > yaw_tolerance:
                self._log(f"Warning: Yaw deviation. Expected {target_yaw:.4f}, got {yaw:.4f}, correcting")
                yaw = target_yaw
            
            return {
                "target_position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "target_yaw": float(yaw),
                "motion_params": motion_params,
                "reasoning": reasoning,
                "verification": verification,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing move_to_place response: {e}")
            return None

    # ==================== Release Planning ====================
    
    def analyze_release_feedback(
        self,
        actual_z: float,
        target_z: float,
        contact_threshold_mm: float = 0.3,
        descend_step: float = 0.005,
        lift_step: float = 0.01,
        attempt_number: int = 1,
        max_attempts: int = 10,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Analyze placement feedback and decide release action.
        
        Args:
            actual_z: Current TCP Z position (meters)
            target_z: Target surface Z position (meters)
            contact_threshold_mm: Threshold for contact detection (mm)
            descend_step: Step size for descending (meters)
            lift_step: Step size for lifting before release (meters)
            attempt_number: Current attempt number
            max_attempts: Maximum attempts
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - contact_detected: bool
            - confidence: float
            - analysis: {z_error_mm, contact_state}
            - action: {type, delta_z, reason}
            - reasoning: str
        """
        from .prompts import get_release_prompt, build_release_context
        
        # Build context
        context = build_release_context(
            actual_z=actual_z,
            target_z=target_z,
            contact_threshold_mm=contact_threshold_mm,
            descend_step=descend_step,
            lift_step=lift_step,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_release_prompt(context)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Release Prompt - System]")
            print(system_prompt)
            print("\n[LLM Release Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM release analysis failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_release_summary(response)
        
        # Parse response
        result = self._parse_release_response(response, descend_step, lift_step)
        
        if result is None:
            return False, None, "Failed to parse LLM release response"
        
        return True, result, None
    
    def _print_release_summary(self, response: Dict):
        """Print release analysis summary."""
        contact_detected = response.get("contact_detected", False)
        confidence = response.get("confidence", 0.0)
        analysis = response.get("analysis", {})
        action = response.get("action", {})
        reasoning = response.get("reasoning", "N/A")
        
        z_error_mm = analysis.get("z_error_mm", 0)
        contact_state = analysis.get("contact_state", "unknown")
        action_type = action.get("type", "unknown")
        delta_z = action.get("delta_z", 0)
        
        print(f"  [LLM Release] Contact: {contact_state} (z_error: {z_error_mm:+.1f} mm)")
        print(f"  [LLM Release] Action: {action_type}, delta_z: {delta_z*1000:+.1f} mm")
        print(f"  [LLM Release] Confidence: {confidence:.2f}")
        print(f"  [LLM Release] Reason: {action.get('reason', 'N/A')}")
    
    def _parse_release_response(
        self, 
        response: Dict,
        descend_step: float,
        lift_step: float,
    ) -> Optional[Dict]:
        """Parse LLM response for release decision."""
        try:
            contact_detected = response.get("contact_detected", False)
            confidence = response.get("confidence", 0.0)
            analysis = response.get("analysis", {})
            action = response.get("action", {})
            reasoning = response.get("reasoning", "")
            
            action_type = action.get("type", "descend")
            delta_z = action.get("delta_z", 0.0)
            
            # Normalize action type (no "release" allowed, convert to appropriate action)
            if action_type == "release":
                # Convert "release" to "descend" (keep going down)
                action_type = "descend"
                delta_z = -descend_step
            
            # Validate and clamp delta_z
            if action_type == "descend":
                # Should be negative (going down)
                delta_z = -abs(delta_z) if delta_z != 0 else -descend_step
                delta_z = max(delta_z, -0.02)  # Max 2cm down
            elif action_type == "lift_then_release":
                # Should be positive (going up)
                delta_z = abs(delta_z) if delta_z != 0 else lift_step
                delta_z = min(delta_z, 0.02)  # Max 2cm up
            
            # Normalize contact_state (no "just_contact")
            contact_state = analysis.get("contact_state", "no_contact")
            if contact_state == "just_contact":
                contact_state = "no_contact"
            
            return {
                "contact_detected": bool(contact_detected),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "analysis": {
                    "z_error_mm": float(analysis.get("z_error_mm", 0)),
                    "contact_state": contact_state,
                },
                "action": {
                    "type": action_type,
                    "delta_z": float(delta_z),
                    "reason": action.get("reason", ""),
                },
                "reasoning": reasoning,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing release response: {e}")
            return None

    # ==================== Scene Analysis ====================
    
    def analyze_scene(
        self,
        detected_bricks: List[Dict[str, Any]],
        target_position: List[float],
        placed_count: int = 0,
        initial_brick_count: Optional[int] = None,
        proximity_threshold: float = 0.08,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Analyze scene for multi-brick stacking task.
        
        Args:
            detected_bricks: List of detected brick info, each with:
                - position: [x, y, z] in base_link frame
                - yaw: orientation (optional)
            target_position: [x, y, z] target placement position
            placed_count: Number of bricks already placed
            initial_brick_count: Initial number of bricks (if known)
            proximity_threshold: Distance threshold for "at target" (meters)
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - task_complete: bool
            - confidence: float
            - scene_state: {total_bricks_detected, bricks_at_target, bricks_to_grasp, estimated_stack_height}
            - next_action: {type, target_brick_index, reason}
            - analysis: {target_area_status, completion_criteria_met, reasoning}
        """
        from .prompts import get_scene_analysis_prompt, build_scene_analysis_context
        
        # Get brick config
        brick_height = self.brick_config.get("size_LWH", [0.11, 0.05, 0.025])[2]
        stack_increment = brick_height + 0.003  # 3mm margin
        
        # Build context
        context = build_scene_analysis_context(
            detected_bricks=detected_bricks,
            target_position=target_position,
            placed_count=placed_count,
            initial_brick_count=initial_brick_count,
            proximity_threshold=proximity_threshold,
            brick_height=brick_height,
            stack_increment=stack_increment,
        )
        
        # Generate prompt
        system_prompt, user_prompt = get_scene_analysis_prompt(context)
        
        # Debug: print prompt if enabled
        if self.debug_config.get("print_llm_prompt", False):
            print("\n" + "=" * 60)
            print("[LLM Scene Analysis Prompt - System]")
            print(system_prompt)
            print("\n[LLM Scene Analysis Prompt - User]")
            print(user_prompt)
            print("=" * 60 + "\n")
        
        # Call LLM
        success, response, error = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if not success:
            self._log(f"LLM scene analysis failed: {error}")
            return False, None, error
        
        # Debug: print response
        if self.debug_config.get("print_llm_response", True):
            self._print_scene_analysis_summary(response)
        
        # Parse response
        result = self._parse_scene_analysis_response(response, len(detected_bricks))
        
        if result is None:
            return False, None, "Failed to parse LLM scene analysis response"
        
        return True, result, None
    
    def _print_scene_analysis_summary(self, response: Dict):
        """Print scene analysis summary."""
        task_complete = response.get("task_complete", False)
        confidence = response.get("confidence", 0.0)
        scene_state = response.get("scene_state", {})
        next_action = response.get("next_action", {})
        analysis = response.get("analysis", {})
        
        total = scene_state.get("total_bricks_detected", 0)
        at_target = scene_state.get("bricks_at_target", 0)
        to_grasp = scene_state.get("bricks_to_grasp", 0)
        
        status = "✓ COMPLETE" if task_complete else "→ IN PROGRESS"
        print(f"  [LLM Scene] {status} (confidence: {confidence:.2f})")
        print(f"  [LLM Scene] Bricks: {total} total, {at_target} at target, {to_grasp} to grasp")
        
        action_type = next_action.get("type", "unknown")
        if action_type == "grasp":
            idx = next_action.get("target_brick_index", -1)
            print(f"  [LLM Scene] Next: Grasp brick #{idx}")
        elif action_type == "complete":
            print(f"  [LLM Scene] Next: Task complete!")
        else:
            print(f"  [LLM Scene] Next: {action_type} - {next_action.get('reason', 'N/A')}")
    
    def _parse_scene_analysis_response(
        self, 
        response: Dict,
        num_detected: int,
    ) -> Optional[Dict]:
        """Parse LLM response for scene analysis."""
        try:
            task_complete = response.get("task_complete", False)
            confidence = response.get("confidence", 0.0)
            scene_state = response.get("scene_state", {})
            next_action = response.get("next_action", {})
            analysis = response.get("analysis", {})
            
            action_type = next_action.get("type", "error")
            target_idx = next_action.get("target_brick_index")
            
            # Validate target_brick_index
            if action_type == "grasp":
                if target_idx is None or not isinstance(target_idx, int):
                    self._log("Warning: grasp action but no valid target_brick_index")
                    # Try to find a valid brick (first one away from target)
                    target_idx = 0 if num_detected > 0 else None
                elif target_idx < 0 or target_idx >= num_detected:
                    self._log(f"Warning: target_brick_index {target_idx} out of range [0, {num_detected})")
                    target_idx = 0 if num_detected > 0 else None
            
            return {
                "task_complete": bool(task_complete),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "scene_state": {
                    "total_bricks_detected": int(scene_state.get("total_bricks_detected", num_detected)),
                    "bricks_at_target": int(scene_state.get("bricks_at_target", 0)),
                    "bricks_to_grasp": int(scene_state.get("bricks_to_grasp", num_detected)),
                    "estimated_stack_height": int(scene_state.get("estimated_stack_height", 0)),
                },
                "next_action": {
                    "type": action_type,
                    "target_brick_index": target_idx,
                    "reason": next_action.get("reason", ""),
                },
                "analysis": {
                    "target_area_status": analysis.get("target_area_status", "unknown"),
                    "completion_criteria_met": analysis.get("completion_criteria_met", False),
                    "reasoning": analysis.get("reasoning", ""),
                },
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing scene analysis response: {e}")
            return None        
# ==================== Factory Function ====================

def create_grasp_planner(config_path: Optional[str] = None) -> LLMGraspPlanner:
    """
    Create a grasp planner with default or specified config.
    
    Args:
        config_path: Path to config file (default: config/llm_config.json)
        
    Returns:
        Configured LLMGraspPlanner instance
    """
    return LLMGraspPlanner(config_path=config_path)