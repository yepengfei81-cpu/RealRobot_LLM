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
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Analyze grasp attempt using sensor feedback.
        
        Args:
            brick_position: [x, y, z] target grasp position
            brick_yaw: brick yaw angle
            tcp_position: [x, y, z] actual TCP position after grasp
            gripper_effort: gripper motor current after closing (A)
            gripper_gap_after_close: gripper gap after closing (m)
            brick_size: [L, W, H] brick dimensions
            attempt_number: current attempt number
            max_attempts: maximum retry attempts
            
        Returns:
            Tuple of (success, result_dict, error_message)
            
            result_dict contains:
            - grasp_success: bool
            - confidence: float
            - analysis: {effort_indicates_contact, position_at_brick_center, failure_mode}
            - adjustment: {needed: bool, delta_z: float, reason: str}
            - reasoning: str
        """
        if brick_size is None:
            brick_size = self.default_brick_size
        
        effort_threshold = self.grasp_config.get("grasp_effort_threshold", 2.0)
        
        # Build context
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
            self._print_grasp_feedback_summary(response)
        
        # Parse response
        result = self._parse_grasp_feedback_response(response)
        
        if result is None:
            return False, None, "Failed to parse LLM feedback response"
        
        return True, result, None
    
    def _print_grasp_feedback_summary(self, response: Dict):
        """Print grasp feedback analysis summary."""
        grasp_success = response.get("grasp_success", False)
        confidence = response.get("confidence", 0.0)
        analysis = response.get("analysis", {})
        adjustment = response.get("adjustment", {})
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
        
        print(f"  [LLM Feedback] Analysis: {reasoning}")
    
    def _parse_grasp_feedback_response(self, response: Dict) -> Optional[Dict]:
        """Parse LLM response for grasp feedback analysis."""
        try:
            grasp_success = response.get("grasp_success", False)
            confidence = response.get("confidence", 0.0)
            analysis = response.get("analysis", {})
            adjustment = response.get("adjustment", {})
            reasoning = response.get("reasoning", "")
            
            # Validate and clamp delta_z
            delta_z = adjustment.get("delta_z", 0.0)
            max_adjustment = 0.05  # 5cm max
            delta_z = float(np.clip(delta_z, -max_adjustment, max_adjustment))
            
            # If grasp successful, no adjustment needed
            if grasp_success:
                delta_z = 0.0
                adjustment["needed"] = False
            
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
                "reasoning": reasoning,
                "raw_response": response,
            }
            
        except Exception as e:
            self._log(f"Error parsing grasp feedback response: {e}")
            return None
            
    # ==================== Place Planning (Placeholder) ====================
    
    def plan_place(
        self,
        current_position: List[float],
        place_position: List[float],
        place_yaw: float,
        **kwargs
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan placement position.
        
        TODO: Implement when needed. For now, use config target position.
        """
        self._log("Place planning not implemented, using config fallback")
        
        place_config = self.config.get("place", {})
        target = place_config.get("target_position", place_position)
        target_yaw = place_config.get("target_yaw", place_yaw)
        
        return True, {
            "target_position": target,
            "target_yaw": target_yaw,
            "reasoning": "Using config place position",
            "is_fallback": True,
        }, None


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