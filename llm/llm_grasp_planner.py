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


class LLMGraspPlanner:
    """
    LLM-based Grasp Planner for real robot.
    
    This planner uses LLM to compute grasp poses based on sensor observations.
    Currently supports:
    - Pre-grasp planning: Move to hover position above brick
    
    Future phases (can be added later):
    - Descend planning: Fine-tune descent trajectory
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
        self.debug_config = self.config.get("debug", {})
        
        # Default brick size from config
        self.default_brick_size = self.brick_config.get("size_LWH", [0.11, 0.05, 0.025])
        self.hover_height = self.grasp_config.get("hover_height", 0.15)
        
        if self.verbose:
            print(f"[LLM Planner] Initialized (hover_height={self.hover_height}m)")
    
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
            gripper_max_opening=self.config.get("robot", {}).get("gripper_max_opening", 0.08),
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
        
        # Debug: print response (simplified format)
        if self.debug_config.get("print_llm_response", True):
            self._print_llm_response_summary(response)
        
        # Parse response
        result = self._parse_pre_grasp_response(response, brick_position, brick_size)
        
        if result is None:
            return False, None, "Failed to parse LLM response"
        
        return True, result, None
    
    def _print_llm_response_summary(self, response: Dict):
        """Print a simplified summary of LLM response"""
        target_pose = response.get("target_pose", {})
        xyz = target_pose.get("xyz", [0, 0, 0])
        yaw = target_pose.get("yaw", 0)
        reasoning = response.get("reasoning", "N/A")
        
        print(f"  [LLM] Target: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m, yaw={np.degrees(yaw):.1f}°")
        print(f"  [LLM] Reasoning: {reasoning}")
    
    def _parse_pre_grasp_response(
        self, 
        response: Dict, 
        brick_position: List[float],
        brick_size: List[float],
    ) -> Optional[Dict]:
        """
        Parse LLM response for pre-grasp planning.
        
        Validates the response and extracts target pose.
        """
        try:
            target_pose = response.get("target_pose", {})
            xyz = target_pose.get("xyz", [])
            yaw = target_pose.get("yaw", 0.0)
            reasoning = response.get("reasoning", "")
            verification = response.get("verification", {})
            
            # Validate xyz
            if len(xyz) != 3:
                self._log(f"Invalid xyz format: {xyz}")
                return None
            
            # Basic sanity checks
            expected_z = brick_position[2] + brick_size[2] / 2 + self.hover_height
            z_tolerance = 0.05  # Allow 5cm tolerance
            
            if abs(xyz[2] - expected_z) > z_tolerance:
                self._log(f"Warning: Z deviation large. Expected ~{expected_z:.4f}, got {xyz[2]:.4f}")
                # Don't fail, just warn - LLM might have different calculation
            
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
    
    # ==================== Future Phase Placeholders ====================
    
    def plan_descend(
        self,
        current_position: List[float],
        brick_position: List[float],
        brick_yaw: float,
        **kwargs
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Plan descent trajectory to grasp brick.
        
        TODO: Implement when needed. For now, use geometric calculation.
        """
        self._log("Descend planning not implemented, using geometric fallback")
        
        # Simple geometric fallback: go straight down to brick position
        target_position = [brick_position[0], brick_position[1], brick_position[2]]
        
        return True, {
            "target_position": target_position,
            "target_yaw": brick_yaw,
            "reasoning": "Geometric fallback: direct descent to brick center",
            "is_fallback": True,
        }, None
    
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


# ==================== Test Code ====================

if __name__ == "__main__":
    """Test the LLM grasp planner"""
    
    print("=" * 60)
    print("Testing LLM Grasp Planner")
    print("=" * 60)
    
    # Create planner
    planner = create_grasp_planner()
    
    # Test pre-grasp planning with sample data
    test_brick_position = [0.35, 0.15, 0.95]  # Example position
    test_brick_yaw = 0.5  # ~28.6 degrees
    
    print(f"\nTest input:")
    print(f"  Brick position: {test_brick_position}")
    print(f"  Brick yaw: {test_brick_yaw:.4f} rad ({np.degrees(test_brick_yaw):.1f}°)")
    
    print("\nCalling LLM for pre-grasp planning...")
    success, result, error = planner.plan_pre_grasp(
        brick_position=test_brick_position,
        brick_yaw=test_brick_yaw,
    )
    
    if success:
        print("\n✓ Pre-grasp planning successful!")
        print(f"  Target position: {result['target_position']}")
        print(f"  Target yaw: {result['target_yaw']:.4f} rad")
    else:
        print(f"\n✗ Pre-grasp planning failed: {error}")