"""
Descend Prompt for Real Robot

This prompt guides the LLM to plan the descent trajectory for grasping.
The robot TCP descends from hover position to grasp position with proper gripper opening.
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

DESCEND_REPLY_TEMPLATE = """
{
  "target_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "yaw": <float yaw in radians>
  },
  "gripper": {
    "gap": <float gripper opening width in meters>
  },
  "reasoning": "Brief explanation of the descent calculation (max 50 words)",
  "verification": {
    "xy_aligned_with_brick": true,
    "z_at_grasp_height": true,
    "gripper_gap_sufficient": true,
    "yaw_aligned_with_brick": true
  }
}
""".strip()


# ==================== Descend Prompt Builder ====================

class DescendPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for descend phase.
    
    Goal: Plan TCP descent from hover position to grasp position,
    and determine proper gripper opening width.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Descend to Grasp Position"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Robot initialization and calibration",
            "Head camera brick detection and localization",
            "Pre-grasp position planning completed",
            "TCP moved to hover position above brick",
            "Hand-eye camera fine positioning completed",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Calculate gripper opening and descent height (current task)",
            "Open gripper and descend to grasp position",
            "Close gripper to grasp brick",
            "Lift brick and move to placement position",
        ]
    
    def _get_role_name(self) -> str:
        return "Descend Grasp Planning Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Calculate TCP descent height for optimal grasp",
            "Determine gripper opening width based on brick size",
            "Maintain XY alignment with brick center during descent",
            "Ensure safe descent without collision",
        ]
    
    def _get_specific_task(self) -> str:
        brick_pos = self.brick.get('position', [0, 0, 0])
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_yaw = self.brick.get('yaw', 0)
        
        gripper = self.context.get('gripper', {})
        clearance = gripper.get('clearance', 0.01)
        
        brick_width = brick_size[1]
        
        return f"""
Calculate the descend pose and gripper opening for grasping the brick.

**Given (from hand-eye fine positioning):**
- Brick grasp position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m
- Brick size (L×W×H): [{brick_size[0]:.3f}, {brick_size[1]:.3f}, {brick_size[2]:.3f}] m
- Brick yaw angle: {brick_yaw:.4f} rad ({math.degrees(brick_yaw):.1f}°)
- Gripper clearance: {clearance:.3f} m

**IMPORTANT - The position provided is already the correct grasp height:**
- The hand-eye camera system has already calculated the optimal TCP grasp position
- The Z coordinate ({brick_pos[2]:.4f}m) is the TARGET grasp height - use it directly
- Do NOT modify the Z value

**Required Output:**
- TCP X: {brick_pos[0]:.4f} m (same as input)
- TCP Y: {brick_pos[1]:.4f} m (same as input)  
- TCP Z: {brick_pos[2]:.4f} m (same as input - already correct grasp height)
- TCP yaw: {brick_yaw:.4f} rad (same as input)
- Gripper gap: brick_width + clearance = {brick_width:.3f} + {clearance:.3f} = {brick_width + clearance:.4f} m
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_pos = self.brick.get('position', [0, 0, 0])
        gripper = self.context.get('gripper', {})
        clearance = gripper.get('clearance', 0.01)
        
        return f"""
**Grasp Height - Already Calculated:**
- The hand-eye camera + calibration system has computed the correct grasp position
- The input Z coordinate ({brick_pos[2]:.4f}m) accounts for:
  - Camera-to-gripper transformation
  - Brick surface detection
  - TCP offset to grasp center
- Simply use the provided Z value as-is

**Gripper Opening Calculation:**
- Gripper gap = brick_width + clearance
- Brick width: {brick_size[1]:.3f} m
- Clearance: {clearance:.3f} m
- Total gap: {brick_size[1] + clearance:.4f} m
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        brick_pos = self.brick.get('position', [0, 0, 0])
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_yaw = self.brick.get('yaw', 0)
        
        gripper = self.context.get('gripper', {})
        clearance = gripper.get('clearance', 0.01)
        
        brick_width = brick_size[1]
        target_gap = brick_width + clearance
        
        return [
            {
                "title": "Extract Input Parameters",
                "content": f"""
- Brick grasp position: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}) m
- Brick yaw: {brick_yaw:.4f} rad
- Brick width (W): {brick_width:.3f} m
- Gripper clearance: {clearance:.3f} m
""".strip()
            },
            {
                "title": "Determine TCP Position",
                "content": f"""
- TCP X = brick_pos[0] = {brick_pos[0]:.4f} m
- TCP Y = brick_pos[1] = {brick_pos[1]:.4f} m
- TCP Z = brick_pos[2] = {brick_pos[2]:.4f} m (use directly, already correct)
- TCP yaw = brick_yaw = {brick_yaw:.4f} rad
""".strip()
            },
            {
                "title": "Calculate Gripper Opening",
                "content": f"""
- Gripper gap = brick_width + clearance
- Gripper gap = {brick_width:.3f} + {clearance:.3f}
- Gripper gap = {target_gap:.4f} m
""".strip()
            },
            {
                "title": "Final Output",
                "content": f"""
- xyz: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}]
- yaw: {brick_yaw:.4f}
- gripper.gap: {target_gap:.4f}
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return DESCEND_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        brick_pos = self.brick.get('position', [0, 0, 0])
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_yaw = self.brick.get('yaw', 0)
        
        gripper = self.context.get('gripper', {})
        clearance = gripper.get('clearance', 0.01)
        
        target_gap = brick_size[1] + clearance
        
        return [
            f"xyz[0] must equal {brick_pos[0]:.4f} (input X)",
            f"xyz[1] must equal {brick_pos[1]:.4f} (input Y)",
            f"xyz[2] must equal {brick_pos[2]:.4f} (input Z - do NOT modify)",
            f"yaw must equal {brick_yaw:.4f} rad",
            f"gripper.gap must equal {target_gap:.4f} (brick_width + clearance)",
            "Output JSON only, no markdown code blocks",
            "All numbers in floating-point format with at least 4 decimal places",
        ]


# ==================== Main Function ====================

def get_descend_prompt(context: Dict[str, Any], 
                       attempt_idx: int = 0, 
                       feedback: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate descend prompt for real robot.
    """
    builder = DescendPromptBuilder(context, attempt_idx, feedback)
    return builder.build()


# ==================== Context Builder Helper ====================

def build_descend_context(
    brick_position: List[float],
    brick_yaw: float,
    tcp_position: Optional[List[float]] = None,
    brick_size: Optional[List[float]] = None,
    gripper_max_opening: float = 0.08,
    gripper_clearance: float = 0.01,
    z_compensation: float = 0.0,
) -> Dict[str, Any]:
    """
    Build context dictionary for descend prompt.
    
    Args:
        brick_position: [x, y, z] brick TOP SURFACE position from hand-eye camera
                       Note: Z is already compensated by DynamicZCompensator
        brick_yaw: brick yaw angle in radians
        tcp_position: [x, y, z] current TCP position (at hover)
        brick_size: [L, W, H] brick dimensions
        gripper_max_opening: maximum gripper opening (meters)
        gripper_clearance: clearance between gripper and brick (meters)
        z_compensation: Z compensation value applied by DynamicZCompensator (for context)
        
    Returns:
        Context dictionary for prompt builder
    """
    # Default brick size
    if brick_size is None:
        brick_size = [0.11, 0.05, 0.025]  # L, W, H
    
    # Default TCP position if not provided
    if tcp_position is None:
        tcp_position = [brick_position[0], brick_position[1], brick_position[2] + 0.15]
    
    return {
        "robot": {
            "dof": 6,
        },
        "gripper": {
            "max_opening": gripper_max_opening,
            "clearance": gripper_clearance,
        },
        "brick": {
            "position": brick_position,
            "size_LWH": brick_size,
            "yaw": brick_yaw,
        },
        "tcp": {
            "x": tcp_position[0],
            "y": tcp_position[1],
            "z": tcp_position[2],
        },
        "z_compensation": z_compensation,
    }