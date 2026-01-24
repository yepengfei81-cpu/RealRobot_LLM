"""
Pre-Grasp Prompt for Real Robot

This prompt guides the LLM to plan a safe pre-grasp pose directly above the brick.
The robot TCP should be positioned above the brick center with proper orientation.
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder, compute_brick_top_z


# ==================== Reply Template ====================

PRE_GRASP_REPLY_TEMPLATE = """
{
  "target_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "yaw": <float yaw in radians>
  },
  "reasoning": "Brief explanation of the calculation (max 50 words)",
  "verification": {
    "xy_aligned_with_brick": true,
    "z_above_brick_with_clearance": true,
    "yaw_aligned_with_brick": true
  }
}
""".strip()


# ==================== Pre-Grasp Prompt Builder ====================

class PreGraspPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for pre-grasp phase.
    
    Goal: Plan TCP pose directly above brick center at hover height.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Pre-Grasp Position Planning"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Robot initialization and calibration",
            "Head camera brick detection and localization",
            "Brick position and orientation estimation",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Pre-grasp position planning (current task)",
            "Move to pre-grasp position",
            "Hand-eye camera fine positioning",
            "Descend and grasp",
            "Lift and place",
        ]
    
    def _get_role_name(self) -> str:
        return "Pre-Grasp Position Planning Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Analyze brick position and orientation from sensor data",
            "Calculate safe pre-grasp position directly above brick",
            "Ensure proper TCP orientation for top-down approach",
            "Verify the planned pose is within robot workspace",
        ]
    
    def _get_specific_task(self) -> str:
        brick_pos = self.brick.get('position', [0, 0, 0])
        brick_yaw = self.brick.get('yaw', 0)
        hover_height = self.constraints.get('hover_height', 0.15)
        
        return f"""
Calculate the pre-grasp pose for the robot TCP to position directly above the brick.

**Given:**
- Brick center position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m
- Brick yaw angle: {brick_yaw:.4f} rad ({math.degrees(brick_yaw):.1f}°)
- Required hover height above brick: {hover_height:.3f} m

**Required:**
- TCP X, Y should align with brick center X, Y
- TCP Z should be at: brick_top_z + hover_height
- TCP yaw should align with brick yaw for proper grasp approach
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        hover_height = self.constraints.get('hover_height', 0.15)
        brick_size = self.brick.get('size_LWH', [0.20, 0.095, 0.06])
        
        return f"""
**Pre-Grasp Position Calculation:**
- Hover height: {hover_height:.3f} m above brick top surface
- Brick height: {brick_size[2]:.3f} m
- TCP target Z = brick_center_z + (brick_height / 2) + hover_height

**Orientation for Top-Down Grasp:**
- For top-down approach, the gripper should point downward
- TCP yaw should match brick yaw to align gripper with brick orientation
- This ensures the gripper fingers will be parallel to brick width direction

**Real Robot Considerations:**
- No simulation fallback - output must be accurate
- Position will be directly sent to real robot
- Ensure the pose is reachable and safe
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        brick_pos = self.brick.get('position', [0, 0, 0])
        brick_size = self.brick.get('size_LWH', [0.20, 0.095, 0.06])
        brick_yaw = self.brick.get('yaw', 0)
        hover_height = self.constraints.get('hover_height', 0.15)
        
        brick_top_z = brick_pos[2] + brick_size[2] / 2
        target_z = brick_top_z + hover_height
        
        return [
            {
                "title": "Extract Brick Information",
                "content": f"""
- Brick center: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}) m
- Brick size LWH: ({brick_size[0]:.3f}, {brick_size[1]:.3f}, {brick_size[2]:.3f}) m
- Brick yaw: {brick_yaw:.4f} rad
""".strip()
            },
            {
                "title": "Calculate Target Position",
                "content": f"""
- Target X = brick_center_x = {brick_pos[0]:.4f} m
- Target Y = brick_center_y = {brick_pos[1]:.4f} m
- Brick top Z = brick_center_z + height/2 = {brick_pos[2]:.4f} + {brick_size[2]/2:.4f} = {brick_top_z:.4f} m
- Target Z = brick_top_z + hover_height = {brick_top_z:.4f} + {hover_height:.3f} = {target_z:.4f} m
""".strip()
            },
            {
                "title": "Determine Target Orientation",
                "content": f"""
- For top-down grasp, TCP should point downward
- TCP yaw should align with brick yaw: {brick_yaw:.4f} rad
- This aligns gripper opening direction with brick width
""".strip()
            },
            {
                "title": "Verify and Output",
                "content": f"""
- Final target position: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {target_z:.4f}) m
- Final target yaw: {brick_yaw:.4f} rad
- Verify: X,Y aligned with brick center ✓
- Verify: Z is {hover_height:.3f}m above brick top ✓
- Verify: Yaw aligned with brick ✓
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return PRE_GRASP_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        brick_pos = self.brick.get('position', [0, 0, 0])
        brick_size = self.brick.get('size_LWH', [0.20, 0.095, 0.06])
        brick_yaw = self.brick.get('yaw', 0)
        hover_height = self.constraints.get('hover_height', 0.15)
        
        brick_top_z = brick_pos[2] + brick_size[2] / 2
        target_z = brick_top_z + hover_height
        
        return [
            f"xyz[0] (X) must equal brick center X: {brick_pos[0]:.4f}",
            f"xyz[1] (Y) must equal brick center Y: {brick_pos[1]:.4f}",
            f"xyz[2] (Z) must equal brick_top_z + hover_height: {target_z:.4f}",
            f"yaw must equal brick yaw: {brick_yaw:.4f} rad",
            "Output JSON only, no markdown code blocks",
            "All numbers in floating-point format with at least 4 decimal places",
        ]


# ==================== Main Function ====================

def get_pre_grasp_prompt(context: Dict[str, Any], 
                         attempt_idx: int = 0, 
                         feedback: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate pre-grasp prompt for real robot.
    """
    builder = PreGraspPromptBuilder(context, attempt_idx, feedback)
    return builder.build()

def build_pre_grasp_context(
    brick_position: List[float],
    brick_yaw: float,
    tcp_position: Optional[List[float]] = None,
    tcp_orientation: Optional[List[float]] = None,
    brick_size: Optional[List[float]] = None,
    hover_height: float = 0.15,
    gripper_max_opening: float = 0.08,
    z_compensation: float = 0.0,
) -> Dict[str, Any]:
    """
    Build context dictionary for pre-grasp prompt.
    
    Args:
        brick_position: [x, y, z] brick center position in base_link frame (meters)
                       Note: Z is already compensated by DynamicZCompensator
        brick_yaw: brick yaw angle in radians
        tcp_position: [x, y, z] current TCP position (optional)
        tcp_orientation: [roll, pitch, yaw] current TCP orientation (optional)
        brick_size: [L, W, H] brick dimensions (optional, uses default)
        hover_height: height above brick for hover position (meters)
        gripper_max_opening: maximum gripper opening (meters)
        z_compensation: Z compensation value applied by DynamicZCompensator (for context)
        
    Returns:
        Context dictionary for prompt builder
    """
    # Default brick size
    if brick_size is None:
        brick_size = [0.11, 0.05, 0.025]  # L, W, H
    
    # Default TCP position if not provided
    if tcp_position is None:
        tcp_position = [0.4, 0.0, 1.0]
    
    # Default TCP orientation if not provided  
    if tcp_orientation is None:
        tcp_orientation = [0.0, 0.0, 0.0]
    
    return {
        "robot": {
            "dof": 6,
        },
        "gripper": {
            "max_opening": gripper_max_opening,
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
            "roll": tcp_orientation[0],
            "pitch": tcp_orientation[1],
            "yaw": tcp_orientation[2],
        },
        "hover_height": hover_height,
        "z_compensation": z_compensation,
    }