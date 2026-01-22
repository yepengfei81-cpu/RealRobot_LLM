"""
Move to Place Prompt for Real Robot

This prompt guides the LLM to plan the horizontal movement from lift position
to above the placement target, while adjusting yaw to match target orientation.

Key requirements:
- Maintain constant Z height (stay at lift height)
- Move XY to target placement position
- Adjust yaw to target orientation
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

MOVE_TO_PLACE_REPLY_TEMPLATE = """
{
  "target_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "yaw": <float yaw in radians>
  },
  "motion_params": {
    "start_position": [<float x>, <float y>, <float z>],
    "end_position": [<float x>, <float y>, <float z>],
    "xy_distance": <float horizontal distance in meters>,
    "yaw_change": <float yaw change in radians>,
    "motion_type": "horizontal_transfer"
  },
  "reasoning": "Brief explanation of movement calculation (max 50 words)",
  "verification": {
    "z_unchanged": true,
    "xy_at_target": true,
    "yaw_at_target": true,
    "path_collision_free": true
  }
}
""".strip()


# ==================== Move to Place Prompt Builder ====================

class MoveToPlacePromptBuilder(BasePromptBuilder):
    """
    Prompt builder for move-to-place phase.
    
    Goal: Move horizontally from lift position to above placement target,
    adjusting yaw to match target orientation.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Move to Placement Position"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Robot initialization and calibration",
            "Head camera brick detection and localization",
            "Pre-grasp position planning completed",
            "TCP moved to hover position above brick",
            "Hand-eye camera fine positioning completed",
            "Descended to grasp position",
            "Gripper closed and brick grasped successfully",
            "Brick lifted to safe transport height",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Move horizontally to above placement target (current task)",
            "Descend to place brick",
            "Open gripper and release brick",
            "Retreat to safe position",
        ]
    
    def _get_role_name(self) -> str:
        return "Placement Transfer Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Plan horizontal movement from lift position to placement target",
            "Maintain safe Z height during horizontal transfer",
            "Adjust yaw angle to match placement target orientation",
            "Ensure collision-free path during transfer",
        ]
    
    def _get_specific_task(self) -> str:
        tcp = self.context.get('tcp', {})
        tcp_x = tcp.get('x', 0)
        tcp_y = tcp.get('y', 0)
        tcp_z = tcp.get('z', 0)
        tcp_yaw = tcp.get('yaw', 0)
        
        place_config = self.context.get('place', {})
        target_x = place_config.get('target_x', 0.54)
        target_y = place_config.get('target_y', -0.04)
        target_yaw = place_config.get('target_yaw', 0.0)
        
        # Calculate horizontal distance
        dx = target_x - tcp_x
        dy = target_y - tcp_y
        xy_distance = math.sqrt(dx**2 + dy**2)
        yaw_change = target_yaw - tcp_yaw
        
        return f"""
Calculate the target TCP pose for moving to above the placement position.

**Current State (after lift):**
- TCP position: [{tcp_x:.4f}, {tcp_y:.4f}, {tcp_z:.4f}] m
- TCP yaw: {tcp_yaw:.4f} rad ({math.degrees(tcp_yaw):.1f}°)
- Brick is grasped and lifted to safe height

**Placement Target:**
- Target X: {target_x:.4f} m
- Target Y: {target_y:.4f} m
- Target yaw: {target_yaw:.4f} rad ({math.degrees(target_yaw):.1f}°)

**Movement Calculation:**
- Horizontal distance: {xy_distance:.4f} m
- Yaw change: {yaw_change:.4f} rad ({math.degrees(yaw_change):.1f}°)

**IMPORTANT - Horizontal Transfer with Yaw Adjustment:**
- Z coordinate must remain UNCHANGED (stay at lift height)
- X coordinate changes from {tcp_x:.4f} to {target_x:.4f}
- Y coordinate changes from {tcp_y:.4f} to {target_y:.4f}
- Yaw angle changes from {tcp_yaw:.4f} to {target_yaw:.4f} rad

**Required Output:**
- TCP X: {target_x:.4f} m (target placement X)
- TCP Y: {target_y:.4f} m (target placement Y)
- TCP Z: {tcp_z:.4f} m (unchanged, stay at lift height)
- TCP yaw: {target_yaw:.4f} rad (target placement orientation)
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        tcp = self.context.get('tcp', {})
        tcp_x = tcp.get('x', 0)
        tcp_y = tcp.get('y', 0)
        tcp_z = tcp.get('z', 0)
        tcp_yaw = tcp.get('yaw', 0)
        
        place_config = self.context.get('place', {})
        target_x = place_config.get('target_x', 0.54)
        target_y = place_config.get('target_y', -0.04)
        target_yaw = place_config.get('target_yaw', 0.0)
        
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        
        return f"""
**Horizontal Transfer Strategy:**
- Move at constant height to avoid obstacles
- Current Z ({tcp_z:.4f}m) is safe transport height
- XY movement covers horizontal distance to target

**Yaw Adjustment:**
- Current yaw: {tcp_yaw:.4f} rad ({math.degrees(tcp_yaw):.1f}°)
- Target yaw: {target_yaw:.4f} rad ({math.degrees(target_yaw):.1f}°)
- Yaw adjusted during horizontal transfer for efficiency

**Safety Considerations:**
- Maintain Z height to clear obstacles on table/ground
- Smooth yaw transition prevents brick from swinging
- Brick dimensions: L={brick_size[0]:.3f}, W={brick_size[1]:.3f}, H={brick_size[2]:.3f} m

**Motion Planning:**
- This is a combined XY translation + yaw rotation
- Z remains constant throughout
- Robot controller handles path interpolation
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        tcp = self.context.get('tcp', {})
        tcp_x = tcp.get('x', 0)
        tcp_y = tcp.get('y', 0)
        tcp_z = tcp.get('z', 0)
        tcp_yaw = tcp.get('yaw', 0)
        
        place_config = self.context.get('place', {})
        target_x = place_config.get('target_x', 0.54)
        target_y = place_config.get('target_y', -0.04)
        target_yaw = place_config.get('target_yaw', 0.0)
        
        dx = target_x - tcp_x
        dy = target_y - tcp_y
        xy_distance = math.sqrt(dx**2 + dy**2)
        
        return [
            {
                "title": "Extract Current and Target State",
                "content": f"""
Current TCP:
- Position: [{tcp_x:.4f}, {tcp_y:.4f}, {tcp_z:.4f}] m
- Yaw: {tcp_yaw:.4f} rad

Target placement:
- Position: [{target_x:.4f}, {target_y:.4f}] m (XY only)
- Yaw: {target_yaw:.4f} rad
""".strip()
            },
            {
                "title": "Calculate Movement Parameters",
                "content": f"""
- ΔX = {target_x:.4f} - {tcp_x:.4f} = {dx:.4f} m
- ΔY = {target_y:.4f} - {tcp_y:.4f} = {dy:.4f} m
- Horizontal distance = √(ΔX² + ΔY²) = {xy_distance:.4f} m
- Yaw change = {target_yaw:.4f} - {tcp_yaw:.4f} = {target_yaw - tcp_yaw:.4f} rad
""".strip()
            },
            {
                "title": "Determine Target Pose",
                "content": f"""
- Target X = {target_x:.4f} m
- Target Y = {target_y:.4f} m
- Target Z = {tcp_z:.4f} m (unchanged from lift)
- Target yaw = {target_yaw:.4f} rad
""".strip()
            },
            {
                "title": "Verify Movement Safety",
                "content": f"""
- Z unchanged: ✓ (maintains safe height)
- XY at target: ✓ (above placement position)
- Yaw at target: ✓ (ready for placement)
- Final pose: [{target_x:.4f}, {target_y:.4f}, {tcp_z:.4f}], yaw={target_yaw:.4f}
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return MOVE_TO_PLACE_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        tcp = self.context.get('tcp', {})
        tcp_z = tcp.get('z', 0)
        
        place_config = self.context.get('place', {})
        target_x = place_config.get('target_x', 0.54)
        target_y = place_config.get('target_y', -0.04)
        target_yaw = place_config.get('target_yaw', 0.0)
        
        return [
            f"xyz[0] must equal {target_x:.4f} (target X)",
            f"xyz[1] must equal {target_y:.4f} (target Y)",
            f"xyz[2] must equal {tcp_z:.4f} (unchanged Z from lift)",
            f"yaw must equal {target_yaw:.4f} rad (target orientation)",
            "motion_type must be 'horizontal_transfer'",
            "Output JSON only, no markdown code blocks",
            "All numbers in floating-point format with at least 4 decimal places",
        ]


# ==================== Main Function ====================

def get_move_to_place_prompt(
    context: Dict[str, Any], 
    attempt_idx: int = 0, 
    feedback: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate move-to-place prompt for real robot.
    
    Args:
        context: Context dictionary with tcp, place info
        attempt_idx: Retry attempt number
        feedback: Error feedback from previous attempt
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = MoveToPlacePromptBuilder(context, attempt_idx, feedback)
    return builder.build()


# ==================== Context Builder Helper ====================

def build_move_to_place_context(
    tcp_position: List[float],
    tcp_yaw: float,
    target_position: List[float],
    target_yaw: float,
    brick_size: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Helper function to build context dictionary for move-to-place prompt.
    
    Args:
        tcp_position: [x, y, z] current TCP position (after lift)
        tcp_yaw: current TCP yaw angle in radians
        target_position: [x, y, z] target placement position
        target_yaw: target yaw angle for placement in radians
        brick_size: [L, W, H] brick dimensions (optional)
        
    Returns:
        Context dictionary ready for get_move_to_place_prompt()
    """
    if brick_size is None:
        brick_size = [0.11, 0.05, 0.025]
    
    return {
        "robot": {
            "dof": 6,
        },
        "gripper": {
            "state": "closed",
            "holding_brick": True,
        },
        "tcp": {
            "x": tcp_position[0],
            "y": tcp_position[1],
            "z": tcp_position[2],
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": tcp_yaw,
        },
        "brick": {
            "size_LWH": brick_size,
            "grasped": True,
        },
        "place": {
            "target_x": target_position[0],
            "target_y": target_position[1],
            "target_z": target_position[2],
            "target_yaw": target_yaw,
        },
        "constraints": {
            "maintain_z": True,
            "horizontal_transfer": True,
        },
    }