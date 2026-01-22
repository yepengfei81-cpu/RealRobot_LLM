"""
Lift Prompt for Real Robot

This prompt guides the LLM to plan the lift trajectory after successful grasp.
The robot TCP lifts the brick to a safe height before moving to placement position.
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

LIFT_REPLY_TEMPLATE = """
{
  "target_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "yaw": <float yaw in radians>
  },
  "lift_params": {
    "lift_height": <float lift height from current position in meters>,
    "target_z": <float absolute target Z coordinate in meters>
  },
  "reasoning": "Brief explanation of lift calculation (max 50 words)",
  "verification": {
    "xy_unchanged": true,
    "z_above_grasp": true,
    "sufficient_clearance": true,
    "yaw_unchanged": true
  }
}
""".strip()


# ==================== Lift Prompt Builder ====================

class LiftPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for lift phase.
    
    Goal: Lift the grasped brick to a safe height while maintaining
    XY position and orientation.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Lift Brick After Grasp"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Robot initialization and calibration",
            "Head camera brick detection and localization",
            "Pre-grasp position planning completed",
            "TCP moved to hover position above brick",
            "Hand-eye camera fine positioning completed",
            "Descended to grasp position",
            "Gripper closed and brick grasped successfully",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Lift brick to safe height (current task)",
            "Move to placement position",
            "Descend to place brick",
            "Open gripper and release brick",
            "Retreat to safe position",
        ]
    
    def _get_role_name(self) -> str:
        return "Lift Control Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Calculate safe lift height above grasp position",
            "Maintain XY alignment during vertical lift",
            "Ensure sufficient clearance from obstacles",
            "Preserve gripper orientation for stable transport",
        ]
    
    def _get_specific_task(self) -> str:
        tcp = self.context.get('tcp', {})
        tcp_x = tcp.get('x', 0)
        tcp_y = tcp.get('y', 0)
        tcp_z = tcp.get('z', 0)
        tcp_yaw = tcp.get('yaw', 0)
        
        lift_config = self.context.get('lift', {})
        lift_height = lift_config.get('lift_height', 0.10)
        
        target_z = tcp_z + lift_height
        
        return f"""
Calculate the target TCP pose for lifting the grasped brick.

**Current State (after successful grasp):**
- TCP position: [{tcp_x:.4f}, {tcp_y:.4f}, {tcp_z:.4f}] m
- TCP yaw: {tcp_yaw:.4f} rad ({math.degrees(tcp_yaw):.1f}°)
- Brick is grasped and attached to gripper
- Lift height required: {lift_height:.3f} m

**Lift Calculation:**
- Target Z = Current Z + Lift Height
- Target Z = {tcp_z:.4f} + {lift_height:.3f} = {target_z:.4f} m

**IMPORTANT - Pure Vertical Lift:**
- X and Y coordinates must remain UNCHANGED
- Only Z coordinate changes (increases)
- Yaw angle must remain UNCHANGED
- This ensures stable brick transport without swaying

**Required Output:**
- TCP X: {tcp_x:.4f} m (unchanged)
- TCP Y: {tcp_y:.4f} m (unchanged)
- TCP Z: {target_z:.4f} m (lifted)
- TCP yaw: {tcp_yaw:.4f} rad (unchanged)
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        tcp = self.context.get('tcp', {})
        tcp_z = tcp.get('z', 0)
        
        lift_config = self.context.get('lift', {})
        lift_height = lift_config.get('lift_height', 0.10)
        min_clearance = lift_config.get('min_clearance', 0.05)
        
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_height = brick_size[2]
        
        return f"""
**Lift Strategy:**
- Vertical lift only: Change Z, keep X/Y/yaw constant
- This prevents brick from swaying or slipping during transport
- Lift height ({lift_height:.3f}m) provides clearance for obstacle avoidance

**Safety Considerations:**
- Current grasp Z: {tcp_z:.4f} m
- Lift height: {lift_height:.3f} m
- Target Z after lift: {tcp_z + lift_height:.4f} m
- Brick height: {brick_height:.3f} m
- Minimum clearance from ground: {min_clearance:.3f} m

**Stability During Lift:**
- Maintaining constant XY prevents pendulum motion
- Maintaining constant yaw prevents torsional stress on grasp
- Smooth vertical motion minimizes inertial forces
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        tcp = self.context.get('tcp', {})
        tcp_x = tcp.get('x', 0)
        tcp_y = tcp.get('y', 0)
        tcp_z = tcp.get('z', 0)
        tcp_yaw = tcp.get('yaw', 0)
        
        lift_config = self.context.get('lift', {})
        lift_height = lift_config.get('lift_height', 0.10)
        
        target_z = tcp_z + lift_height
        
        return [
            {
                "title": "Extract Current TCP State",
                "content": f"""
- Current TCP X: {tcp_x:.4f} m
- Current TCP Y: {tcp_y:.4f} m
- Current TCP Z: {tcp_z:.4f} m (grasp height)
- Current TCP yaw: {tcp_yaw:.4f} rad
""".strip()
            },
            {
                "title": "Calculate Lift Parameters",
                "content": f"""
- Lift height: {lift_height:.3f} m
- Target Z = Current Z + Lift Height
- Target Z = {tcp_z:.4f} + {lift_height:.3f} = {target_z:.4f} m
""".strip()
            },
            {
                "title": "Determine Target Pose",
                "content": f"""
- Target X = Current X = {tcp_x:.4f} m (unchanged)
- Target Y = Current Y = {tcp_y:.4f} m (unchanged)
- Target Z = {target_z:.4f} m (lifted)
- Target yaw = Current yaw = {tcp_yaw:.4f} rad (unchanged)
""".strip()
            },
            {
                "title": "Verify Lift Safety",
                "content": f"""
- XY unchanged: ✓ (stable vertical lift)
- Z increased by {lift_height:.3f} m: ✓ (sufficient clearance)
- Yaw unchanged: ✓ (no torsional stress)
- Final pose: [{tcp_x:.4f}, {tcp_y:.4f}, {target_z:.4f}], yaw={tcp_yaw:.4f}
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return LIFT_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        tcp = self.context.get('tcp', {})
        tcp_x = tcp.get('x', 0)
        tcp_y = tcp.get('y', 0)
        tcp_z = tcp.get('z', 0)
        tcp_yaw = tcp.get('yaw', 0)
        
        lift_config = self.context.get('lift', {})
        lift_height = lift_config.get('lift_height', 0.10)
        
        target_z = tcp_z + lift_height
        
        return [
            f"xyz[0] must equal {tcp_x:.4f} (unchanged X)",
            f"xyz[1] must equal {tcp_y:.4f} (unchanged Y)",
            f"xyz[2] must equal {target_z:.4f} (current Z + lift height)",
            f"yaw must equal {tcp_yaw:.4f} rad (unchanged)",
            f"lift_params.lift_height must equal {lift_height:.3f}",
            f"lift_params.target_z must equal {target_z:.4f}",
            "Output JSON only, no markdown code blocks",
            "All numbers in floating-point format with at least 4 decimal places",
        ]


# ==================== Main Function ====================

def get_lift_prompt(context: Dict[str, Any], 
                    attempt_idx: int = 0, 
                    feedback: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate lift prompt for real robot.
    
    Args:
        context: Context dictionary with tcp, lift, brick info
        attempt_idx: Retry attempt number
        feedback: Error feedback from previous attempt
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = LiftPromptBuilder(context, attempt_idx, feedback)
    return builder.build()


# ==================== Context Builder Helper ====================

def build_lift_context(
    tcp_position: List[float],
    tcp_yaw: float,
    brick_position: Optional[List[float]] = None,
    brick_yaw: Optional[float] = None,
    brick_size: Optional[List[float]] = None,
    lift_height: float = 0.10,
    min_clearance: float = 0.05,
) -> Dict[str, Any]:
    """
    Helper function to build context dictionary for lift prompt.
    
    Args:
        tcp_position: [x, y, z] current TCP position (at grasp height)
        tcp_yaw: current TCP yaw angle in radians
        brick_position: [x, y, z] brick position (optional, for reference)
        brick_yaw: brick yaw angle (optional, for reference)
        brick_size: [L, W, H] brick dimensions (optional)
        lift_height: height to lift from current position (meters)
        min_clearance: minimum clearance from obstacles (meters)
        
    Returns:
        Context dictionary ready for get_lift_prompt()
    """
    if brick_size is None:
        brick_size = [0.11, 0.05, 0.025]
    
    if brick_position is None:
        brick_position = tcp_position.copy()
    
    if brick_yaw is None:
        brick_yaw = tcp_yaw
    
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
            "position": brick_position,
            "size_LWH": brick_size,
            "yaw": brick_yaw,
            "grasped": True,
        },
        "lift": {
            "lift_height": lift_height,
            "min_clearance": min_clearance,
        },
        "constraints": {
            "maintain_xy": True,
            "maintain_yaw": True,
            "vertical_lift_only": True,
        },
    }