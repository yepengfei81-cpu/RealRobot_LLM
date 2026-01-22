"""
Grasp Feedback Prompt for Real Robot

This prompt enables LLM to analyze grasp attempt results and suggest corrections.
Uses sensor feedback (gripper effort, position) to determine grasp success
and compute adjustment if needed.
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

GRASP_FEEDBACK_REPLY_TEMPLATE = """
{
  "grasp_success": <boolean>,
  "confidence": <float 0.0-1.0>,
  "analysis": {
    "effort_indicates_contact": <boolean>,
    "position_at_brick_center": <boolean>,
    "failure_mode": "<string: 'none' | 'too_high' | 'too_low' | 'xy_misaligned' | 'unknown'>"
  },
  "adjustment": {
    "needed": <boolean>,
    "delta_z": <float in meters, positive=up, negative=down>,
    "reason": "<string explaining the adjustment>"
  },
  "reasoning": "Brief analysis of the grasp attempt (max 80 words)"
}
""".strip()


# ==================== Grasp Feedback Prompt Builder ====================

class GraspFeedbackPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for grasp feedback analysis.
    
    Goal: Analyze sensor data after grasp attempt to determine success
    and suggest Z-axis corrections if needed.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Grasp Feedback Analysis"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Head camera brick detection",
            "Pre-grasp hover positioning",
            "Hand-eye fine XY alignment",
            "Descend to estimated grasp height",
            "Close gripper (grasp attempt)",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Analyze grasp success from sensor feedback (current task)",
            "If failed: adjust Z and retry",
            "If success: lift and place brick",
        ]
    
    def _get_role_name(self) -> str:
        return "Grasp Feedback Analysis Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Analyze gripper effort/current to detect successful grasp",
            "Determine failure mode (too high, too low, misaligned)",
            "Compute Z adjustment for retry if needed",
            "Ensure safe operation within retry limits",
        ]
    
    def _get_specific_task(self) -> str:
        # Extract sensor data
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap_after_close = gripper.get('gap_after_close', 0.0)
        
        tcp = self.context.get('tcp', {})
        tcp_z = tcp.get('z', 0.0)
        
        brick = self.brick
        brick_pos = brick.get('position', [0, 0, 0])
        brick_size = brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_width = brick_size[1]
        brick_height = brick_size[2]
        
        attempt = self.context.get('attempt', {})
        attempt_num = attempt.get('number', 1)
        max_attempts = attempt.get('max', 3)
        
        return f"""
Analyze the grasp attempt and determine if adjustment is needed.

**Sensor Feedback After Grasp Attempt #{attempt_num}/{max_attempts}:**
- Gripper effort (motor current): {effort:.3f} A
- Success threshold: effort > {effort_threshold:.1f} A indicates object contact
- Gripper gap after closing: {gap_after_close:.4f} m
- Expected gap if brick grasped: ~{brick_width:.3f} m (brick width)

**Position Data:**
- Current TCP Z: {tcp_z:.4f} m
- Target brick position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m
- Brick dimensions (L×W×H): [{brick_size[0]:.3f}, {brick_size[1]:.3f}, {brick_size[2]:.3f}] m

**Analysis Rules:**

1. **SUCCESS** (grasp_success = true):
   - effort > {effort_threshold:.1f} A (contact detected)
   - gap ≈ brick_width ({brick_width:.3f} m) ± 0.015 m

2. **TOO HIGH** (gripper closed above brick):
   - effort ≈ 0 (no contact)
   - gap ≈ 0 (gripper closed on air)
   - Adjustment: delta_z = -{brick_height:.3f} to -{brick_height*1.5:.3f} m (descend by brick height)

3. **TOO LOW** (gripper hit table):
   - effort > threshold (contact with table)
   - gap ≈ 0 (fingers blocked by table)
   - Adjustment: delta_z = +0.01 to +0.02 m (raise slightly)

4. **PARTIAL GRASP** (brick edge or unstable):
   - effort > threshold but lower than expected
   - gap slightly different from brick_width
   - May need XY adjustment (flag for re-detection)

**Your Task:**
Analyze the sensor data and determine:
1. Is the grasp successful?
2. If not, what is the failure mode?
3. What Z adjustment is needed for retry?
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap = gripper.get('gap_after_close', 0.0)
        
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_width = brick_size[1]
        brick_height = brick_size[2]
        
        return f"""
**Sensor Interpretation Guide:**

```
Effort vs Gap Analysis:
┌────────────────┬─────────────┬─────────────────────────────────┐
│ Effort         │ Gap         │ Interpretation                  │
├────────────────┼─────────────┼─────────────────────────────────┤
│ > {effort_threshold:.1f} A        │ ≈ {brick_width:.3f} m    │ ✓ SUCCESS - brick grasped       │
│ ≈ 0 A          │ ≈ 0 m       │ ✗ TOO HIGH - missed brick       │
│ > {effort_threshold:.1f} A        │ ≈ 0 m       │ ✗ TOO LOW - hit table           │
│ > {effort_threshold:.1f} A        │ < {brick_width:.3f} m    │ ? PARTIAL - edge grasp          │
└────────────────┴─────────────┴─────────────────────────────────┘
```

**Current Readings:**
- Effort: {effort:.3f} A (threshold: {effort_threshold:.1f} A)
- Gap: {gap:.4f} m (expected for brick: {brick_width:.3f} m)

**Adjustment Guidelines:**
- If TOO HIGH: descend by {brick_height:.3f}m ~ {brick_height*1.5:.3f}m (brick height or more)
- If TOO LOW: raise by 0.01m ~ 0.02m (small increment)
- Max single adjustment: ±0.05m for safety
- Do not exceed {self.context.get('attempt', {}).get('max', 3)} total attempts
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap = gripper.get('gap_after_close', 0.0)
        brick_width = self.brick.get('size_LWH', [0.11, 0.05, 0.025])[1]
        brick_height = self.brick.get('size_LWH', [0.11, 0.05, 0.025])[2]
        
        effort_ok = effort > effort_threshold
        gap_ok = abs(gap - brick_width) < 0.015
        
        return [
            {
                "title": "Step 1: Analyze Effort",
                "content": f"""
- Measured effort: {effort:.3f} A
- Threshold: {effort_threshold:.1f} A
- Contact detected: {"YES" if effort_ok else "NO"}
""".strip()
            },
            {
                "title": "Step 2: Analyze Gap",
                "content": f"""
- Gap after close: {gap:.4f} m
- Expected (brick width): {brick_width:.3f} m
- Difference: {abs(gap - brick_width):.4f} m
- Gap matches brick: {"YES" if gap_ok else "NO"}
""".strip()
            },
            {
                "title": "Step 3: Determine Outcome",
                "content": f"""
- Effort OK: {"✓" if effort_ok else "✗"}
- Gap OK: {"✓" if gap_ok else "✗"}
- If both OK → SUCCESS
- If effort=0 and gap=0 → TOO HIGH (missed)
- If effort>0 and gap=0 → TOO LOW (table contact)
""".strip()
            },
            {
                "title": "Step 4: Compute Adjustment (if needed)",
                "content": f"""
- If TOO HIGH: delta_z = -{brick_height:.3f} m (descend)
- If TOO LOW: delta_z = +0.015 m (raise)
- If SUCCESS: delta_z = 0, no adjustment needed
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return GRASP_FEEDBACK_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        return [
            "grasp_success: true only if effort > threshold AND gap ≈ brick_width",
            "confidence: 0.0-1.0, higher if sensor readings are clear",
            "failure_mode: 'none', 'too_high', 'too_low', 'xy_misaligned', or 'unknown'",
            "delta_z: positive = move up, negative = move down, range [-0.05, 0.05] m",
            "adjustment.needed must be false if grasp_success is true",
            "Output JSON only, no markdown code blocks",
        ]


# ==================== Main Function ====================

def get_grasp_feedback_prompt(context: Dict[str, Any], 
                              attempt_idx: int = 0, 
                              feedback: Optional[str] = None) -> Tuple[str, str]:
    """Generate grasp feedback analysis prompt."""
    builder = GraspFeedbackPromptBuilder(context, attempt_idx, feedback)
    return builder.build()


# ==================== Context Builder Helper ====================

def build_grasp_feedback_context(
    brick_position: List[float],
    brick_yaw: float,
    brick_size: List[float],
    tcp_position: List[float],
    gripper_effort: float,
    gripper_gap_after_close: float,
    effort_threshold: float = 2.0,
    attempt_number: int = 1,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Build context for grasp feedback prompt.
    
    Args:
        brick_position: [x, y, z] target grasp position
        brick_yaw: brick yaw angle
        brick_size: [L, W, H] brick dimensions
        tcp_position: [x, y, z] actual TCP position after grasp attempt
        gripper_effort: gripper motor current after closing (A)
        gripper_gap_after_close: gripper gap after closing (m)
        effort_threshold: threshold for detecting contact (default 2.0 A)
        attempt_number: current attempt number
        max_attempts: maximum retry attempts
    """
    return {
        "robot": {"dof": 6},
        "gripper": {
            "effort": gripper_effort,
            "effort_threshold": effort_threshold,
            "gap_after_close": gripper_gap_after_close,
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
        "attempt": {
            "number": attempt_number,
            "max": max_attempts,
        },
    }