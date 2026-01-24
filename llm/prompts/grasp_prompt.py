"""
Grasp Feedback Prompt for Real Robot

This prompt enables LLM to analyze grasp attempt results.
Z compensation is handled by DynamicZCompensator, not by LLM learning.
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
    "gap_matches_brick": <boolean>,
    "failure_mode": "<string: 'none' | 'too_high' | 'too_low' | 'xy_misaligned' | 'unknown'>"
  },
  "reasoning": "Brief analysis of the grasp attempt (max 80 words)"
}
""".strip()


# ==================== Grasp Feedback Prompt Builder ====================

class GraspFeedbackPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for grasp feedback analysis.
    
    Goal: Analyze sensor data after grasp attempt to determine success.
    Z compensation is pre-applied by DynamicZCompensator, no learning needed.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Grasp Feedback Analysis"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Head camera brick detection (with DynamicZCompensator)",
            "Pre-grasp hover positioning",
            "Hand-eye fine XY alignment",
            "Descend to grasp height (Z compensated)",
            "Close gripper (grasp attempt)",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Analyze grasp success from sensor feedback (current task)",
        ]
    
    def _get_role_name(self) -> str:
        return "Grasp Feedback Analysis Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Analyze gripper effort/current to detect successful grasp",
            "Determine failure mode if grasp failed",
        ]
    
    def _get_specific_task(self) -> str:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap_after_close = gripper.get('gap_after_close', 0.0)
        
        brick = self.brick
        brick_pos = brick.get('position', [0, 0, 0])
        brick_size = brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_width = brick_size[1]
        
        z_compensation = self.context.get('z_compensation', 0.0)
        
        return f"""
Analyze the grasp attempt and determine if it succeeded.

**Sensor Feedback:**
- Gripper effort (motor current): {effort:.3f} A
- Success threshold: effort > {effort_threshold:.1f} A indicates object contact
- Gripper gap after closing: {gap_after_close:.4f} m
- Expected gap if brick grasped: ~{brick_width:.3f} m (brick width)

**Position Data:**
- Target brick position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m
- Z compensation applied: {z_compensation*1000:+.1f} mm (from DynamicZCompensator)

**Analysis Rules:**

1. **SUCCESS** (grasp_success = true):
   - effort > {effort_threshold:.1f} A (contact detected)
   - gap ≈ brick_width ({brick_width:.3f} m) ± 0.015 m

2. **TOO HIGH** (gripper closed above brick):
   - effort ≈ 0 (no contact)
   - gap ≈ 0 (gripper closed on air)

3. **TOO LOW** (gripper hit table):
   - effort > threshold (contact with table)
   - gap ≈ 0 (fingers blocked by table)
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap = gripper.get('gap_after_close', 0.0)
        brick_width = self.brick.get('size_LWH', [0.11, 0.05, 0.025])[1]
        
        return f"""
**Sensor Interpretation:**

```
Effort vs Gap Analysis:
┌────────────────┬─────────────┬─────────────────────────────────┐
│ Effort         │ Gap         │ Interpretation                  │
├────────────────┼─────────────┼─────────────────────────────────┤
│ > {effort_threshold:.1f} A        │ ≈ {brick_width:.3f} m    │ ✓ SUCCESS - brick grasped       │
│ ≈ 0 A          │ ≈ 0 m       │ ✗ TOO HIGH - missed brick       │
│ > {effort_threshold:.1f} A        │ ≈ 0 m       │ ✗ TOO LOW - hit table           │
└────────────────┴─────────────┴─────────────────────────────────┘
```

**Current Readings:**
- Effort: {effort:.3f} A (threshold: {effort_threshold:.1f} A)
- Gap: {gap:.4f} m (expected for brick: {brick_width:.3f} m)
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap = gripper.get('gap_after_close', 0.0)
        brick_width = self.brick.get('size_LWH', [0.11, 0.05, 0.025])[1]
        
        effort_ok = effort > effort_threshold
        gap_ok = abs(gap - brick_width) < 0.015
        
        return [
            {
                "title": "Step 1: Analyze Sensor Feedback",
                "content": f"""
- Measured effort: {effort:.3f} A (threshold: {effort_threshold:.1f} A) → {"Contact" if effort_ok else "No contact"}
- Gap after close: {gap:.4f} m (expected: {brick_width:.3f} m) → {"Matches brick" if gap_ok else "Mismatch"}
""".strip()
            },
            {
                "title": "Step 2: Determine Outcome",
                "content": f"""
- SUCCESS: effort > threshold AND gap ≈ brick_width
- TOO_HIGH: effort ≈ 0 AND gap ≈ 0
- TOO_LOW: effort > threshold AND gap ≈ 0
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
    z_compensation: float = 0.0,
    attempt_number: int = 1,
    max_attempts: int = 1,
    # Deprecated parameters (kept for compatibility, ignored)
    compensation_history: Optional[List[float]] = None,
    current_compensation: float = 0.0,
    applied_compensation: float = 0.0,
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
        z_compensation: Z compensation from DynamicZCompensator
        attempt_number: current attempt number (unused, kept for compatibility)
        max_attempts: maximum retry attempts (unused, kept for compatibility)
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
        "z_compensation": z_compensation,
    }