"""
Grasp Feedback Prompt for Real Robot

This prompt enables LLM to analyze grasp attempt results and suggest corrections.
Uses sensor feedback (gripper effort, position) to determine grasp success
and compute adjustment if needed.

Now includes historical compensation data for adaptive learning.
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
  "learning": {
    "recommended_compensation": <float in meters, suggested offset for future grasps>,
    "compensation_confidence": <float 0.0-1.0>
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
    
    Now includes historical compensation learning.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Grasp Feedback Analysis with Adaptive Learning"
    
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
            "Learn from historical compensation data",
            "If failed: compute adjusted Z using learned compensation",
            "If success: update recommended compensation for future",
        ]
    
    def _get_role_name(self) -> str:
        return "Adaptive Grasp Feedback Analysis Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Analyze gripper effort/current to detect successful grasp",
            "Determine failure mode (too high, too low, misaligned)",
            "Learn from historical compensation data to improve accuracy",
            "Compute Z adjustment that incorporates learned patterns",
            "Recommend compensation value for future grasp attempts",
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
        
        # Historical compensation data
        compensation = self.context.get('compensation', {})
        history = compensation.get('history', [])
        current_compensation = compensation.get('current', 0.0)
        applied_this_grasp = compensation.get('applied_this_grasp', 0.0)
        
        # Format history
        history_str = ""
        if history:
            history_str = "\n**Historical Compensation Data (from previous successful grasps):**\n"
            for i, h in enumerate(history[-5:]):  # Show last 5
                history_str += f"  - Grasp {i+1}: adjustment = {h*1000:+.1f} mm\n"
            history_str += f"  - Current learned compensation: {current_compensation*1000:+.1f} mm\n"
            history_str += f"  - Applied to this grasp: {applied_this_grasp*1000:+.1f} mm\n"
        else:
            history_str = "\n**No historical compensation data yet (first grasp attempt).**\n"
        
        return f"""
Analyze the grasp attempt and determine if adjustment is needed.
Use historical compensation data to make smarter adjustments.

**Sensor Feedback After Grasp Attempt #{attempt_num}/{max_attempts}:**
- Gripper effort (motor current): {effort:.3f} A
- Success threshold: effort > {effort_threshold:.1f} A indicates object contact
- Gripper gap after closing: {gap_after_close:.4f} m
- Expected gap if brick grasped: ~{brick_width:.3f} m (brick width)

**Position Data:**
- Current TCP Z: {tcp_z:.4f} m
- Target brick position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}] m
- Brick dimensions (L×W×H): [{brick_size[0]:.3f}, {brick_size[1]:.3f}, {brick_size[2]:.3f}] m
{history_str}

**Analysis Rules:**

1. **SUCCESS** (grasp_success = true):
   - effort > {effort_threshold:.1f} A (contact detected)
   - gap ≈ brick_width ({brick_width:.3f} m) ± 0.015 m
   - Update recommended_compensation based on any runtime adjustments

2. **TOO HIGH** (gripper closed above brick):
   - effort ≈ 0 (no contact)
   - gap ≈ 0 (gripper closed on air)
   - **Use historical data**: If previous grasps needed ~X mm descent, suggest similar
   - Adjustment: delta_z should incorporate learned compensation

3. **TOO LOW** (gripper hit table):
   - effort > threshold (contact with table)
   - gap ≈ 0 (fingers blocked by table)
   - Adjustment: delta_z = +0.01 to +0.02 m (raise slightly)

**Adaptive Learning Rules:**
- If this is attempt #1 and we have history, the compensation was already applied
- If adjustment is still needed, the NEW delta_z should be ADDITIONAL to what was applied
- On success: recommended_compensation = applied_this_grasp + any runtime delta_z
- On failure: recommended_compensation should be updated based on observed error

**Your Task:**
1. Analyze the sensor data to determine grasp success
2. If failed, compute delta_z considering historical patterns
3. Update recommended_compensation for future grasps
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap = gripper.get('gap_after_close', 0.0)
        
        brick_size = self.brick.get('size_LWH', [0.11, 0.05, 0.025])
        brick_width = brick_size[1]
        brick_height = brick_size[2]
        
        compensation = self.context.get('compensation', {})
        current_compensation = compensation.get('current', 0.0)
        applied_this_grasp = compensation.get('applied_this_grasp', 0.0)
        
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

**Adaptive Compensation Logic:**
```
┌─────────────────────────────────────────────────────────────────┐
│ FIRST GRASP (no history):                                       │
│   - Use standard adjustment: -{brick_height:.3f}m for too_high  │
│   - Record result for future learning                           │
├─────────────────────────────────────────────────────────────────┤
│ SUBSEQUENT GRASPS (with history):                               │
│   - Compensation {applied_this_grasp*1000:+.1f} mm was pre-applied           │
│   - If still too_high: add MORE descent (delta_z negative)      │
│   - If too_low: reduce compensation (delta_z positive)          │
│   - Update recommended_compensation accordingly                 │
├─────────────────────────────────────────────────────────────────┤
│ ON SUCCESS:                                                     │
│   - recommended_compensation = total offset that worked         │
│   - This becomes the baseline for next grasp                    │
└─────────────────────────────────────────────────────────────────┘
```

**Compensation Calculation:**
- Current learned compensation: {current_compensation*1000:+.1f} mm
- Applied this grasp: {applied_this_grasp*1000:+.1f} mm
- If SUCCESS: recommended_compensation = {applied_this_grasp*1000:+.1f} mm (keep what worked)
- If TOO HIGH: recommended_compensation = {applied_this_grasp*1000:+.1f} mm + delta_z (more descent)
- If TOO LOW: recommended_compensation = {applied_this_grasp*1000:+.1f} mm + delta_z (less descent)
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        gripper = self.context.get('gripper', {})
        effort = gripper.get('effort', 0.0)
        effort_threshold = gripper.get('effort_threshold', 2.0)
        gap = gripper.get('gap_after_close', 0.0)
        brick_width = self.brick.get('size_LWH', [0.11, 0.05, 0.025])[1]
        brick_height = self.brick.get('size_LWH', [0.11, 0.05, 0.025])[2]
        
        compensation = self.context.get('compensation', {})
        current_compensation = compensation.get('current', 0.0)
        applied_this_grasp = compensation.get('applied_this_grasp', 0.0)
        history = compensation.get('history', [])
        
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
                "title": "Step 2: Review Compensation History",
                "content": f"""
- Historical adjustments: {len(history)} records
- Learned compensation: {current_compensation*1000:+.1f} mm
- Applied to this attempt: {applied_this_grasp*1000:+.1f} mm
- Was the applied compensation sufficient?
""".strip()
            },
            {
                "title": "Step 3: Determine Outcome & Compute Adjustment",
                "content": f"""
- If SUCCESS: recommended_compensation = applied amount ({applied_this_grasp*1000:+.1f} mm)
- If TOO HIGH: need additional descent, delta_z = -{brick_height*1000:.1f} to -{brick_height*1.5*1000:.1f} mm
  → recommended_compensation = {applied_this_grasp*1000:.1f} + delta_z
- If TOO LOW: need to raise, delta_z = +10 to +20 mm
  → recommended_compensation = {applied_this_grasp*1000:.1f} + delta_z
""".strip()
            },
            {
                "title": "Step 4: Output Learning Update",
                "content": f"""
- recommended_compensation: the Z offset to apply to FUTURE grasps
- compensation_confidence: how certain we are (higher if consistent history)
- This value will be used by the system in the next grasp attempt
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return GRASP_FEEDBACK_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        compensation = self.context.get('compensation', {})
        applied = compensation.get('applied_this_grasp', 0.0)
        
        return [
            "grasp_success: true only if effort > threshold AND gap ≈ brick_width",
            "confidence: 0.0-1.0, higher if sensor readings are clear",
            "failure_mode: 'none', 'too_high', 'too_low', 'xy_misaligned', or 'unknown'",
            "delta_z: ADDITIONAL adjustment needed NOW, range [-0.05, 0.05] m",
            f"recommended_compensation: total offset for FUTURE grasps (current applied: {applied*1000:+.1f} mm)",
            "compensation_confidence: 0.0-1.0, higher if pattern is consistent",
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
        attempt_number: current attempt number
        max_attempts: maximum retry attempts
        compensation_history: list of past successful compensation values
        current_compensation: current learned compensation value
        applied_compensation: compensation applied to this grasp attempt
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
        "compensation": {
            "history": compensation_history or [],
            "current": current_compensation,
            "applied_this_grasp": applied_compensation,
        },
    }