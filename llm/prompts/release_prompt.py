"""
Release Prompt for Real Robot

This prompt enables LLM to analyze placement feedback and decide:
- Whether to release the gripper (brick touching surface)
- Or adjust Z position (descend more or lift slightly)

Contact detection uses TWO methods:
1. Z position error (z_error > threshold_mm)
2. Arm effort (total arm current > effort_threshold)
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

RELEASE_REPLY_TEMPLATE = """
{
  "contact_detected": <boolean, true if z_error > threshold OR arm_effort > effort_threshold>,
  "confidence": <float 0.0-1.0>,
  "analysis": {
    "z_error_mm": <float, positive=above target>,
    "arm_effort": <float or null, total arm current in Amps>,
    "contact_state": "<string: 'pressing' | 'effort_contact' | 'no_contact'>"
  },
  "action": {
    "type": "<string: 'descend' | 'lift_then_release'>",
    "delta_z": <float in meters, negative for descend, positive for lift>,
    "reason": "<string explaining the action>"
  },
  "reasoning": "Brief analysis (max 60 words)"
}
""".strip()


# ==================== Release Prompt Builder ====================

class ReleasePromptBuilder(BasePromptBuilder):
    """
    Prompt builder for release decision.
    
    Goal: Analyze TCP Z position feedback AND arm effort to determine if brick
    is touching surface and decide release action.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Release Decision"
    
    def _get_completed_steps(self) -> List[str]:
        return [
            "Head camera brick detection",
            "Pre-grasp hover positioning",
            "Hand-eye fine XY alignment",
            "Descend and grasp brick",
            "Lift brick",
            "Move to place position",
            "Descend to place surface",
        ]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Analyze contact state (current task)",
            "If contact: release gripper",
            "If no contact: descend more",
            "Retreat after release",
        ]
    
    def _get_role_name(self) -> str:
        return "Placement Release Expert"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Analyze TCP Z position vs target to detect surface contact",
            "Monitor arm effort (current) for force-based contact detection",
            "Determine if brick is pressing on surface (safe to release)",
            "Compute Z adjustment if not yet in contact",
            "Ensure gentle placement without dropping brick",
        ]
    
    def _get_specific_task(self) -> str:
        tcp = self.context.get('tcp', {})
        actual_z = tcp.get('actual_z', 0.0)
        target_z = tcp.get('target_z', 0.0)
        z_error = tcp.get('z_error', 0.0)
        z_error_mm = z_error * 1000
        
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.8)
        descend_step = thresholds.get('descend_step', 0.005)
        lift_step = thresholds.get('lift_step', 0.01)
        
        # 获取电流信息
        arm_effort = self.context.get('arm_effort', None)
        arm_effort_threshold = self.context.get('arm_effort_threshold', 12.0)
        arm_effort_str = f"{arm_effort:.2f}A" if arm_effort is not None else "N/A"
        
        attempt = self.context.get('attempt', {})
        attempt_num = attempt.get('number', 1)
        max_attempts = attempt.get('max', 10)
        
        # 判断是否超过电流阈值
        effort_exceeded = arm_effort is not None and arm_effort > arm_effort_threshold
        z_exceeded = z_error_mm > contact_threshold_mm
        
        return f"""
Analyze the placement state and decide the release action.

**TCP Position Feedback (Attempt #{attempt_num}/{max_attempts}):**
- Actual TCP Z: {actual_z:.4f} m
- Target surface Z: {target_z:.4f} m
- Z Error: {z_error_mm:+.1f} mm (positive = actual is ABOVE target)

**Arm Effort Feedback:**
- Current arm effort: {arm_effort_str}
- Effort threshold: {arm_effort_threshold:.1f} A
- Effort exceeded: {"YES" if effort_exceeded else "NO"}

**Decision Rules (IMPORTANT - follow strictly):**

1. **ARM EFFORT EXCEEDED** (arm_effort > {arm_effort_threshold:.1f} A):
   - High force detected on arm joints
   - This indicates the brick is pressing hard on the surface
   - Action: Lift +{lift_step*1000:.0f} mm, then release
   - action.type = "lift_then_release", delta_z = +{lift_step}
   - contact_state = "effort_contact"

2. **Z PRESSING on surface** (z_error > +{contact_threshold_mm:.1f} mm):
   - Actual Z is MORE THAN {contact_threshold_mm:.1f} mm ABOVE target
   - This means the brick is blocked by the surface
   - Action: Lift +{lift_step*1000:.0f} mm, then release
   - action.type = "lift_then_release", delta_z = +{lift_step}
   - contact_state = "pressing"

3. **NO CONTACT yet** (z_error ≤ +{contact_threshold_mm:.1f} mm AND arm_effort ≤ {arm_effort_threshold:.1f} A):
   - Neither Z error nor arm effort indicates contact
   - The brick has NOT reached the surface yet
   - Action: Descend by {descend_step*1000:.0f} mm
   - action.type = "descend", delta_z = -{descend_step}
   - contact_state = "no_contact"

**CRITICAL:** 
- If arm_effort > {arm_effort_threshold:.1f} A → ALWAYS lift_then_release (highest priority)
- If z_error > +{contact_threshold_mm:.1f} mm → lift_then_release
- Otherwise → descend

**Current situation:**
- Z error = {z_error_mm:+.1f} mm (threshold: {contact_threshold_mm:.1f} mm)
- Arm effort = {arm_effort_str} (threshold: {arm_effort_threshold:.1f} A)
- Decision: {"EFFORT CONTACT - lift then release" if effort_exceeded else ("Z PRESSING - lift then release" if z_exceeded else "NO CONTACT - descend more")}
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.8)
        descend_step = thresholds.get('descend_step', 0.005)
        lift_step = thresholds.get('lift_step', 0.01)
        arm_effort_threshold = self.context.get('arm_effort_threshold', 12.0)
        
        return f"""
**Contact Detection Logic (Dual Method):**

```
Method 1: Z Position Error
  Z Error = Actual_Z - Target_Z
  If z_error > +{contact_threshold_mm:.1f} mm → PRESSING

Method 2: Arm Effort (Force)
  If arm_effort > {arm_effort_threshold:.1f} A → EFFORT_CONTACT

┌─────────────────────┬─────────────────┬─────────────────────────────┐
│ Condition           │ Contact State   │ Action                      │
├─────────────────────┼─────────────────┼─────────────────────────────┤
│ effort > {arm_effort_threshold:.1f}A        │ EFFORT_CONTACT  │ lift_then_release (+{lift_step*1000:.0f}mm)  │
│ z_error > +{contact_threshold_mm:.1f}mm     │ PRESSING        │ lift_then_release (+{lift_step*1000:.0f}mm)  │
│ otherwise           │ NO_CONTACT      │ descend (-{descend_step*1000:.0f}mm)           │
└─────────────────────┴─────────────────┴─────────────────────────────┘
```

**Priority:** Arm effort detection has HIGHER priority than Z position.
Even if z_error is small, high arm effort means contact.

**Only TWO possible actions:**
1. "descend" with delta_z = -{descend_step} (negative, go down)
2. "lift_then_release" with delta_z = +{lift_step} (positive, go up then release)
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        tcp = self.context.get('tcp', {})
        z_error = tcp.get('z_error', 0.0)
        z_error_mm = z_error * 1000
        
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.8)
        descend_step = thresholds.get('descend_step', 0.005)
        lift_step = thresholds.get('lift_step', 0.01)
        
        arm_effort = self.context.get('arm_effort', None)
        arm_effort_threshold = self.context.get('arm_effort_threshold', 12.0)
        arm_effort_str = f"{arm_effort:.2f}A" if arm_effort is not None else "N/A"
        
        effort_exceeded = arm_effort is not None and arm_effort > arm_effort_threshold
        z_exceeded = z_error_mm > contact_threshold_mm
        
        return [
            {
                "title": "Step 1: Check Arm Effort (Priority)",
                "content": f"""
- Arm effort = {arm_effort_str}
- Threshold = {arm_effort_threshold:.1f} A
- Is {arm_effort_str} > {arm_effort_threshold:.1f}A? → {"YES, EFFORT CONTACT" if effort_exceeded else "NO, check Z error"}
""".strip()
            },
            {
                "title": "Step 2: Check Z Error",
                "content": f"""
- Z error = {z_error_mm:+.1f} mm
- Threshold for PRESSING: > +{contact_threshold_mm:.1f} mm
- Is {z_error_mm:+.1f} > +{contact_threshold_mm:.1f}? → {"YES, Z PRESSING" if z_exceeded else "NO, keep descending"}
""".strip()
            },
            {
                "title": "Step 3: Decide Action",
                "content": f"""
- If EFFORT CONTACT or PRESSING: lift_then_release, delta_z = +{lift_step}
- If NO CONTACT: descend, delta_z = -{descend_step}
- Current decision: {"lift_then_release" if (effort_exceeded or z_exceeded) else "descend"}
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return RELEASE_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.8)
        descend_step = thresholds.get('descend_step', 0.005)
        lift_step = thresholds.get('lift_step', 0.01)
        arm_effort_threshold = self.context.get('arm_effort_threshold', 12.0)
        
        return [
            f"contact_detected: true if arm_effort > {arm_effort_threshold:.1f}A OR z_error > +{contact_threshold_mm:.1f}mm",
            "confidence: 0.0-1.0",
            "contact_state: 'effort_contact', 'pressing', or 'no_contact'",
            "action.type: ONLY 'descend' or 'lift_then_release'",
            f"For 'descend': delta_z = -{descend_step} (negative)",
            f"For 'lift_then_release': delta_z = +{lift_step} (positive)",
            "Output JSON only, no markdown code blocks",
        ]


# ==================== Main Function ====================

def get_release_prompt(context: Dict[str, Any], 
                       attempt_idx: int = 0, 
                       feedback: Optional[str] = None) -> Tuple[str, str]:
    """Generate release decision prompt."""
    builder = ReleasePromptBuilder(context, attempt_idx, feedback)
    return builder.build()


# ==================== Context Builder Helper ====================

def build_release_context(
    actual_z: float,
    target_z: float,
    contact_threshold_mm: float = 0.8,
    descend_step: float = 0.005,
    lift_step: float = 0.01,
    attempt_number: int = 1,
    max_attempts: int = 10,
    arm_effort: Optional[float] = None,
    arm_effort_threshold: float = 12.0,
) -> Dict[str, Any]:
    """
    Build context for release prompt.
    
    Args:
        actual_z: Current TCP Z position (meters)
        target_z: Target surface Z position (meters)
        contact_threshold_mm: Z error threshold for contact detection (mm)
        descend_step: Step size for descending (meters)
        lift_step: Step size for lifting (meters)
        attempt_number: Current attempt number
        max_attempts: Maximum attempts
        arm_effort: Total arm effort/current (Amps), None if unavailable
        arm_effort_threshold: Threshold for effort-based contact (Amps)
    """
    z_error = actual_z - target_z
    
    return {
        "tcp": {
            "actual_z": actual_z,
            "target_z": target_z,
            "z_error": z_error,
        },
        "thresholds": {
            "contact_mm": contact_threshold_mm,
            "descend_step": descend_step,
            "lift_step": lift_step,
        },
        "attempt": {
            "number": attempt_number,
            "max": max_attempts,
        },
        "arm_effort": arm_effort,
        "arm_effort_threshold": arm_effort_threshold,
        "brick": {
            "size_LWH": [0.11, 0.05, 0.025],
        },
    }