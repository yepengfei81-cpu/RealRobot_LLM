"""
Release Prompt for Real Robot

This prompt enables LLM to analyze placement feedback and decide:
- Whether to release the gripper (brick touching surface)
- Or adjust Z position (descend more or lift slightly)
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

RELEASE_REPLY_TEMPLATE = """
{
  "contact_detected": <boolean, true only if z_error > threshold>,
  "confidence": <float 0.0-1.0>,
  "analysis": {
    "z_error_mm": <float, positive=above target>,
    "contact_state": "<string: 'pressing' | 'no_contact'>"
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
    
    Goal: Analyze TCP Z position feedback to determine if brick
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
        contact_threshold_mm = thresholds.get('contact_mm', 0.6)
        descend_step = thresholds.get('descend_step', 0.008)
        lift_step = thresholds.get('lift_step', 0.005)
        
        attempt = self.context.get('attempt', {})
        attempt_num = attempt.get('number', 1)
        max_attempts = attempt.get('max', 10)
        
        return f"""
Analyze the placement state and decide the release action.

**TCP Position Feedback (Attempt #{attempt_num}/{max_attempts}):**
- Actual TCP Z: {actual_z:.4f} m
- Target surface Z: {target_z:.4f} m
- Z Error: {z_error_mm:+.1f} mm (positive = actual is ABOVE target)

**Decision Rules (IMPORTANT - follow strictly):**

1. **PRESSING on surface** (z_error > +{contact_threshold_mm:.1f} mm):
   - Actual Z is MORE THAN {contact_threshold_mm:.1f} mm ABOVE target
   - This means the brick is blocked by the surface
   - Action: Lift +{lift_step*1000:.0f} mm, then release
   - action.type = "lift_then_release", delta_z = +{lift_step}

2. **NO CONTACT yet** (z_error ≤ 0 mm):
   - Actual Z is AT or BELOW target (not blocked)
   - The brick has NOT reached the surface yet
   - Action: Descend by {descend_step*1000:.0f} mm
   - action.type = "descend", delta_z = -{descend_step}
   - Continue descending until z_error becomes positive

**CRITICAL:** 
- There is NO "just_contact" or "release" action type!
- If z_error is between 0 and +{contact_threshold_mm:.1f} mm, treat as NO CONTACT (descend more)
- Only when z_error > +{contact_threshold_mm:.1f} mm, lift then release

**Current situation:**
- Z error = {z_error_mm:+.1f} mm
- {"PRESSING - lift then release" if z_error_mm > contact_threshold_mm else "NO CONTACT - descend more"}
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.6)
        descend_step = thresholds.get('descend_step', 0.008)
        lift_step = thresholds.get('lift_step', 0.005)
        
        return f"""
**Contact Detection Logic (Simplified):**

```
Z Error = Actual_Z - Target_Z

┌─────────────────────┬─────────────────┬─────────────────────────────┐
│ Z Error             │ Contact State   │ Action                      │
├─────────────────────┼─────────────────┼─────────────────────────────┤
│ > +{contact_threshold_mm:.1f} mm           │ PRESSING        │ lift_then_release (+{lift_step*1000:.0f}mm)  │
│ ≤ 0 mm              │ NO_CONTACT      │ descend (-{descend_step*1000:.0f}mm)           │
│ 0 ~ +{contact_threshold_mm:.1f} mm         │ NO_CONTACT      │ descend (-{descend_step*1000:.0f}mm)           │
└─────────────────────┴─────────────────┴─────────────────────────────┘
```

**Key insight:**
- Positive z_error means actual position is HIGHER than target
- When robot tries to go down but actual Z stays high → surface is blocking
- We keep descending until z_error exceeds +{contact_threshold_mm:.1f} mm

**Only TWO possible actions:**
1. "descend" with delta_z = -{descend_step} (negative, go down)
2. "lift_then_release" with delta_z = +{lift_step} (positive, go up then release)

**NO "release" action with delta_z = 0!**
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        tcp = self.context.get('tcp', {})
        z_error = tcp.get('z_error', 0.0)
        z_error_mm = z_error * 1000
        
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.6)
        descend_step = thresholds.get('descend_step', 0.008)
        lift_step = thresholds.get('lift_step', 0.005)
        
        is_pressing = z_error_mm > contact_threshold_mm
        
        return [
            {
                "title": "Step 1: Calculate Z Error",
                "content": f"""
- Z error = actual_z - target_z = {z_error_mm:+.1f} mm
- Positive error = actual is ABOVE target (blocked by surface)
- Zero or negative error = actual is AT or BELOW target (no contact)
""".strip()
            },
            {
                "title": "Step 2: Compare with Threshold",
                "content": f"""
- Threshold for PRESSING: > +{contact_threshold_mm:.1f} mm
- Current z_error: {z_error_mm:+.1f} mm
- Is {z_error_mm:+.1f} > +{contact_threshold_mm:.1f}? → {"YES, PRESSING" if is_pressing else "NO, keep descending"}
""".strip()
            },
            {
                "title": "Step 3: Decide Action",
                "content": f"""
- If PRESSING (z_error > +{contact_threshold_mm:.1f}): lift_then_release, delta_z = +{lift_step}
- If NOT pressing (z_error ≤ +{contact_threshold_mm:.1f}): descend, delta_z = -{descend_step}
- Current decision: {"lift_then_release" if is_pressing else "descend"}
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return RELEASE_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        thresholds = self.context.get('thresholds', {})
        contact_threshold_mm = thresholds.get('contact_mm', 0.6)
        descend_step = thresholds.get('descend_step', 0.008)
        lift_step = thresholds.get('lift_step', 0.005)
        
        return [
            "contact_detected: true ONLY if z_error > +{:.1f} mm".format(contact_threshold_mm),
            "confidence: 0.0-1.0",
            f"contact_state: ONLY 'pressing' or 'no_contact' (NO 'just_contact')",
            f"action.type: ONLY 'descend' or 'lift_then_release' (NO 'release')",
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
    contact_threshold_mm: float = 0.6,
    descend_step: float = 0.005,
    lift_step: float = 0.01,
    attempt_number: int = 1,
    max_attempts: int = 10,
) -> Dict[str, Any]:
    """
    Build context for release prompt.
    
    Args:
        actual_z: Current TCP Z position (meters)
        target_z: Target surface Z position (meters)
        contact_threshold_mm: Threshold for contact detection (mm)
        descend_step: Step size for descending (meters)
        lift_step: Step size for lifting before release (meters)
        attempt_number: Current attempt number
        max_attempts: Maximum attempts
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
        "brick": {
            "size_LWH": [0.11, 0.05, 0.025],
        },
    }