"""
Scene Analysis Prompt for Multi-Brick Stacking

This prompt enables LLM to analyze the current scene state:
- Count detected bricks
- Identify which bricks are at target location (already stacked)
- Determine if task is complete
- Select next brick to grasp
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Reply Template ====================

SCENE_ANALYSIS_REPLY_TEMPLATE = """
{
  "task_complete": <boolean>,
  "confidence": <float 0.0-1.0>,
  "scene_state": {
    "total_bricks_detected": <int>,
    "bricks_at_target": <int, bricks within proximity_threshold of target>,
    "bricks_to_grasp": <int, remaining bricks away from target>,
    "estimated_stack_height": <int, number of bricks stacked>
  },
  "next_action": {
    "type": "<string: 'grasp' | 'complete' | 'error'>",
    "target_brick_index": <int or null, ORIGINAL index in detected_bricks list>,
    "reason": "<string explaining the decision>"
  },
  "analysis": {
    "target_area_status": "<string: 'empty' | 'has_stack' | 'complete'>",
    "completion_criteria_met": <boolean>,
    "reasoning": "<string, max 80 words>"
  }
}
""".strip()


# ==================== Scene Analysis Prompt Builder ====================

class SceneAnalysisPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for multi-brick scene analysis.
    
    Goal: Analyze detected bricks to determine task state and next action.
    """
    
    def _get_current_phase_name(self) -> str:
        return "Scene Analysis for Multi-Brick Stacking"
    
    def _get_completed_steps(self) -> List[str]:
        placed = self.context.get('task_state', {}).get('placed_count', 0)
        if placed == 0:
            return ["Task initialized"]
        return [f"Placed {placed} brick(s) at target location"]
    
    def _get_pending_steps(self) -> List[str]:
        return [
            "Analyze current scene (current task)",
            "Determine if task is complete",
            "If not complete: select next brick to grasp",
            "Execute grasp-place cycle",
        ]
    
    def _get_role_name(self) -> str:
        return "Multi-Brick Stacking Scene Analyzer"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Count and locate all detected bricks",
            "Identify bricks already at target location (stacked)",
            "Determine if stacking task is complete",
            "Select the next brick to grasp (if task not complete)",
            "Ensure safe and efficient brick selection",
        ]
    
    def _get_specific_task(self) -> str:
        bricks = self.context.get('detected_bricks', [])
        target = self.context.get('target_position', {})
        target_xy = [target.get('x', 0.54), target.get('y', -0.04)]
        proximity_threshold = self.context.get('proximity_threshold', 0.08)
        
        task_state = self.context.get('task_state', {})
        placed_count = task_state.get('placed_count', 0)
        initial_count = task_state.get('initial_brick_count', len(bricks))
        
        # Format brick list with clear at_target marking
        brick_info = ""
        for brick in bricks:
            idx = brick.get('original_index', 0)
            pos = brick.get('position', [0, 0, 0])
            dist = brick.get('distance_to_target', 0)
            at_target = dist < proximity_threshold
            status = "⚠️ AT TARGET - DO NOT GRASP" if at_target else "✓ Available to grasp"
            brick_info += f"  - Brick {idx}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m, "
            brick_info += f"distance_to_target={dist:.3f} m → {status}\n"
        
        if not brick_info:
            brick_info = "  (No bricks detected)\n"
        
        return f"""
Analyze the current scene and determine the next action for brick stacking.

**Target Placement Location:**
- XY Position: [{target_xy[0]:.3f}, {target_xy[1]:.3f}] m
- Proximity threshold: {proximity_threshold:.3f} m (bricks within this distance are considered "at target")

**Task Progress:**
- Bricks placed so far: {placed_count}
- Initial brick count: {initial_count}
- Expected remaining: {initial_count - placed_count}

**Detected Bricks (with ORIGINAL indices):**
{brick_info}

**CRITICAL RULES:**
1. Count how many bricks are near the target (distance < {proximity_threshold:.3f} m) - these are ALREADY STACKED
2. NEVER select a brick that is "AT TARGET" - it's already placed!
3. Task is complete if: no bricks available to grasp (all at target)
4. If not complete: select a brick with distance_to_target >= {proximity_threshold:.3f} m
5. Return the ORIGINAL brick index (the number shown after "Brick")

**Your Task:**
- If all bricks are at target → return type="complete"
- If bricks remain to grasp → return type="grasp" with the ORIGINAL index of a brick NOT at target
""".strip()
    
    def _get_phase_specific_knowledge(self) -> str:
        proximity_threshold = self.context.get('proximity_threshold', 0.08)
        brick_height = self.context.get('brick_height', 0.025)
        stack_increment = self.context.get('stack_increment', 0.028)
        
        return f"""
**Multi-Brick Stacking Logic:**

```
Scene Analysis Decision Tree:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Count bricks at target (distance < {proximity_threshold:.3f} m) - STACKED     │
│ 2. Count bricks away from target (distance >= {proximity_threshold:.3f} m) - TO GRASP │
├─────────────────────────────────────────────────────────────────┤
│ IF no bricks to grasp (all at target) → COMPLETE                │
│ IF bricks to grasp > 0 → SELECT one that is NOT at target       │
│ IF no bricks detected → ERROR                                   │
└─────────────────────────────────────────────────────────────────┘
```

**⚠️ CRITICAL: Brick Selection Rules:**
- A brick with distance < {proximity_threshold:.3f} m is AT TARGET (already stacked)
- NEVER select a brick that is at target - you'd be picking up your own stack!
- Only select bricks with distance >= {proximity_threshold:.3f} m

**Brick Selection Strategy (for bricks NOT at target):**
- Prefer bricks farther from target (less risk of disturbing stack)
- Any brick NOT at target is valid to grasp

**Stacking Height Calculation:**
- Each brick height: {brick_height:.3f} m
- Stack increment (with margin): {stack_increment:.3f} m
- Stack of N bricks ≈ N × {stack_increment:.3f} m above surface

**Completion Criteria:**
- All bricks consolidated at target location
- No isolated bricks remaining elsewhere (no bricks with distance >= {proximity_threshold:.3f} m)
""".strip()
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        bricks = self.context.get('detected_bricks', [])
        proximity_threshold = self.context.get('proximity_threshold', 0.08)
        
        # Pre-calculate for thinking steps
        at_target_bricks = [b for b in bricks if b.get('distance_to_target', 999) < proximity_threshold]
        graspable_bricks = [b for b in bricks if b.get('distance_to_target', 999) >= proximity_threshold]
        
        at_target_indices = [b.get('original_index', '?') for b in at_target_bricks]
        graspable_indices = [b.get('original_index', '?') for b in graspable_bricks]
        
        return [
            {
                "title": "Step 1: Classify Bricks",
                "content": f"""
- Total detected: {len(bricks)}
- At target (distance < {proximity_threshold:.3f} m): {len(at_target_bricks)} → indices: {at_target_indices}
- Available to grasp (distance >= {proximity_threshold:.3f} m): {len(graspable_bricks)} → indices: {graspable_indices}
""".strip()
            },
            {
                "title": "Step 2: Check Completion",
                "content": f"""
- Bricks available to grasp: {len(graspable_bricks)}
- Task complete? → {"YES (all at target)" if len(graspable_bricks) == 0 else "NO (bricks remain)"}
""".strip()
            },
            {
                "title": "Step 3: Select Next Brick (if not complete)",
                "content": f"""
- If not complete, select from graspable bricks: {graspable_indices}
- Choose the one farthest from target (safest)
- Return its ORIGINAL index
- ⚠️ DO NOT return indices: {at_target_indices} (these are at target!)
""".strip()
            },
        ]
    
    def _get_output_template(self) -> str:
        return SCENE_ANALYSIS_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        proximity_threshold = self.context.get('proximity_threshold', 0.08)
        bricks = self.context.get('detected_bricks', [])
        
        at_target_indices = [b.get('original_index', '?') for b in bricks 
                           if b.get('distance_to_target', 999) < proximity_threshold]
        graspable_indices = [b.get('original_index', '?') for b in bricks 
                            if b.get('distance_to_target', 999) >= proximity_threshold]
        
        constraints = [
            "task_complete: true only if NO bricks available to grasp",
            f"bricks_at_target: count of bricks with distance < {proximity_threshold:.3f} m",
            "next_action.type: 'grasp' if bricks remain, 'complete' if done, 'error' if problem",
            f"target_brick_index: MUST be one of {graspable_indices} if type='grasp'",
            f"⚠️ FORBIDDEN indices (at target): {at_target_indices} - NEVER select these!",
            "Output JSON only, no markdown code blocks",
        ]
        return constraints


# ==================== Main Function ====================

def get_scene_analysis_prompt(context: Dict[str, Any], 
                              attempt_idx: int = 0, 
                              feedback: Optional[str] = None) -> Tuple[str, str]:
    """Generate scene analysis prompt."""
    builder = SceneAnalysisPromptBuilder(context, attempt_idx, feedback)
    return builder.build()


# ==================== Context Builder Helper ====================

def build_scene_analysis_context(
    detected_bricks: List[Dict[str, Any]],
    target_position: List[float],
    placed_count: int = 0,
    initial_brick_count: Optional[int] = None,
    proximity_threshold: float = 0.08,
    brick_height: float = 0.025,
    stack_increment: float = 0.028,
) -> Dict[str, Any]:
    """
    Build context for scene analysis prompt.
    
    Args:
        detected_bricks: List of detected brick info, each with:
            - position: [x, y, z] in base_link frame
            - (optional) yaw: orientation
        target_position: [x, y, z] target placement position
        placed_count: Number of bricks already placed
        initial_brick_count: Initial number of bricks (if known)
        proximity_threshold: Distance threshold for "at target" (meters)
        brick_height: Single brick height (meters)
        stack_increment: Height increment per stacked brick (meters)
    
    Returns:
        Context dictionary for prompt builder
    """
    target_xy = target_position[:2] if len(target_position) >= 2 else [0.54, -0.04]
    
    # Calculate distance to target for each brick, PRESERVE original index
    enriched_bricks = []
    for i, brick in enumerate(detected_bricks):
        pos = brick.get('position', [0, 0, 0])
        dx = pos[0] - target_xy[0]
        dy = pos[1] - target_xy[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        enriched_bricks.append({
            'original_index': i,  # 保留原始索引！
            'position': pos,
            'yaw': brick.get('yaw', 0),
            'distance_to_target': distance,
            'at_target': distance < proximity_threshold,  # 明确标记是否在目标位置
        })
    
    # Sort by distance (farthest first) but keep original_index
    enriched_bricks.sort(key=lambda b: b['distance_to_target'], reverse=True)
    
    if initial_brick_count is None:
        initial_brick_count = len(detected_bricks) + placed_count
    
    return {
        "detected_bricks": enriched_bricks,
        "target_position": {
            "x": target_xy[0],
            "y": target_xy[1],
            "z": target_position[2] if len(target_position) > 2 else 0.89,
        },
        "task_state": {
            "placed_count": placed_count,
            "initial_brick_count": initial_brick_count,
        },
        "proximity_threshold": proximity_threshold,
        "brick_height": brick_height,
        "stack_increment": stack_increment,
    }