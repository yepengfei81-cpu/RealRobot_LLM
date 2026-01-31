"""
Scene Analysis Prompt for Multi-Brick Stacking (v4)

Detection Logic:
1. rel_h < 0.032m (normal flat height range):
   - area >= 45cm² → FLAT (normal)
   - area < 45cm² → OCCLUDED (being stacked on)

2. rel_h >= 0.032m (abnormal height):
   - area >= 45cm² → STACKED_UPPER (on top of another brick)
   - area 25-45cm² → SIDE (侧放)
   - area < 25cm² → UPRIGHT (竖放)
"""

import math
from typing import Dict, Any, Optional, Tuple, List

from .base_prompt import BasePromptBuilder


# ==================== Constants ====================

DEFAULT_BRICK_SIZE_LWH = [0.11, 0.05, 0.025]  # L=110mm, W=50mm, H=25mm
NORMAL_BRICK_AREA_CM2 = 45.0  # 正常平放砖块面积阈值
SIDE_AREA_THRESHOLD = 30.0    # 侧放砖块面积阈值 (L*H ≈ 27.5cm²)
HEIGHT_THRESHOLD = 0.032      # 高度阈值 (略高于正常平放 H/2=0.0125m)
ANOMALY_PLACE_OFFSET_Y = 0.10  # 默认 Y 偏移
BRICK_SAFE_DISTANCE = 0.10    # 砖块之间的安全距离 (略大于砖块对角线)


# ==================== Reply Template ====================

SCENE_ANALYSIS_REPLY_TEMPLATE = """{
  "task_complete": <bool>,
  "confidence": <float 0-1>,
  "anomaly_detected": <bool>,
  "scene_state": {
    "total_bricks_detected": <int>,
    "bricks_at_target": <int>,
    "bricks_to_grasp": <int>,
    "stacked_anomaly_count": <int>,
    "pose_anomaly_count": <int>,
    "occluded_count": <int>
  },
  "next_action": {
    "type": "<'remove_anomaly'|'grasp'|'complete'|'error'>",
    "target_brick_index": <int or null>,
    "place_position": <[x,y,z] or null>,
    "reason": "<string>"
  },
  "bricks_status": [
    {"index": <int>, "pose": "<flat|side|upright|stacked_upper|stacked_lower>", "status": "<normal|anomaly_stacking|anomaly_pose|occluded|at_target>", "graspable": <bool>, "reasoning": "<brief>"}
  ],
  "analysis": {
    "stacking_pairs": "<list of [upper_idx, lower_idx] pairs or []>",
    "place_position_reasoning": "<explain why this position is safe>",
    "reasoning": "<string>"
  }
}"""


# ==================== Prompt Builder ====================

class SceneAnalysisPromptBuilder(BasePromptBuilder):
    """Scene analysis with height-first, area-second detection logic."""
    
    def _get_current_phase_name(self) -> str:
        return "Scene Analysis"
    
    def _get_completed_steps(self) -> List[str]:
        placed = self.context.get('task_state', {}).get('placed_count', 0)
        return [f"Placed {placed} brick(s)"] if placed > 0 else ["Task started"]
    
    def _get_pending_steps(self) -> List[str]:
        return ["Classify bricks", "Detect stacking pairs", "Find safe place position", "Decide action"]
    
    def _get_role_name(self) -> str:
        return "Brick Scene Analyzer"
    
    def _get_main_responsibilities(self) -> List[str]:
        return [
            "Classify each brick by rel_h first, then area",
            "Detect stacking pairs (upper + lower)",
            "Calculate safe place_position avoiding other bricks",
            "Select next action based on anomaly priority",
        ]
    
    def _get_specific_task(self) -> str:
        bricks = self.context.get('detected_bricks', [])
        target = self.context.get('target_position', {})
        target_xy = [target.get('x', 0.54), target.get('y', -0.04)]
        proximity = self.context.get('proximity_threshold', 0.08)
        ground_z = self.context.get('ground_z', 0.86)
        brick_size = self.context.get('brick_size_LWH', DEFAULT_BRICK_SIZE_LWH)
        L, W, H = brick_size
        
        # 理论面积计算
        flat_area = L * W * 10000   # 平放: 11*5 = 55 cm²
        side_area = L * H * 10000   # 侧放: 11*2.5 = 27.5 cm²
        upright_area = W * H * 10000  # 竖放: 5*2.5 = 12.5 cm²
        
        # Build brick table
        brick_rows = []
        for b in bricks:
            idx = b.get('original_index', 0)
            pos = b.get('position', [0, 0, 0])
            rel_h = b.get('relative_height', pos[2] - ground_z)
            area = b.get('area_cm2', 0)
            dist = b.get('distance_to_target', 0)
            at_tgt = "Y" if dist < proximity else "N"
            
            brick_rows.append(f"#{idx}: rel_h={rel_h:.4f}m area={area:.1f}cm² pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] dist={dist:.3f}m at_tgt={at_tgt}")
        
        brick_table = "\n".join(brick_rows) if brick_rows else "(none)"
        
        return f"""Analyze each brick and decide next action.

**Brick dimensions:** L={L*100:.1f}cm, W={W*100:.1f}cm, H={H*100:.1f}cm
**Ground Z:** {ground_z:.3f}m | **Target XY:** [{target_xy[0]:.3f}, {target_xy[1]:.3f}]
**Height threshold:** {HEIGHT_THRESHOLD}m (normal flat brick rel_h ≈ 0.012-0.025m)
**Safe distance between bricks:** {BRICK_SAFE_DISTANCE}m

**Theoretical areas:**
- FLAT (top view): {flat_area:.1f}cm²
- SIDE (side view): {side_area:.1f}cm²  
- UPRIGHT (end view): {upright_area:.1f}cm²

=== CLASSIFICATION LOGIC ===

**Step 1: Check rel_h (relative height from ground)**

IF rel_h < {HEIGHT_THRESHOLD}m → Normal height range (flat on ground)
  - area >= {NORMAL_BRICK_AREA_CM2:.0f}cm² → FLAT, status=normal, graspable=true
  - area < {NORMAL_BRICK_AREA_CM2:.0f}cm² → OCCLUDED (stacked_lower), status=occluded, graspable=false

IF rel_h >= {HEIGHT_THRESHOLD}m → Abnormal height (elevated or tilted)
  - area >= {NORMAL_BRICK_AREA_CM2:.0f}cm² → STACKED_UPPER, status=anomaly_stacking, graspable=true
  - area {SIDE_AREA_THRESHOLD:.0f}-{NORMAL_BRICK_AREA_CM2:.0f}cm² → SIDE, status=anomaly_pose, graspable=false
  - area < {SIDE_AREA_THRESHOLD:.0f}cm² → UPRIGHT, status=anomaly_pose, graspable=false

**Step 2: Detect stacking pairs**

For each STACKED_UPPER brick, find its paired OCCLUDED brick:
- Look for an OCCLUDED brick with similar XY position (distance < 0.10m)
- Record as stacking_pair: [upper_idx, lower_idx]

**Step 3: Check at_target status**

If FLAT brick is at target (dist < {proximity}m):
- Change status to "at_target", graspable=false

**Step 4: Calculate safe place_position for remove_anomaly**

When removing a stacked_upper brick, you need to place it at a SAFE position:

1. **Baseline:** Start with [upper.x, upper.y + {ANOMALY_PLACE_OFFSET_Y}, {ground_z + H}]
   (Move +{ANOMALY_PLACE_OFFSET_Y*100:.0f}cm in Y direction)

2. **Check for collisions:** 
   For each other brick (excluding the stacked_lower being freed):
   - Calculate XY distance from candidate position to that brick
   - If distance < {BRICK_SAFE_DISTANCE}m, position is NOT safe

3. **If baseline is not safe, adjust:**
   - Try [upper.x, upper.y + 0.15, z] (move further in Y)
   - Try [upper.x + 0.10, upper.y + 0.10, z] (add X offset)
   - Try [upper.x - 0.10, upper.y + 0.10, z] (negative X offset)
   - Choose the first position that is safe

4. **Place Z height:** Always use ground_z + H/2 = {ground_z + H:.3f}m

=== ACTION PRIORITY ===

1. If stacking_pairs exist → type="remove_anomaly"
   - target = stacked_upper brick index
   - place_position = [calculated safe position]
   - Explain in place_position_reasoning why this position is chosen
   
2. If only pose anomalies (SIDE/UPRIGHT) and no graspable FLAT → type="error"

3. If all graspable bricks are at_target → type="complete"

4. Otherwise → type="grasp"
   - target = first graspable FLAT brick not at target

**Detected Bricks:**
{brick_table}"""
    
    def _get_phase_specific_knowledge(self) -> str:
        brick_size = self.context.get('brick_size_LWH', DEFAULT_BRICK_SIZE_LWH)
        L, W, H = brick_size
        ground_z = self.context.get('ground_z', 0.86)
        
        return f"""**Quick Reference Table:**

| rel_h | area | pose | status | graspable |
|-------|------|------|--------|-----------|
| <{HEIGHT_THRESHOLD}m | ≥{NORMAL_BRICK_AREA_CM2:.0f}cm² | flat | normal/at_target | true/false |
| <{HEIGHT_THRESHOLD}m | <{NORMAL_BRICK_AREA_CM2:.0f}cm² | stacked_lower | occluded | false |
| ≥{HEIGHT_THRESHOLD}m | ≥{NORMAL_BRICK_AREA_CM2:.0f}cm² | stacked_upper | anomaly_stacking | true |
| ≥{HEIGHT_THRESHOLD}m | {SIDE_AREA_THRESHOLD:.0f}-{NORMAL_BRICK_AREA_CM2:.0f}cm² | side | anomaly_pose | false |
| ≥{HEIGHT_THRESHOLD}m | <{SIDE_AREA_THRESHOLD:.0f}cm² | upright | anomaly_pose | false |

**Safe Place Position Calculation:**

Given stacked_upper at position [x, y, z], check these candidate positions in order:

| Priority | Candidate Position | Description |
|----------|-------------------|-------------|
| 1 | [x, y+0.10, {ground_z + H:.3f}] | Baseline: +10cm Y |
| 2 | [x, y+0.15, {ground_z + H:.3f}] | Extended: +15cm Y |
| 3 | [x+0.10, y+0.10, {ground_z + H:.3f}] | Diagonal: +10cm X, +10cm Y |
| 4 | [x-0.10, y+0.10, {ground_z + H:.3f}] | Diagonal: -10cm X, +10cm Y |

For each candidate, verify:
- Distance to ALL other bricks >= {BRICK_SAFE_DISTANCE}m
- Choose the first safe position

**Distance formula:** sqrt((x1-x2)² + (y1-y2)²)

**Example:**
If brick at [0.52, -0.01, z] is stacked_upper, and there's another brick at [0.52, 0.08, z]:
- Baseline [0.52, 0.09, z]: distance to [0.52, 0.08] = 0.01m < 0.12m → NOT SAFE
- Extended [0.52, 0.14, z]: distance = 0.06m < 0.12m → NOT SAFE  
- Diagonal [0.62, 0.09, z]: distance = sqrt(0.10² + 0.01²) = 0.10m < 0.12m → NOT SAFE
- Diagonal [0.42, 0.09, z]: distance = sqrt(0.10² + 0.01²) = 0.10m < 0.12m → NOT SAFE
- Try [0.52, 0.20, z]: distance = 0.12m = 0.12m → SAFE (barely)"""
    
    def _get_thinking_steps(self) -> List[Dict[str, str]]:
        bricks = self.context.get('detected_bricks', [])
        ground_z = self.context.get('ground_z', 0.86)
        brick_size = self.context.get('brick_size_LWH', DEFAULT_BRICK_SIZE_LWH)
        H = brick_size[2]
        
        # Pre-classify for hint
        flat_normal = []
        occluded = []
        stacked_upper = []
        side = []
        upright = []
        
        stacked_upper_pos = None  # 记录 stacked_upper 的位置
        
        for b in bricks:
            idx = b.get('original_index', 0)
            area = b.get('area_cm2', 0)
            rel_h = b.get('relative_height', 0)
            pos = b.get('position', [0, 0, 0])
            
            if rel_h < HEIGHT_THRESHOLD:
                if area >= NORMAL_BRICK_AREA_CM2:
                    flat_normal.append(f"#{idx}[{pos[0]:.2f},{pos[1]:.2f}]")
                else:
                    occluded.append(f"#{idx}(area={area:.0f})")
            else:
                if area >= NORMAL_BRICK_AREA_CM2:
                    stacked_upper.append(f"#{idx}(rel_h={rel_h:.3f})")
                    stacked_upper_pos = pos
                elif area >= SIDE_AREA_THRESHOLD:
                    side.append(f"#{idx}(area={area:.0f},rel_h={rel_h:.3f})")
                else:
                    upright.append(f"#{idx}(area={area:.0f},rel_h={rel_h:.3f})")
        
        # Check for stacking pairs and suggest safe position
        stacking_hint = ""
        position_hint = ""
        if occluded and stacked_upper and stacked_upper_pos:
            stacking_hint = f"\n⚠️ Likely stacking: {len(stacked_upper)} upper + {len(occluded)} lower"
            
            # Calculate baseline position
            baseline_pos = [stacked_upper_pos[0], stacked_upper_pos[1] + ANOMALY_PLACE_OFFSET_Y, ground_z + H]
            
            # Check distances to all other bricks
            all_positions = [(b['original_index'], b['position']) for b in bricks]
            conflicts = []
            for idx, pos in all_positions:
                dist = math.sqrt((baseline_pos[0] - pos[0])**2 + (baseline_pos[1] - pos[1])**2)
                if dist < BRICK_SAFE_DISTANCE:
                    conflicts.append(f"#{idx}(dist={dist:.2f}m)")
            
            if conflicts:
                position_hint = f"\n⚠️ Baseline position may conflict with: {', '.join(conflicts)}"
                position_hint += f"\n   Consider adjusting place_position!"
            else:
                position_hint = f"\n✓ Baseline position [{baseline_pos[0]:.2f}, {baseline_pos[1]:.2f}, {baseline_pos[2]:.3f}] appears safe"
        
        analysis = f"""Pre-classification (verify with your analysis):
- FLAT normal: {', '.join(flat_normal) or 'none'}
- OCCLUDED (stacked_lower): {', '.join(occluded) or 'none'}  
- STACKED_UPPER: {', '.join(stacked_upper) or 'none'}
- SIDE: {', '.join(side) or 'none'}
- UPRIGHT: {', '.join(upright) or 'none'}{stacking_hint}{position_hint}"""
        
        return [{"title": "Initial classification", "content": analysis}]
    
    def _get_output_template(self) -> str:
        return SCENE_ANALYSIS_REPLY_TEMPLATE
    
    def _get_output_constraints(self) -> List[str]:
        return [
            f"Use rel_h threshold {HEIGHT_THRESHOLD}m to split normal vs abnormal height",
            "Then use area to distinguish specific pose",
            "For stacking: record [upper_idx, lower_idx] in stacking_pairs",
            f"place_position MUST be safe: distance to all bricks >= {BRICK_SAFE_DISTANCE}m",
            "Explain your place_position choice in place_position_reasoning",
            "Output JSON only, no markdown"
        ]


# ==================== Main Functions ====================

def get_scene_analysis_prompt(context: Dict[str, Any], 
                              attempt_idx: int = 0, 
                              feedback: Optional[str] = None) -> Tuple[str, str]:
    """Generate scene analysis prompt."""
    builder = SceneAnalysisPromptBuilder(context, attempt_idx, feedback)
    return builder.build()


def build_scene_analysis_context(
    detected_bricks: List[Dict[str, Any]],
    target_position: List[float],
    placed_count: int = 0,
    initial_brick_count: Optional[int] = None,
    proximity_threshold: float = 0.08,
    brick_height: float = 0.025,
    stack_increment: float = 0.028,
    normal_brick_area_cm2: float = NORMAL_BRICK_AREA_CM2,
    ground_z: float = 0.86,
    brick_size_LWH: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build context for scene analysis prompt."""
    if brick_size_LWH is None:
        brick_size_LWH = DEFAULT_BRICK_SIZE_LWH
    
    target_xy = target_position[:2] if len(target_position) >= 2 else [0.54, -0.04]
    
    # Enrich brick data
    enriched_bricks = []
    for i, brick in enumerate(detected_bricks):
        pos = brick.get('position', [0, 0, 0])
        if hasattr(pos, 'tolist'):
            pos = pos.tolist()
        
        dx = pos[0] - target_xy[0]
        dy = pos[1] - target_xy[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        z_height = brick.get('z_height', pos[2])
        
        # 优先使用已计算的 relative_height
        relative_height = brick.get('relative_height', None)
        if relative_height is None:
            relative_height = z_height - ground_z
        
        enriched_bricks.append({
            'original_index': i,
            'position': pos,
            'yaw': brick.get('yaw', 0),
            'area_cm2': brick.get('area_cm2', 55.0),
            'area_pixels': brick.get('area_pixels', 0),
            'z_height': z_height,
            'relative_height': relative_height,
            'distance_to_target': distance,
            'at_target': distance < proximity_threshold,
        })
    
    # Sort by rel_h descending (highest first) - helps see stacking
    enriched_bricks.sort(key=lambda b: b['relative_height'], reverse=True)
    
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
        "normal_brick_area_cm2": normal_brick_area_cm2,
        "ground_z": ground_z,
        "brick_size_LWH": brick_size_LWH,
    }