"""
Base Prompt Builder for Real Robot LLM Planning

This module provides a reusable base class for constructing structured prompts
following a 6-part structure:
1. Current Environment State
2. Memory Information
3. Role Definition
4. Knowledge Base
5. Thinking Chain
6. Output Format
"""

from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BasePromptBuilder(ABC):
    """
    Abstract base class for building structured LLM prompts.
    
    All specific prompt builders (pre_grasp, descend, etc.) should inherit from this class
    and implement the abstract methods to provide phase-specific content.
    """
    
    # Default system prompt - can be overridden by subclasses
    DEFAULT_SYSTEM_PROMPT = (
        "You are a precise robotics motion planning expert for real robot control. "
        "Your output will be directly used as robot control commands. "
        "Always output valid JSON only, with precise numbers in meters/radians. "
        "Do not include markdown code blocks or any additional text."
    )
    
    def __init__(self, context: Dict[str, Any], attempt_idx: int = 0, feedback: Optional[str] = None):
        """
        Initialize the prompt builder.
        
        Args:
            context: Dictionary containing all necessary information for planning
            attempt_idx: Number of retry attempts
            feedback: Error feedback from previous attempts
        """
        self.context = context
        self.attempt_idx = attempt_idx
        self.feedback = feedback
        
        # Extract common context sections
        self.robot = context.get("robot", {})
        self.gripper = context.get("gripper", {})
        self.brick = context.get("brick", {})
        self.tcp = context.get("tcp", {})
        self.constraints = context.get("constraints", {})
        self.task = context.get("task", {})
    
    # ==================== Part 1: Environment State ====================
    
    def _build_robot_state(self) -> str:
        """Build robot arm state description"""
        return f"""
**Robot Arm Status:**
- Arm DOF: {self.robot.get('dof', 6)} joints
- Current TCP position (m): [{self.tcp.get('x', 0):.4f}, {self.tcp.get('y', 0):.4f}, {self.tcp.get('z', 0):.4f}]
- Current TCP orientation RPY (rad): [{self.tcp.get('roll', 0):.4f}, {self.tcp.get('pitch', 0):.4f}, {self.tcp.get('yaw', 0):.4f}]
""".strip()
    
    def _build_gripper_state(self) -> str:
        """Build gripper state description"""
        return f"""
**Gripper Status:**
- Gripper max opening (m): {self.gripper.get('max_opening', 0.08):.4f}
- Current gripper state: {self.gripper.get('state', 'open')}
""".strip()
    
    def _build_brick_state(self) -> str:
        """Build brick state description"""
        pos = self.brick.get('position', [0, 0, 0])
        size = self.brick.get('size_LWH', [0.20, 0.095, 0.06])
        return f"""
**Target Brick Status:**
- Brick center position (m): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]
- Brick size LWH (m): [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}] (Length x Width x Height)
- Brick yaw angle (rad): {self.brick.get('yaw', 0):.4f}
- Brick top surface z (m): {pos[2] + size[2] / 2:.4f}
""".strip()
    
    def _build_constraints_state(self) -> str:
        """Build constraints description"""
        return f"""
**Planning Constraints:**
- Hover height above brick (m): {self.constraints.get('hover_height', 0.15):.3f}
- Safety clearance (m): {self.constraints.get('safety_clearance', 0.05):.3f}
""".strip()
    
    def build_environment_state(self) -> str:
        """
        Build Part 1: Current Environment State
        Can be overridden by subclasses to add phase-specific state information.
        """
        sections = [
            self._build_robot_state(),
            self._build_gripper_state(),
            self._build_brick_state(),
            self._build_constraints_state(),
        ]
        
        # Allow subclasses to add extra state sections
        extra = self._get_extra_environment_state()
        if extra:
            sections.append(extra)
        
        return "## (1) Current Environment State\n\n" + "\n\n".join(sections)
    
    def _get_extra_environment_state(self) -> Optional[str]:
        """Override this method to add phase-specific environment state"""
        return None
    
    # ==================== Part 2: Memory Information ====================
    
    @abstractmethod
    def _get_current_phase_name(self) -> str:
        """Return the name of the current planning phase"""
        pass
    
    @abstractmethod
    def _get_completed_steps(self) -> list:
        """Return list of completed steps"""
        pass
    
    @abstractmethod
    def _get_pending_steps(self) -> list:
        """Return list of pending steps"""
        pass
    
    def build_memory_info(self) -> str:
        """Build Part 2: Memory Information"""
        completed = "\n".join([f"- ✓ {step}" for step in self._get_completed_steps()])
        pending = "\n".join([f"- → {step}" for step in self._get_pending_steps()])
        
        feedback_text = f"Previous attempt rejected, reason: {self.feedback}" if self.feedback else "No historical error feedback"
        
        return f"""
## (2) Memory Information

**Task Progress:**
- Current execution phase: {self._get_current_phase_name()}
- Attempt number: {self.attempt_idx + 1}

**Completed Steps:**
{completed}

**Pending Steps:**
{pending}

**Error Feedback:**
{feedback_text}
""".strip()
    
    # ==================== Part 3: Role Definition ====================
    
    @abstractmethod
    def _get_role_name(self) -> str:
        """Return the role name for this agent"""
        pass
    
    @abstractmethod
    def _get_main_responsibilities(self) -> list:
        """Return list of main responsibilities"""
        pass
    
    @abstractmethod
    def _get_specific_task(self) -> str:
        """Return the specific task description"""
        pass
    
    def build_role_definition(self) -> str:
        """Build Part 3: Role Definition"""
        responsibilities = "\n".join([f"- {r}" for r in self._get_main_responsibilities()])
        
        return f"""
## (3) Role Definition

You are a **{self._get_role_name()}**, working on real robot control.

**Main Responsibilities:**
{responsibilities}

**Specific Task:**
{self._get_specific_task()}
""".strip()
    
    # ==================== Part 4: Knowledge Base ====================
    
    def _get_coordinate_system_knowledge(self) -> str:
        """Standard coordinate system knowledge"""
        return """
**Coordinate System:**
- World coordinate system: X-forward, Y-left, Z-up (right-hand rule)
- Robot base frame: aligned with world frame
- TCP coordinate system: z-axis along gripper centerline, x-axis towards gripper opening direction
""".strip()
    
    def _get_brick_geometry_knowledge(self) -> str:
        """Standard brick geometry knowledge"""
        size = self.brick.get('size_LWH', [0.20, 0.095, 0.06])
        return f"""
**Brick Geometry:**
- Standard brick size: L={size[0]:.3f}m (length), W={size[1]:.3f}m (width), H={size[2]:.3f}m (height)
- Brick yaw angle: rotation of brick length direction relative to world X-axis
- Brick center: geometric center of the brick
- Brick top surface: center_z + height/2
""".strip()
    
    @abstractmethod
    def _get_phase_specific_knowledge(self) -> str:
        """Return phase-specific knowledge"""
        pass
    
    def build_knowledge_base(self) -> str:
        """Build Part 4: Knowledge Base"""
        sections = [
            self._get_coordinate_system_knowledge(),
            self._get_brick_geometry_knowledge(),
            self._get_phase_specific_knowledge(),
        ]
        
        return "## (4) Knowledge Base\n\n" + "\n\n".join(sections)
    
    # ==================== Part 5: Thinking Chain ====================
    
    @abstractmethod
    def _get_thinking_steps(self) -> list:
        """Return list of thinking steps, each as a dict with 'title' and 'content'"""
        pass
    
    def build_thinking_chain(self) -> str:
        """Build Part 5: Thinking Chain"""
        steps = self._get_thinking_steps()
        formatted_steps = []
        
        for i, step in enumerate(steps, 1):
            formatted_steps.append(f"**Step {i}: {step['title']}**\n{step['content']}")
        
        return "## (5) Thinking Chain\n\n" + "\n\n".join(formatted_steps)
    
    # ==================== Part 6: Output Format ====================
    
    @abstractmethod
    def _get_output_template(self) -> str:
        """Return the JSON output template"""
        pass
    
    @abstractmethod
    def _get_output_constraints(self) -> list:
        """Return list of output constraints"""
        pass
    
    def build_output_format(self) -> str:
        """Build Part 6: Output Format"""
        constraints = "\n".join([f"{i+1}. {c}" for i, c in enumerate(self._get_output_constraints())])
        
        return f"""
## (6) Output Format

**Output Description:**
Your output will be directly used as real robot control commands. Must be precise and accurate.

**Strict Constraints:**
{constraints}

**Output Template:**
{self._get_output_template()}

**Important Reminders:**
- Numbers must use floating-point format (e.g., 0.15, not .15)
- Coordinate units: meters (m), angle units: radians (rad)
- Output JSON only, no markdown code blocks or explanatory text
- If calculation is uncertain, explain in reasoning field
""".strip()
    
    # ==================== Main Build Method ====================
    
    def build(self) -> Tuple[str, str]:
        """
        Build the complete prompt.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        sections = [
            self.build_environment_state(),
            self.build_memory_info(),
            self.build_role_definition(),
            self.build_knowledge_base(),
            self.build_thinking_chain(),
            self.build_output_format(),
        ]
        
        user_prompt = "\n\n".join(sections)
        system_prompt = self.get_system_prompt()
        
        return system_prompt, user_prompt
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt. Override to customize.
        """
        return self.DEFAULT_SYSTEM_PROMPT


# ==================== Helper Functions ====================

def format_position(pos: list, precision: int = 4) -> str:
    """Format position list as string"""
    return f"[{pos[0]:.{precision}f}, {pos[1]:.{precision}f}, {pos[2]:.{precision}f}]"


def format_orientation(rpy: list, precision: int = 4) -> str:
    """Format RPY orientation as string"""
    return f"[{rpy[0]:.{precision}f}, {rpy[1]:.{precision}f}, {rpy[2]:.{precision}f}]"


def compute_brick_top_z(brick_pos: list, brick_size: list) -> float:
    """Compute brick top surface z coordinate"""
    return brick_pos[2] + brick_size[2] / 2