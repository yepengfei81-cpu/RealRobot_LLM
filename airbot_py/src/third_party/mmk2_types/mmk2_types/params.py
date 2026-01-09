from dataclasses import dataclass, field
from typing import List, Union, Optional
from mmk2_types.types import (
    MMK2Components,
    ServoCommandTypes,
    ComponentTypes,
    ControllerTypes,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from bidict import bidict
from enum import Enum


@dataclass
class RecordParams:
    """Parameters for recording a trajectory."""

    duration: float = 0.0
    frequency: float = 1.0  # factor to multiply the speed
    mode: str = "overwrite"  # overwrite, append
    stop: bool = False  #  stop current recording, if True, other parameters are ignored


@dataclass
class ReplayParams:
    """Parameters for replaying a trajectory."""

    loop: bool = False  # loop the replay
    frequency: float = 1.0  # factor to multiply the speed
    wait: bool = False  # wait for finishing the replay
    stop: bool = False  # stop current replay


@dataclass
class TeleopParams:
    """Parameters for teleoperation."""

    speed: float = 1.0
    angular_speed: float = 1.0
    wait: bool = True


@dataclass
class ManualParams:
    """Parameters for manual control."""

    direction: str = ""
    stop: bool = False


@dataclass
class MoveServoParams:
    """Parameters for servo control."""

    # TODO: has usefull values for all fields in server side while None for client side?
    # or should all use the former?
    # TODO: remove header
    header: Header = field(default_factory=Header)
    # TODO: change to command_type and remove 'servo' prefix
    servo_type: ServoCommandTypes = ServoCommandTypes.JOINT_POSITION
    servo_backend: ControllerTypes = ControllerTypes.FORWARD_POSITION
    component_type: ComponentTypes = ComponentTypes.UNKNOWN

    def __post_init__(self):
        if isinstance(self.servo_type, int):
            self.servo_type = self.get_command_type_mapping()[self.servo_type]
        if isinstance(self.servo_backend, int):
            self.servo_backend = self.get_servo_backend_mapping()[self.servo_backend]

    @staticmethod
    def get_command_type_mapping():
        return bidict(
            {
                0: ServoCommandTypes.JOINT_POSITION,
                1: ServoCommandTypes.JOINT_VELOCITY,
                2: ServoCommandTypes.POSE,
                3: ServoCommandTypes.TWIST,
            }
        )

    @staticmethod
    def get_servo_backend_mapping():
        return bidict(
            {
                0: ControllerTypes.FORWARD_POSITION,
                1: ControllerTypes.TRAJECTORY,
            }
        )


@dataclass
class ForwardPositionParams:
    """Parameters for forward position control."""

    max_velocity: float = 0.1


@dataclass
class TrajectoryParams:
    """Parameters for trajectory control."""

    start_state: Optional[JointState] = None
    planning_pipelines: List[str] = field(default_factory=lambda: ["ompl"])
    max_velocity_scaling_factor: float = 1.0
    max_acceleration_scaling_factor: float = 1.0
    allowed_start_tolerance: float = 0.1
    allowed_goal_duration_margin: float = 1.5
    allowed_execution_duration_scaling: float = 2.0
    interpolate_space: str = "joint"  # cartesian, joint
    execute: bool = True
    wait: bool = True


@dataclass
class TrackingParams:
    """Parameters for tracking control."""

    max_velocity: float = 0.1
    max_effort: float = 0.1


@dataclass
class ParallelGripperParams:
    """Parameters for parallel gripper control."""

    goal_tolerance: float = 0.01
    # Allow stalling will make the action server return success
    # if the gripper stalls when moving to the goal
    allow_stalling: bool = False
    stall_timeout: float = 1.0
    stall_velocity_threshold: float = 0.001
    max_velocity: float = 0.1
    max_effort: float = 10.0
    wait: bool = False


class NavigationMode(Enum):
    Free = 0
    StrictVirtualTrack = 1
    PriorityVirtualTrack = 2
    FollowPathPoints = 3
    # auto set by the backward param
    ReverseWalk = 4
    StrictVirtualTrackReverseWalk = 5


class MoveParams(Enum):
    NoParam = 0
    Appending = 1
    NoSmooth = 4
    Precise = 16
    WithYaw = 32
    ReturnUnreachableDirectly = 64
    WithFailRetryCount = 512
    FindPathIgnoringDynamicObstacles = 1024
    WithDirectedVirtualTrack = 2048


@dataclass
class BaseControlParams:
    """Parameters for base control."""

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0
    wait: bool = False
    backward: bool = False
    navigation_mode: int = NavigationMode.Free.value
    move_params: List[str] = field(default_factory=lambda: [])
    speed_ratio: float = 1.0
    fail_retry_count: int = 0


@dataclass
class BuildMapParams:
    """Parameters for base build map."""

    move_to_origin: bool = False
    stop: bool = False


@dataclass
class BaseChargeStationParams:
    """Parameters for base dock."""

    navigation_mode: int = NavigationMode.Free.value
    move_to_dock: bool = False


@dataclass
class TeleopBagParams:
    """Parameters for teleoperation bag."""

    stop: bool = False


@dataclass
class TeleopVRParams:
    """Parameters for teleoperation VR."""

    spine_mode: int = 0
    stop: bool = False


@dataclass
class TeleopJoystickParams:
    """Parameters for teleoperation joystick."""

    initial_component: MMK2Components = MMK2Components.LEFT_ARM
    stop: bool = False


ParamType = Union[
    TrajectoryParams,
    MoveServoParams,
    ManualParams,
    RecordParams,
    ReplayParams,
    BaseControlParams,
    BuildMapParams,
    BaseChargeStationParams,
    None,
]
