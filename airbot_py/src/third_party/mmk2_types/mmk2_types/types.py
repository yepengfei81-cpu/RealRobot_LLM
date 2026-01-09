from dataclasses import dataclass, field
from typing import List, Union, Dict, Set
import numpy as np
from enum import Enum


@dataclass
class Time:
    sec: int = 0
    nanosec: int = 0


@dataclass
class PosePro:
    position: List[float] = field(default_factory=list)
    orientation: List[float] = field(default_factory=list)
    pose_link: str = ""
    reference_link: str = ""
    stamp: Time = field(default_factory=Time)


class SystemParameters(Enum):
    ROBOT_NAME = "robot_name"
    DOMAIN_ID = "domain_id"
    VERSION = "version"
    ROBOT_ID = "robot_id"


class ComponentKind(Enum):
    """Component kind, use '-' to separate multi words instead of '_'."""

    ARM = "arm"
    EEF = "eef"
    GPIO = "gpio"
    CAMERA = "camera"
    HEAD = "head"
    SPINE = "spine"
    BASE = "base"
    OTHER = "other"


class ComponentPos(Enum):
    LEFT = "left"
    RIGHT = "right"


class MMK2Components(Enum):
    LEFT_ARM = "_".join([ComponentPos.LEFT.value, ComponentKind.ARM.value])
    RIGHT_ARM = "_".join([ComponentPos.RIGHT.value, ComponentKind.ARM.value])
    LEFT_ARM_EEF = "_".join([LEFT_ARM, ComponentKind.EEF.value])
    RIGHT_ARM_EEF = "_".join([RIGHT_ARM, ComponentKind.EEF.value])
    LEFT_ARM_GPIO = "_".join([LEFT_ARM, ComponentKind.GPIO.value])
    RIGHT_ARM_GPIO = "_".join([RIGHT_ARM, ComponentKind.GPIO.value])
    HEAD = ComponentKind.HEAD.value
    HEAD_GPIO = "_".join([HEAD, ComponentKind.GPIO.value])
    SPINE = ComponentKind.SPINE.value
    SPINE_GPIO = "_".join([SPINE, ComponentKind.GPIO.value])
    BASE = ComponentKind.BASE.value
    # TODO: _ARM_CAMERA?
    LEFT_CAMERA = "_".join([ComponentPos.LEFT.value, ComponentKind.CAMERA.value])
    RIGHT_CAMERA = "_".join([ComponentPos.RIGHT.value, ComponentKind.CAMERA.value])
    HEAD_CAMERA = "_".join([HEAD, ComponentKind.CAMERA.value])
    # TODO: should use an embedded structure, e.g.
    # ARM has sub-components: eef, gpio
    # or maybe flattened is better which
    # hides the sub-components in the name prefix
    # and decoupling software from hardware components
    OTHER = ComponentKind.OTHER.value


# get the sub-component of a component, e.g. left_arm_eef is the sub-component of left_arm
COMPONENT_SUB: Dict[ComponentKind, Dict[MMK2Components, MMK2Components]] = {}
for comp in MMK2Components:
    removed = set(MMK2Components)
    removed.remove(comp)
    for comp_r in removed:
        if comp.value in comp_r.value:
            sub_comp = ComponentKind(comp_r.value.replace(comp.value + "_", ""))
            if sub_comp not in COMPONENT_SUB:
                COMPONENT_SUB[sub_comp] = {}
            COMPONENT_SUB[sub_comp][comp] = comp_r

KIND_COMPONENTS: Dict[ComponentKind, Set[MMK2Components]] = {}
for comp in MMK2Components:
    kind = ComponentKind(comp.value.split("_")[-1])
    if kind not in KIND_COMPONENTS:
        KIND_COMPONENTS[kind] = set()
    KIND_COMPONENTS[kind].add(comp)
ONE_KIND_COMPONENTS: Set[MMK2Components] = {
    comps.copy().pop() for comps in KIND_COMPONENTS.values() if len(comps) == 1
}


class ComponentTypes(Enum):
    NONE = "none"
    EEF_COMMON = "eef"
    EEF_G2 = "G2"
    EEF_INSPIRE_HAND = "inspire_hand"
    ARM_PLAY_LONG = "play_long"
    ARM_PLAY_SHORT = "play_short"
    CAMERA_REALSENSE = "realsense"
    CAMERA_ORBBEC = "orbbec"
    UNKNOWN = "unknown"
    SPINE_DEV = "/dev/spine"
    HEAD = "head"
    BASE = "192.168.11.1"


@dataclass
class MMK2ComponentsGroup:  # TODO: change to Enum
    ARMS = (MMK2Components.LEFT_ARM, MMK2Components.RIGHT_ARM)
    EEFS = (MMK2Components.LEFT_ARM_EEF, MMK2Components.RIGHT_ARM_EEF)
    LEFT_ARM_AND_EEF = (MMK2Components.LEFT_ARM, MMK2Components.LEFT_ARM_EEF)
    RIGHT_ARM_AND_EEF = (MMK2Components.RIGHT_ARM, MMK2Components.RIGHT_ARM_EEF)
    ARMS_EEFS = ARMS + EEFS
    ARMS_HEAD = ARMS + (MMK2Components.HEAD,)
    ARMS_EEFS_HEAD = ARMS_EEFS + (MMK2Components.HEAD,)
    GPIOS = (
        MMK2Components.LEFT_ARM_GPIO,
        MMK2Components.RIGHT_ARM_GPIO,
        MMK2Components.HEAD_GPIO,
        MMK2Components.SPINE_GPIO,
    )
    JOINTS = ARMS_EEFS_HEAD + (MMK2Components.SPINE,)
    HEAD_SPINE = (MMK2Components.HEAD, MMK2Components.SPINE)
    CAMERAS = (
        MMK2Components.LEFT_CAMERA,
        MMK2Components.RIGHT_CAMERA,
        MMK2Components.HEAD_CAMERA,
    )


class ControllerTypes(Enum):
    NONE = "none"
    TRAJECTORY = "trajectory"
    FORWARD_POSITION = "forward_position"
    TRACKING = "tracking"
    SAFETY_PASSTHROUGH = "safety_passthrough"
    SOLO = ""  # e.g. for GPIO, EEF, etc.


class MMK2ControlModes(Enum):
    # a control mode can have multi-type controllers
    TRAJECTORY = ControllerTypes.TRAJECTORY.value
    FORWARD_POSITION = ControllerTypes.FORWARD_POSITION.value
    TRACKING = ControllerTypes.TRACKING.value
    SOLO = ControllerTypes.SOLO.value
    SERVO = "servo"
    BASIC = "basic"
    GPIO = "gpios"
    NONE = "none"


class ServoCommandTypes(Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    POSE = "pose"
    TWIST = "twist"


@dataclass
class JointNames:  # TODO: change to Enum
    left_arm: List[str] = field(
        default_factory=lambda: [f"left_arm_joint{i}" for i in range(1, 7)]
    )
    right_arm: List[str] = field(
        default_factory=lambda: [f"right_arm_joint{i}" for i in range(1, 7)]
    )
    left_arm_eef: List[str] = field(
        default_factory=lambda: ["left_arm_eef_gripper_joint"]
    )
    right_arm_eef: List[str] = field(
        default_factory=lambda: ["right_arm_eef_gripper_joint"]
    )
    spine: List[str] = field(default_factory=lambda: ["slide_joint"])
    head: List[str] = field(
        default_factory=lambda: [f"head_{pos}_joint" for pos in ["yaw", "pitch"]]
    )

    def __post_init__(self):
        for comp in self.__dict__.keys():
            MMK2Components(comp)


@dataclass
class TopicNames:  # TODO: change to Enum
    joint_state: str = "/joint_states"
    velocity: str = "/cmd_vel"
    command: str = "{prefix}/commands"
    controller_command: str = "{component}_{controller}_controller/commands"
    # TODO: change the format of tracking to fit controller_command
    tracking: str = "{component}_leader/joint_states"
    servo: str = "{component}_servo_node/delta_{servo_type}_cmds"
    image: str = (
        "/{component}/{kind}/image_raw"  # kind: color, depth, aligned_depth_to_color
    )
    joy: str = "/joy"

    @staticmethod
    def get_topic_controller_command(
        component: MMK2Components, controller: ControllerTypes
    ):
        return TopicNames.controller_command.format(
            component=component.value, controller=controller.value
        ).replace("__", "_")


@dataclass
class Pose3D:
    x: float = 0
    y: float = 0
    theta: float = 0


@dataclass
class Twist3D:
    x: float = 0
    y: float = 0  # useless for two-wheel robot
    omega: float = 0


@dataclass
class BaseState:
    pose: Pose3D = field(default_factory=Pose3D)
    velocity: Twist3D = field(default_factory=Twist3D)
    odometry: float = 0.0
    stamp: Time = field(default_factory=Time)


class ImageTypes(Enum):
    COLOR = "color"
    DEPTH = "depth"
    ALIGNED_DEPTH_TO_COLOR = "aligned_depth_to_color"


@dataclass
class Image:
    """Image data."""

    data: Dict[ImageTypes, np.ndarray] = field(default_factory=dict)
    stamp: Time = field(default_factory=Time)


GoalType_AIRBOT = Union[
    str,
    PosePro,
    Pose3D,
    Twist3D,
]


class GoalStatus:
    class Status(Enum):
        SUCCESS = 0
        INVALID_TARGET = 1
        INVALID_PARAM = 2
        INVALID_MODE = 3
        NEAR_SINGULARITY = 4
        COLLISION = 5
        TIMEOUT = 6
        UNKNOWN = 7

    def __init__(self, status: Union[int, Status] = Status.SUCCESS):
        if isinstance(status, int):
            status = self.Status(status)
        self._status = status

    @property
    def status(self) -> Status:
        return self._status

    @property
    def value(self) -> int:
        return self._status.value

    @property
    def success(self) -> bool:
        return bool(self)

    def __bool__(self) -> bool:
        return self.status == self.Status.SUCCESS

    # def __mul__(self, scalar) -> bool:
    #     return bool(bool(self) * scalar)

    # def __imul__(self, scalar) -> bool:
    #     return bool(bool(self) * scalar)


class MoveOrder(Enum):

    SAME = 0
    BY_KEY = 1


# @dataclass
# class Position:
#     x: float = 0
#     y: float = 0
#     z: float = 0


# @dataclass
# class Orientation:
#     x: float = 0
#     y: float = 0
#     z: float = 0
#     w: float = 0


# @dataclass
# class JointState:
#     name: List[str] = field(default_factory=list)
#     position: List[float] = field(default_factory=list)
#     velocity: List[float] = field(default_factory=list)
#     effort: List[float] = field(default_factory=list)
#     stamp: float = 0.0


# class RobotState:
#     """
#     Representation of a robot’s state. At the lowest level, a state is a collection of variables. Each variable has a name and can have position, velocity, acceleration associated to it. Often variables correspond to joint names as well. Operations are allowed at variable level, joint level and joint group level.
#     """

#     def __init__(self):
#         pass

#     def get_frame_transform(self, frame_id: str, reference_frame_id: str) -> np.ndarray:
#         """
#         Get the transformation matrix from the reference frame to the frame identified by frame_id. If frame_id was not found, frame_found is set to false and an identity transform is returned.

#         Parameters:
#             frame_id (str) – The id of the frame to get the transform for.
#             reference_frame_id (str) – The id of the reference frame.

#         Returns:
#             numpy.ndarray – The transformation matrix from the reference frame to the frame identified by frame_id.

#         """
#         pass

#     def get_joint_group_accelerations(self, joint_group_name: str) -> np.ndarray:
#         """
#         For a given group, get the acceleration values of the variables that make up the group.

#         Parameters:
#             joint_group_name (str) – The name of the joint model group to copy the accelerations for.

#         Returns:
#             numpy.ndarray – The accelerations of the joints in the joint model group.
#         """
#         pass

#     def get_joint_group_positions(self, joint_group_name: str) -> np.ndarray:
#         """
#         For a given group, get the position values of the variables that make up the group.

#         Parameters:
#             joint_group_name (str) – The name of the joint model group to copy the positions for.

#         Returns:
#             numpy.ndarray – The positions of the joints in the joint model group.
#         """
#         pass

#     def get_joint_group_velocities(self, joint_group_name: str) -> np.ndarray:
#         """
#         For a given group, get the velocity values of the variables that make up the group.

#         Parameters:
#             joint_group_name (str) – The name of the joint model group to copy the velocities for.

#         Returns:
#             numpy.ndarray – The velocities of the joints in the joint model group.
#         """
#         pass

#     def get_pose(self, link_name: str, reference_link_name: str) -> Pose:
#         """
#         Get the pose of the link identified by link_name in the reference frame of the model.

#         Parameters:
#             link_name (str) – The name of the link to get the pose for.
#             reference_link_name (str) – The name of the reference link.

#         Returns:
#             numpy.ndarray – The pose of the link in the reference frame of the model.
#         """
#         pass

#     @property
#     def joint_positions(self) -> dict:
#         """
#         Get the joint positions of all joint groups.

#         Returns:
#             dict – The joint positions of all joint groups.
#         """
#         pass

#     @property
#     def joint_velocities(self) -> dict:
#         """
#         Get the joint velocities of all joint groups.

#         Returns:
#             dict – The joint velocities of all joint groups.
#         """
#         pass

#     @property
#     def joint_accelerations(self) -> dict:
#         """
#         Get the joint accelerations of all joint groups.

#         Returns:
#             dict – The joint accelerations of all joint groups.
#         """
#         pass

#     def get_stamp(self, component: Optional[RobotComponents] = None) -> float:
#         """
#         Get the timestamp of the robot state.
#         If a component is specified, the timestamp of the component is returned,
#         otherwise the timestamp shared by arms, eefs and head will return.

#         Returns:
#             float – The timestamp of the robot state.
#         """
#         pass

#     def get_base_pose(self):
#         pass

#     def get_base_velocity(self):
#         pass


# @dataclass
# class ArmStatus:
#     """Status of the arm."""

#     error_codes: List[int] = field(default_factory=list)


# @dataclass
# class BaseStatus:
#     """Status of the base."""

#     is_building_map: bool = False
#     power: float = 0.0
#     error_codes: List[int] = field(default_factory=list)


# @dataclass
# class EndEffectorStatus:
#     """Status of the end-effector."""

#     error_codes: List[int] = field(default_factory=list)


# class RobotStatus:
#     """Status of the robot."""

#     has_error: bool = False
#     error_components: List[RobotComponents] = field(default_factory=list)
#     base_status: BaseStatus = BaseStatus()
#     left_arm_status: ArmStatus = ArmStatus()
#     right_arm_status: ArmStatus = ArmStatus()


# # class TrigerOptions(Enum):
# #     BASE_BUILD_MAP_START = 0
# #     BASE_BUILD_MAP_STOP = 1


# StatusType = Union[RobotStatus, ArmStatus, BaseStatus, EndEffectorStatus]
