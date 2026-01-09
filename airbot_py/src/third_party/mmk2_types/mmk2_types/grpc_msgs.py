from typing import Union
from airbot_grpc.mmk2_pb2 import *
from airbot_grpc.common_pb2 import *
from airbot_grpc.airbot_play_pb2 import MoveServoParams

GoalMsg = Union[Pose, JointState, Twist, Pose3D, Twist3D, str, float, None]
ParamMsg = Union[
    TrajectoryParams,
    MoveServoParams,
    TrackingParams,
    ParamList,
    RecordParams,
    ReplayParams,
    BaseControlParams,
    BuildMapParams,
    BaseChargeStationParams,
    Pose3D,
    Twist3D,
]
