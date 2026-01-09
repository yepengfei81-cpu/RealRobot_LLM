from __future__ import annotations

import threading
import time
from threading import Lock

import grpc
import functools
from collections.abc import Sequence

from airbot_py.airbot_grpc import airbot_play_pb2, airbot_play_pb2_grpc, common_pb2


def check_valid(timeout: float = 0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance, *args, **kwargs):
            if not instance.valid(timeout):
                return None
            return func(instance, *args, **kwargs)

        return wrapper

    return decorator


class AirbotPlay:
    """
    Airbot Play class to control the robot arm
    """

    def __init__(self, ip: str = "localhost", port: int = 50051) -> None:
        """
        Create a new AirbotPlay object

        Args:
            ip (str): IP address of the server
            port (int): Port number of the server
        """
        self.ip = ip
        self.port = port
        self.params: common_pb2.ParamList | None = None
        self.state: airbot_play_pb2.RobotState | None = None

        self.servo_type = None  # Use None to indicate no new command
        self.servo_arm_target = None
        self.servo_gripper_target = None
        self.servo_mutex = Lock()
        self.state_update_time = None
        self.grpc_stub = airbot_play_pb2_grpc.AirbotPlayServiceStub(
            grpc.insecure_channel(f"{self.ip}:{self.port}")
        )
        self.stop_event = threading.Event()
        self.get_thread = threading.Thread(target=self._get)
        self.get_thread.start()
        self.set_thread = threading.Thread(target=self._set, args=(100,))
        self.set_thread.start()

        self.params = self.grpc_stub.CreateRobot(self._get_a_header()).params

    def valid(self, timeout=0.1) -> bool:
        return (
            self.state is not None
            and self.state.is_valid
            and time.time() - self.state_update_time < timeout
        )

    def initialize(self):
        pass

    def shutdown(self):
        self.stop_event.set()
        self.get_thread.join()
        self.set_thread.join()

    def __enter__(self):
        ### Blocking context manager
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def _get(self):
        it = self.grpc_stub.GetRobotState(self._get_a_header())
        while not self.stop_event.is_set():
            self.state = next(it)
            self.state_update_time = time.time()

    def _get_a_header(self) -> common_pb2.Header:
        current_time = time.time_ns()
        return common_pb2.Header(
            stamp=common_pb2.Time(
                sec=current_time // int(1e9), nanosec=current_time % int(1e9)
            ),
            caller_id="airbot_play_py",
        )

    def _set(self, freq: int = 100):
        self.grpc_stub.MoveServo(self._gen_servo_params(freq))

    def _gen_servo_params(self, freq: int = 100):
        while not self.stop_event.is_set():
            time.sleep(1 / freq)
            with self.servo_mutex:
                if self.servo_type is None:
                    continue
                if (
                    self.servo_arm_target is not None
                    and self.servo_gripper_target is not None
                ):
                    params = airbot_play_pb2.MoveServoParams(
                        header=self._get_a_header(),
                        servo_type=self.servo_type,
                        servo_arm_target=self.servo_arm_target,
                        servo_gripper_target=self.servo_gripper_target,
                    )
                elif self.servo_arm_target is not None:
                    params = airbot_play_pb2.MoveServoParams(
                        header=self._get_a_header(),
                        servo_type=self.servo_type,
                        servo_arm_target=self.servo_arm_target,
                    )
                elif self.servo_gripper_target is not None:
                    params = airbot_play_pb2.MoveServoParams(
                        header=self._get_a_header(),
                        servo_type=self.servo_type,
                        servo_gripper_target=self.servo_gripper_target,
                    )
            self.servo_arm_target = None
            self.servo_gripper_target = None
            self.servo_type = None
            yield params

    def trigger_log(self) -> bool:
        result = self.grpc_stub.TriggerLog(self._get_a_header())
        return result.value == common_pb2.GoalStatus.SUCCESS

    def get_system_params(
        self,
        params_name_list: Sequence[str] = (
            "ArmElement",
            "FirmwareVersion",
            "HardwareVersion",
            "SN",
            "ArmSN",
            "ProductFlag",
            "ErrorList",
            "arm_type",
            "end_effector",
            "arm_sn_code",
        ),
    ) -> dict:
        """
        Fetch system parameters via a gRPC call to the GetSystemParams method.

        Args:
        params_name_list (list[str]): A list of strings containing the names of the system parameters to fetch.
            Available values:   [
                                    "ArmElement", "FirmwareVersion", "HardwareVersion",\
                                    "SN", "ArmSN", "ProductFlag", "ErrorList", "arm_type",\
                                    "end_effector","arm_sn_code"
                                ]

        Returns:
        dict: A dictionary containing the requested system parameters.
        """
        result = self.grpc_stub.GetSystemParams(
            common_pb2.GetSystemParamRequest(
                header=self._get_a_header(), param_names=params_name_list
            )
        ).params
        return result

    def launch_robot(self) -> bool:
        """Launch the robot.

        Returns:
            bool: the result of the action
        """
        params = self.grpc_stub.LaunchRobot(self._get_a_header())
        if len(params.params) == 0:
            return False
        else:
            self.params = params.params
            return True

    def calibrate_robot(self) -> bool:
        """Calibrate the zero position of the robot.

        Returns:
            bool: the result of the action
        """
        result = self.grpc_stub.CalibrateRobot(self._get_a_header())
        return result.value == common_pb2.GoalStatus.SUCCESS

    def calibrate_confirm(self) -> bool:
        """Confirm the calibration of the robot.

        Returns:
            bool: the result of the action
        """
        result = self.grpc_stub.CalibrateConfirm(self._get_a_header())
        return result.value == common_pb2.GoalStatus.SUCCESS

    def shutdown_robot(self) -> bool:
        """Shut down the robot.

        Returns:
            bool: the result of the action
        """
        result = self.grpc_stub.ShutdownRobot(self._get_a_header())
        return result.value == common_pb2.GoalStatus.SUCCESS

    def iap(self) -> bool:
        """Prepare to update the robot firmware.

        Currently this method just power on the robot.

        Returns:
            bool: the result of the action
        """
        result = self.grpc_stub.IAP(self._get_a_header())
        return result.value == common_pb2.GoalStatus.SUCCESS

    def set_robot_mode(
        self, mode: airbot_play_pb2.AirbotMode, param: int | None = None
    ) -> bool:
        """Set the robot mode.

        Args:
            mode (airbot_play_pb2.AirbotMode): The mode to set the robot to.
            param (int): tbd

        Returns:
            bool: the result of the action
        """
        result = self.grpc_stub.SetRobotMode(
            airbot_play_pb2.SetRobotModeRequest(
                header=self._get_a_header(), target_mode=mode, param=param
            )
        )
        return result.value == common_pb2.GoalStatus.SUCCESS

    def move_to_postion(
        self, arm_waypoints: list[airbot_play_pb2.ArmWayPoint], wait: bool = True
    ) -> common_pb2.GoalStatus:
        """Move the robot to a given position.

        Args:
            arm_waypoints (list[airbot_play_pb2.ArmWayPoint]): A list of waypoints to move the robot to.
            wait (bool, optional): Blocking call. Defaults to True.

        Returns:
            common_pb2.GoalStatus: the result of the action
        """
        result = self.grpc_stub.MoveToPosition(
            airbot_play_pb2.MoveToPositionRequest(
                header=self._get_a_header(), arm_waypoints=arm_waypoints, wait=wait
            )
        )
        return result

    ### API wrappers for getter

    # @check_valid(timeout=0.1)
    def get_current_end(self) -> float | None:
        """
        Get current gripper end.

        Returns:
            float: Current gripper end. Range from 0 to 1. 0 is close, 1 is open.
        """
        return (
            self.state.gripper_states.position[0]
            if (
                self.params.get("eef_type", None) is not None
                and self.params["eef_type"] != ""
            )
            else None
        )

    # @check_valid(timeout=0.1)
    def get_current_end_v(self) -> float | None:
        """
        Get current gripper end velocity.

        Returns:
            float: Current gripper end velocity. In radian/s.
        """
        return (
            self.state.gripper_states.velocity[0]
            if (
                self.params.get("eef_type", None) is not None
                and self.params["eef_type"] != ""
            )
            else None
        )

    # @check_valid(timeout=0.1)
    def get_current_end_t(self) -> float | None:
        """
        Get current gripper end torque.

        Returns:
            float: Current gripper end torque. In Nm.
        """
        return (
            self.state.gripper_states.effort[0]
            if (
                self.params.get("eef_type", None) is not None
                and self.params["eef_type"] != ""
            )
            else None
        )

    # @check_valid(timeout=0.1)
    def get_current_end_temperature(self) -> float | None:
        """
        Get current gripper end temperature.

        Returns:
            float: Current gripper end temperature. In Celsius.
        """
        return (
            self.state.gripper_states.temperature[0]
            if (
                self.params.get("eef_type", None) is not None
                and self.params["eef_type"] != ""
            )
            else None
        )

    # @check_valid(timeout=0.1)
    def get_current_end_error_code(self) -> int | None:
        """
        Get current gripper end error code.

        Returns:
            int: Current gripper end error code.
        """
        return (
            self.state.gripper_states.error_code[0]
            if (
                self.params.get("eef_type", None) is not None
                and self.params["eef_type"] != ""
            )
            else None
        )

    # @check_valid(timeout=0.1)
    def get_current_joint_error_code(self) -> list[int]:
        """
        Get current joint error code.

        Returns:
            List[int]: Current joint error code. Size of 6.
        """
        return self.state.joint_states.error_code

    # @check_valid(timeout=0.1)
    def get_current_joint_q(self) -> list[float]:
        """
        Get current joint motor radian.

        Returns:
            List[float]: Current joint motor radian. Size of 6. In radian.
        """
        return self.state.joint_states.position

    # @check_valid(timeout=0.1)
    def get_current_joint_t(self) -> list[float]:
        """
        Get current joint motor torque.

        Returns:
            List[float]: Current joint motor torque. Size of 6.
        """
        return self.state.joint_states.effort

    # @check_valid(timeout=0.1)
    def get_current_joint_temperature(self) -> list[float]:
        """
        Get current joint motor temperature.

        Returns:
            List[float]: Current joint motor temperature. Size of 6. In Celsius.
        """
        return self.state.joint_states.temperature

    # @check_valid(timeout=0.1)
    def get_current_joint_v(self) -> list[float]:
        """
        Get current joint motor velocity.

        Returns:
            List[float]: Current joint motor velocity. Size of 6. In radian/s.
        """
        return self.state.joint_states.velocity

    # @check_valid(timeout=0.1)
    def get_current_pose(self) -> list[list[float]]:
        """
        Get current pose.

        Returns:
            List[List[float]]: Current pose. [[x, y, z], [x, y, z, w]]. In meter and quaternion.
        """
        return [
            [
                self.state.eef_pose.position.x,
                self.state.eef_pose.position.y,
                self.state.eef_pose.position.z,
            ],
            [
                self.state.eef_pose.orientation.x,
                self.state.eef_pose.orientation.y,
                self.state.eef_pose.orientation.z,
                self.state.eef_pose.orientation.w,
            ],
        ]

    # @check_valid(timeout=0.1)
    def get_current_rotation(self) -> list[float]:
        """
        Get current rotation.

        Returns:
            List[float]: Current rotation. [x, y, z, w]. Quaternion.
        """
        return [
            self.state.eef_pose.orientation.x,
            self.state.eef_pose.orientation.y,
            self.state.eef_pose.orientation.z,
            self.state.eef_pose.orientation.w,
        ]

    # @check_valid(timeout=0.1)
    def get_current_translation(self) -> list[float]:
        """
        Get current translation.

        Returns:
            List[float]: Current translation. [x, y, z]. In meter.
        """
        return [
            self.state.eef_pose.position.x,
            self.state.eef_pose.position.y,
            self.state.eef_pose.position.z,
        ]

    # @check_valid(timeout=0.1)
    def get_sn(self) -> str:
        """
        Get robot serial number.

        Returns:
            str: Serial number.
        """
        return self.params.get("SN", "")

    # @check_valid(timeout=0.1)
    def get_current_state(self) -> str:
        """
        Get current state.

        Returns:
            str: Current state. Can be "OFFLINE", "ONLINE_IDLE", "ONLINE_SERVO", "ONLINE_TRAJ", "DEMONSTRATE_PREP", \
                "DEMONSTRATING", "REPLAY_WAITING", "REPLAY_REACHING", "REPLAYING", "REPLAY_PAUSED", "SLAVE_WAITING", \
                "SLAVE_REACHING", "SLAVE_MOVING", "LOW_LEVEL", "MANUAL".
        """
        # return str(self.state.current_state)
        return airbot_play_pb2.AirbotMode.Name(self.state.current_state)

    ### API wrappers for setter

    def set_target_end(self, target_end: float, blocking: bool = False) -> bool:
        """
        Set target end.

        Args:
            target_end (float): Target end. Range from 0 to 1. 0 is close, 1 is open.
            blocking (bool, optional): Wait for the set target end moving finish. Defaults to False.


        Returns:
            bool: True if the set target end request send successful when wait is False or the set target end moving \
                finish when wait is True.
        """
        if self.params.get("eef_type", None) is None or self.params["eef_type"] == "":
            print("No gripper.")
            return False
        wait = blocking  # fit to 2.9
        gripper_target = airbot_play_pb2.GripperTarget(target_position=[target_end])
        if wait:
            arm_waypoint = airbot_play_pb2.ArmWayPoint(gripper_target=gripper_target)
            result = self.move_to_postion([arm_waypoint], wait=wait).value
            current_end = self.get_current_end()
            assert current_end is not None
            while abs(current_end - target_end) > 0.1:
                current_end = self.get_current_end()
                assert current_end is not None
                time.sleep(0.01)
            return result == common_pb2.GoalStatus.SUCCESS
        else:
            self.servo_mutex.acquire()
            self.servo_gripper_target = gripper_target
            self.servo_type = airbot_play_pb2.MoveServoParams.JOINT_POSITION
            self.servo_mutex.release()
            return True

    def set_target_joint_q(
        self,
        target_joint_q: list[float],
        use_planning: bool = True,
        vel: float = 0.2,
        blocking: bool = True,
        acceleration: float = 0.2,
    ) -> bool:
        """
        Set target joint motor radian.

        Args:
            target_joint_q (List[float]): Target joint motor radian. Size of 6. In radian.
            use_planning (bool, optional): Use planning or not. Defaults to True.
            vel (float, optional): Max velocity scaling factor. Defaults to 0.2. Only support in ONLINE_TRAJ mode \
                temporarily.
            blocking (bool, optional): Wait for the set target joint motor radian moving finish. Defaults to True.
            acceleration (float, optional): Max acceleration scaling factor. Defaults to 0.2. Only support in \
                ONLINE_TRAJ mode temporarily.

        Returns:
            bool: True if the set target joint motor radian request send successful when wait is False or the set \
                target joint motor radian moving finish when wait is True.
        """
        wait = blocking  # fit to 2.9
        velocity = vel
        if use_planning or blocking:  # trajectory planning
            arm_target = airbot_play_pb2.ArmTarget(
                target_joint_value=target_joint_q,
            )
            arm_waypoint = airbot_play_pb2.ArmWayPoint(
                arm_target=arm_target, speed_scale=velocity, acc_scale=acceleration
            )
            result = self.move_to_postion([arm_waypoint], wait=wait).value
            return result == common_pb2.GoalStatus.SUCCESS
        # servo
        self.servo_mutex.acquire()
        self.servo_arm_target = airbot_play_pb2.ArmTarget(
            target_joint_value=target_joint_q,
        )
        self.servo_type = airbot_play_pb2.MoveServoParams.JOINT_POSITION
        self.servo_mutex.release()
        return True

    def set_target_joint_q_waypoints(
        self,
        waypoints: list[list[float]],
        blocking: bool = True,
        vel: float = 0.2,
        acceleration: float = 0.2,
    ) -> bool:
        """
        Set target joint motor radian waypoints.

        Args:
            waypoints (List[List[float]]): Target joint motor radian waypoints. Size of 6. In radian.
            blocking (bool, optional): Wait for the set target joint motor radian waypoints moving finish. Defaults \
                to True.
            vel (float, optional): Max velocity scaling factor. Defaults to 0.2. Only support in ONLINE_TRAJ mode \
                temporarily.
            acceleration (float, optional): Max acceleration scaling factor. Defaults to 0.2. Only support in \
                ONLINE_TRAJ mode temporarily.

        Returns:
            bool: True if the set target joint motor radian waypoints request send successful when wait is False or \
                the set target joint motor radian waypoints moving finish when wait is True.
        """
        wait = blocking  # fit to 2.9
        velocity = vel
        arm_waypoints = []
        for waypoint in waypoints:
            arm_target = airbot_play_pb2.ArmTarget(
                target_joint_value=waypoint,
            )
            arm_waypoint = airbot_play_pb2.ArmWayPoint(
                arm_target=arm_target, speed_scale=velocity, acc_scale=acceleration
            )
            arm_waypoints.append(arm_waypoint)
        result = self.move_to_postion(arm_waypoints, wait=wait).value
        return result == common_pb2.GoalStatus.SUCCESS

    def set_target_pose(
        self,
        target_pose: list[list[float]],
        use_planning: bool = True,
        vel: float = 0.2,
        blocking: bool = False,
        acceleration: float = 0.2,
    ) -> bool:
        """
        Set target pose.

        Args:
            target_pose (List[List[float]]): Target pose. [[x, y, z], [x, y, z, w]]. In meter and quaternion.
            use_planning (bool, optional): Use planning or not. Defaults to True.
            vel (float, optional): Max velocity scaling factor. Defaults to 0.2. Only support in ONLINE_TRAJ mode \
                temporarily.
            blocking (bool, optional): Wait for the set target pose moving finish. Defaults to True.
            acceleration (float, optional): Max acceleration scaling factor. Defaults to 0.2. Only support in\
                ONLINE_TRAJ mode temporarily.

        Returns:
            bool: True if the set target pose request send successful when wait is False or the set target pose \
                moving finish when wait is True.
        """
        wait = blocking  # fit to 2.9
        velocity = vel  # fit to 2.9
        if use_planning or blocking:  # trajectory planning
            pose = common_pb2.Pose(
                position=common_pb2.Position(
                    x=target_pose[0][0], y=target_pose[0][1], z=target_pose[0][2]
                ),
                orientation=common_pb2.Orientation(
                    x=target_pose[1][0],
                    y=target_pose[1][1],
                    z=target_pose[1][2],
                    w=target_pose[1][3],
                ),
                pose_link="end_link",
                reference_link="base_link",
            )
            arm_target = airbot_play_pb2.ArmTarget(
                target_pose=pose,
            )
            arm_waypoint = airbot_play_pb2.ArmWayPoint(
                arm_target=arm_target, speed_scale=velocity, acc_scale=acceleration
            )
            result = self.move_to_postion([arm_waypoint], wait=wait).value
            return result == common_pb2.GoalStatus.SUCCESS
        # servo
        self.servo_mutex.acquire()
        self.servo_arm_target = airbot_play_pb2.ArmTarget(
            target_pose=common_pb2.Pose(
                position=common_pb2.Position(
                    x=target_pose[0][0], y=target_pose[0][1], z=target_pose[0][2]
                ),
                orientation=common_pb2.Orientation(
                    x=target_pose[1][0],
                    y=target_pose[1][1],
                    z=target_pose[1][2],
                    w=target_pose[1][3],
                ),
                pose_link="end_link",
                reference_link="base_link",
            ),
        )
        self.servo_type = airbot_play_pb2.MoveServoParams.CART_POSITION
        self.servo_mutex.release()
        return True

    def set_target_pose_waypoints(
        self,
        waypoints: list[list[list[float]]],
        blocking: bool = True,
        vel: float = 0.25,
        acceleration: float = 0.25,
    ) -> bool:
        """
        Set target pose waypoints.

        Args:
            waypoints (List[List[List[float]]]): Target pose waypoints. [[x, y, z], [x, y, z, w]]. In meter and \
                quaternion.
            blocking (bool, optional): Wait for the set target pose waypoints moving finish. Defaults to True.
            vel (float, optional): Max velocity scaling factor. Defaults to 0.25. Only support in ONLINE_TRAJ mode \
                temporarily.
            acceleration (float, optional): Max acceleration scaling factor. Defaults to 0.25. Only support in \
                ONLINE_TRAJ mode temporarily.

        Returns:
            bool: True if the set target pose waypoints request send successful when wait is False or the set target \
                pose waypoints moving finish when wait is True.
        """
        wait = blocking  # fit to 2.9
        velocity = vel
        arm_waypoints = []
        for waypoint in waypoints:
            pose = common_pb2.Pose(
                position=common_pb2.Position(
                    x=waypoint[0][0], y=waypoint[0][1], z=waypoint[0][2]
                ),
                orientation=common_pb2.Orientation(
                    x=waypoint[1][0],
                    y=waypoint[1][1],
                    z=waypoint[1][2],
                    w=waypoint[1][3],
                ),
                pose_link="end_link",
                reference_link="base_link",
            )
            arm_target = airbot_play_pb2.ArmTarget(
                target_pose=pose,
            )
            arm_waypoint = airbot_play_pb2.ArmWayPoint(
                arm_target=arm_target, speed_scale=velocity, acc_scale=acceleration
            )
            arm_waypoints.append(arm_waypoint)
        result = self.move_to_postion(arm_waypoints, wait=wait).value
        return result == common_pb2.GoalStatus.SUCCESS

    def set_target_rotation(
        self,
        target_rotation: list[float],
        use_planning: bool = True,
        vel: float = 0.2,
        blocking: bool = True,
        acceleration: float = 0.2,
    ) -> bool:
        """
        Set target rotation.

        Args:
            target_rotation (List[float]): Target rotation. [x, y, z, w]. Quaternion.
            use_planning (bool, optional): Use planning or not. Defaults to True.
            vel (float, optional): Max velocity scaling factor. Defaults to 0.2. Only support in ONLINE_TRAJ mode \
                temporarily.
            blocking (bool, optional): Wait for the set target rotation moving finish. Defaults to True.
            acceleration (float, optional): Max acceleration scaling factor. Defaults to 0.2. Only support in \
                ONLINE_TRAJ mode temporarily.

        Returns:
            bool: True if the set target rotation request send successful when wait is False or the set target \
                rotation moving finish when wait is True.
        """
        return self.set_target_pose(
            [self.get_current_translation(), target_rotation],
            use_planning=use_planning,
            vel=vel,
            blocking=blocking,
            acceleration=acceleration,
        )

    def set_target_translation(
        self,
        target_translation: list[float],
        use_planning: bool = True,
        vel: float = 0.2,
        blocking: bool = True,
        acceleration: float = 0.2,
    ) -> bool:
        """
        Set target translation.

        Args:
            target_translation (List[float]): Target translation. [x, y, z]. In meter.
            use_planning (bool, optional): Use planning or not. Defaults to True.
            vel (float, optional): Max velocity scaling factor. Defaults to 0.2. Only support in ONLINE_TRAJ mode \
                temporarily.
            blocking (bool, optional): Wait for the set target translation moving finish. Defaults to True.
            acceleration (float, optional): Max acceleration scaling factor. Defaults to 0.2. Only support in \
                ONLINE_TRAJ mode temporarily.

        Returns:
            bool: True if the set target translation request send successful when wait is False or the set target \
                translation moving finish when wait is True.
        """
        return self.set_target_pose(
            [target_translation, self.get_current_rotation()],
            use_planning=use_planning,
            vel=vel,
            blocking=blocking,
            acceleration=acceleration,
        )

    ### API wrappers for mode changes

    def manual_mode(self) -> bool:
        """
        Enter manual mode. (From online idle mode or offline idle mode enter demonstate prep mode.)

        Returns:
            bool : True if the manual mode request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.DEMONSTRATE_PREP:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.ONLINE_IDLE:
            if not self.set_robot_mode(airbot_play_pb2.AirbotMode.ONLINE_IDLE):
                return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.DEMONSTRATE_PREP)

    def offline_mode(self) -> bool:
        """
        Enter offline mode. (From online idle mode or manual idle mode enter replay waiting mode.)

        Returns:
            bool: True if the offline mode request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAY_WAITING:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.ONLINE_IDLE:
            if not self.set_robot_mode(airbot_play_pb2.AirbotMode.ONLINE_IDLE):
                return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.REPLAY_WAITING)

    def online_mode(self) -> bool:
        """
        Enter online mode. (Enter online idle mode.)

        Returns:
            bool: True if the online mode request send successful.
        """
        if self.state.current_state != airbot_play_pb2.AirbotMode.ONLINE_IDLE:
            return self.set_robot_mode(airbot_play_pb2.AirbotMode.ONLINE_IDLE)
        return True

    def slave_mode(self, master_domain_id: int) -> bool:
        """
        Enter slave mode. (From online idle mode enter slave waiting mode.)

        Returns:
            bool: True if the slave mode request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.SLAVE_WAITING:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.ONLINE_IDLE:
            if not self.set_robot_mode(airbot_play_pb2.AirbotMode.ONLINE_IDLE):
                return False
        return self.set_robot_mode(
            airbot_play_pb2.AirbotMode.SLAVE_WAITING, param=master_domain_id
        )

    def record_start(self) -> bool:
        """
        Start record trajectory. Be sure you are in demonstrate prep mode. Will switch from demonstrate prep mode to \
            demonstrating mode.

        Returns:
            bool: True if the record start request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.DEMONSTRATING:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.DEMONSTRATE_PREP:
            print("Please enter manual mode first.")
            return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.DEMONSTRATING)

    def record_stop(self) -> bool:
        """
        Stop record trajectory. Be sure you are in demonstrating mode. Will switch from demonstrating mode to \
            demonstrate prep mode.

        Returns:
            bool: True if the record stop request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.DEMONSTRATE_PREP:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.DEMONSTRATING:
            print("Please start record first.")
            return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.DEMONSTRATE_PREP)

    def replay_start(self) -> bool:
        """
        Start replay. Be sure you are in offline mode.

        Returns:
            bool: True if the replay start request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAYING:
            return True
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAY_WAITING:
            if not self.set_robot_mode(airbot_play_pb2.AirbotMode.REPLAY_REACHING):
                return False
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAY_REACHING:
            start_time = time.time()
            while self.state.current_state != airbot_play_pb2.AirbotMode.REPLAY_REACHED:
                if time.time() - start_time > 5:
                    print("Replay reaching timeout.")
                    return False
                time.sleep(0.1)
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAY_REACHED:
            return self.set_robot_mode(airbot_play_pb2.AirbotMode.REPLAYING)
        print("Please enter offline mode first.")
        return False

    def replay_pause(self) -> bool:
        """
        Pause replay. Be sure you are in replaying mode. Will switch from replaying mode to replay paused mode.

        Returns:
            bool: True if the replay pause request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAY_PAUSED:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.REPLAYING:
            print("Please start replay first.")
            return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.REPLAY_PAUSED)

    def replay_resume(self) -> bool:
        """
        Resume replay. Be sure you are in replay paused mode. Will switch from replay paused mode to replaying mode.

        Returns:
            bool: True if the replay resume request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.REPLAYING:
            return True
        if self.state.current_state != airbot_play_pb2.AirbotMode.REPLAY_PAUSED:
            print("Please pause replay first.")
            return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.REPLAYING)

    def follow_start(self) -> bool:
        """
        Start follow. Be sure you are in offline mode.

        Returns:
            bool: True if the follow start request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.SLAVE_MOVING:
            return True
        if self.state.current_state == airbot_play_pb2.AirbotMode.SLAVE_WAITING:
            if not self.set_robot_mode(airbot_play_pb2.AirbotMode.SLAVE_REACHING):
                return False
        if self.state.current_state == airbot_play_pb2.AirbotMode.SLAVE_REACHING:
            start_time = time.time()
            while self.state.current_state != airbot_play_pb2.AirbotMode.SLAVE_REACHED:
                if time.time() - start_time > 5:
                    print("Follow reaching timeout.")
                    return False
                time.sleep(0.1)
        if self.state.current_state == airbot_play_pb2.AirbotMode.SLAVE_REACHED:
            return self.set_robot_mode(airbot_play_pb2.AirbotMode.SLAVE_MOVING)
        print("Please enter offline mode first.")
        return False

    def follow_stop(self) -> bool:
        """
        Stop follow. Be sure you are in slave moving mode. Will switch from slave moving mode to slave waiting mode.

        Returns:
            bool: True if the follow stop request send successful.
        """
        if self.state.current_state == airbot_play_pb2.AirbotMode.ONLINE_IDLE:
            return True
        if self.state.current_state not in [
            airbot_play_pb2.AirbotMode.SLAVE_MOVING,
            airbot_play_pb2.AirbotMode.SLAVE_MOVING_ADJUST,
        ]:
            print("Please start follow first.")
            return False
        return self.set_robot_mode(airbot_play_pb2.AirbotMode.ONLINE_IDLE)


def create_agent(ip: str = "localhost", port: int = 50051) -> AirbotPlay:
    """
    Create an AirbotPlay object

    Args:
        ip (str): IP address of the server
        port (int): Port number of the server

    Returns:
        AirbotPlay: AirbotPlay object
    """
    return AirbotPlay(ip, port)
