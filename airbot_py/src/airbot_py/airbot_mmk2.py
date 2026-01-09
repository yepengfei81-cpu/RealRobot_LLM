from typing import List, Dict, Union, Optional, Callable, Any, Mapping
import time
import logging
import asyncio
from threading import Thread
import numpy as np
import cv2
import grpc

from mmk2_types.grpc_msgs import (
    ParamMsg,
    GoalMsg,
    Header,
    ParamList,
    GoalStatus,
    JointState,
    Param,
    ArrayStamped,
)
from mmk2_types.types import (
    SystemParameters,
    MMK2Components,
    Image,
    Time,
    ImageTypes,
)

from airbot_grpc.mmk2_pb2_grpc import MMK2ServiceStub
from airbot_grpc.mmk2_pb2 import (
    RobotState,
    GoalRepeated,
    Goal,
    GoalRequest,
    SetGoalRequest,
    GetImageResponse,
    GetImageRequest,
    Resources,
    EnableResourcesRequest,
    ListenToRequest,
    ListenToResponse,
)
from airbot_grpc import common_pb2
from collections import defaultdict


ResourcesType = Dict[MMK2Components, Dict[str, str]]
SingleGoalType = Union[GoalMsg, List[GoalMsg], int, float]
GoalType = Optional[Union[Mapping[MMK2Components, SingleGoalType], SingleGoalType]]


logging.basicConfig(level=logging.DEBUG)


class ClientBackend:
    event_loop = None

    def __init__(
        self,
        ip: str,
        port: int,
        name: Optional[str],
        domain_id: Optional[int],
    ):
        self.name = name if name is not None else "mmk2_ros2_interface"
        self.domain_id = domain_id if domain_id is not None else -1
        self._version = None
        self.ip = ip
        self.port = port
        self.logger = logging.getLogger(f"mmk2_client_{name}")
        self.robot_state = None
        self._topic_msg = {}
        self._stop_listen = True
        self._is_running = True
        self._listen_task = None
        self._streamed = {}
        self._stream_response = {}

        self._imdecode_cfg = {
            ImageTypes.COLOR: cv2.IMREAD_COLOR,
            ImageTypes.DEPTH: cv2.IMREAD_ANYDEPTH,
            ImageTypes.ALIGNED_DEPTH_TO_COLOR: cv2.IMREAD_ANYDEPTH,
        }

        self.stub_usual = MMK2ServiceStub(
            grpc.insecure_channel(f"{self.ip}:{self.port}")
        )
        self.create_robot()
        self.run_thread = Thread(target=asyncio.run, args=(self._run(),), daemon=True)
        self.run_thread.start()
        while self.robot_state is None:
            self.logger.info("Waiting for robot states...")
            time.sleep(0.2)
        self.logger.info(
            f"Created robot {self.name} with domain id {self.domain_id} and version {self._version}"
        )

    def create_robot(self) -> ParamList:
        self.logger.info(f"Creating robot: {self.name} with domain id {self.domain_id}")
        response: ParamList = self.stub_usual.create_robot(
            ParamList(
                params={
                    SystemParameters.ROBOT_NAME.value: self.name,
                    SystemParameters.DOMAIN_ID.value: str(self.domain_id),
                }
            )
        )
        self.domain_id = response.params[SystemParameters.DOMAIN_ID.value]
        self._version = response.params[SystemParameters.VERSION.value]
        return response

    async def get_robot_state(self) -> None:
        """
        Get the current state of the robot.

        Returns:
            RobotState – The current state of the robot.
        """
        req = self.get_header()
        features = self.stub_async.get_robot_state(req)
        last_stamp = time.time()
        ref_time = time.time()
        cnt = 0
        self.logger.info("Start getting robot state")
        async for feature in features:
            feature: RobotState
            # self.logger.info(
            #     f"{self.port}:current joint names {feature.joint_state.name}"
            # )
            # self.logger.info(
            #     f"{self.port}:current joint position {feature.joint_state.position}"
            # )
            joint_stamp = feature.stamp.sec + feature.stamp.nanosec / 1e9
            current_stamp = time.time()
            period = current_stamp - last_stamp
            last_stamp = current_stamp
            # self.logger.info(f"{self.port}: joint stamp {joint_stamp}")
            # self.logger.info(f"{self.port}: current stamp {current_stamp}")
            cnt += 1
            if current_stamp - ref_time >= 5:
                # self.logger.info(f"frequency: {cnt/5} Hz")
                # self.logger.info(f"period: {period}")
                # self.logger.info(f"delay (time unsync): {current_stamp - joint_stamp}")
                cnt = 0
                ref_time = current_stamp

            # self.logger.info(" ")
            self.robot_state = feature
            if not self._is_running:
                break
        self.logger.info(f"exit: get_robot_state")

    def set_goal(
        self,
        goal: GoalType,
        param: Union[Dict[MMK2Components, ParamMsg], ParamMsg],
    ) -> GoalStatus:
        # set for all components the same param
        if not goal and goal != 0:  # None or empty dict
            if isinstance(param, dict):
                goal = {comp: None for comp in param.keys()}
            else:
                goal = {MMK2Components.OTHER: None}
        elif not isinstance(goal, dict):
            goal = {MMK2Components.OTHER: goal}

        if not isinstance(param, dict):
            param_dic = {}
            for comp in goal:
                param_dic[comp] = param
            param = param_dic

        goals_map = {}
        for comp, g in goal.items():
            # construct goal and param for each component
            param_to_send = self._set_field(param[comp], Param())[0]
            if isinstance(g, (list, tuple)):
                goal_rep = GoalRepeated()
                name = self._set_field(g[0], Goal())[1]
                handle: list = getattr(goal_rep, name)
                handle.extend(g)
                goal_to_send = GoalRequest(goal_repeated=goal_rep, param=param_to_send)
                # input(f"goals_to_send: {goals_to_send}")
            else:
                if g is None:
                    goal_to_send = GoalRequest(param=param_to_send)
                else:
                    if isinstance(goal, int):
                        goal = float(goal)
                    goal_sgl: Goal = self._set_field(g, Goal())[0]
                    assert len(goal_sgl.ListFields()) > 0, "Goal is empty"
                    goal_to_send = GoalRequest(goal=goal_sgl, param=param_to_send)
                # input(f"goal_to_send: {goal_to_send}")
            goals_map[comp.value] = goal_to_send
            # input(f"param_to_send: {param_to_send}")

        header = self.get_header()
        goal_req = SetGoalRequest(goals_map=goals_map, header=header)
        self.goal_req = goal_req
        # start = time.time()
        goal_status = self.stub_usual.set_goal(goal_req)
        # self.logger.info(f"Set goal time: {time.time() - start}")
        return goal_status

    # def get_last_goal_status(self) -> GoalStatus:
    #     pass

    # def get_robot_status(self, component: Optional[MMK2Components] = None):
    #     pass

    # def set_system_parameters(self, params: SystemParameters) -> SystemParameters:
    #     pass

    def _to_str_comp(self, comp_dict: Dict[MMK2Components, Any]) -> Dict[str, Any]:
        return {comp.value: value for comp, value in comp_dict.items()}

    def _to_all_str(self, comp_dict: Dict[MMK2Components, Any]) -> Dict[str, str]:
        return {comp.value: value.value for comp, value in comp_dict.items()}

    def __get_get_image_request(
        self, comp_types: Dict[MMK2Components, List[ImageTypes]]
    ):
        comps = []
        types = []
        for comp, types_list in comp_types.items():
            for t in types_list:
                comps.append(comp.value)
                types.append(t.value)
        return GetImageRequest(components=comps, image_types=types)

    def get_image(
        self, comp_types: Dict[MMK2Components, List[ImageTypes]]
    ) -> Dict[MMK2Components, Image]:
        request = self.__get_get_image_request(comp_types)
        fn = self.get_image.__name__
        if fn not in self._streamed:
            response: GetImageResponse = self.stub_usual.get_image_once(request=request)
        else:
            while (response := self._stream_response.get(fn, None)) is None:
                self.logger.info("Waiting for streamed images")
                time.sleep(0.1)

        images = defaultdict(Image)
        for comp, image in response.images.items():
            for type, img in image.data.items():
                component = MMK2Components(comp)
                img = self._image_convert(img)
                type = ImageTypes(type)
                if len(img.shape) == 1:  # compressed
                    img = cv2.imdecode(img, self._imdecode_cfg[type])
                images[component].data[type] = img
                images[component].stamp = Time(image.stamp.sec, image.stamp.nanosec)
        return images

    def listen_to(self, names: List[str], enable: bool = True):
        if isinstance(names, str):
            names = [names]
        if enable:
            self._listen_task = self.event_loop.create_task(
                self._listen_to(names, enable)
            )
        else:
            # TODO: support for stop specified names
            stop_task = self.event_loop.create_task(self._listen_to(names, enable))
            self._wait_tasks((self._listen_task, stop_task))
            # self.logger.info(f"Stopped listening to {names}")
            # for name in names:
            #     self._topic_msg.pop(name)
            self._topic_msg = {}
            self.logger.info(f"Stopped listening to all topics")

    def get_listened(self, name: str):
        return self._topic_msg.get(name, None)

    def enable_resources(self, resources: ResourcesType) -> ResourcesType:
        resources_pb = {}
        for comp, params in resources.items():
            resources_pb[comp.value] = ParamList(params=params)
        result: Resources = self.stub_usual.enable_resources(
            EnableResourcesRequest(resources=Resources(data=resources_pb))
        )
        result_dict = {}
        for comp, value in result.data.items():
            result_dict[MMK2Components(comp)] = dict(value.params)

        return result_dict

    def enable_stream(self, func: Callable, *args, enable: bool = True, **kwargs):
        fn = func.__name__
        if enable:
            assert fn not in self._streamed, f"{fn} is already enabled"
            # self._streamed[fn] = self.event_loop.create_task(
            #     getattr(self, f"_{fn}")(*args, **kwargs)
            # )
            self._streamed[fn] = self.event_loop.create_task(
                self._get_image(*args, **kwargs)
            )
        else:
            assert fn in self._streamed, f"{fn} is not enabled"
            self._streamed.pop(fn).cancel()

    def close(self):
        self._is_running = False
        if hasattr(self, "run_thread"):
            self.run_thread.join()

    async def _listen_to(self, names: List[str], enable: bool):
        # error: server will only receive one
        # async with self.stub_async.listen_to(
        #     ListenToRequest(name=names, enable=enable)
        # ) as features:
        features = self.stub_async.listen_to(ListenToRequest(name=names, enable=enable))
        async for feature in features:
            feature: ListenToResponse
            for topic, msg in feature.joint_state.items():
                self._topic_msg[topic] = msg
            for topic, msg in feature.array_stamped.items():
                self._topic_msg[topic] = msg
            if not self._is_running:
                break
        self.logger.info("exit: _listen_to")
        # features.cancel()

    def _set_field(
        self, value: Union[GoalMsg, ParamMsg], reference: Union[Goal, Param]
    ) -> tuple:
        for field in reference.DESCRIPTOR.fields:
            field_name = field.name
            field_type = str(type(getattr(reference, field_name)))
            value_type = str(type(value))
            if field_type == value_type:
                output = reference.__class__(**{field_name: value})
                break
        else:
            raise ValueError(f"Invalid value: {value} with type: {type(value)}")
        return output, field_name

    def _wait_tasks(self, tasks: List[asyncio.Task]) -> None:
        for task in tasks:
            while not task.done():
                time.sleep(0.1)

    def _get_clock(self) -> tuple:
        clock = time.time()
        clock_sec = int(clock)
        clock_nsec = int((clock - clock_sec) * 1e9)
        return clock_sec, clock_nsec

    def get_header(self) -> Header:
        clock = self._get_clock()
        stamp = common_pb2.Time(sec=clock[0], nanosec=clock[1])
        header = Header(stamp=stamp)
        return header

    def _image_convert(self, image: common_pb2.NumpyArray) -> np.ndarray:
        return np.frombuffer(image.data, dtype=np.dtype(image.dtype)).reshape(
            image.shape
        )

    async def _run(self) -> None:
        async with grpc.aio.insecure_channel(f"{self.ip}:{self.port}") as channel:
            self.stub_async = MMK2ServiceStub(channel)
            if ClientBackend.event_loop is None:
                ClientBackend.event_loop = asyncio.get_event_loop()
            self.logger.info("Async Client Ready")
            # start get robot state stream
            task_get_robot_state = asyncio.create_task(self.get_robot_state())
            await task_get_robot_state
            if self._listen_task is not None:
                await self._listen_task
            self.logger.info("Client exit")

    def __del__(self):
        self.close()

    async def _get_image(self, comp_types: Dict[MMK2Components, List[ImageTypes]]):
        features = self.stub_async.get_image_stream(
            self.__get_get_image_request(comp_types)
        )
        last_stamp = time.time()
        cnt = 0
        async for feature in features:
            self._stream_response[self.get_image.__name__] = feature
            # current_stamp = time.time()
            # self.logger.info(f"Get image frequency: {1 / (current_stamp - last_stamp)}")
            # last_stamp = current_stamp
            cnt += 1
            # self.logger.info(f"Get image stream count: {cnt}")


class AirbotMMK2:
    def __init__(
        self,
        ip: str = "192.168.11.200",
        port: int = 50055,
        name: Optional[str] = None,
        domain_id: Optional[int] = None,
    ):
        self._backend = ClientBackend(ip, port, name, domain_id)

    def set_goal(
        self,
        goal: GoalType,
        param: Union[Dict[MMK2Components, ParamMsg], ParamMsg],
    ) -> GoalStatus:
        return self._backend.set_goal(goal, param)

    def get_robot_state(self) -> Optional[RobotState]:
        return self._backend.robot_state

    def get_image(
        self, comp_type: Dict[MMK2Components, List[ImageTypes]]
    ) -> Dict[MMK2Components, Image]:
        return self._backend.get_image(comp_type)

    def listen_to(self, names: List[str], enable: bool = True):
        return self._backend.listen_to(names, enable)

    def get_listened(self, name: str) -> Optional[Union[JointState, ArrayStamped]]:
        return self._backend.get_listened(name)

    def get_header(self) -> Header:
        return self._backend.get_header()

    def enable_resources(self, resources: ResourcesType) -> ResourcesType:
        return self._backend.enable_resources(resources)

    def enable_stream(self, func: Callable, *args, enable: bool = True, **kwargs):
        return self._backend.enable_stream(func, *args, enable=enable, **kwargs)

    @staticmethod
    def get_joint_values_by_names(
        joint_states: JointState, names: List[str], key: str = "position"
    ) -> List[float]:
        joint_values = []
        joint_names = list(joint_states.name)
        for name in names:
            index = joint_names.index(name)
            if key == "position":
                joint_values.append(list(joint_states.position)[index])
            elif key == "velocity":
                joint_values.append(list(joint_states.velocity)[index])
            elif key == "effort":
                joint_values.append(list(joint_states.effort)[index])
        return joint_values

    def close(self):
        return self._backend.close()
