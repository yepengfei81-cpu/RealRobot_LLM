from mmk2_types.types import MMK2Components, ImageTypes, ControllerTypes, JointNames
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
    GoalStatus,
    BaseControlParams,
    BuildMapParams,
    Pose3D,
    Twist3D,
    BaseChargeStationParams,
    ArrayStamped,
    Pose,
    Position,
    Orientation,
)
from airbot_py.airbot_mmk2 import AirbotMMK2
from pprint import pprint
import logging
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create an instance of the AirbotMMK2 class
# change the ip address to the ip address of the robot
# mmk2 = AirbotMMK2(port=50055)
mmk2 = AirbotMMK2(ip="192.168.11.200")


start_joint_action = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

stop_joint_action = {
    MMK2Components.LEFT_ARM: JointState(position=[1.52, -2.1, 2.0, 1.4, 0.1, -0.62]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

start_arm_pose = {
    MMK2Components.LEFT_ARM: Pose(
        position=Position(x=0.457, y=0.221, z=1.147),
        orientation=Orientation(x=-0.584, y=0.394, z=0.095, w=0.700),
    ),
    MMK2Components.RIGHT_ARM: Pose(
        position=Position(x=0.457, y=-0.221, z=1.147),
        orientation=Orientation(x=0.584, y=0.394, z=-0.095, w=0.700),
    ),
}


def get_robot_state():
    robot_state = mmk2.get_robot_state()
    if robot_state is None:
        logger.error("Failed to get robot state")
        return
    for _ in range(5):
        print("Robot state:")
        print("stamps:")
        pprint(robot_state.stamp)
        pprint(robot_state.joint_state.header.stamp)
        print("current_time", time.time())
        print("joint states:")
        pprint(robot_state.joint_state.name)
        pprint(robot_state.joint_state.position)
        pprint(robot_state.joint_state.velocity)
        pprint(robot_state.joint_state.effort)
        # get left arm joint values in order of joint names
        print(
            f"{MMK2Components.LEFT_ARM} joint position: {mmk2.get_joint_values_by_names(robot_state.joint_state, JointNames().left_arm, 'position')}"
        )
        print("base state:")
        pose = robot_state.base_state.pose
        print(f"x: {pose.x}, y: {pose.y}, theta: {pose.theta}")
        vel = robot_state.base_state.velocity
        print(f"linear: {vel.x}, angular: {vel.omega}")

        print("robot poses:")
        pprint(robot_state.robot_pose.robot_pose)
        # get robot pose of right arm
        print(
            f"{MMK2Components.RIGHT_ARM} pose: {robot_state.robot_pose.robot_pose[MMK2Components.RIGHT_ARM.value]}"
        )
        time.sleep(1)
        print("\n")


def control_trajectory_full():
    if (
        mmk2.set_goal(start_joint_action, TrajectoryParams()).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")

    time.sleep(5)


def control_traj_servo_separate():
    freq = 20
    time_sec = 5
    action_ref = stop_joint_action.copy()
    if (
        mmk2.set_goal(
            {MMK2Components.SPINE: action_ref.pop(MMK2Components.SPINE)},
            TrajectoryParams(),
        ).value
        == GoalStatus.Status.SUCCESS
    ):
        for _ in range(freq * time_sec):
            start = time.time()
            if (
                mmk2.set_goal(action_ref, MoveServoParams()).value
                != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set goal")
            time.sleep(max(0, 1 / freq - (time.time() - start)))
    else:
        logger.error("Failed to move spine")


def control_arm_poses():
    # if you want to set other poses, make sure you know
    # the target link and reference link
    # poses = mmk2.get_robot_state().robot_pose.robot_pose
    # target_pose = (  # set arm poses to the current values
    #     {
    #         MMK2Components.LEFT_ARM: poses[MMK2Components.LEFT_ARM.value],
    #         MMK2Components.RIGHT_ARM: poses[MMK2Components.RIGHT_ARM.value],
    #     },
    # )
    target_pose = start_arm_pose
    if (
        mmk2.set_goal(target_pose, TrajectoryParams()).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set poses")


def control_move_base_pose():
    # move the robot base to zero pose
    base_param = BaseControlParams()
    if (
        mmk2.set_goal(
            {MMK2Components.BASE: Pose3D(x=0, y=0, theta=0)},
            base_param,
        ).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")


def control_rotate_base():
    # rotate 0.5 rad
    base_param = BaseControlParams()
    if (
        mmk2.set_goal(
            {MMK2Components.BASE: -0.5},
            base_param,
        ).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")


def control_charge_base():
    # move the robot base to the charging station to charge
    base_param = BaseChargeStationParams(move_to_dock=True)
    if (
        mmk2.set_goal(
            {MMK2Components.BASE: Pose3D(x=0.5, y=0, theta=0)},
            base_param,
        ).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")


def control_move_base_velocity():

    freq = 20
    time_sec = 2

    for _ in range(freq * time_sec):
        start = time.time()
        print(f"stamp: {start}")
        if (
            mmk2.set_goal(
                {MMK2Components.BASE: Twist3D(x=0.1, omega=-0.5)},
                BaseControlParams(),
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")
        print(f"time cost: {time.time() - start}")
        # time.sleep(1 / freq)


def control_build_map():
    build_map_param = BuildMapParams()
    if (
        mmk2.set_goal(
            None,
            build_map_param,
        ).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")

    # move the base around to build the map and then
    # set current pose to zero pose in the map
    if (
        mmk2.set_goal(
            Pose3D(x=0, y=0, theta=0),
            build_map_param,
        ).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")

    # stop build map
    build_map_param.stop = True
    if (
        mmk2.set_goal(
            None,
            build_map_param,
        ).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")


def get_images():
    import cv2

    # enable cameras
    result = mmk2.enable_resources(
        {
            MMK2Components.HEAD_CAMERA: {  # realsense case
                "camera_type": "REALSENSE",
                "rgb_camera.color_profile": "640,480,30",
                # "enable_depth": "true",
                # "align_depth.enable": "true",
                # "depth_module.depth_profile": "640,480,30",
                # "serial_no": "'123456'",  # multiple rs-cameras case
            },
            # MMK2Components.LEFT_CAMERA: {  # realsense case
            #     "camera_type": "REALSENSE",
            #     "rgb_camera.color_profile": "640,480,30",
            #     "enable_depth": "false",
            #     # "depth_module.depth_profile": "640,480,30",
            #     # "align_depth.enable": "false",
            # "serial_no": "'123456'",  # multiple rs-cameras case
            # },
            # MMK2Components.RIGHT_CAMERA: {  # realsense case
            #     "camera_type": "REALSENSE",
            #     "rgb_camera.color_profile": "640,480,30",
            #     "enable_depth": "false",
            #     # "depth_module.depth_profile": "640,480,30",
            #     # "align_depth.enable": "false",
            # "serial_no": "'123456'",  # multiple rs-cameras case
            # },
            # MMK2Components.LEFT_CAMERA: { # USB camera case
            #     "camera_type": "USB",
            #     "video_device": "/dev/left_camera",
            #     "image_width": "640",
            #     "image_height": "480",
            #     "framerate": "25",
            # },
            # MMK2Components.RIGHT_CAMERA: { # USB camera case
            #     "camera_type": "USB",
            #     "video_device": "/dev/right_camera",
            #     "image_width": "640",
            #     "image_height": "480",
            #     "framerate": "25",
            # },
            # MMK2Components.HEAD_CAMERA: {  # orbbec case
            #     "camera_type": "ORBBEC_GEMINI2",
            #     "color_width": "640",
            #     "color_height": "480",
            #     "color_fps": "30",
            #     "enable_depth": "false",
            #     # "depth_width": "640",
            #     # "depth_height": "480",
            #     # "depth_fps": "30",
            #     # "enable_depth_scale": "false",
            #     # "enable_ir": "false",
            # },
        }
    )
    assert (
        MMK2Components.OTHER not in result
    ), f"Failed to enable cameras: {result[MMK2Components.OTHER]}"

    image_goal = {
        # MMK2Components.HEAD_CAMERA: [ImageTypes.COLOR],
        MMK2Components.HEAD_CAMERA: [
            ImageTypes.COLOR,
            ImageTypes.ALIGNED_DEPTH_TO_COLOR,
        ],
        # MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR],
        # MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR],
    }
    # mmk2.enable_stream(mmk2.get_image, image_goal)
    time.sleep(5)
    while True:
        # get images
        start = time.perf_counter()
        comp_images = mmk2.get_image(image_goal)
        # print(f"time cost: {time.perf_counter() - start}")
        for comp, images in comp_images.items():
            for img_type, img in images.data.items():
                if img.shape[0] == 1:
                    print(f"{comp} got no image")
                    time.sleep(1)
                    break
                # print(f"{comp} got {img_type} image with shape {img.shape}")
                cv2.imshow(f"{comp.value}:{img_type.value}", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def listen_to():
    from mmk2_types.types import MMK2ComponentsGroup, TopicNames

    comp_action_topic = {
        comp: TopicNames.tracking.format(component=comp.value)
        for comp in MMK2ComponentsGroup.ARMS
    }
    comp_action_topic.update(
        {
            comp: TopicNames.controller_command.format(
                component=comp.value, controller=ControllerTypes.FORWARD_POSITION.value
            )
            for comp in MMK2ComponentsGroup.HEAD_SPINE
        }
    )

    time.sleep(3)
    mmk2.listen_to(list(comp_action_topic.values()))
    time.sleep(1)
    for _ in range(10):
        for comp in MMK2ComponentsGroup.ARMS:
            result = mmk2.get_listened(comp_action_topic[comp])
            if isinstance(result, JointState):
                stamp = result.header.stamp
                print(
                    f"comp: {comp.value} stamp: {stamp.sec}.{stamp.nanosec} position: {result.position} velocity: {result.velocity} effort: {result.effort}"
                )
            elif result is None:
                logger.warning(
                    f"You must start teleoprating to get the action data of {comp}"
                )
            else:
                raise ValueError(f"Unexpected data type: {type(result)}")
        for comp in MMK2ComponentsGroup.HEAD_SPINE:
            result = mmk2.get_listened(comp_action_topic[comp])
            if isinstance(result, ArrayStamped):
                print(f"comp: {comp.value} data: {result.data}")
            elif result is None:
                logger.warning(
                    f"You must start teleoprating to get the action data of {comp}"
                )
            else:
                raise ValueError(f"Unexpected data type: {type(result)}")
        print("current_time:", time.time())
        print("\n")
        time.sleep(0.01)


"""******run the functions******"""

get_robot_state()
# control_trajectory_full()
# control_traj_servo_separate()
# control_arm_poses()
# control_move_base_pose()
# control_rotate_base()
# control_move_base_velocity()
# control_build_map()
# control_charge_base()
# get_images()
# listen_to()
