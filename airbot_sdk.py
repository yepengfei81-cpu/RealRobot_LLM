from mmk2_types.types import MMK2Components, ImageTypes
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
    GoalStatus,
    BaseControlParams,
    BuildMapParams,
    Pose3D,
    Twist3D,
    Pose,
    Position,
    Orientation,
)
from airbot_py.airbot_mmk2 import AirbotMMK2
import logging
import time
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 预设姿态
START_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

STOP_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[1.52, -2.1, 2.0, 1.4, 0.1, -0.62]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}


class MMK2RealRobot:
    """MMK2 机器人控制类"""
    
    def __init__(self, ip="192.168.11.200"):
        """初始化机器人连接"""
        self.mmk2 = AirbotMMK2(ip=ip)
        self.camera = self.Camera(self.mmk2)  # 相机实例化
        logger.info(f"已连接到机器人: {ip}")

    # ========== 已验证可用：头部控制 ==========
    def set_robot_head_pose(self, yaw=0.0, pitch=0.0):
        """
        设置头部姿态
        Args:
            yaw: 左右旋转角度（弧度），正数向左
            pitch: 上下旋转角度（弧度），正数向下
        """
        head_action = {
            MMK2Components.HEAD: JointState(position=[yaw, pitch]),
        }
        if (
            self.mmk2.set_goal(head_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set head pose")
        else:
            logger.info(f"头部姿态设置: yaw={yaw:.2f}, pitch={pitch:.2f}")

    # ========== 已验证可用：相机类 ==========
    class Camera:
        """相机迭代器类"""
        def __init__(self, mmk2_handler):
            self.mmk2 = mmk2_handler
            self._init_camera()
            self.image_goal = {
                MMK2Components.HEAD_CAMERA: [
                    ImageTypes.COLOR,
                    ImageTypes.ALIGNED_DEPTH_TO_COLOR,
                ],
                MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR],
                MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR],
            }
            print("等待相机初始化...")
            time.sleep(5)
            print("相机初始化完成")
            
        def _init_camera(self):
            """初始化相机硬件"""
            print("正在启动相机服务...")
            result = self.mmk2.enable_resources({
                MMK2Components.HEAD_CAMERA: {
                    "camera_type": "REALSENSE",
                    "serial_no": "'242322078139'",
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "true",
                    "depth_module.depth_profile": "640,480,15",
                    "align_depth.enable": "true",
                },
                MMK2Components.LEFT_CAMERA: {
                    "camera_type": "USB",
                    "video_device": "/dev/left_camera",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25",
                },
                MMK2Components.RIGHT_CAMERA: {
                    "camera_type": "USB",
                    "video_device": "/dev/right_camera",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25",
                },
            })
            print(f"相机服务启动结果: {result}")
            if MMK2Components.OTHER in result:
                print(f"警告: 相机初始化可能有问题: {result[MMK2Components.OTHER]}")
        
        def __iter__(self):
            return self
            
        def __next__(self):
            """获取最新图像数据"""
            comp_images = self.mmk2.get_image(self.image_goal)
            
            img_head = None
            img_depth = None
            img_left = None 
            img_right = None

            for comp, images in comp_images.items():
                for img_type, img in images.data.items():
                    if img is None or len(img.shape) < 2:
                        continue
                    if img.shape[0] <= 1 or img.shape[1] <= 1:
                        continue
                    
                    if comp == MMK2Components.HEAD_CAMERA:
                        if img_type == ImageTypes.COLOR:
                            img_head = img
                        elif img_type == ImageTypes.ALIGNED_DEPTH_TO_COLOR:
                            img_depth = cv2.resize(img, (640, 480))
                    elif comp == MMK2Components.LEFT_CAMERA:
                        img_left = cv2.resize(img, (640, 480))
                    elif comp == MMK2Components.RIGHT_CAMERA:
                        img_right = cv2.resize(img, (640, 480))
            
            return img_head, img_depth, img_left, img_right

    # ========== 以下来自 airbot_mmk2_examples.py ==========
    
    def get_robot_state(self):
        """获取机器人状态"""
        return self.mmk2.get_robot_state()

    def control_trajectory_full(self, joint_action=None):
        """全关节轨迹控制"""
        if joint_action is None:
            joint_action = START_JOINT_ACTION
        if (
            self.mmk2.set_goal(joint_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")
            return False
        return True

    def control_traj_servo_separate(self, joint_action=None):
        """分离式轨迹伺服控制"""
        if joint_action is None:
            joint_action = STOP_JOINT_ACTION
        freq = 20
        time_sec = 5
        action_ref = joint_action.copy()
        if (
            self.mmk2.set_goal(
                {MMK2Components.SPINE: action_ref.pop(MMK2Components.SPINE)},
                TrajectoryParams(),
            ).value
            == GoalStatus.Status.SUCCESS
        ):
            for _ in range(freq * time_sec):
                start = time.time()
                if (
                    self.mmk2.set_goal(action_ref, MoveServoParams()).value
                    != GoalStatus.Status.SUCCESS
                ):
                    logger.error("Failed to set goal")
                time.sleep(max(0, 1 / freq - (time.time() - start)))
        else:
            logger.error("Failed to move spine")

    def control_arm_poses(self, target_pose=None):
        """控制机械臂末端位姿"""
        if target_pose is None:
            target_pose = {
                MMK2Components.LEFT_ARM: Pose(
                    position=Position(x=0.457, y=0.221, z=1.147),
                    orientation=Orientation(x=-0.584, y=0.394, z=0.095, w=0.700),
                ),
                MMK2Components.RIGHT_ARM: Pose(
                    position=Position(x=0.457, y=-0.221, z=1.147),
                    orientation=Orientation(x=0.584, y=0.394, z=-0.095, w=0.700),
                ),
            }
        if (
            self.mmk2.set_goal(target_pose, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set poses")

    def control_move_base_pose(self, x=0, y=0, theta=0):
        """控制底盘移动到指定位姿"""
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Pose3D(x=x, y=y, theta=theta)},
                base_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_rotate_base(self, angle=-0.5):
        """控制底盘旋转"""
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: angle},
                base_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    # ========== 夹爪控制 ==========
    def set_gripper(self, left_pos=None, right_pos=None):
        """
        控制左右夹爪开合
        Args:
            left_pos: 左夹爪位置 (0.0=闭合, 1.0=张开), None表示不控制
            right_pos: 右夹爪位置 (0.0=闭合, 1.0=张开), None表示不控制
        """
        gripper_action = {}
        if left_pos is not None:
            gripper_action[MMK2Components.LEFT_ARM_EEF] = JointState(position=[left_pos])
        if right_pos is not None:
            gripper_action[MMK2Components.RIGHT_ARM_EEF] = JointState(position=[right_pos])
        
        if not gripper_action:
            logger.warning("未指定任何夹爪位置")
            return False
        
        if (
            self.mmk2.set_goal(gripper_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set gripper")
            return False
        logger.info(f"夹爪设置: left={left_pos}, right={right_pos}")
        return True

    def open_gripper(self, left=True, right=True):
        """张开夹爪"""
        left_pos = 1.0 if left else None
        right_pos = 1.0 if right else None
        return self.set_gripper(left_pos, right_pos)

    def close_gripper(self, left=True, right=True):
        """闭合夹爪"""
        left_pos = 0.0 if left else None
        right_pos = 0.0 if right else None
        return self.set_gripper(left_pos, right_pos)
    
    def control_move_base_velocity(self, linear_x=0.0, angular_z=0.0):
        """控制底盘速度"""
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Twist3D(x=linear_x, omega=angular_z)},
                BaseControlParams(),
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def reset_to_start(self):
        """重置到起始姿态"""
        return self.control_trajectory_full(START_JOINT_ACTION)

    def reset_to_stop(self):
        """重置到停止姿态"""
        return self.control_trajectory_full(STOP_JOINT_ACTION)


# ========== 测试代码 ==========
if __name__ == "__main__":
    mmk2 = MMK2RealRobot(ip="192.168.11.200")
    
    # 测试头部控制
    mmk2.set_robot_head_pose(0, 0)
    mmk2.open_gripper()
    
    # 测试获取机器人状态
    state = mmk2.get_robot_state()
    if state:
        print("机器人状态获取成功")    