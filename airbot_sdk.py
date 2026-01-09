      
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
from scipy.spatial.transform import Rotation
import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


START_ALL_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
}

START_LEFT_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
}

START_RIGHT_JOINT_ACTION = {
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    # MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.0, 0.0, -1.57, 0.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
}

STOP_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[1.52, -2.1, 2.0, 1.4, 0.1, -0.62]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

TEST_JOINT_ACTION = {
    MMK2Components.LEFT_ARM: JointState(position=[1.197, -1.082,  0.71,   2.595,  1.415,  0.499]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.197, -1.082,  0.71,   -2.595,  -1.415,  -0.499]),
    # MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, -0.8]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

class MMK2RealRobot():
    def __init__(self, ip="192.168.11.200"):
        self.mmk2 = AirbotMMK2(ip)
        self._joint_order_cache = None
        self.camera = self.Camera(self.mmk2)  # 相机实例化

    def reset_start(self, arm_type):
        if arm_type == 'left_arm':
            self.control_trajectory_full(START_LEFT_JOINT_ACTION)
        elif arm_type == 'right_arm':
            self.control_trajectory_full(START_RIGHT_JOINT_ACTION)
        elif arm_type == "all_arm":
            self.control_trajectory_full(START_ALL_JOINT_ACTION)
        else:
            logger.error(f"Invalid arm type: {arm_type}")
        print(f"回到{arm_type}起始姿态，当前状态为：")
        self.printMessage()

    def reset_stop(self):
        self.control_trajectory_full(STOP_JOINT_ACTION)
        # self.set_move_base_zero()
        print("机器人重置，回到双臂停止姿态，当前状态为：")
        self.printMessage()

    @property
    def head_pose(self):
        return self.get_all_joint_states()[0]

    @property
    def base_pose(self):
        return self.get_all_joint_states()[-1]

    @property
    def spine_position(self):
        return self.get_all_joint_states()[1]

    @property
    def left_arm_joints(self):
        return self.get_all_joint_states()[2]

    @property
    def right_arm_joints(self):
        return self.get_all_joint_states()[3]

    @property
    def left_arm_pose(self):
        """返回的是一个列表  [pos[X,Y,Z], quant[qx,qy,qz,qw]]"""
        return self.get_all_joint_states()[4]

    @property
    def right_arm_pose(self):
        """返回的是一个列表  [pos[X,Y,Z], quant[qx,qy,qz,qw]]"""
        return self.get_all_joint_states()[5]

    def get_robot_current_state(self):
        state = self.mmk2.get_robot_state()
        if not state:
            return None
        
        joint_dict = {
            name: {
                'position': pos,
                'velocity': vel,
                'effort': eff
            } for name, pos, vel, eff in zip(
                state.joint_state.name,
                state.joint_state.position,
                state.joint_state.velocity,
                state.joint_state.effort
            )
        }
        
        base_dict = {
            'x': state.base_state.pose.x,
            'y': state.base_state.pose.y,
            'theta': state.base_state.pose.theta,
        }
        
        left_ee_pose  = state.robot_pose.robot_pose['left_arm']
        right_ee_pose = state.robot_pose.robot_pose['right_arm']
        ee_pose_dict = {
            'left_arm': {
                'position': [left_ee_pose.position.x, left_ee_pose.position.y, left_ee_pose.position.z],
                'orientation': [left_ee_pose.orientation.x, left_ee_pose.orientation.y, left_ee_pose.orientation.z, left_ee_pose.orientation.w]
            },
            'right_arm': {
                'position': [right_ee_pose.position.x, right_ee_pose.position.y, right_ee_pose.position.z],
                'orientation': [right_ee_pose.orientation.x, right_ee_pose.orientation.y, right_ee_pose.orientation.z, right_ee_pose.orientation.w]
            }
        }

        return {
            'joints': joint_dict,
            'base': base_dict,
            'ee_poses': ee_pose_dict
        }

    def get_all_joint_states(self):
        state = self.get_robot_current_state()
        if not state:
            return None
        
        # 初始化关节顺序缓存
        if self._joint_order_cache is None:
            self._init_joint_order_cache(state['joints'])
        
        head = [
            state['joints']['head_yaw_joint']['position'],
            state['joints']['head_pitch_joint']['position']
        ]
        
        lift = state['joints']['slide_joint']['position']
        left_arm  = [state['joints'][f'left_arm_joint{i+1}']['position'] 
                   for i in range(6)]
        right_arm = [state['joints'][f'right_arm_joint{i+1}']['position'] 
                    for i in range(6)]
        
        left_arm_eef  = [self.get_robot_current_state()['ee_poses']['left_arm']['position'],
                         self.get_robot_current_state()['ee_poses']['left_arm']['orientation']]
        right_arm_eef = [self.get_robot_current_state()['ee_poses']['right_arm']['position'],
                         self.get_robot_current_state()['ee_poses']['right_arm']['orientation']]

        left_gripper = [state['joints']['left_arm_eef_gripper_joint']['position']]
        right_gripper = [state['joints']['right_arm_eef_gripper_joint']['position']]

        base_pose = [
            state['base']['x'],
            state['base']['y'],
            state['base']['theta']
        ]
        
        return (head, lift, 
                left_arm, right_arm,
                left_arm_eef, right_arm_eef,
                left_gripper, right_gripper, 
                base_pose)

    def get_base_pose(self):
        base_pos_dict = self.get_robot_current_state()['base']
        pose = [base_pos_dict["x"], base_pos_dict["y"], base_pos_dict["theta"]]
        return pose

    def quaternion_to_rotation_matrix(self, quaternion):
        x, y, z, w = quaternion
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),   2*(x*z + y*w)],
            [2*(x*y + z*w),   1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
        ])

    def rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数 (x,y,z,w 顺序)"""
        from scipy.spatial.transform import Rotation as R_scipy
        return R_scipy.from_matrix(R).as_quat()

    def get_arm_ee_pose(self, arm_type='left_arm'):
        if arm_type == 'left_arm':
            return self.left_arm_pose        
        elif arm_type == 'right_arm':
            return self.right_arm_pose
        else:
            logger.error(f"Invalid arm type: {arm_type}")
            return None,None

        # ee_pose = state['ee_poses'][arm_type]
        # 顺序是 xyzw
        # rotation_matrix = self.quaternion_to_rotation_matrix(ee_pose['orientation'])
        
        # transform = np.eye(4)
        # transform[:3, :3] = rotation_matrix
        # transform[:3, 3] = ee_pose['position']
        # return transform

    def _init_joint_order_cache(self, joints_dict):
        """初始化关节顺序验证"""
        required_joints = [
            'head_yaw_joint', 'head_pitch_joint',
            'slide_joint',
            *[f'left_arm_joint{i+1}' for i in range(6)],
            *[f'right_arm_joint{i+1}' for i in range(6)],
            'left_arm_eef_gripper_joint', 'right_arm_eef_gripper_joint'
        ]
        
        missing = [j for j in required_joints if j not in joints_dict]
        if missing:
            raise KeyError(f"Missing required joints: {missing}")
        
        self._joint_order_cache = required_joints

    class Camera:
        """相机迭代器类"""
        def __init__(self, mmk2_handler):
            self.mmk2 = mmk2_handler
            self._init_camera()  # 初始化相机服务
            self.image_goal = {
                MMK2Components.HEAD_CAMERA: [
                    ImageTypes.COLOR,
                    ImageTypes.ALIGNED_DEPTH_TO_COLOR,
                ],
                MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR],
                MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR],
            }
            print("等待相机初始化...")
            time.sleep(5)  # 等待相机初始化完成
            print("相机初始化完成")
            
        def _init_camera(self):
            """初始化相机硬件"""
            print("正在启动相机服务...")
            result = self.mmk2.enable_resources({
                MMK2Components.HEAD_CAMERA: {
                    "serial_no": "'242322078139'",
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "true",
                    "depth_module.depth_profile": "640,480,15",
                    "align_depth.enable": "true",
                },
                MMK2Components.LEFT_CAMERA: {
                    "camera_type": "USB",
                    "video_device": "/dev/video8",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25",
                },
                MMK2Components.RIGHT_CAMERA: {
                    "camera_type": "USB",
                    "video_device": "/dev/video0",
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
                    # 检查图像是否有效
                    if img is None or len(img.shape) < 2:
                        continue
                    if img.shape[0] <= 1 or img.shape[1] <= 1:
                        continue
                    
                    if comp == MMK2Components.HEAD_CAMERA:
                        if img_type == ImageTypes.COLOR:
                            img_head = img
                        elif img_type == ImageTypes.ALIGNED_DEPTH_TO_COLOR:
                            img_depth = cv2.resize(img, (1280, 720))
                    elif comp == MMK2Components.LEFT_CAMERA:
                        img_left = cv2.resize(img, (1280, 720))
                    elif comp == MMK2Components.RIGHT_CAMERA:
                        img_right = cv2.resize(img, (1280, 720))
            
            return img_head, img_depth, img_left, img_right


    def set_robot_eef(self, eef_type, value):
        """设置机械臂末端执行器状态 0为开 1为关"""
        if eef_type == "left_arm":
            eef_action = {
                MMK2Components.LEFT_ARM_EEF: JointState(position=[value]),
            }
            if(
                self.mmk2.set_goal(eef_action, TrajectoryParams()).value
                != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set eef")
        elif eef_type == "right_arm":
            eef_action = {
                MMK2Components.RIGHT_ARM_EEF: JointState(position=[value]),
            }
            if(
                self.mmk2.set_goal(eef_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set eef")
        elif eef_type == "all_arm":
            eef_action = {
                MMK2Components.LEFT_ARM_EEF:  JointState(position=[value]),
                MMK2Components.RIGHT_ARM_EEF: JointState(position=[value]),
            }
            if (
                self.mmk2.set_goal(eef_action, TrajectoryParams()).value
              != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set eef")
        else:
            logger.error("Invalid eef type")

    def set_robot_spine(self, value=0.0):
        """设置脊柱状态, 正数向下，负数向上"""
        value = max(min(value, 1.0), -0.1)
        print(value)
        spine_action = {
            MMK2Components.SPINE: JointState(position=[value]),
        }
        if (
            self.mmk2.set_goal(spine_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set spine")

    def set_robot_head_pose(self, yaw=0.0, pitch=0.0):
        """设置头部姿态,单位为弧度,yaw为左右,pitch为上下"""
        spine_action = {
            MMK2Components.HEAD: JointState(position=[yaw, pitch]),
        }
        if (
            self.mmk2.set_goal(spine_action, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set spine")

    def control_robot_arm_pose_updown(self, arm_type='left_arm', updown=0.0):
        """控制机械臂末端沿机器人base坐标系Z轴移动
        Args:
            arm_type: 机械臂类型 left_arm/right_arm
            updown: 移动量（米）正数向下，负数向上
        """
        # 参数校验和类型判断
        if arm_type not in ['left_arm', 'right_arm']:
            logger.error(f"Invalid arm type: {arm_type}")
            return

        current_arm_pose = self.get_arm_ee_pose(arm_type)
        if current_arm_pose is None:
            logger.error("Failed to get arm pose")
            return

        updown = max(min(updown, 0.1), -0.1)
        # 修改Z轴位置
        current_arm_pose[2, 3] += updown
        position = current_arm_pose[:3,3]
        new_quat = self.rotation_matrix_to_quaternion(current_arm_pose[:3, :3])
        new_pose = Pose(
            position=Position(x=position[0], y=position[1], z=position[2]),
            orientation=Orientation(
                x=new_quat[0],
                y=new_quat[1],
                z=new_quat[2],
                w=new_quat[3]
            )
        )

        component = MMK2Components.LEFT_ARM if arm_type == 'left_arm' else MMK2Components.RIGHT_ARM
        result = self.mmk2.set_goal(
            {component: new_pose},
            TrajectoryParams()
        )
        if result.value != GoalStatus.Status.SUCCESS:
            logger.error(f"Failed to set {arm_type} poses")

    def control_robot_arm_pose_rotate(self, arm_type='left_arm', rotate=0.0):
        """机械臂末端绕机器人base坐标系Z轴旋转
           rotate 正数时逆时针旋转  负数时顺时针旋转
        """
        if arm_type in ['left_arm', 'right_arm']:
            current_arm_pose = self.get_arm_ee_pose(arm_type)
            if current_arm_pose is None:
                logger.error("Failed to get arm pose")
                return
            
            # 限制旋转幅度在±0.1弧度（约±5.7度）以内
            rotate = max(min(rotate, 0.1), -0.1)
            
            # 获取当前旋转矩阵和位置
            R_current = current_arm_pose[:3, :3]
            position  = current_arm_pose[:3, 3]
            
            # 创建绕Z轴旋转矩阵
            R_z = np.array([
                [np.cos(rotate), -np.sin(rotate), 0],
                [np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1]
            ])
            R_new = R_z @ R_current
            new_quat = self.rotation_matrix_to_quaternion(R_new)

            new_pose = Pose(
                position=Position(x=position[0], y=position[1], z=position[2]),
                orientation=Orientation(
                    x=new_quat[0],
                    y=new_quat[1],
                    z=new_quat[2],
                    w=new_quat[3]
                )
            )
            
            component = MMK2Components.LEFT_ARM if arm_type == 'left_arm' else MMK2Components.RIGHT_ARM
            result = self.mmk2.set_goal(
                {component: new_pose},
                TrajectoryParams()
            )
            if result.value != GoalStatus.Status.SUCCESS:
                logger.error(f"Failed to set {arm_type} poses")

    def control_trajectory_full(self, joint_action):
        if (
            self.mmk2.set_goal(
                joint_action, 
                TrajectoryParams(
                    max_velocity_scaling_factor=0.8,
                    max_acceleration_scaling_factor=0.8,
                )
            ).value != GoalStatus.Status.SUCCESS 
        ):
            logger.error("Failed to set goal")
            return False
        # time.sleep(5)
        return True

    def control_arm_joints(self, arm_type, joint_action):
        """控制机械臂关节, arm_type为left或right, joint_action为关节角度列表"""
        if arm_type == "left_arm":
            arm_action = {
                MMK2Components.LEFT_ARM: JointState(position=joint_action),
            }
            if (
                self.mmk2.set_goal(
                    arm_action, 
                    TrajectoryParams(
                        max_velocity_scaling_factor=0.8,
                        max_acceleration_scaling_factor=0.8,
                    )
                ).value != GoalStatus.Status.SUCCESS 
            ):
                logger.error("Failed to set arm joints")
        elif arm_type == "right_arm":
            arm_action = {
                MMK2Components.RIGHT_ARM: JointState(position=joint_action),
            }
            if (
                self.mmk2.set_goal(
                    arm_action, 
                    TrajectoryParams(
                        max_velocity_scaling_factor=0.8,
                        max_acceleration_scaling_factor=0.8,
                    )
                ).value != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set arm joints")
        elif arm_type == "all_arm":
            arm_action = {
                MMK2Components.LEFT_ARM:  JointState(position=joint_action[0]),
                MMK2Components.RIGHT_ARM: JointState(position=joint_action[1]),
            }
            if (
                self.mmk2.set_goal(
                    arm_action, 
                    TrajectoryParams(
                        max_velocity_scaling_factor=0.8,
                        max_acceleration_scaling_factor=0.8,
                    )
                ).value != GoalStatus.Status.SUCCESS
            ):
                logger.error("Failed to set arm joints")
        else:
            logger.error("Invalid arm type")
        time.sleep(0.3)

    def control_arm_pose(self, arm_type, pose_base):
        """控制机械臂末端移动:
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入pose:  [X,Y,Z,qx,qy,qz,qw]
            双臂输入pose:  [left,right]
        """
        if arm_type == "left_arm":
            arm_ee_pose = {
                MMK2Components.LEFT_ARM: Pose(
                    position=Position(x=pose_base[0], y=pose_base[1], z=pose_base[2]),
                    orientation=Orientation(x=pose_base[3], y=pose_base[4], 
                                            z=pose_base[5], w=pose_base[6]),
                )
            }
        elif arm_type == "right_arm":
            arm_ee_pose = {
                MMK2Components.RIGHT_ARM: Pose(
                    position=Position(x=pose_base[0], y=pose_base[1], z=pose_base[2]),
                    orientation=Orientation(x=pose_base[3], y=pose_base[4], 
                                            z=pose_base[5], w=pose_base[6]),
                )
            }
        elif arm_type == "all_arm":
            arm_ee_pose = {
                MMK2Components.LEFT_ARM: Pose(
                    position=Position(x=pose_base[0][0], y=pose_base[0][1], z=pose_base[0][2]),
                    orientation=Orientation(x=pose_base[0][3], y=pose_base[0][4], 
                                            z=pose_base[0][5], w=pose_base[0][6]),
                ),
                MMK2Components.RIGHT_ARM: Pose(
                    position=Position(x=pose_base[1][0], y=pose_base[1][1], z=pose_base[1][2]),
                    orientation=Orientation(x=pose_base[1][3], y=pose_base[1][4], 
                                            z=pose_base[1][5], w=pose_base[1][6]),
                )
            }
        else:
            logger.error("Invalid arm type")
        
        if (
            self.mmk2.set_goal(arm_ee_pose, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to control_arm_pose")

        time.sleep(2)

    def control_arm_pose_waypoints(self, arm_type, waypoints):
        """控制机械臂末端移动，输入轨迹点基于机器人base坐标系: 
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入waypoints:  [[X,Y,Z,qx,qy,qz,qw],...]
            双臂输入waypoints:  [left_waypoints,right_waypoints]
        """
        pose_sequence = []
        if arm_type in ['left_arm', 'right_arm']:
            for wp in waypoints:
                if len(wp) != 7:
                    raise ValueError("轨迹点需要7个参数 [X,Y,Z,qx,qy,qz,qw]")
                    
                pose_sequence.append(Pose(
                    position=Position(x=wp[0], y=wp[1], z=wp[2]),
                    orientation=Orientation(x=wp[3], y=wp[4], z=wp[5], w=wp[6])
                ))
        elif arm_type == "all_arm":
            left_pose_sequence = []
            for wp in waypoints[0]:
                if len(wp)!= 7:
                    raise ValueError("轨迹点需要7个参数 [X,Y,Z,qx,qy,qz,qw]")

                left_pose_sequence.append(Pose(
                    position=Position(x=wp[0], y=wp[1], z=wp[2]),
                    orientation=Orientation(x=wp[3], y=wp[4], z=wp[5], w=wp[6])
                ))

            right_pose_sequence = []
            for wp in waypoints[1]:
                if len(wp)!= 7:
                    raise ValueError("轨迹点需要7个参数 [X,Y,Z,qx,qy,qz,qw]")

                right_pose_sequence.append(Pose(
                    position=Position(x=wp[0], y=wp[1], z=wp[2]),
                    orientation=Orientation(x=wp[3], y=wp[4], z=wp[5], w=wp[6])
                ))
        else:
            logger.error("Invalid arm type")

        if arm_type == "left_arm":
            arm_ee_pose = {MMK2Components.LEFT_ARM: pose_sequence}
        elif arm_type == "right_arm":
            arm_ee_pose = {MMK2Components.RIGHT_ARM: pose_sequence}
        elif arm_type == "all_arm":
            arm_ee_pose = {
                    MMK2Components.LEFT_ARM: left_pose_sequence,
                    MMK2Components.RIGHT_ARM: right_pose_sequence,
                }
        else:
            logger.error("Invalid arm type")
        
        if (
            self.mmk2.set_goal(arm_ee_pose, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to control_arm_pose")

        # time.sleep(2)

    def control_traj_servo_separate(self, joint_action):
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

    def control_move_base_pose(self, base_pose=[0,0,0]):
        # move the robot base to zero pose
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Pose3D(x=base_pose[0], y=base_pose[1], theta=base_pose[2])},
                base_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_robot_move_base_updown(self, updown=0.0):
        """控制底盘相对移动
        Args:
            updown: 正数向前 负数向后 最小分辨率0.1m
        """
        base_pose = self.get_all_joint_states()[-1]
        base_pose[0] +=  updown
        print(base_pose)
        self.control_move_base_pose(base_pose)

    def control_rotate_base(self, rotate_rad=-0.5):
        base_param = BaseControlParams()
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: rotate_rad},
                base_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_move_base_velocity(self, velocity=[0,0]):
        if (
            self.mmk2.set_goal(
                {MMK2Components.BASE: Twist3D(x=velocity[0], omega=velocity[1])},
                BaseControlParams(),
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def control_build_map(self):
        build_map_param = BuildMapParams()
        if (
            self.mmk2.set_goal(
                None,
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

        # move the base around to build the map and then
        # set current pose to zero pose in the map
        if (
            self.mmk2.set_goal(
                Pose3D(x=0, y=0, theta=0),
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

        # stop build map
        build_map_param.stop = True
        if (
            self.mmk2.set_goal(
                None,
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")

    def set_move_base_zero(self):
        build_map_param = BuildMapParams()
        if (
            self.mmk2.set_goal(
                Pose3D(x=0, y=0, theta=0),
                build_map_param,
            ).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to set goal")
        time.sleep(3)

    def printMessage(self):
        print("-" * 50)
        head, lift, left_arm, right_arm, left_arm_eef, right_arm_eef, left_gripper, right_gripper, xy_yaw = self.get_all_joint_states()
        print("real_robot.get_all_joint_states:")
        print("head                         : ", np.array(head))
        print("spine positon                : ", np.array(self.spine_position))
        print("left_arm_joints              : ", np.array(self.left_arm_joints))
        print("right_arm_joints             : ", np.array(self.right_arm_joints))
        print("left_arm_eef_pos             : ", np.array(left_arm_eef[0]))
        print("left_arm_eef_orientation     : ", np.array(left_arm_eef[1]))
        print("right_arm_eef_pos            : ", np.array(right_arm_eef[0]))
        print("right_arm_eef_orientation    : ", np.array(right_arm_eef[1]))
        print("left_gripper  : ", np.array(left_gripper))
        print("right_gripper : ", np.array(right_gripper))
        print("xy_yaw        : ", np.array(xy_yaw))

    def control_arm_joint_waypoints(self, arm_type, waypoints):
        """控制机械臂末端移动:
            arm_type: 机械臂类型 left_arm/right_arm
            单臂输入waypoints:  [[0,1,2,3,4,5,6],...]
            双臂输入waypoints:  [left_waypoints,right_waypoints]
        """
        joint_sequence = []
        if arm_type in ['left_arm', 'right_arm']:
            for wp in waypoints:
                if len(wp) != 6:
                    raise ValueError("关节角度需要6个参数")
                joint_sequence.append(JointState(position=wp))
        elif arm_type == "all_arm":
            left_joint_sequence = []
            for wp in waypoints[0]:
                if len(wp)!= 6:
                    raise ValueError("关节角度需要6个参数")
                left_joint_sequence.append(JointState(position=wp))
            right_joint_sequence = []
            for wp in waypoints[1]:
                if len(wp)!= 6:
                    raise ValueError("关节角度需要6个参数")
                right_joint_sequence.append(JointState(position=wp))
        else:
            logger.error("Invalid arm type")

        if arm_type == "left_arm":
            arm_joints = {MMK2Components.LEFT_ARM: joint_sequence}
        elif arm_type == "right_arm":
            arm_joints = {MMK2Components.RIGHT_ARM: joint_sequence}
        elif arm_type == "all_arm":
            arm_joints = {
                    MMK2Components.LEFT_ARM: left_joint_sequence,
                    MMK2Components.RIGHT_ARM: right_joint_sequence,
                }
        else:
            logger.error("Invalid arm type")
        
        if (
            self.mmk2.set_goal(arm_joints, TrajectoryParams()).value
            != GoalStatus.Status.SUCCESS
        ):
            logger.error("Failed to control_arm_pose")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # mmk2 = MMK2RealRobot()
    # mmk2 = MMK2RealRobot(ip="172.25.15.162")
    mmk2 = MMK2RealRobot(ip="192.168.11.200")
    mmk2.show_head_camera(show_depth=True)
    # mmk2.reset_start("right_arm")
    # mmk2.reset_start("left_arm")
    # mmk2.set_robot_head_pose(yaw=0, pitch=-1.08)
    # mmk2.set_robot_head_pose(yaw=0, pitch=0.16)
    # mmk2.set_robot_head_pose(yaw=0, pitch=0)
    # mmk2.reset_stop()
    # mmk2.set_move_base_zero()
    # mmk2.printMessage()

    