"""
手眼标定结果验证工具

验证方法：
1. 重投影误差：检测标定板，通过手眼变换计算其在基座标系的位置
2. 触碰验证：让机械臂末端移动到标定板原点位置
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R

from mmk2_types.types import MMK2Components
from mmk2_types.grpc_msgs import Pose, Position, Orientation
from airbot_sdk import MMK2RealRobot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 棋盘格参数
CHECKERBOARD = {
    'rows': 6,
    'cols': 9,
    'square_size': 0.022
}

CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")

# 增量控制参数
DELTA_POS = 0.01      # 平移增量：1cm
DELTA_ROT = 2.0       # 旋转增量：2度

# 悬停高度
HOVER_HEIGHT = 0.20   # 20cm


class ArmController:
    """机械臂增量控制器"""
    
    def __init__(self, robot: MMK2RealRobot, use_left_arm: bool = True):
        self.robot = robot
        self.use_left_arm = use_left_arm
        self.arm_component = MMK2Components.LEFT_ARM if use_left_arm else MMK2Components.RIGHT_ARM
        self.arm_name = "left_arm" if use_left_arm else "right_arm"
        
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_euler = np.array([0.0, 0.0, 0.0])
        
        self._update_current_pose()
    
    def _update_current_pose(self) -> bool:
        try:
            poses = self.robot.get_arm_end_poses()
            if poses is None:
                return False
            
            arm_key = 'left_arm' if self.use_left_arm else 'right_arm'
            if arm_key not in poses:
                return False
            
            pose = poses[arm_key]
            self.current_pos = np.array(pose['position'])
            quat = pose['orientation']
            self.current_euler = R.from_quat(quat).as_euler('xyz', degrees=True)
            return True
        except Exception as e:
            logger.error(f"读取末端位姿失败: {e}")
            return False
    
    def get_current_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取当前位姿，返回 (R_gripper2base, t_gripper2base)"""
        if not self._update_current_pose():
            return None
        
        quat = R.from_euler('xyz', self.current_euler, degrees=True).as_quat()
        R_mat = R.from_quat(quat).as_matrix()
        t = self.current_pos.reshape(3, 1)
        return R_mat, t
    
    def get_current_quat(self) -> np.ndarray:
        """获取当前四元数"""
        return R.from_euler('xyz', self.current_euler, degrees=True).as_quat()
    
    def move_delta(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0) -> bool:
        if not self._update_current_pose():
            logger.error("无法读取当前位姿")
            return False
        
        new_pos = self.current_pos + np.array([dx, dy, dz])
        new_euler = self.current_euler + np.array([droll, dpitch, dyaw])
        new_quat = R.from_euler('xyz', new_euler, degrees=True).as_quat()
        
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
                orientation=Orientation(x=float(new_quat[0]), y=float(new_quat[1]), 
                                        z=float(new_quat[2]), w=float(new_quat[3]))
            )
        }
        
        try:
            self.robot.control_arm_poses(target_pose)
            self.current_pos = new_pos
            self.current_euler = new_euler
            return True
        except Exception as e:
            logger.error(f"移动异常: {e}")
            return False
    
    def move_to_position_with_down_orientation(self, position: np.ndarray) -> bool:
        """
        移动到指定位置，末端朝下（X轴向下）
        
        末端坐标系定义：
        - X 轴：夹爪指向方向（抓取时朝下）
        - Y/Z 轴：夹爪开合方向
        
        参考 test_get_position.py 中的逻辑：
        从单位姿态（X朝前）旋转到 X 朝下，需要绕 Y 轴旋转 +90°
        """
        # X轴朝下的姿态：绕Y轴旋转90度
        # 这样夹爪的X轴指向下方（-Z方向）
        r_down = R.from_euler('y', np.pi / 2)  # 绕Y轴旋转90度
        down_quat = r_down.as_quat()  # [x, y, z, w]
        
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                orientation=Orientation(x=float(down_quat[0]), y=float(down_quat[1]), 
                                        z=float(down_quat[2]), w=float(down_quat[3]))
            )
        }
        
        try:
            self.robot.control_arm_poses(target_pose)
            return True
        except Exception as e:
            logger.error(f"移动异常: {e}")
            return False
    
    def move_to_pose(self, position: np.ndarray, quaternion: np.ndarray) -> bool:
        """移动到指定位姿"""
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                orientation=Orientation(x=float(quaternion[0]), y=float(quaternion[1]), 
                                        z=float(quaternion[2]), w=float(quaternion[3]))
            )
        }
        
        try:
            self.robot.control_arm_poses(target_pose)
            return True
        except Exception as e:
            logger.error(f"移动异常: {e}")
            return False


class HandEyeVerifier:
    """手眼标定验证器"""
    
    def __init__(self, robot_ip: str, use_left_arm: bool = True):
        print(f"[验证工具] 连接机器人 {robot_ip}...")
        self.robot = MMK2RealRobot(ip=robot_ip)
        self.mmk2 = self.robot.mmk2
        
        self.use_left_arm = use_left_arm
        self.arm_component = MMK2Components.LEFT_ARM if use_left_arm else MMK2Components.RIGHT_ARM
        self.arm_name = "left_arm" if use_left_arm else "right_arm"
        self.camera_name = "Left Hand-Eye Camera" if use_left_arm else "Right Hand-Eye Camera"
        
        # 机械臂控制器
        self.arm_controller = ArmController(self.robot, use_left_arm)
        
        # 设置图像获取
        from mmk2_types.types import ImageTypes
        self.image_goal = {
            MMK2Components.HEAD_CAMERA: [],
            MMK2Components.LEFT_CAMERA: [ImageTypes.COLOR] if use_left_arm else [],
            MMK2Components.RIGHT_CAMERA: [] if use_left_arm else [ImageTypes.COLOR],
        }
        
        # 加载标定参数
        self.intrinsics = self._load_intrinsics()
        self.extrinsics = self._load_extrinsics()
        
        if self.intrinsics is None:
            raise ValueError("内参文件不存在！")
        if self.extrinsics is None:
            raise ValueError("外参文件不存在！")
        
        # 相机内参矩阵
        self.camera_matrix = np.array([
            [self.intrinsics['fx'], 0, self.intrinsics['cx']],
            [0, self.intrinsics['fy'], self.intrinsics['cy']],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(self.intrinsics['dist_coeffs'])
        
        # 手眼变换矩阵 (相机到末端)
        self.T_cam2gripper = self._build_transform_matrix(
            np.array(self.extrinsics['rotation_matrix']),
            np.array(self.extrinsics['translation'])
        )
        
        # 3D 棋盘格角点
        self.objp = np.zeros((CHECKERBOARD['rows'] * CHECKERBOARD['cols'], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD['cols'], 0:CHECKERBOARD['rows']].T.reshape(-1, 2)
        self.objp *= CHECKERBOARD['square_size']
        
        print(f"[验证工具] 加载完成")
        print(f"  手眼平移: {self.extrinsics['translation']}")
        print(f"  手眼欧拉角: {self.extrinsics['euler_xyz_deg']}")
    
    def _build_transform_matrix(self, R_mat: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t.flatten()
        return T
    
    def _get_intrinsics_path(self) -> Path:
        side = "left" if self.use_left_arm else "right"
        return CALIB_DIR / f"hand_eye_intrinsics_{side}.json"
    
    def _get_extrinsics_path(self) -> Path:
        side = "left" if self.use_left_arm else "right"
        return CALIB_DIR / f"hand_eye_extrinsics_{side}.json"
    
    def _load_intrinsics(self) -> Optional[dict]:
        path = self._get_intrinsics_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_extrinsics(self) -> Optional[dict]:
        path = self._get_extrinsics_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_image(self) -> Optional[np.ndarray]:
        comp_images = self.mmk2.get_image(self.image_goal)
        target_comp = MMK2Components.LEFT_CAMERA if self.use_left_arm else MMK2Components.RIGHT_CAMERA
        
        for comp, images in comp_images.items():
            if comp == target_comp:
                for img_type, img in images.data.items():
                    if img is not None and len(img.shape) >= 2 and img.shape[0] > 1:
                        return cv2.resize(img, (640, 480))
        return None
    
    def _detect_board(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD['cols'], CHECKERBOARD['rows']), flags)
        
        vis = image.copy()
        
        if not ret:
            return False, None, None, vis
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, self.camera_matrix, self.dist_coeffs)
        
        if not ret:
            return False, None, None, vis
        
        R_mat, _ = cv2.Rodrigues(rvec)
        
        # 绘制棋盘格和坐标轴
        cv2.drawChessboardCorners(vis, (CHECKERBOARD['cols'], CHECKERBOARD['rows']), corners, True)
        
        axis_length = CHECKERBOARD['square_size'] * 3
        axis_points = np.float32([
            [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]
        ])
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int).reshape(-1, 2)
        
        origin = tuple(img_points[0])
        cv2.line(vis, origin, tuple(img_points[1]), (0, 0, 255), 3)
        cv2.line(vis, origin, tuple(img_points[2]), (0, 255, 0), 3)
        cv2.line(vis, origin, tuple(img_points[3]), (255, 0, 0), 3)
        
        return True, R_mat, tvec, vis
    
    def compute_board_in_base(self, R_board2cam: np.ndarray, t_board2cam: np.ndarray,
                               R_gripper2base: np.ndarray, t_gripper2base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算标定板原点在基座标系中的位置
        
        变换链: Board -> Camera -> Gripper -> Base
        T_board2base = T_gripper2base @ T_cam2gripper @ T_board2cam
        """
        T_board2cam = self._build_transform_matrix(R_board2cam, t_board2cam)
        T_gripper2base = self._build_transform_matrix(R_gripper2base, t_gripper2base)
        
        T_board2base = T_gripper2base @ self.T_cam2gripper @ T_board2cam
        
        R_board2base = T_board2base[:3, :3]
        t_board2base = T_board2base[:3, 3]
        
        return R_board2base, t_board2base
    
    def compute_board_center_in_base(self, R_board2cam: np.ndarray, t_board2cam: np.ndarray,
                                      R_gripper2base: np.ndarray, t_gripper2base: np.ndarray) -> np.ndarray:
        """
        计算标定板中心在基座标系中的位置
        标定板原点是左上角第一个内角点，中心需要偏移
        """
        # 标定板中心相对于原点的偏移（在标定板坐标系中）
        center_offset = np.array([
            (CHECKERBOARD['cols'] - 1) * CHECKERBOARD['square_size'] / 2,
            (CHECKERBOARD['rows'] - 1) * CHECKERBOARD['square_size'] / 2,
            0.0,
            1.0  # 齐次坐标
        ])
        
        T_board2cam = self._build_transform_matrix(R_board2cam, t_board2cam)
        T_gripper2base = self._build_transform_matrix(R_gripper2base, t_gripper2base)
        
        T_board2base = T_gripper2base @ self.T_cam2gripper @ T_board2cam
        
        # 标定板中心在基座标系中的位置
        center_in_base = T_board2base @ center_offset
        
        return center_in_base[:3]
    
    def _handle_arm_control(self, key: int) -> bool:
        """处理机械臂控制按键，返回是否处理了按键"""
        # 平移控制
        if key == ord('w'):
            self.arm_controller.move_delta(dx=DELTA_POS)
            return True
        elif key == ord('s'):
            self.arm_controller.move_delta(dx=-DELTA_POS)
            return True
        elif key == ord('a'):
            self.arm_controller.move_delta(dy=DELTA_POS)
            return True
        elif key == ord('e'):
            self.arm_controller.move_delta(dy=-DELTA_POS)
            return True
        elif key == ord('r'):
            self.arm_controller.move_delta(dz=DELTA_POS)
            return True
        elif key == ord('f'):
            self.arm_controller.move_delta(dz=-DELTA_POS)
            return True
        # 旋转控制
        elif key == ord('j'):
            self.arm_controller.move_delta(droll=DELTA_ROT)
            return True
        elif key == ord('l'):
            self.arm_controller.move_delta(droll=-DELTA_ROT)
            return True
        elif key == ord('i'):
            self.arm_controller.move_delta(dpitch=DELTA_ROT)
            return True
        elif key == ord('k'):
            self.arm_controller.move_delta(dpitch=-DELTA_ROT)
            return True
        elif key == ord('u'):
            self.arm_controller.move_delta(dyaw=DELTA_ROT)
            return True
        elif key == ord('o'):
            self.arm_controller.move_delta(dyaw=-DELTA_ROT)
            return True
        return False
    
    def verify_interactive(self):
        """交互式验证"""
        print("\n" + "=" * 60)
        print(f"Hand-Eye Calibration Verification - {self.camera_name}")
        print("=" * 60)
        print("Instructions:")
        print("  - Use keyboard to move arm until chessboard is visible")
        print("  - Press 'v' to verify (show board position in base frame)")
        print("  - Press 'h' to hover above board CENTER (gripper down)")
        print("  - Press 'g' to hover above board ORIGIN (gripper down)")
        print("  - Press 't' to touch board origin (CAREFUL!)")
        print("  - Press 'q' to quit")
        print("-" * 60)
        print("Arm Control Keys:")
        print("  W/S : +/- X (forward/backward)")
        print("  A/E : +/- Y (left/right)")
        print("  R/F : +/- Z (up/down)")
        print("  J/L : +/- Roll")
        print("  I/K : +/- Pitch")
        print("  U/O : +/- Yaw")
        print(f"  Step: {DELTA_POS*100:.1f}cm / {DELTA_ROT:.1f}deg")
        print("=" * 60 + "\n")
        
        window_name = f"Hand-Eye Verification - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        board_pos_base = None      # 标定板原点
        board_center_base = None   # 标定板中心
        board_R_base = None
        
        while True:
            img = self._get_image()
            if img is None:
                continue
            
            ret, R_board2cam, t_board2cam, vis = self._detect_board(img)
            
            # 获取末端位姿
            arm_result = self.arm_controller.get_current_pose()
            
            # 显示状态
            status = "Chessboard: "
            status += "DETECTED" if ret else "NOT DETECTED"
            status += " | EEF: "
            status += "OK" if arm_result else "N/A"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示控制说明
            cv2.putText(vis, "Move: W/S(X) A/E(Y) R/F(Z) | Rotate: J/L(Roll) I/K(Pitch) U/O(Yaw)", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(vis, "'v':verify 'h':hover(center) 'g':hover(origin) 't':touch 'q':quit", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            # 显示当前末端位姿
            if arm_result:
                pos = self.arm_controller.current_pos
                euler = self.arm_controller.current_euler
                pose_text = f"EEF Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m"
                cv2.putText(vis, pose_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                euler_text = f"EEF Euler: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]deg"
                cv2.putText(vis, euler_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            if ret and arm_result:
                R_gripper2base, t_gripper2base = arm_result
                
                # 计算标定板原点
                board_R_base, board_pos_base = self.compute_board_in_base(
                    R_board2cam, t_board2cam, R_gripper2base, t_gripper2base
                )
                
                # 计算标定板中心
                board_center_base = self.compute_board_center_in_base(
                    R_board2cam, t_board2cam, R_gripper2base, t_gripper2base
                )
                
                # 显示标定板在基座标系的位置
                origin_text = f"Board Origin: [{board_pos_base[0]:.4f}, {board_pos_base[1]:.4f}, {board_pos_base[2]:.4f}]m"
                cv2.putText(vis, origin_text, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                center_text = f"Board Center: [{board_center_base[0]:.4f}, {board_center_base[1]:.4f}, {board_center_base[2]:.4f}]m"
                cv2.putText(vis, center_text, (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 计算标定板在相机坐标系的距离
                dist_cam = np.linalg.norm(t_board2cam)
                dist_text = f"Distance to camera: {dist_cam:.3f}m"
                cv2.putText(vis, dist_text, (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cv2.imshow(window_name, vis)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            
            # 处理机械臂控制按键
            if self._handle_arm_control(key):
                continue
            
            if key == ord('v'):
                # 验证：打印详细信息
                if ret and arm_result:
                    print("\n" + "-" * 50)
                    print("[验证] 当前检测结果:")
                    print(f"  标定板在相机坐标系: t={t_board2cam.flatten()}")
                    print(f"  末端在基座标系: t={t_gripper2base.flatten()}")
                    print(f"  标定板原点在基座标系: t={board_pos_base}")
                    print(f"  标定板中心在基座标系: t={board_center_base}")
                    
                    euler = R.from_matrix(board_R_base).as_euler('xyz', degrees=True)
                    print(f"  标定板姿态 (euler xyz): {euler}")
                    print("-" * 50)
                else:
                    print("[验证] 未检测到标定板或无法获取末端位姿")
            
            elif key == ord('h'):
                # 悬停在标定板中心上方（末端朝下）
                if board_center_base is not None:
                    target_pos = board_center_base.copy()
                    target_pos[2] += HOVER_HEIGHT  # 悬停高度 20cm
                    
                    print(f"\n[悬停-中心] 移动到标定板中心上方 {HOVER_HEIGHT*100:.0f}cm")
                    print(f"  目标位置: {target_pos}")
                    print(f"  末端姿态: 朝下 (Z轴向下)")
                    
                    if self.arm_controller.move_to_position_with_down_orientation(target_pos):
                        print("[悬停-中心] 移动指令已发送")
                    else:
                        print("[悬停-中心] 移动失败")
                else:
                    print("[悬停-中心] 请先检测标定板")
            
            elif key == ord('g'):
                # 悬停在标定板原点上方（末端朝下）
                if board_pos_base is not None:
                    target_pos = board_pos_base.copy()
                    target_pos[2] += HOVER_HEIGHT
                    
                    print(f"\n[悬停-原点] 移动到标定板原点上方 {HOVER_HEIGHT*100:.0f}cm")
                    print(f"  目标位置: {target_pos}")
                    print(f"  末端姿态: 朝下 (Z轴向下)")
                    
                    if self.arm_controller.move_to_position_with_down_orientation(target_pos):
                        print("[悬停-原点] 移动指令已发送")
                    else:
                        print("[悬停-原点] 移动失败")
                else:
                    print("[悬停-原点] 请先检测标定板")
            
            elif key == ord('t'):
                # 触碰标定板原点 - 危险操作，需要确认
                if board_pos_base is not None:
                    print("\n" + "!" * 50)
                    print("[警告] 即将移动机械臂到标定板原点!")
                    print(f"  标定板原点: {board_pos_base}")
                    print(f"  悬停高度: {HOVER_HEIGHT*100:.0f}cm")
                    print(f"  最终高度: 标定板上方 2cm")
                    confirm = input("  确认执行? (yes/no): ").strip().lower()
                    
                    if confirm == 'yes':
                        # 先悬停
                        hover_pos = board_pos_base.copy()
                        hover_pos[2] += HOVER_HEIGHT
                        
                        print(f"[触碰] 1. 先悬停到 {hover_pos}")
                        if self.arm_controller.move_to_position_with_down_orientation(hover_pos):
                            print("[触碰] 悬停完成，3秒后下降...")
                            time.sleep(3)
                            
                            # 下降到标定板上方2cm
                            touch_pos = board_pos_base.copy()
                            touch_pos[2] += 0.02
                            
                            print(f"[触碰] 2. 下降到 {touch_pos}")
                            if self.arm_controller.move_to_position_with_down_orientation(touch_pos):
                                print("[触碰] 完成! 请检查末端是否对准标定板原点")
                            else:
                                print("[触碰] 下降失败")
                        else:
                            print("[触碰] 悬停失败")
                    else:
                        print("[触碰] 已取消")
                else:
                    print("[触碰] 请先检测标定板")
        
        cv2.destroyAllWindows()
    
    def compute_reprojection_error(self):
        """计算多帧的重投影误差（一致性测试）"""
        print("\n" + "=" * 60)
        print("Reprojection Error Test (Consistency Test)")
        print("=" * 60)
        print("Test: Move arm to different poses, check if board position stays consistent")
        print("Use keyboard to move arm, then press 'c' to capture")
        print("Arm Control: W/S(X) A/E(Y) R/F(Z) J/L(Roll) I/K(Pitch) U/O(Yaw)")
        print("Press 'c' to capture, 'r' to reset, 'q' to quit")
        print("=" * 60 + "\n")
        
        window_name = "Reprojection Error Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        errors = []
        reference_pos = None
        
        while True:
            img = self._get_image()
            if img is None:
                continue
            
            ret, R_board2cam, t_board2cam, vis = self._detect_board(img)
            arm_result = self.arm_controller.get_current_pose()
            
            # 显示状态
            status = f"Samples: {len(errors)}"
            if errors and len(errors) > 1:
                status += f" | Mean Error: {np.mean(errors[1:]):.4f}m ({np.mean(errors[1:])*100:.2f}cm)"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示控制说明
            cv2.putText(vis, "Move: W/S(X) A/E(Y) R/F(Z) | Rotate: J/L(Roll) I/K(Pitch) U/O(Yaw)", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(vis, "'c':capture  'r':reset  'q':quit", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # 显示参考位置
            if reference_pos is not None:
                ref_text = f"Reference: [{reference_pos[0]:.4f}, {reference_pos[1]:.4f}, {reference_pos[2]:.4f}]m"
                cv2.putText(vis, ref_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 显示当前末端位姿
            if arm_result:
                pos = self.arm_controller.current_pos
                pose_text = f"EEF: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m"
                cv2.putText(vis, pose_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.imshow(window_name, vis)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            
            # 处理机械臂控制按键
            if self._handle_arm_control(key):
                continue
            
            if key == ord('r'):
                # 重置
                errors = []
                reference_pos = None
                print("[重置] 已清除所有采样数据")
            
            elif key == ord('c') and ret and arm_result:
                R_gripper2base, t_gripper2base = arm_result
                _, t_board2base = self.compute_board_in_base(
                    R_board2cam, t_board2cam, R_gripper2base, t_gripper2base
                )
                
                if reference_pos is None:
                    reference_pos = t_board2base.copy()
                    errors.append(0.0)
                    print(f"[采集 #{len(errors)}] 参考位置: {reference_pos}")
                else:
                    error = np.linalg.norm(t_board2base - reference_pos)
                    errors.append(error)
                    print(f"[采集 #{len(errors)}] 位置: {t_board2base}")
                    print(f"           误差: {error:.4f}m ({error*100:.2f}cm)")
        
        if len(errors) > 1:
            print("\n" + "=" * 50)
            print(f"[结果] 采集 {len(errors)} 帧 (第1帧为参考)")
            print(f"  平均误差: {np.mean(errors[1:]):.4f}m ({np.mean(errors[1:])*100:.2f}cm)")
            print(f"  最大误差: {np.max(errors[1:]):.4f}m ({np.max(errors[1:])*100:.2f}cm)")
            print(f"  标准差: {np.std(errors[1:]):.4f}m ({np.std(errors[1:])*100:.2f}cm)")
            print("=" * 50)
            
            if np.mean(errors[1:]) < 0.01:
                print("✓ 标定质量: 优秀 (平均误差 < 1cm)")
            elif np.mean(errors[1:]) < 0.02:
                print("✓ 标定质量: 良好 (平均误差 < 2cm)")
            elif np.mean(errors[1:]) < 0.03:
                print("△ 标定质量: 一般 (平均误差 < 3cm)")
            else:
                print("✗ 标定质量: 较差 (平均误差 >= 3cm)，建议重新标定")
        
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hand-Eye Calibration Verification Tool')
    parser.add_argument("--ip", type=str, default="192.168.11.200")
    parser.add_argument("--left", action="store_true", help="Verify left hand-eye camera")
    parser.add_argument("--right", action="store_true", help="Verify right hand-eye camera")
    args = parser.parse_args()
    
    if args.left:
        use_left_arm = True
    elif args.right:
        use_left_arm = False
    else:
        print("\n请选择验证哪个手眼相机:")
        print("  1. Left Hand-Eye Camera")
        print("  2. Right Hand-Eye Camera")
        choice = input("选择 (1/2): ").strip()
        use_left_arm = (choice == '1')
    
    verifier = HandEyeVerifier(robot_ip=args.ip, use_left_arm=use_left_arm)
    
    print("\n验证模式:")
    print("  1. 交互式验证 (推荐)")
    print("  2. 重投影误差测试 (一致性测试)")
    mode = input("选择 (1/2): ").strip()
    
    if mode == '1':
        verifier.verify_interactive()
    elif mode == '2':
        verifier.compute_reprojection_error()
    else:
        verifier.verify_interactive()


if __name__ == '__main__':
    main()

# """根据欧拉角计算旋转矩阵和四元数"""

# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import json

# # 你记录的参数
# # translation = [-0.2219, 0.0404, 0.1859]
# # euler_xyz_deg = [125.02, 30.50, 99.99]
# translation = [-0.2875, 0.0387, 0.1833]
# euler_xyz_deg = [122.82, 30.34, 93.97]

# # 计算旋转
# rotation = R.from_euler('xyz', euler_xyz_deg, degrees=True)

# # 旋转矩阵
# rotation_matrix = rotation.as_matrix()

# # 四元数 [x, y, z, w]
# quaternion = rotation.as_quat()

# # 构造完整的外参
# extrinsics = {
#     'translation': translation,
#     'rotation_matrix': rotation_matrix.tolist(),
#     'quaternion': quaternion.tolist(),  # [x, y, z, w]
#     'euler_xyz_deg': euler_xyz_deg,
#     'num_samples': 70  # 根据你之前的采样数
# }

# print("=" * 50)
# print("计算结果:")
# print("=" * 50)
# print(f"translation: {translation}")
# print(f"euler_xyz_deg: {euler_xyz_deg}")
# print(f"quaternion (xyzw): [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
# print(f"rotation_matrix:")
# for row in rotation_matrix:
#     print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
# print("=" * 50)

# # 保存到文件
# output_path = "/home/ypf/qiuzhiarm_LLM/calibration/hand_eye_extrinsics_left.json"
# with open(output_path, 'w') as f:
#     json.dump(extrinsics, f, indent=2)
# print(f"\n已保存到: {output_path}")