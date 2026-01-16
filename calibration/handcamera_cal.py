"""
手眼标定工具
1. 先标定相机内参
2. 再标定手眼外参（相机相对末端的变换）

使用方法：
1. 准备棋盘格标定板（9x6内角点，每格22mm）
2. 运行脚本，按提示操作
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from mmk2_types.types import MMK2Components
from mmk2_types.grpc_msgs import (
    Pose,
    Position,
    Orientation,
)
from airbot_py.airbot_mmk2 import AirbotMMK2
from airbot_sdk import MMK2RealRobot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 棋盘格参数
CHECKERBOARD = {
    'rows': 6,           # 内角点行数
    'cols': 9,           # 内角点列数
    'square_size': 0.022  # 每格边长（米）
}

# 保存路径
CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")

# 增量控制参数
DELTA_POS = 0.01      # 平移增量：1cm
DELTA_ROT = 2.0       # 旋转增量：2度


class CameraIntrinsicCalibrator:
    """相机内参标定器"""
    
    def __init__(self, checkerboard: dict = CHECKERBOARD):
        self.rows = checkerboard['rows']
        self.cols = checkerboard['cols']
        self.square_size = checkerboard['square_size']
        
        # 3D 角点坐标（标定板坐标系）
        self.objp = np.zeros((self.rows * self.cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # 存储数据
        self.obj_points: List[np.ndarray] = []  # 3D 点
        self.img_points: List[np.ndarray] = []  # 2D 点
        self.image_size: Optional[Tuple[int, int]] = None
    
    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
        """检测棋盘格角点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.image_size = (gray.shape[1], gray.shape[0])
        
        # 检测角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), flags)
        
        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 绘制结果
        vis = image.copy()
        cv2.drawChessboardCorners(vis, (self.cols, self.rows), corners, ret)
        
        return ret, corners, vis
    
    def add_sample(self, corners: np.ndarray):
        """添加一组标定数据"""
        self.obj_points.append(self.objp)
        self.img_points.append(corners)
        print(f"[内参标定] 已采集 {len(self.obj_points)} 组数据")
    
    def calibrate(self) -> Optional[dict]:
        """执行标定"""
        if len(self.obj_points) < 10:
            print(f"[内参标定] 数据不足，需要至少 10 组，当前 {len(self.obj_points)} 组")
            return None
        
        print(f"[内参标定] 开始标定，使用 {len(self.obj_points)} 组数据...")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.image_size, None, None
        )
        
        if not ret:
            print("[内参标定] 标定失败")
            return None
        
        # 计算重投影误差
        total_error = 0
        for i in range(len(self.obj_points)):
            img_points_reproj, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.img_points[i], img_points_reproj, cv2.NORM_L2)
            total_error += error ** 2
        mean_error = np.sqrt(total_error / (len(self.obj_points) * self.rows * self.cols))
        
        result = {
            'fx': float(mtx[0, 0]),
            'fy': float(mtx[1, 1]),
            'cx': float(mtx[0, 2]),
            'cy': float(mtx[1, 2]),
            'dist_coeffs': dist.flatten().tolist(),
            'image_size': list(self.image_size),
            'reprojection_error': float(mean_error),
            'num_samples': len(self.obj_points),
        }
        
        print(f"[内参标定] 完成!")
        print(f"  fx={result['fx']:.2f}, fy={result['fy']:.2f}")
        print(f"  cx={result['cx']:.2f}, cy={result['cy']:.2f}")
        print(f"  重投影误差: {mean_error:.4f} 像素")
        
        return result


class HandEyeCalibrator:
    """手眼标定器（Eye-in-Hand）"""
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 checkerboard: dict = CHECKERBOARD):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rows = checkerboard['rows']
        self.cols = checkerboard['cols']
        self.square_size = checkerboard['square_size']
        
        # 3D 角点
        self.objp = np.zeros((self.rows * self.cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # 存储数据
        self.R_gripper2base: List[np.ndarray] = []  # 末端到基座的旋转
        self.t_gripper2base: List[np.ndarray] = []  # 末端到基座的平移
        self.R_board2cam: List[np.ndarray] = []     # 标定板到相机的旋转
        self.t_board2cam: List[np.ndarray] = []     # 标定板到相机的平移
    
    def find_board_pose(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """检测标定板并计算位姿"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), flags)
        
        vis = image.copy()
        
        if not ret:
            return False, None, None, vis
        
        # 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 计算位姿
        ret, rvec, tvec = cv2.solvePnP(
            self.objp, corners, self.camera_matrix, self.dist_coeffs
        )
        
        if not ret:
            return False, None, None, vis
        
        # 绘制坐标轴
        cv2.drawChessboardCorners(vis, (self.cols, self.rows), corners, True)
        axis_length = self.square_size * 3
        axis_points = np.float32([
            [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]
        ])
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int).reshape(-1, 2)
        
        origin = tuple(img_points[0])
        cv2.line(vis, origin, tuple(img_points[1]), (0, 0, 255), 3)  # X - 红
        cv2.line(vis, origin, tuple(img_points[2]), (0, 255, 0), 3)  # Y - 绿
        cv2.line(vis, origin, tuple(img_points[3]), (255, 0, 0), 3)  # Z - 蓝
        
        R_mat, _ = cv2.Rodrigues(rvec)
        
        return True, R_mat, tvec, vis
    
    def add_sample(self, R_board2cam: np.ndarray, t_board2cam: np.ndarray,
                   end_pose: dict):
        """添加一组标定数据"""
        # 从末端位姿提取旋转和平移
        pos = end_pose['position']
        quat = end_pose['orientation']
        
        R_end = R.from_quat(quat).as_matrix()
        t_end = np.array([[pos[0]], [pos[1]], [pos[2]]])
        
        self.R_gripper2base.append(R_end)
        self.t_gripper2base.append(t_end)
        self.R_board2cam.append(R_board2cam)
        self.t_board2cam.append(t_board2cam)
        
        print(f"[手眼标定] 已采集 {len(self.R_gripper2base)} 组数据")
    
    def calibrate(self) -> Optional[dict]:
        """执行手眼标定"""
        if len(self.R_gripper2base) < 10:
            print(f"[手眼标定] 数据不足，需要至少 10 组，当前 {len(self.R_gripper2base)} 组")
            return None
        
        print(f"[手眼标定] 开始标定，使用 {len(self.R_gripper2base)} 组数据...")
        
        # OpenCV 手眼标定
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_board2cam, self.t_board2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        # 转换为四元数
        r = R.from_matrix(R_cam2gripper)
        quat = r.as_quat()  # [x, y, z, w]
        euler = r.as_euler('xyz', degrees=True)
        
        result = {
            'translation': t_cam2gripper.flatten().tolist(),
            'rotation_matrix': R_cam2gripper.tolist(),
            'quaternion': quat.tolist(),  # [x, y, z, w]
            'euler_xyz_deg': euler.tolist(),
            'num_samples': len(self.R_gripper2base),
        }
        
        print(f"[手眼标定] 完成!")
        print(f"  平移: [{result['translation'][0]:.4f}, {result['translation'][1]:.4f}, {result['translation'][2]:.4f}]m")
        print(f"  欧拉角 (xyz): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]°")
        
        return result


class ArmController:
    """机械臂增量控制器 - 使用 airbot_sdk"""
    
    def __init__(self, robot: MMK2RealRobot, use_left_arm: bool = True):
        self.robot = robot
        self.use_left_arm = use_left_arm
        self.arm_component = MMK2Components.LEFT_ARM if use_left_arm else MMK2Components.RIGHT_ARM
        self.arm_name = "left_arm" if use_left_arm else "right_arm"
        
        # 当前末端位姿 (xyz + euler_xyz)
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_euler = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw (degrees)
        
        # 初始化当前位姿
        self._update_current_pose()
    
    def _update_current_pose(self) -> bool:
        """从机器人读取当前末端位姿"""
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
    
    def get_current_pose_dict(self) -> Optional[dict]:
        """获取当前位姿（字典格式）"""
        self._update_current_pose()
        quat = R.from_euler('xyz', self.current_euler, degrees=True).as_quat()
        return {
            'position': self.current_pos.tolist(),
            'orientation': quat.tolist()  # [x, y, z, w]
        }
    
    def move_delta(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0) -> bool:
        """
        增量移动末端
        Args:
            dx, dy, dz: 平移增量（米）
            droll, dpitch, dyaw: 旋转增量（度）
        """
        # 先更新当前位姿
        if not self._update_current_pose():
            logger.error("无法读取当前位姿")
            return False
        
        # 计算新位姿
        new_pos = self.current_pos + np.array([dx, dy, dz])
        new_euler = self.current_euler + np.array([droll, dpitch, dyaw])
        
        # 欧拉角转四元数
        new_quat = R.from_euler('xyz', new_euler, degrees=True).as_quat()  # [x, y, z, w]
        
        # 构造目标位姿 - 使用 control_arm_poses 的格式
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])),
                orientation=Orientation(x=float(new_quat[0]), y=float(new_quat[1]), 
                                        z=float(new_quat[2]), w=float(new_quat[3]))
            )
        }
        
        # 发送指令 - 使用 airbot_sdk 的 control_arm_poses
        try:
            self.robot.control_arm_poses(target_pose)
            self.current_pos = new_pos
            self.current_euler = new_euler
            logger.info(f"移动到: pos=[{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]")
            return True
        except Exception as e:
            logger.error(f"移动异常: {e}")
            return False


class CalibrationTool:
    """标定工具主程序"""
    
    def __init__(self, robot_ip: str, use_left_arm: bool = True):
        print(f"[标定工具] 连接机器人 {robot_ip}...")
        
        # 使用 MMK2RealRobot 替代直接使用 AirbotMMK2
        self.robot = MMK2RealRobot(ip=robot_ip)
        self.mmk2 = self.robot.mmk2  # 保留对底层的引用，用于获取图像
        
        self.use_left_arm = use_left_arm
        self.arm_name = "left_arm" if use_left_arm else "right_arm"
        self.camera_name = "Left Hand-Eye Camera" if use_left_arm else "Right Hand-Eye Camera"
        
        # 相机已在 MMK2RealRobot 中初始化，设置 image_goal
        self.image_goal = {
            MMK2Components.HEAD_CAMERA: [],
            MMK2Components.LEFT_CAMERA: [],
            MMK2Components.RIGHT_CAMERA: [],
        }
        from mmk2_types.types import ImageTypes
        self.image_goal[MMK2Components.LEFT_CAMERA] = [ImageTypes.COLOR]
        self.image_goal[MMK2Components.RIGHT_CAMERA] = [ImageTypes.COLOR]
        
        # 机械臂控制器 - 使用 MMK2RealRobot
        self.arm_controller = ArmController(self.robot, use_left_arm)
        
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        
        # 加载已有的内参（如果存在）
        self.intrinsics = self._load_intrinsics()
    
    def _init_camera(self):
        """初始化相机"""
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
        print("等待相机初始化...")
        time.sleep(5)
        print("相机初始化完成")
        
        self.image_goal = {
            MMK2Components.HEAD_CAMERA: [],
            MMK2Components.LEFT_CAMERA: [],
            MMK2Components.RIGHT_CAMERA: [],
        }
        from mmk2_types.types import ImageTypes
        self.image_goal[MMK2Components.LEFT_CAMERA] = [ImageTypes.COLOR]
        self.image_goal[MMK2Components.RIGHT_CAMERA] = [ImageTypes.COLOR]
    
    def _get_image(self) -> Optional[np.ndarray]:
        """获取当前手眼相机图像"""
        from mmk2_types.types import ImageTypes
        comp_images = self.mmk2.get_image(self.image_goal)
        
        target_comp = MMK2Components.LEFT_CAMERA if self.use_left_arm else MMK2Components.RIGHT_CAMERA
        
        for comp, images in comp_images.items():
            if comp == target_comp:
                for img_type, img in images.data.items():
                    if img is not None and len(img.shape) >= 2 and img.shape[0] > 1:
                        return cv2.resize(img, (640, 480))
        return None
    
    def _get_intrinsics_path(self) -> Path:
        """获取内参文件路径"""
        side = "left" if self.use_left_arm else "right"
        return CALIB_DIR / f"hand_eye_intrinsics_{side}.json"
    
    def _get_extrinsics_path(self) -> Path:
        """获取外参文件路径"""
        side = "left" if self.use_left_arm else "right"
        return CALIB_DIR / f"hand_eye_extrinsics_{side}.json"
        
    def _load_intrinsics(self) -> Optional[dict]:
        """加载内参"""
        path = self._get_intrinsics_path()
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"[标定工具] 已加载 {self.camera_name} 内参: fx={data['fx']:.2f}, fy={data['fy']:.2f}")
                return data
        return None
    
    def _save_intrinsics(self, data: dict):
        """保存内参"""
        path = self._get_intrinsics_path()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[标定工具] {self.camera_name} 内参已保存到 {path}")
    
    def _save_extrinsics(self, data: dict):
        """保存外参"""
        path = self._get_extrinsics_path()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[标定工具] {self.camera_name} 外参已保存到 {path}")
    
    def calibrate_intrinsics(self):
        """标定相机内参"""
        print("\n" + "=" * 60)
        print(f"{self.camera_name} Intrinsic Calibration")
        print("=" * 60)
        print("Instructions:")
        print("  - Place chessboard in camera view")
        print("  - Press 'c' to capture (when chessboard detected)")
        print("  - Press 'b' to begin calibration (need >= 10 samples)")
        print("  - Press 'q' to quit")
        print("  - Recommend: 15-20 samples with different angles")
        print("=" * 60 + "\n")
        
        calibrator = CameraIntrinsicCalibrator()
        window_name = f"Intrinsic Calibration - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            img = self._get_image()
            if img is None:
                continue
            
            ret, corners, vis = calibrator.find_corners(img)
            
            status = f"Captured: {len(calibrator.obj_points)}"
            status += " | Chessboard detected!" if ret else " | Chessboard not detected"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, "'c':capture  'b':calibrate  'q':quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(window_name, vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and ret:
                calibrator.add_sample(corners)
            elif key == ord('b'):  # 改为 'b' (begin calibration)
                result = calibrator.calibrate()
                if result:
                    self.intrinsics = result
                    self._save_intrinsics(result)
                    print("\n内参标定完成，按 'q' 退出")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def calibrate_hand_eye(self):
        """标定手眼外参（带键盘控制机械臂）"""
        if self.intrinsics is None:
            print(f"[错误] 请先标定 {self.camera_name} 内参！")
            return
        
        print("\n" + "=" * 60)
        print(f"{self.camera_name} Hand-Eye Calibration")
        print("=" * 60)
        print("Instructions:")
        print("  - Fix chessboard on table (DO NOT MOVE!)")
        print("  - Use keyboard to move arm, keep chessboard in view")
        print("  - Press 'c' to capture (when chessboard detected)")
        print("  - Press 'b' to begin calibration (need >= 10 samples)")
        print("  - Press 'q' to quit")
        print("-" * 60)
        print("Arm Control Keys:")
        print("  W/S : +/- X (forward/backward)")
        print("  A/D : +/- Y (left/right)")
        print("  R/F : +/- Z (up/down)")
        print("  J/L : +/- Roll")
        print("  I/K : +/- Pitch")
        print("  U/O : +/- Yaw")
        print(f"  Step: {DELTA_POS*100:.1f}cm / {DELTA_ROT:.1f}deg")
        print("=" * 60 + "\n")
        
        camera_matrix = np.array([
            [self.intrinsics['fx'], 0, self.intrinsics['cx']],
            [0, self.intrinsics['fy'], self.intrinsics['cy']],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(self.intrinsics['dist_coeffs'])
        
        calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)
        window_name = f"Hand-Eye Calibration - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            img = self._get_image()
            if img is None:
                continue
            
            ret, R_board, t_board, vis = calibrator.find_board_pose(img)
            
            # 获取当前末端位姿
            arm_pose = self.arm_controller.get_current_pose_dict()
            pose_str = "OK" if arm_pose else "N/A"
            
            # 显示状态
            status1 = f"Captured: {len(calibrator.R_gripper2base)} | EEF: {pose_str}"
            status1 += " | Chessboard detected!" if ret else " | Chessboard not detected"
            cv2.putText(vis, status1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示控制说明
            cv2.putText(vis, "Move: W/S(X) A/D(Y) R/F(Z) | Rotate: J/L(Roll) I/K(Pitch) U/O(Yaw)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(vis, "'c':capture  'b':calibrate  'q':quit", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # 显示当前位姿
            if arm_pose:
                pos = arm_pose['position']
                euler = self.arm_controller.current_euler
                pose_text = f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m"
                cv2.putText(vis, pose_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                euler_text = f"Euler: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]deg"
                cv2.putText(vis, euler_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.imshow(window_name, vis)
            
            key = cv2.waitKey(30) & 0xFF
            
            # 采集数据
            if key == ord('c') and ret and arm_pose:
                calibrator.add_sample(R_board, t_board, arm_pose)
            
            # 执行标定 - 改为 'b'
            elif key == ord('b'):
                result = calibrator.calibrate()
                if result:
                    self._save_extrinsics(result)
                    print("\n手眼标定完成！")
            
            # 退出
            elif key == ord('q'):
                break
            
            # ========== 机械臂控制 ==========
            # 平移控制
            elif key == ord('w'):  # +X
                self.arm_controller.move_delta(dx=DELTA_POS)
            elif key == ord('s'):  # -X
                self.arm_controller.move_delta(dx=-DELTA_POS)
            elif key == ord('a'):  # +Y
                self.arm_controller.move_delta(dy=DELTA_POS)
            elif key == ord('d'):  # -Y
                self.arm_controller.move_delta(dy=-DELTA_POS)
            elif key == ord('r'):  # +Z
                self.arm_controller.move_delta(dz=DELTA_POS)
            elif key == ord('f'):  # -Z
                self.arm_controller.move_delta(dz=-DELTA_POS)
            
            # 旋转控制
            elif key == ord('j'):  # +Roll
                self.arm_controller.move_delta(droll=DELTA_ROT)
            elif key == ord('l'):  # -Roll
                self.arm_controller.move_delta(droll=-DELTA_ROT)
            elif key == ord('i'):  # +Pitch
                self.arm_controller.move_delta(dpitch=DELTA_ROT)
            elif key == ord('k'):  # -Pitch
                self.arm_controller.move_delta(dpitch=-DELTA_ROT)
            elif key == ord('u'):  # +Yaw
                self.arm_controller.move_delta(dyaw=DELTA_ROT)
            elif key == ord('o'):  # -Yaw
                self.arm_controller.move_delta(dyaw=-DELTA_ROT)
        
        cv2.destroyAllWindows()
    
    def run(self):
        """运行标定工具"""
        print("\n" + "=" * 60)
        print(f"Hand-Eye Calibration Tool - Selected: {self.camera_name}")
        print("=" * 60)
        print("Options:")
        print("  1. Calibrate camera intrinsics")
        print("  2. Calibrate hand-eye extrinsics")
        print("  3. Full calibration (intrinsics + extrinsics)")
        print("  q. Quit")
        print("=" * 60)
        
        choice = input("Enter option: ").strip()
        
        if choice == '1':
            self.calibrate_intrinsics()
        elif choice == '2':
            self.calibrate_hand_eye()
        elif choice == '3':
            self.calibrate_intrinsics()
            if self.intrinsics:
                self.calibrate_hand_eye()
        elif choice == 'q':
            print("Quit")
        else:
            print("Invalid option")


def select_camera() -> bool:
    """选择要标定的相机，返回 True 表示左臂，False 表示右臂"""
    print("\n" + "=" * 60)
    print("Select Hand-Eye Camera")
    print("=" * 60)
    print("  1. Left Hand-Eye Camera (Left Arm)")
    print("  2. Right Hand-Eye Camera (Right Arm)")
    print("=" * 60)
    
    while True:
        choice = input("Enter option (1/2): ").strip()
        if choice == '1':
            print("Selected: Left Hand-Eye Camera")
            return True
        elif choice == '2':
            print("Selected: Right Hand-Eye Camera")
            return False
        else:
            print("Invalid option, please enter 1 or 2")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hand-Eye Calibration Tool')
    parser.add_argument("--ip", type=str, default="192.168.11.200")
    parser.add_argument("--left", action="store_true", help="Use left hand-eye camera")
    parser.add_argument("--right", action="store_true", help="Use right hand-eye camera")
    args = parser.parse_args()
    
    # 确定使用哪个相机
    if args.left:
        use_left_arm = True
    elif args.right:
        use_left_arm = False
    else:
        use_left_arm = select_camera()
    
    tool = CalibrationTool(robot_ip=args.ip, use_left_arm=use_left_arm)
    tool.run()


if __name__ == '__main__':
    main()