# filepath: /home/ypf/qiuzhiarm_LLM/test_get_position.py
"""
砖块检测器 - 两阶段抓取流程
阶段1: 头部深度相机粗定位
阶段2: 手眼相机精确分割

按键: s-头部分割 g-移动机械臂 e-手眼分割 p-打印位姿 t-TF详情 r-重置 q-退出
"""

import sys
from pathlib import Path

# Add parent directory to path for importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ypf/sam3-main")

import cv2
import argparse
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import gc
from typing import Optional, List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

from airbot_sdk import MMK2RealRobot, START_JOINT_ACTION
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tf_client import TFClient
from mmk2_types.types import MMK2Components, ImageTypes
from mmk2_types.grpc_msgs import Pose, Position, Orientation, TrajectoryParams, GoalStatus

# 配置常量
HEAD_INTRINSICS = {'fx': 607.15, 'fy': 607.02, 'cx': 324.25, 'cy': 248.46}
BRICK_SIZE = {'length': 0.11, 'width': 0.05, 'height': 0.025}
HOVER_HEIGHT = 0.15
CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")
COLORS = [(0,255,0), (255,0,0), (0,255,255), (255,0,255), (0,165,255), (255,255,0)]


class SAM3Segmenter:
    """SAM3 分割器（单例）"""
    _instance = None
    
    def __new__(cls, checkpoint_path: str, confidence: float = 0.5):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("[SAM3] 加载模型...")
            cls._instance.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            cls._instance.processor = Sam3Processor(cls._instance.model, resolution=1008, confidence_threshold=confidence)
            torch.cuda.empty_cache()
            gc.collect()
            print("[SAM3] 加载完成")
        return cls._instance
    
    def segment(self, img_bgr: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        state = self.processor.set_image(pil_img)
        out = self.processor.set_text_prompt(state=state, prompt=prompt)
        masks = out["masks"].cpu().numpy() if out["masks"] is not None else None
        torch.cuda.empty_cache()
        return masks


def estimate_orientation(mask: np.ndarray) -> Tuple[float, Optional[Dict]]:
    """从掩码估算朝向"""
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
    
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0, None
    
    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), angle = rect
    long_angle = angle + 90 if w < h else angle
    yaw = -np.radians(long_angle)
    
    # 归一化到 [-π/2, π/2]
    while yaw > np.pi/2: yaw -= np.pi
    while yaw < -np.pi/2: yaw += np.pi
    
    return float(yaw), {
        'center': (cx, cy), 'size': (w, h), 'angle': angle,
        'long_edge': max(w, h), 'short_edge': min(w, h),
        'box_points': cv2.boxPoints(rect)
    }


class DynamicZCompensator:
    """动态 Z 轴补偿器"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.calibration_points: List[Dict] = []
        self.model_type = "linear"  # linear, quadratic
        self.coefficients = {
            'intercept': 0.0,
            'x_coeff': 0.0,
            'y_coeff': 0.0,
            'xy_coeff': 0.0,
            'x2_coeff': 0.0,
            'y2_coeff': 0.0,
        }
        self.enabled = False
        self._load_config()
    
    def _load_config(self):
        """从配置文件加载"""
        if not self.config_path.exists():
            print("[Z补偿] 配置文件不存在，使用默认值")
            return
        
        try:
            with open(self.config_path) as f:
                data = json.load(f)
            
            dz = data.get('dynamic_z_compensation', {})
            self.enabled = dz.get('enabled', False)
            self.model_type = dz.get('model', 'linear')
            self.coefficients = dz.get('coefficients', self.coefficients)
            self.calibration_points = dz.get('calibration_points', [])
            
            if self.enabled:
                print(f"[Z补偿] 已加载动态补偿模型: {self.model_type}")
                print(f"  系数: {self.coefficients}")
                print(f"  标定点数: {len(self.calibration_points)}")
        except Exception as e:
            print(f"[Z补偿] 加载配置失败: {e}")
    
    def _save_config(self):
        """保存到配置文件"""
        # 读取现有配置
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
        else:
            data = {
                "offset_xyz": [0.0, 0.0, 0.0],
                "std_xyz": [0.0, 0.0, 0.0],
                "num_samples": 0
            }
        
        # 更新动态补偿部分
        data['dynamic_z_compensation'] = {
            'enabled': self.enabled,
            'model': self.model_type,
            'coefficients': self.coefficients,
            'calibration_points': self.calibration_points
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Z补偿] 已保存到 {self.config_path}")
    
    def compute_compensation(self, x: float, y: float) -> float:
        """根据 X, Y 坐标计算 Z 补偿值"""
        if not self.enabled:
            return 0.0
        
        c = self.coefficients
        
        if self.model_type == "linear":
            # Z_comp = a + b*X + c*Y
            return c['intercept'] + c.get('x_coeff', 0.0) * x + c['y_coeff'] * y
        
        elif self.model_type == "quadratic":
            # Z_comp = a + b*X + c*Y + d*XY + e*X² + f*Y²
            return (c['intercept'] + 
                    c.get('x_coeff', 0.0) * x + 
                    c['y_coeff'] * y +
                    c.get('xy_coeff', 0.0) * x * y +
                    c.get('x2_coeff', 0.0) * x * x +
                    c.get('y2_coeff', 0.0) * y * y)
        
        return 0.0
    
    def add_calibration_point(self, detected_pos: np.ndarray, actual_pos: np.ndarray):
        """添加一个标定点"""
        point = {
            'detected': {
                'x': float(detected_pos[0]),
                'y': float(detected_pos[1]),
                'z': float(detected_pos[2])
            },
            'actual': {
                'x': float(actual_pos[0]),
                'y': float(actual_pos[1]),
                'z': float(actual_pos[2])
            },
            'z_error': float(actual_pos[2] - detected_pos[2]),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.calibration_points.append(point)
        print(f"[Z补偿] 已添加标定点 #{len(self.calibration_points)}")
        print(f"  检测位置: [{detected_pos[0]:.4f}, {detected_pos[1]:.4f}, {detected_pos[2]:.4f}]")
        print(f"  实际位置: [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")
        print(f"  Z误差:    {point['z_error']:+.4f}m")
        
        self._save_config()
    
    def fit_model(self, model_type: str = "linear") -> bool:
        """拟合补偿模型"""
        if len(self.calibration_points) < 3:
            print(f"[Z补偿] 标定点不足（需要至少3个，当前{len(self.calibration_points)}个）")
            return False
        
        self.model_type = model_type
        
        # 提取数据
        X = np.array([[p['detected']['x'], p['detected']['y']] for p in self.calibration_points])
        z_errors = np.array([p['z_error'] for p in self.calibration_points])
        
        print(f"\n[Z补偿] 拟合 {model_type} 模型，使用 {len(self.calibration_points)} 个标定点")
        print("-" * 60)
        
        # 打印标定点
        print(f"{'#':<3} {'X':>8} {'Y':>8} {'Z检测':>8} {'Z实际':>8} {'Z误差':>8}")
        for i, p in enumerate(self.calibration_points):
            print(f"{i+1:<3} {p['detected']['x']:>8.4f} {p['detected']['y']:>8.4f} "
                  f"{p['detected']['z']:>8.4f} {p['actual']['z']:>8.4f} {p['z_error']:>+8.4f}")
        print("-" * 60)
        
        if model_type == "linear":
            # Z_error = a + b*X + c*Y
            # 使用最小二乘法
            A = np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1]])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, z_errors, rcond=None)
            
            self.coefficients = {
                'intercept': float(coeffs[0]),
                'x_coeff': float(coeffs[1]),
                'y_coeff': float(coeffs[2]),
            }
            
            print(f"线性模型: Z_comp = {coeffs[0]:.6f} + {coeffs[1]:.6f}*X + {coeffs[2]:.6f}*Y")
            
        elif model_type == "quadratic":
            # Z_error = a + b*X + c*Y + d*XY + e*X² + f*Y²
            x, y = X[:, 0], X[:, 1]
            A = np.column_stack([np.ones(len(x)), x, y, x*y, x*x, y*y])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, z_errors, rcond=None)
            
            self.coefficients = {
                'intercept': float(coeffs[0]),
                'x_coeff': float(coeffs[1]),
                'y_coeff': float(coeffs[2]),
                'xy_coeff': float(coeffs[3]),
                'x2_coeff': float(coeffs[4]),
                'y2_coeff': float(coeffs[5]),
            }
            
            print(f"二次模型: Z_comp = {coeffs[0]:.6f} + {coeffs[1]:.6f}*X + {coeffs[2]:.6f}*Y")
            print(f"                  + {coeffs[3]:.6f}*XY + {coeffs[4]:.6f}*X² + {coeffs[5]:.6f}*Y²")
        
        # 计算拟合误差
        fitted_errors = np.array([self.compute_compensation(p['detected']['x'], p['detected']['y']) 
                                   for p in self.calibration_points])
        residuals = z_errors - fitted_errors
        rmse = np.sqrt(np.mean(residuals**2))
        max_err = np.max(np.abs(residuals))
        
        print(f"\n拟合质量:")
        print(f"  RMSE:     {rmse*1000:.2f}mm")
        print(f"  最大误差: {max_err*1000:.2f}mm")
        
        # 显示每个点的拟合效果
        print(f"\n{'#':<3} {'Z误差':>8} {'拟合值':>8} {'残差':>8}")
        for i, (ze, fe) in enumerate(zip(z_errors, fitted_errors)):
            print(f"{i+1:<3} {ze:>+8.4f} {fe:>+8.4f} {ze-fe:>+8.4f}")
        
        self.enabled = True
        self._save_config()
        
        print("-" * 60)
        print("[Z补偿] 模型已保存并启用")
        return True
    
    def clear_calibration_points(self):
        """清除所有标定点"""
        self.calibration_points = []
        self.enabled = False
        self._save_config()
        print("[Z补偿] 已清除所有标定点")
    
    def print_status(self):
        """打印当前状态"""
        print(f"\n" + "=" * 60)
        print("[动态Z补偿状态]")
        print("-" * 60)
        print(f"  启用状态: {'是' if self.enabled else '否'}")
        print(f"  模型类型: {self.model_type}")
        print(f"  标定点数: {len(self.calibration_points)}")
        if self.enabled:
            print(f"  系数: {self.coefficients}")
        print("=" * 60)


class HeadCameraCalculator:
    """头部相机位置计算器"""
    
    def __init__(self, z_compensator: Optional[DynamicZCompensator] = None):
        self.fx, self.fy = HEAD_INTRINSICS['fx'], HEAD_INTRINSICS['fy']
        self.cx, self.cy = HEAD_INTRINSICS['cx'], HEAD_INTRINSICS['cy']
        self.z_compensator = z_compensator
        
        # 加载静态外参偏移补偿
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            try:
                with open(offset_path) as f:
                    data = json.load(f)
                    self.offset = np.array(data['offset_xyz'])
                    print(f"[头部相机] 加载静态偏移: X={self.offset[0]:+.4f}, Y={self.offset[1]:+.4f}, Z={self.offset[2]:+.4f}")
            except Exception as e:
                print(f"[头部相机] 加载偏移文件失败: {e}")
        else:
            print("[头部相机] 未找到偏移补偿文件，使用默认值")
    
    def compute(self, masks: np.ndarray, depth: np.ndarray, tf_matrix: Optional[np.ndarray],
                apply_dynamic_compensation: bool = True) -> List[Dict]:
        """
        计算砖块位置
        
        Args:
            apply_dynamic_compensation: 是否应用动态Z补偿（标定时设为False）
        """
        results = []
        h, w = depth.shape
        
        for i, mask in enumerate(masks):
            mask = mask[0] if len(mask.shape) == 3 else mask
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            ys, xs = np.where(mask_bool)
            px, py = np.mean(xs), np.mean(ys)
            yaw_cam, rect = estimate_orientation(mask_bool)
            
            # 深度计算
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
            valid = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
            valid = valid[valid > 0]
            if len(valid) == 0:
                continue
            
            z = np.median(valid) / 1000.0
            pos_cam = np.array([(px - self.cx) * z / self.fx, (py - self.cy) * z / self.fy, z])
            
            if tf_matrix is not None:
                pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3]
                # 应用静态偏移补偿
                pos_base = pos_base + self.offset
                yaw_base = yaw_cam + np.arctan2(tf_matrix[1,0], tf_matrix[0,0])
                yaw_base = yaw_base + np.pi
            else:
                pos_base, yaw_base = pos_cam, yaw_cam
            
            # 应用动态 Z 补偿
            z_compensation = 0.0
            if apply_dynamic_compensation and self.z_compensator and self.z_compensator.enabled:
                z_compensation = self.z_compensator.compute_compensation(pos_base[0], pos_base[1])
                pos_base[2] += z_compensation
            
            # 归一化 yaw
            while yaw_base > np.pi: yaw_base -= 2*np.pi
            while yaw_base < -np.pi: yaw_base += 2*np.pi
            
            results.append({
                'id': i+1, 'position': pos_base, 'yaw': yaw_base, 'yaw_deg': np.degrees(yaw_base),
                'yaw_camera': yaw_cam, 'pixel_center': (int(px), int(py)), 'depth_m': z, 'rect_info': rect,
                'z_compensation': z_compensation,
                'position_raw': pos_base - np.array([0, 0, z_compensation])  # 未补偿的原始位置
            })
        
        return sorted(results, key=lambda x: x['depth_m'])


class HandEyeCalculator:
    """手眼相机位置计算器"""
    
    def __init__(self, intrinsics: dict, extrinsics: dict):
        self.fx, self.fy = intrinsics['fx'], intrinsics['fy']
        self.cx, self.cy = intrinsics['cx'], intrinsics['cy']
        
        self.T_cam2gripper = np.eye(4)
        self.T_cam2gripper[:3, :3] = np.array(extrinsics['rotation_matrix'])
        self.T_cam2gripper[:3, 3] = np.array(extrinsics['translation'])
        
        print(f"[HandEye] fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
    
    def compute(self, masks: np.ndarray, shape: Tuple[int,int], 
                R_g2b: np.ndarray, t_g2b: np.ndarray,
                reference_z: Optional[float] = None) -> List[Dict]:
        results = []
        h, w = shape
        T_g2b = np.eye(4)
        T_g2b[:3, :3], T_g2b[:3, 3] = R_g2b, t_g2b.flatten()
        T_cam2base = T_g2b @ self.T_cam2gripper
        
        for i, mask in enumerate(masks):
            mask = mask[0] if len(mask.shape) == 3 else mask
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            ys, xs = np.where(mask_bool)
            px, py = np.mean(xs), np.mean(ys)
            yaw_cam, rect = estimate_orientation(mask_bool)
            
            z = self._estimate_depth_from_reference_z(px, py, reference_z, T_cam2base)
            
            pos_cam = np.array([(px - self.cx) * z / self.fx, (py - self.cy) * z / self.fy, z])
            pos_base = (T_cam2base @ np.append(pos_cam, 1))[:3]
            yaw_base = yaw_cam + np.arctan2(T_cam2base[1,0], T_cam2base[0,0])
            
            while yaw_base > np.pi: yaw_base -= 2*np.pi
            while yaw_base < -np.pi: yaw_base += 2*np.pi
            
            results.append({
                'id': i+1, 'position': pos_base, 'yaw': yaw_base, 'yaw_deg': np.degrees(yaw_base),
                'yaw_camera': yaw_cam, 'pixel_center': (int(px), int(py)), 
                'estimated_depth_m': z, 'rect_info': rect,
                'used_reference': reference_z is not None
            })
        
        return sorted(results, key=lambda x: x['estimated_depth_m'])

    def _estimate_depth_from_reference_z(self, px: float, py: float, 
                                          target_z: float, T_cam2base: np.ndarray) -> float:
        R = T_cam2base[:3, :3]
        t = T_cam2base[:3, 3]
        
        nx = (px - self.cx) / self.fx
        ny = (py - self.cy) / self.fy
        
        coeff = R[2, 0] * nx + R[2, 1] * ny + R[2, 2]
        
        if abs(coeff) < 1e-6:
            return 0.3
        
        z_cam = (target_z - t[2]) / coeff
        z_cam = max(0.05, min(1.0, z_cam))
        
        return z_cam
    
    
class ArmController:
    """机械臂控制器"""
    def __init__(self, robot: MMK2RealRobot, use_left_arm: bool = True):
        self.robot = robot
        self.use_left_arm = use_left_arm
        self.arm_component = MMK2Components.LEFT_ARM if use_left_arm else MMK2Components.RIGHT_ARM
        self.other_arm_component = MMK2Components.RIGHT_ARM if use_left_arm else MMK2Components.LEFT_ARM
        self.arm_name = "左臂" if use_left_arm else "右臂"
        self.arm_key = 'left_arm' if use_left_arm else 'right_arm'
        self.other_arm_key = 'right_arm' if use_left_arm else 'left_arm'
    
    def compute_grasp_orientation(self, yaw: float) -> Tuple[float, float, float, float]:
        r_base = R.from_euler('y', np.pi / 2)
        r_yaw = R.from_euler('z', yaw)
        r_final = r_yaw * r_base
        quat = r_final.as_quat()
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def move_to_hover(self, brick_position: np.ndarray, brick_yaw: float, hover_height: float = 0.25) -> bool:
        target_x = float(brick_position[0])
        target_y = float(brick_position[1])
        target_z = float(brick_position[2]) + hover_height
        
        ox, oy, oz, ow = self.compute_grasp_orientation(brick_yaw)
        
        print(f"\n[{self.arm_name}] 移动到砖块上方")
        print(f"  目标位置: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]m")
        
        poses = self.robot.get_arm_end_poses()
        if poses is None or self.other_arm_key not in poses:
            if self.use_left_arm:
                other_pose = Pose(
                    position=Position(x=0.457, y=-0.221, z=1.147),
                    orientation=Orientation(x=0.584, y=0.394, z=-0.095, w=0.700),
                )
            else:
                other_pose = Pose(
                    position=Position(x=0.457, y=0.221, z=1.147),
                    orientation=Orientation(x=-0.584, y=0.394, z=0.095, w=0.700),
                )
        else:
            other = poses[self.other_arm_key]
            other_pose = Pose(
                position=Position(x=other['position'][0], y=other['position'][1], z=other['position'][2]),
                orientation=Orientation(x=other['orientation'][0], y=other['orientation'][1], 
                                        z=other['orientation'][2], w=other['orientation'][3]),
            )
        
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=target_x, y=target_y, z=target_z),
                orientation=Orientation(x=ox, y=oy, z=oz, w=ow),
            ),
            self.other_arm_component: other_pose,
        }
        
        try:
            self.robot.control_arm_poses(target_pose)
            print(f"[{self.arm_name}] 移动指令已发送")
            return True
        except Exception as e:
            print(f"[{self.arm_name}] 移动失败: {e}")
            return False


class BrickDetector:
    """砖块检测主程序"""
    def __init__(self, ip: str, checkpoint: str, prompt: str, tf_host: str, tf_port: int, use_left: bool):
        self.prompt = prompt
        self.use_left = use_left
        self.arm_key = 'left_arm' if use_left else 'right_arm'
        
        # 初始化
        print(f"[Init] 连接 {ip}...")
        self.robot = MMK2RealRobot(ip=ip)
        self.robot.set_robot_head_pose(0, -1.08)
        self.robot.set_spine(0.05)
        
        self.segmenter = SAM3Segmenter(checkpoint)
        
        # 初始化动态Z补偿器
        self.z_compensator = DynamicZCompensator(CALIB_DIR / "head_camera_offset.json")
        
        self.head_calc = HeadCameraCalculator(z_compensator=self.z_compensator)
        self.tf_client = TFClient(host=tf_host, port=tf_port, auto_connect=True)
        self.arm_controller = ArmController(self.robot, use_left_arm=use_left)
        
        # 加载手眼标定
        side = "left" if use_left else "right"
        intr_path = CALIB_DIR / f"hand_eye_intrinsics_{side}.json"
        extr_path = CALIB_DIR / f"hand_eye_extrinsics_{side}.json"
        
        self.handeye_calc = None
        if intr_path.exists() and extr_path.exists():
            with open(intr_path) as f: intr = json.load(f)
            with open(extr_path) as f: extr = json.load(f)
            self.handeye_calc = HandEyeCalculator(intr, extr)
        else:
            print(f"[警告] 手眼标定文件不存在")
        
        # 缓存
        self._head_results, self._handeye_results = [], []
        self._head_frame, self._handeye_frame = None, None
        self._head_until, self._handeye_until = 0, 0
        self._tf = None
        
        # 手眼相机补偿调试
        self.handeye_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.offset_step = 0.005
        self._pending_head_offset = None
        
        # 手眼外参校准
        self._handeye_calib_mode = False
        self._handeye_calib_initial_pos = None  # 手眼检测的初始位置
        self._handeye_calib_initial_yaw = None
        
        # Z补偿标定模式
        self._calibration_mode = False
        self._calibration_detected_pos = None  # 检测到的位置（未补偿）
        
        # 最新的 RGB 和 Depth
        self._latest_rgb = None
        self._latest_depth = None
        
        print("=" * 60)
        print("按键说明:")
        print("-" * 60)
        print("基本操作:")
        print("  s - 头部分割  g - 移动到头部检测位置  e - 手眼分割")
        print("  f - 用手眼XY精确移动  d - 下降到砖块  k - 闭合夹爪")
        print("  l - 抬升  m - 移动到放置位置  n - 下降放置  o - 打开夹爪")
        print("  p - 打印位姿  r - 重置  q - 退出")
        print("-" * 60)
        print("Z轴动态补偿标定:")
        print("  b - 开始/结束Z补偿标定模式")
        print("  v - 添加当前位置为标定点")
        print("  7 - 线性模型拟合  8 - 二次模型拟合")
        print("  9 - 清除标定点  0 - 显示补偿状态")
        print("-" * 60)
        print("手眼外参校准 (新功能):")
        print("  c - 开始/结束手眼外参校准模式")
        print("  x - 保存校准后的手眼外参")
        print("-" * 60)
        print("手动微调 (步进5mm):")
        print("  1/2 - Y轴 左/右  3/4 - X轴 前/后  5/6 - Z轴 上/下")
        print("=" * 60)
    
    def _get_handeye_img(self) -> Optional[np.ndarray]:
        goal = {MMK2Components.LEFT_CAMERA if self.use_left else MMK2Components.RIGHT_CAMERA: [ImageTypes.COLOR]}
        comp = MMK2Components.LEFT_CAMERA if self.use_left else MMK2Components.RIGHT_CAMERA
        for c, imgs in self.robot.mmk2.get_image(goal).items():
            if c == comp:
                for _, img in imgs.data.items():
                    if img is not None and img.shape[0] > 1:
                        return cv2.resize(img, (640, 480))
        return None
    
    def _get_arm_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        poses = self.robot.get_arm_end_poses()
        if poses and self.arm_key in poses:
            p = poses[self.arm_key]
            return R.from_quat(p['orientation']).as_matrix(), np.array(p['position'])
        return None
    
    def _draw_results(self, frame: np.ndarray, results: List[Dict], is_handeye: bool = False) -> np.ndarray:
        out = frame.copy()
        for i, b in enumerate(results):
            c = COLORS[i % len(COLORS)]
            px, py = b['pixel_center']
            cv2.circle(out, (px, py), 6, c, -1)
            
            if b['rect_info']:
                box = b['rect_info']['box_points'].astype(np.int32)
                cv2.drawContours(out, [box], 0, c, 2)
                
                yaw = b['yaw_camera']
                sign = -1 if is_handeye else 1
                ex = int(px + 40 * np.cos(sign * yaw))
                ey = int(py + 40 * np.sin(sign * yaw) * (-1 if not is_handeye else 1))
                cv2.arrowedLine(out, (px, py), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
            
            pos = b['position']
            z_comp = b.get('z_compensation', 0)
            txt = f"#{b['id']}: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]"
            if z_comp != 0:
                txt += f" (Zc:{z_comp:+.3f})"
            cv2.putText(out, txt, (px+10, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        return out
    
    def _head_detect(self, rgb: np.ndarray, depth: np.ndarray, apply_compensation: bool = True):
        """头部相机检测"""
        print(f"\n[头部] 分割 '{self.prompt}'...")
        t0 = time.time()
        masks = self.segmenter.segment(rgb, self.prompt)
        
        if masks is None or len(masks) == 0:
            print(f"[头部] 未检测到 ({time.time()-t0:.2f}s)")
            self._head_results = []
            return
        
        tf_mat = self._tf['matrix'] if self._tf else None
        self._head_results = self.head_calc.compute(
            masks, depth, tf_mat, 
            apply_dynamic_compensation=apply_compensation
        )
        print(f"[头部] 检测到 {len(self._head_results)} 个 ({time.time()-t0:.2f}s)")
        
        for b in self._head_results:
            p = b['position']
            z_comp = b.get('z_compensation', 0)
            comp_str = f" (Z补偿:{z_comp:+.4f})" if z_comp != 0 else ""
            print(f"  #{b['id']}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]m yaw={b['yaw_deg']:.1f}°{comp_str}")
        
        self._head_frame = rgb.copy()
        self._head_until = time.time() + 5
        
        # 如果在标定模式，保存未补偿的位置
        if self._calibration_mode and self._head_results:
            self._calibration_detected_pos = self._head_results[0].get('position_raw', 
                                                                        self._head_results[0]['position']).copy()
            print(f"[标定] 已记录检测位置: {self._calibration_detected_pos}")
    
    def _handeye_detect(self):
        if not self.handeye_calc:
            print("[手眼] 未加载标定")
            return
        
        img = self._get_handeye_img()
        pose = self._get_arm_pose()
        if img is None or pose is None:
            print("[手眼] 图像或位姿获取失败")
            return
        
        reference_z = None
        reference_yaw = None
        if self._head_results:
            reference_z = self._head_results[0]['position'][2]
            reference_yaw = self._head_results[0]['yaw']
        
        print(f"\n[手眼] 分割 '{self.prompt}'...")
        t0 = time.time()
        masks = self.segmenter.segment(img, self.prompt)
        
        if masks is None or len(masks) == 0:
            print(f"[手眼] 未检测到 ({time.time()-t0:.2f}s)")
            self._handeye_results = []
            return
        
        self._handeye_results = self.handeye_calc.compute(
            masks, img.shape[:2], pose[0], pose[1], 
            reference_z=reference_z
        )
        
        if reference_yaw is not None:
            for b in self._handeye_results:
                yaw = b['yaw']
                diff = yaw - reference_yaw
                while diff > np.pi: diff -= 2 * np.pi
                while diff < -np.pi: diff += 2 * np.pi
                
                if abs(diff) > np.pi / 2:
                    if yaw > 0:
                        b['yaw'] = yaw - np.pi
                    else:
                        b['yaw'] = yaw + np.pi
                    b['yaw_deg'] = np.degrees(b['yaw'])
        
        print(f"[手眼] 检测到 {len(self._handeye_results)} 个 ({time.time()-t0:.2f}s)")
        
        self._handeye_frame = img.copy()
        self._handeye_until = time.time() + 5

    def _toggle_handeye_calib_mode(self):
        """切换手眼外参校准模式"""
        self._handeye_calib_mode = not self._handeye_calib_mode
        
        if self._handeye_calib_mode:
            print("\n" + "=" * 60)
            print("[手眼外参校准模式] 已开启")
            print("-" * 60)
            print("校准步骤:")
            print("  1. 按 's' 进行头部检测")
            print("  2. 按 'g' 移动机械臂到检测位置上方")
            print("  3. 按 'e' 进行手眼分割（记录手眼检测的 XY）")
            print("  4. 按 'f' 移动到手眼检测的 XY 位置")
            print("  5. 按 'd' 下降到砖块")
            print("  6. 用 1234 微调 XY，直到夹爪精确对准砖块中心")
            print("  7. 按 'x' 保存校准（只计算 XY 偏移，忽略 Z）")
            print("  8. 按 'c' 退出校准模式")
            print("-" * 60)
            print("原理: 手眼相机只提供 XY 定位，Z 来自头部相机")
            print("      所以只校准 XY 偏移，不管 Z 变化")
            print("=" * 60)
            
            self._handeye_calib_initial_pos = None
            self._handeye_calib_initial_yaw = None
        else:
            print("\n[手眼外参校准模式] 已关闭")
            self._handeye_calib_initial_pos = None
            self._handeye_calib_initial_yaw = None

    def _save_handeye_calibration(self):
        """保存手眼外参校准结果 - 使用与可用版本相同的计算方法"""
        if not self._handeye_calib_mode:
            print("[手眼校准] 请先按 'c' 进入校准模式")
            return
        
        if self._handeye_calib_initial_pos is None:
            print("[手眼校准] 请先按 'e' 手眼分割，再按 'f' 移动")
            return
        
        # 获取当前夹爪位姿
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[手眼校准] 无法获取当前位姿")
            return
        
        current = poses[self.arm_key]
        current_pos = np.array(current['position'])
        current_quat = current['orientation']
        
        # 只计算 XY 偏移（手眼相机只负责 XY 定位）
        offset_x = current_pos[0] - self._handeye_calib_initial_pos[0]
        offset_y = current_pos[1] - self._handeye_calib_initial_pos[1]
        
        print("\n" + "=" * 70)
        print("[手眼外参校准结果]")
        print("-" * 70)
        print(f"  手眼检测 XY: [{self._handeye_calib_initial_pos[0]:.4f}, {self._handeye_calib_initial_pos[1]:.4f}]")
        print(f"  实际夹爪 XY: [{current_pos[0]:.4f}, {current_pos[1]:.4f}]")
        print(f"  XY 偏移:     [ΔX={offset_x:+.4f}, ΔY={offset_y:+.4f}]")
        print(f"  偏移距离:    {np.sqrt(offset_x**2 + offset_y**2)*1000:.1f} mm")
        print("-" * 70)
        
        # === 使用与可用版本相同的坐标转换方法 ===
        
        # 当前夹爪姿态
        R_g2b = R.from_quat(current_quat).as_matrix()
        t_g2b = current_pos
        print(f"  夹爪位置 (base): [{t_g2b[0]:.4f}, {t_g2b[1]:.4f}, {t_g2b[2]:.4f}]")
        euler_g = R.from_quat(current_quat).as_euler('xyz', degrees=True)
        print(f"  夹爪欧拉角:      [{euler_g[0]:.1f}, {euler_g[1]:.1f}, {euler_g[2]:.1f}]°")
        
        # base_link 下的补偿向量 (Z=0，只校准XY)
        offset_base = np.array([offset_x, offset_y, 0.0])
        print(f"  补偿向量 (base): [{offset_base[0]:+.4f}, {offset_base[1]:+.4f}, {offset_base[2]:+.4f}]")
        
        # 转换到夹爪坐标系: offset_gripper = R_g2b^T @ offset_base
        offset_gripper = R_g2b.T @ offset_base
        print(f"  补偿向量 (gripper): [{offset_gripper[0]:+.4f}, {offset_gripper[1]:+.4f}, {offset_gripper[2]:+.4f}]")
        
        # 更新手眼外参
        side = "left" if self.use_left else "right"
        extr_path = CALIB_DIR / f"hand_eye_extrinsics_{side}.json"
        
        try:
            with open(extr_path) as f:
                extr = json.load(f)
            
            # 当前外参 translation
            current_trans = np.array(extr['translation'])
            print(f"\n  当前外参 translation: [{current_trans[0]:.4f}, {current_trans[1]:.4f}, {current_trans[2]:.4f}]")
            
            # 新外参 translation = 当前 + gripper坐标系下的偏移
            # 注意：这里直接加 offset_gripper，因为 translation 是在 gripper 坐标系下定义的
            new_trans = current_trans + offset_gripper
            print(f"  建议新 translation:  [{new_trans[0]:.4f}, {new_trans[1]:.4f}, {new_trans[2]:.4f}]")
            
            print("-" * 70)
            print("按 'y' 确认保存，其他键取消")
            
            self._pending_handeye_save = {
                'path': extr_path,
                'old_extr': extr,
                'new_translation': new_trans.tolist(),
                'offset_xy': [offset_x, offset_y],
                'offset_gripper': offset_gripper.tolist(),
            }
            
        except Exception as e:
            print(f"[手眼校准] 读取外参失败: {e}")

    def _confirm_handeye_save(self):
        """确认保存手眼外参"""
        if not hasattr(self, '_pending_handeye_save') or self._pending_handeye_save is None:
            return False
        
        save_info = self._pending_handeye_save
        extr = save_info['old_extr']
        extr['translation'] = save_info['new_translation']
        
        # 备份旧文件
        backup_path = save_info['path'].with_suffix('.json.bak')
        import shutil
        shutil.copy(save_info['path'], backup_path)
        print(f"  [备份] 已保存到 {backup_path}")
        
        # 保存新外参
        with open(save_info['path'], 'w') as f:
            json.dump(extr, f, indent=2)
        
        print(f"  [保存] 新外参已保存到 {save_info['path']}")
        print("=" * 60)
        
        # 重新加载手眼计算器
        side = "left" if self.use_left else "right"
        intr_path = CALIB_DIR / f"hand_eye_intrinsics_{side}.json"
        with open(intr_path) as f:
            intr = json.load(f)
        self.handeye_calc = HandEyeCalculator(intr, extr)
        print("[手眼校准] 已重新加载手眼计算器")
        
        self._pending_handeye_save = None
        return True
    
    def _move_arm(self):
        if not self._head_results:
            print("[移动] 请先按 's' 检测")
            return
        if not self._tf:
            print("[移动] TF 未连接")
            return
        
        b = self._head_results[0]
        pos, yaw = b['position'], b['yaw']
        
        self.arm_controller.move_to_hover(
            brick_position=pos,
            brick_yaw=yaw,
            hover_height=HOVER_HEIGHT
        )
    
    def _move_arm_handeye(self):
        if not self._handeye_results:
            print("[精确移动] 请先按 'e' 进行手眼分割")
            return
        
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[精确移动] 无法获取当前位姿")
            return
        
        current_pos = poses[self.arm_key]['position']
        current_z = current_pos[2]
        
        b = self._handeye_results[0]
        handeye_pos = b['position']
        handeye_yaw = b['yaw']
        
        compensated_pos = np.array([
            handeye_pos[0] + self.handeye_offset['x'],
            handeye_pos[1] + self.handeye_offset['y'],
            handeye_pos[2] + self.handeye_offset['z']
        ])
        
        hover_height = current_z - compensated_pos[2]
        
        print(f"\n[精确移动] 使用手眼 XY，保持当前高度")
        
        # 如果在手眼校准模式，记录手眼检测的 XY 位置
        if self._handeye_calib_mode:
            # 只记录 XY，这是手眼相机实际检测的
            self._handeye_calib_initial_pos = np.array([
                handeye_pos[0],  # 手眼检测的 X
                handeye_pos[1],  # 手眼检测的 Y
            ])
            self._handeye_calib_initial_yaw = handeye_yaw
            print(f"[手眼校准] 已记录手眼检测 XY: [{handeye_pos[0]:.4f}, {handeye_pos[1]:.4f}]")
            print(f"[手眼校准] 按 'd' 下降后用 1234 微调 XY，然后按 'x' 保存")
        
        self.arm_controller.move_to_hover(
            brick_position=compensated_pos,
            brick_yaw=handeye_yaw,
            hover_height=hover_height
        )
    def _move_gripper_z(self, delta_z: float):
        """控制夹爪在 Z 轴方向移动"""
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[Z移动] 无法获取当前位姿")
            return
        
        current = poses[self.arm_key]
        current_pos = current['position']
        current_quat = current['orientation']
        
        new_z = current_pos[2] + delta_z
        
        if self.arm_controller.other_arm_key in poses:
            other = poses[self.arm_controller.other_arm_key]
            other_pose = Pose(
                position=Position(x=other['position'][0], y=other['position'][1], z=other['position'][2]),
                orientation=Orientation(x=other['orientation'][0], y=other['orientation'][1], 
                                        z=other['orientation'][2], w=other['orientation'][3]),
            )
        else:
            other_pose = None
        
        target_pose = {
            self.arm_controller.arm_component: Pose(
                position=Position(x=current_pos[0], y=current_pos[1], z=new_z),
                orientation=Orientation(x=current_quat[0], y=current_quat[1], 
                                        z=current_quat[2], w=current_quat[3]),
            ),
        }
        if other_pose:
            target_pose[self.arm_controller.other_arm_component] = other_pose
        
        try:
            self.robot.control_arm_poses(target_pose)
            print(f"[Z移动] Z: {current_pos[2]:.4f} -> {new_z:.4f} (Δ={delta_z:+.4f})")
        except Exception as e:
            print(f"[Z移动] 失败: {e}")

    def _move_gripper_xy(self, delta_x: float, delta_y: float):
        """控制夹爪在 XY 方向移动"""
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[XY移动] 无法获取当前位姿")
            return
        
        current = poses[self.arm_key]
        current_pos = current['position']
        current_quat = current['orientation']
        
        new_x = current_pos[0] + delta_x
        new_y = current_pos[1] + delta_y
        
        if self.arm_controller.other_arm_key in poses:
            other = poses[self.arm_controller.other_arm_key]
            other_pose = Pose(
                position=Position(x=other['position'][0], y=other['position'][1], z=other['position'][2]),
                orientation=Orientation(x=other['orientation'][0], y=other['orientation'][1], 
                                        z=other['orientation'][2], w=other['orientation'][3]),
            )
        else:
            other_pose = None
        
        target_pose = {
            self.arm_controller.arm_component: Pose(
                position=Position(x=new_x, y=new_y, z=current_pos[2]),
                orientation=Orientation(x=current_quat[0], y=current_quat[1], 
                                        z=current_quat[2], w=current_quat[3]),
            ),
        }
        if other_pose:
            target_pose[self.arm_controller.other_arm_component] = other_pose
        
        try:
            self.robot.control_arm_poses(target_pose)
            if delta_x != 0:
                print(f"[XY移动] X: {current_pos[0]:.4f} -> {new_x:.4f} (Δ={delta_x:+.4f})")
            if delta_y != 0:
                print(f"[XY移动] Y: {current_pos[1]:.4f} -> {new_y:.4f} (Δ={delta_y:+.4f})")
        except Exception as e:
            print(f"[XY移动] 失败: {e}")

    def _toggle_calibration_mode(self):
        """切换标定模式"""
        self._calibration_mode = not self._calibration_mode
        if self._calibration_mode:
            print("\n" + "=" * 60)
            print("[标定模式] 已开启")
            print("-" * 60)
            print("标定步骤:")
            print("  1. 把砖块放到某个位置")
            print("  2. 按 's' 进行头部检测（记录检测位置）")
            print("  3. 按 'g' 移动机械臂到检测位置上方")
            print("  4. 用 1234 微调 XY，5/6 微调 Z，直到夹爪对准砖块中心")
            print("  5. 按 'v' 添加标定点")
            print("  6. 重复步骤 1-5，在不同位置采集多个点（至少3个）")
            print("  7. 按 '7' 或 '8' 拟合模型")
            print("  8. 按 'b' 退出标定模式")
            print("-" * 60)
            print(f"当前已有 {len(self.z_compensator.calibration_points)} 个标定点")
            print("=" * 60)
            self._calibration_detected_pos = None
        else:
            print("\n[标定模式] 已关闭")

    def _add_calibration_point(self):
        """添加当前位置为标定点"""
        if not self._calibration_mode:
            print("[标定] 请先按 'b' 进入标定模式")
            return
        
        if self._calibration_detected_pos is None:
            print("[标定] 请先按 's' 检测砖块")
            return
        
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[标定] 无法获取当前位姿")
            return
        
        # 当前夹爪位置就是实际的砖块位置
        actual_pos = np.array(poses[self.arm_key]['position'])
        
        self.z_compensator.add_calibration_point(
            detected_pos=self._calibration_detected_pos,
            actual_pos=actual_pos
        )
        
        # 清除缓存，准备下一个点
        self._calibration_detected_pos = None
        print(f"\n[提示] 请移动砖块到新位置，然后按 's' 继续采集")

    def _descend_to_brick_surface(self):
        if not self._handeye_results:
            print("[下降] 请先按 'e' 进行手眼分割")
            return
        
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[下降] 无法获取当前位姿")
            return
        
        current = poses[self.arm_key]
        current_pos = current['position']
        current_quat = current['orientation']
        
        b = self._handeye_results[0]
        brick_z = b['position'][2] + self.handeye_offset['z']
        target_z = brick_z
        
        print(f"\n[下降] 移动到砖块表面")
        print(f"  当前高度: {current_pos[2]:.4f}m  目标高度: {target_z:.4f}m")
        
        if self.arm_controller.other_arm_key in poses:
            other = poses[self.arm_controller.other_arm_key]
            other_pose = Pose(
                position=Position(x=other['position'][0], y=other['position'][1], z=other['position'][2]),
                orientation=Orientation(x=other['orientation'][0], y=other['orientation'][1], 
                                        z=other['orientation'][2], w=other['orientation'][3]),
            )
        else:
            other_pose = None
        
        target_pose = {
            self.arm_controller.arm_component: Pose(
                position=Position(x=current_pos[0], y=current_pos[1], z=target_z),
                orientation=Orientation(x=current_quat[0], y=current_quat[1], 
                                        z=current_quat[2], w=current_quat[3]),
            ),
        }
        if other_pose:
            target_pose[self.arm_controller.other_arm_component] = other_pose
        
        try:
            self.robot.control_arm_poses(target_pose)
        except Exception as e:
            print(f"[下降] 失败: {e}")

    def _close_gripper_and_print_effort(self):
        print("\n[夹爪] 闭合夹爪...")
        if self.use_left:
            self.robot.close_gripper(left=True, right=False)
        else:
            self.robot.close_gripper(left=False, right=True)
        
        time.sleep(0.8)
        
        effort = self.robot.get_gripper_effort(left=self.use_left)
        if effort is not None:
            print(f"[夹爪] 电流/力矩: {effort:.4f}")

    def _open_gripper(self):
        print("\n[夹爪] 打开夹爪...")
        if self.use_left:
            self.robot.open_gripper(left=True, right=False)
        else:
            self.robot.open_gripper(left=False, right=True)

    def _lift_to_hover(self):
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            return
        
        current = poses[self.arm_key]
        current_pos = current['position']
        current_quat = current['orientation']
        
        if self._handeye_results:
            brick_z = self._handeye_results[0]['position'][2]
            target_z = brick_z + HOVER_HEIGHT
        else:
            target_z = current_pos[2] + HOVER_HEIGHT
        
        target_pose = {
            self.arm_controller.arm_component: Pose(
                position=Position(x=current_pos[0], y=current_pos[1], z=target_z),
                orientation=Orientation(x=current_quat[0], y=current_quat[1], 
                                        z=current_quat[2], w=current_quat[3]),
            ),
        }
        try:
            self.robot.control_arm_poses(target_pose)
            print(f"[抬升] {current_pos[2]:.4f} -> {target_z:.4f}m")
        except Exception as e:
            print(f"[抬升] 失败: {e}")

    def _normalize_yaw_to_target(self, yaw: float) -> float:
        while yaw > np.pi: yaw -= 2 * np.pi
        while yaw < -np.pi: yaw += 2 * np.pi
        
        if abs(yaw) <= np.pi / 2:
            return 0.0
        else:
            return np.pi if yaw > 0 else -np.pi

    def _compute_place_position(self) -> Optional[Dict]:
        if not self._handeye_results:
            return None
        
        b = self._handeye_results[0]
        brick_pos = b['position']
        brick_yaw = b['yaw']
        
        place_y = brick_pos[1] - 0.08
        place_x = brick_pos[0]
        place_z = brick_pos[2]
        place_yaw = self._normalize_yaw_to_target(brick_yaw)
        
        return {
            'position': np.array([place_x, place_y, place_z]),
            'yaw': place_yaw
        }

    def _move_to_place_hover(self):
        place = self._compute_place_position()
        if place is None:
            print("[移动放置] 请先按 'e' 进行手眼分割")
            return
        
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            return
        
        current_z = poses[self.arm_key]['position'][2]
        hover_height = current_z - place['position'][2]
        
        self.arm_controller.move_to_hover(
            brick_position=place['position'],
            brick_yaw=place['yaw'],
            hover_height=hover_height
        )
        
        self._place_target = place

    def _descend_and_place(self):
        if not hasattr(self, '_place_target') or self._place_target is None:
            print("[放置] 请先按 'm' 移动到放置位置上方")
            return
        
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            return
        
        current = poses[self.arm_key]
        current_pos = current['position']
        target_z = self._place_target['position'][2]
        
        ox, oy, oz, ow = self.arm_controller.compute_grasp_orientation(self._place_target['yaw'])
        
        target_pose = {
            self.arm_controller.arm_component: Pose(
                position=Position(x=current_pos[0], y=current_pos[1], z=target_z),
                orientation=Orientation(x=ox, y=oy, z=oz, w=ow),
            ),
        }
        try:
            self.robot.control_arm_poses(target_pose)
            time.sleep(1.5)
            self._open_gripper()
            self._place_target = None
        except Exception as e:
            print(f"[放置] 失败: {e}")

    def _reset(self):
        print("\n[重置] 回到初始位置...")
        from mmk2_types.grpc_msgs import JointState
        
        arm_action = {
            MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
            MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.SPINE: JointState(position=[0.05]),
        }
        
        try:
            result = self.robot.mmk2.set_goal(arm_action, TrajectoryParams())
            if result.value == GoalStatus.Status.SUCCESS:
                print("[重置] 完成")
        except Exception as e:
            print(f"[重置] 异常: {e}")
    
    def _print_pose(self):
        poses = self.robot.get_arm_end_poses()
        if poses and self.arm_key in poses:
            p = poses[self.arm_key]
            pos, quat = p['position'], p['orientation']
            euler = R.from_quat(quat).as_euler('xyz', degrees=True)
            print(f"\n[{'左' if self.use_left else '右'}臂] 位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m")
            print(f"        欧拉角: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")
    
    def run(self):
        cv2.namedWindow("Head", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HandEye", cv2.WINDOW_NORMAL)
        
        for rgb, depth, _, _ in self.robot.camera:
            if rgb is None or rgb.size == 0:
                continue
            
            self._latest_rgb = rgb
            self._latest_depth = depth
            
            self._tf = self.tf_client.get_transform('base_link', 'head_camera_link')
            
            # 头部显示
            if time.time() < self._head_until and self._head_frame is not None:
                disp = self._draw_results(self._head_frame, self._head_results)
            else:
                disp = rgb.copy()
                cv2.putText(disp, f"'s':detect '{self.prompt}'", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            
            # 标定模式指示
            if self._calibration_mode:
                cv2.putText(disp, "[Z-CALIBRATION MODE]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(disp, f"Points: {len(self.z_compensator.calibration_points)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 手眼校准模式指示
            if self._handeye_calib_mode:
                cv2.putText(disp, "[HANDEYE-CALIB MODE]", (10, 120 if self._calibration_mode else 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                if self._handeye_calib_initial_pos is not None:
                    cv2.putText(disp, f"Init: [{self._handeye_calib_initial_pos[0]:.3f}, {self._handeye_calib_initial_pos[1]:.3f}]", 
                               (10, 150 if self._calibration_mode else 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            # Z补偿状态
            if self.z_compensator.enabled:
                cv2.putText(disp, f"Z-Comp: ON ({self.z_compensator.model_type})", (10, disp.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow("Head", disp)
            
            # 手眼显示
            he_img = self._get_handeye_img()
            if he_img is not None:
                if time.time() < self._handeye_until and self._handeye_frame is not None:
                    he_disp = self._draw_results(self._handeye_frame, self._handeye_results, True)
                else:
                    he_disp = he_img.copy()
                
                offset_text = f"Offset: X={self.handeye_offset['x']:+.3f} Y={self.handeye_offset['y']:+.3f}"
                cv2.putText(he_disp, offset_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 手眼校准模式显示
                if self._handeye_calib_mode:
                    cv2.putText(he_disp, "[HANDEYE-CALIB]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cv2.putText(he_disp, "1234:XY adjust, x:save", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                cv2.imshow("HandEye", he_disp)
            
            # 按键处理
            k = cv2.waitKey(1) & 0xFF
            if hasattr(self, '_pending_handeye_save') and self._pending_handeye_save is not None:
                if k == ord('y'):
                    self._confirm_handeye_save()
                    continue
                elif k != 255:  # 任何其他键取消
                    print("[手眼校准] 已取消保存")
                    self._pending_handeye_save = None
                    continue
            
            # 基本操作
            if k == ord('s') and depth is not None:
                self._head_detect(rgb, depth, apply_compensation=not self._calibration_mode)
            elif k == ord('g'):
                self._move_arm()
            elif k == ord('e'):
                self._handeye_detect()
            elif k == ord('f'):
                self._move_arm_handeye()
            elif k == ord('d'):
                self._descend_to_brick_surface()
            elif k == ord('p'):
                self._print_pose()
            elif k == ord('k'):
                self._close_gripper_and_print_effort()
            elif k == ord('o'):
                self._open_gripper()
            elif k == ord('l'):
                self._lift_to_hover()
            elif k == ord('m'):
                self._move_to_place_hover()
            elif k == ord('n'):
                self._descend_and_place()
            elif k == ord('r'):
                self._reset()
            # 手动微调
            elif k == ord('1'):  # Y+
                self._move_gripper_xy(0, self.offset_step)
            elif k == ord('2'):  # Y-
                self._move_gripper_xy(0, -self.offset_step)
            elif k == ord('3'):  # X+
                self._move_gripper_xy(self.offset_step, 0)
            elif k == ord('4'):  # X-
                self._move_gripper_xy(-self.offset_step, 0)
            elif k == ord('5'):  # Z+
                self._move_gripper_z(self.offset_step)
            elif k == ord('6'):  # Z-
                self._move_gripper_z(-self.offset_step)
            
            # Z补偿标定
            elif k == ord('b'):
                self._toggle_calibration_mode()
            elif k == ord('v'):
                self._add_calibration_point()
            elif k == ord('7'):
                self.z_compensator.fit_model("linear")
            elif k == ord('8'):
                self.z_compensator.fit_model("quadratic")
            elif k == ord('9'):
                self.z_compensator.clear_calibration_points()
            elif k == ord('0'):
                self.z_compensator.print_status()
            
            # 手眼外参校准
            elif k == ord('c'):
                self._toggle_handeye_calib_mode()
            elif k == ord('x'):

                self._save_handeye_calibration()
            elif k in (ord('q'), 27):
                break
        cv2.destroyAllWindows()
        self.tf_client.disconnect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.11.200")
    parser.add_argument("--prompt", default="block, brick, rectangular object")
    parser.add_argument("--checkpoint", default="/home/ypf/sam3-main/checkpoint/sam3.pt")
    parser.add_argument("--tf-host", default="127.0.0.1")
    parser.add_argument("--tf-port", type=int, default=9999)
    parser.add_argument("--right-arm", action="store_true")
    args = parser.parse_args()
    
    BrickDetector(args.ip, args.checkpoint, args.prompt, args.tf_host, args.tf_port, not args.right_arm).run()


if __name__ == '__main__':
    main()