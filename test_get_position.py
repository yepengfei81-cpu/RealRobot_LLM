"""
砖块检测器 - 两阶段抓取流程
阶段1: 头部深度相机粗定位
阶段2: 手眼相机精确分割

按键: s-头部分割 g-移动机械臂 e-手眼分割 p-打印位姿 t-TF详情 r-重置 q-退出
"""

import sys
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


class HeadCameraCalculator:
    """头部相机位置计算器"""
    
    def __init__(self):
        self.fx, self.fy = HEAD_INTRINSICS['fx'], HEAD_INTRINSICS['fy']
        self.cx, self.cy = HEAD_INTRINSICS['cx'], HEAD_INTRINSICS['cy']
        
        # 加载外参偏移补偿
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            try:
                with open(offset_path) as f:
                    data = json.load(f)
                    self.offset = np.array(data['offset_xyz'])
                    print(f"[头部相机] 加载偏移补偿: X={self.offset[0]:+.4f}, Y={self.offset[1]:+.4f}, Z={self.offset[2]:+.4f}")
            except Exception as e:
                print(f"[头部相机] 加载偏移文件失败: {e}")
        else:
            print("[头部相机] 未找到偏移补偿文件，使用默认值")
    
    def compute(self, masks: np.ndarray, depth: np.ndarray, tf_matrix: Optional[np.ndarray]) -> List[Dict]:
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
                # 应用偏移补偿
                pos_base = pos_base + self.offset
                yaw_base = yaw_cam + np.arctan2(tf_matrix[1,0], tf_matrix[0,0])
                yaw_base = yaw_base + np.pi
            else:
                pos_base, yaw_base = pos_cam, yaw_cam
            
            # 归一化 yaw
            while yaw_base > np.pi: yaw_base -= 2*np.pi
            while yaw_base < -np.pi: yaw_base += 2*np.pi
            
            results.append({
                'id': i+1, 'position': pos_base, 'yaw': yaw_base, 'yaw_deg': np.degrees(yaw_base),
                'yaw_camera': yaw_cam, 'pixel_center': (int(px), int(py)), 'depth_m': z, 'rect_info': rect
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
        """
        计算砖块位置
        
        Args:
            masks: 分割掩码
            shape: 图像尺寸 (h, w)
            R_g2b: gripper 到 base 的旋转矩阵
            t_g2b: gripper 到 base 的平移向量
            reference_z: 参考 Z 坐标（来自头部相机），用于约束深度
        """
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
            
            # 深度估算策略
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
        """
        根据目标在 base_link 下的 Z 坐标，反推相机深度
        
        原理：
        pos_base = T_cam2base @ [x_cam, y_cam, z_cam, 1]^T
        其中 x_cam = (px - cx) * z_cam / fx
             y_cam = (py - cy) * z_cam / fy
        
        展开后 pos_base[2] = target_z，解出 z_cam
        """
        # 提取变换矩阵参数
        R = T_cam2base[:3, :3]
        t = T_cam2base[:3, 3]
        
        # 归一化像素坐标
        nx = (px - self.cx) / self.fx
        ny = (py - self.cy) / self.fy
        
        # pos_base[2] = R[2,0]*nx*z + R[2,1]*ny*z + R[2,2]*z + t[2] = target_z
        # z * (R[2,0]*nx + R[2,1]*ny + R[2,2]) = target_z - t[2]
        coeff = R[2, 0] * nx + R[2, 1] * ny + R[2, 2]
        
        if abs(coeff) < 1e-6:
            # 无法求解，使用默认估算
            return 0.3
        
        z_cam = (target_z - t[2]) / coeff
        
        # 限制在合理范围
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
        """计算抓取姿态四元数 - X轴朝下"""
        r_base = R.from_euler('y', np.pi / 2)
        r_yaw = R.from_euler('z', yaw)
        r_final = r_yaw * r_base
        quat = r_final.as_quat()
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def move_to_hover(self, brick_position: np.ndarray, brick_yaw: float, hover_height: float = 0.25) -> bool:
        """移动机械臂到砖块正上方"""
        target_x = float(brick_position[0])
        target_y = float(brick_position[1])
        target_z = float(brick_position[2]) + hover_height
        
        ox, oy, oz, ow = self.compute_grasp_orientation(brick_yaw)
        
        print(f"\n[{self.arm_name}] 移动到砖块上方")
        print(f"  目标位置: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]m")
        print(f"  姿态四元数: [{ox:.4f}, {oy:.4f}, {oz:.4f}, {ow:.4f}]")
        
        poses = self.robot.get_arm_end_poses()
        if poses is None or self.other_arm_key not in poses:
            print(f"[{self.arm_name}] 无法获取另一只臂位姿，使用默认值")
            # 使用 SDK 默认值
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
        
        # 构建目标位姿 - 同时包含两只臂
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
        self.robot.set_spine(0.15)
        
        self.segmenter = SAM3Segmenter(checkpoint)
        self.head_calc = HeadCameraCalculator()
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
        self.offset_step = 0.005  # 5mm 步进
        
        print("=" * 50)
        print("按键说明:")
        print("  s - 头部分割")
        print("  g - 移动到头部检测位置")
        print("  e - 手眼分割")
        print("  f - 用手眼XY精确移动（保持当前高度）")
        print("  p - 打印位姿  r - 重置  q - 退出")
        print("-" * 50)
        print("手眼补偿调试 (步进5mm):")
        print("  1/2 - Y轴 左/右")
        print("  3/4 - X轴 前/后")
        print("  5/6 - Z轴 上/下 (夹爪升降)")
        print("  0   - 重置补偿")
        print("=" * 50)
    
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
                
                # 朝向箭头
                yaw = b['yaw_camera']
                sign = -1 if is_handeye else 1
                ex = int(px + 40 * np.cos(sign * yaw))
                ey = int(py + 40 * np.sin(sign * yaw) * (-1 if not is_handeye else 1))
                cv2.arrowedLine(out, (px, py), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
            
            pos = b['position']
            depth_key = 'estimated_depth_m' if is_handeye else 'depth_m'
            txt = f"#{b['id']}: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] yaw:{b['yaw_deg']:.0f}"
            cv2.putText(out, txt, (px+10, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        return out
    
    def _head_detect(self, rgb: np.ndarray, depth: np.ndarray):
        print(f"\n[头部] 分割 '{self.prompt}'...")
        t0 = time.time()
        masks = self.segmenter.segment(rgb, self.prompt)
        
        if masks is None or len(masks) == 0:
            print(f"[头部] 未检测到 ({time.time()-t0:.2f}s)")
            self._head_results = []
            return
        
        tf_mat = self._tf['matrix'] if self._tf else None
        self._head_results = self.head_calc.compute(masks, depth, tf_mat)
        print(f"[头部] 检测到 {len(self._head_results)} 个 ({time.time()-t0:.2f}s)")
        
        for b in self._head_results:
            p = b['position']
            print(f"  #{b['id']}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]m yaw={b['yaw_deg']:.1f}°")
        
        self._head_frame = rgb.copy()
        self._head_until = time.time() + 5
    
    def _handeye_detect(self):
        if not self.handeye_calc:
            print("[手眼] 未加载标定")
            return
        
        img = self._get_handeye_img()
        pose = self._get_arm_pose()
        if img is None or pose is None:
            print("[手眼] 图像或位姿获取失败")
            return
        
        # 获取参考 Z 值（来自头部相机检测结果）
        reference_z = None
        if self._head_results:
            reference_z = self._head_results[0]['position'][2]
            print(f"[手眼] 使用头部检测的 Z 参考值: {reference_z:.4f}m")
        
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
        print(f"[手眼] 检测到 {len(self._handeye_results)} 个 ({time.time()-t0:.2f}s)")
        print("=" * 70)
        print(f"{'#':<3} {'X':>8} {'Y':>8} {'Z':>8} │ {'Yaw°':>7} │ {'深度':>8} │ {'参考Z'}")
        print("-" * 70)
        for b in self._handeye_results:
            p = b['position']
            ref_mark = "✓" if b.get('used_reference') else "✗"
            print(f"{b['id']:<3} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} │ {b['yaw_deg']:>+7.1f} │ {b['estimated_depth_m']:>8.4f} │ {ref_mark}")
        print("=" * 70)
        
        self._handeye_frame = img.copy()
        self._handeye_until = time.time() + 5

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
        """使用手眼相机的 XY 位置精确移动，Z 保持当前高度"""
        if not self._handeye_results:
            print("[精确移动] 请先按 'e' 进行手眼分割")
            return
        
        # 获取当前机械臂位置
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[精确移动] 无法获取当前位姿")
            return
        
        current_pos = poses[self.arm_key]['position']
        current_quat = poses[self.arm_key]['orientation']
        current_z = current_pos[2]
        
        # 使用手眼检测的 XY 和 yaw
        b = self._handeye_results[0]
        handeye_pos = b['position']
        handeye_yaw = b['yaw']
        
        # 应用补偿
        compensated_pos = np.array([
            handeye_pos[0] + self.handeye_offset['x'],
            handeye_pos[1] + self.handeye_offset['y'],
            handeye_pos[2] + self.handeye_offset['z']
        ])
        
        # 悬停高度 = 当前末端高度 - 砖块高度
        hover_height = current_z - compensated_pos[2]
        
        print(f"\n[精确移动] 使用手眼 XY，保持当前高度")
        print(f"  原始检测: X={handeye_pos[0]:.4f}, Y={handeye_pos[1]:.4f}, Z={handeye_pos[2]:.4f}")
        print(f"  补偿值:   X={self.handeye_offset['x']:+.4f}, Y={self.handeye_offset['y']:+.4f}, Z={self.handeye_offset['z']:+.4f}")
        print(f"  补偿后:   X={compensated_pos[0]:.4f}, Y={compensated_pos[1]:.4f}, Z={compensated_pos[2]:.4f}")
        print(f"  手眼 yaw: {b['yaw_deg']:.1f}°")
        
        # === 输出外参修正计算信息 ===
        if self.handeye_offset['x'] != 0 or self.handeye_offset['y'] != 0 or self.handeye_offset['z'] != 0:
            self._print_extrinsics_correction(current_pos, current_quat)
        
        self.arm_controller.move_to_hover(
            brick_position=compensated_pos,
            brick_yaw=handeye_yaw,
            hover_height=hover_height
        )

    def _print_extrinsics_correction(self, current_pos, current_quat):
        """输出外参修正建议"""
        print("\n" + "=" * 70)
        print("[外参修正计算]")
        print("-" * 70)
        
        # 当前夹爪位姿
        R_g2b = R.from_quat(current_quat).as_matrix()
        t_g2b = np.array(current_pos)
        print(f"  夹爪位置 (base): [{t_g2b[0]:.4f}, {t_g2b[1]:.4f}, {t_g2b[2]:.4f}]")
        euler_g = R.from_quat(current_quat).as_euler('xyz', degrees=True)
        print(f"  夹爪欧拉角:      [{euler_g[0]:.1f}, {euler_g[1]:.1f}, {euler_g[2]:.1f}]°")
        
        # base_link 下的补偿向量
        offset_base = np.array([self.handeye_offset['x'], self.handeye_offset['y'], self.handeye_offset['z']])
        print(f"  补偿向量 (base): [{offset_base[0]:+.4f}, {offset_base[1]:+.4f}, {offset_base[2]:+.4f}]")
        
        # 转换到夹爪坐标系: offset_gripper = R_g2b^T @ offset_base
        offset_gripper = R_g2b.T @ offset_base
        print(f"  补偿向量 (gripper): [{offset_gripper[0]:+.4f}, {offset_gripper[1]:+.4f}, {offset_gripper[2]:+.4f}]")
        
        # 当前外参 translation
        current_trans = self.handeye_calc.T_cam2gripper[:3, 3]
        print(f"\n  当前外参 translation: [{current_trans[0]:.4f}, {current_trans[1]:.4f}, {current_trans[2]:.4f}]")
        
        # 建议的新外参 translation
        new_trans = current_trans + offset_gripper
        print(f"  建议新 translation:  [{new_trans[0]:.4f}, {new_trans[1]:.4f}, {new_trans[2]:.4f}]")
        
        print("-" * 70)
        print("  可直接复制到 hand_eye_extrinsics_left.json:")
        print(f'  "translation": [{new_trans[0]:.4f}, {new_trans[1]:.4f}, {new_trans[2]:.4f}]')
        print("=" * 70)

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
        
        # 获取另一只臂位姿
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
            
            # 如果有手眼检测结果，计算并记录 Z 轴补偿
            if self._handeye_results:
                brick_z = self._handeye_results[0]['position'][2]
                diff = new_z - brick_z
                print(f"  夹爪高度: {new_z:.4f}m, 砖块高度(检测): {brick_z:.4f}m, 差值: {diff:+.4f}m")
                
                # 当夹爪触碰到砖块时，差值就是需要的 Z 补偿
                # 注意：如果夹爪在砖块上方，diff > 0；触碰时应该 diff ≈ 0
                # 但实际上，检测的 Z 是砖块中心，所以触碰时 diff ≈ 砖块高度/2
                print(f"  提示: 当夹爪触碰砖块顶面时，按 'c' 记录当前 Z 差值作为补偿")
        except Exception as e:
            print(f"[Z移动] 失败: {e}")

    def _record_z_offset(self):
        """记录当前夹爪高度与检测砖块高度的差值作为 Z 补偿"""
        if not self._handeye_results:
            print("[Z补偿] 请先按 'e' 进行手眼分割")
            return
        
        poses = self.robot.get_arm_end_poses()
        if not poses or self.arm_key not in poses:
            print("[Z补偿] 无法获取当前位姿")
            return
        
        current_z = poses[self.arm_key]['position'][2]
        brick_z = self._handeye_results[0]['position'][2]
        
        # Z 补偿 = 实际砖块高度 - 检测砖块高度
        # 如果夹爪触碰到砖块顶面，实际砖块高度 ≈ 当前夹爪高度
        z_offset = current_z - brick_z
        self.handeye_offset['z'] = z_offset
        
        print(f"\n[Z补偿] 已记录")
        print(f"  夹爪当前高度: {current_z:.4f}m")
        print(f"  检测砖块高度: {brick_z:.4f}m")
        print(f"  Z 轴补偿值:   {z_offset:+.4f}m")
        print(f"  全部补偿: X={self.handeye_offset['x']:+.4f}, Y={self.handeye_offset['y']:+.4f}, Z={self.handeye_offset['z']:+.4f}")
        print("  按 'f' 查看完整外参修正建议")

    def _descend_to_brick_surface(self):
        """下降到砖块表面"""
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
        
        # 获取砖块位置并应用补偿
        b = self._handeye_results[0]
        brick_z = b['position'][2] + self.handeye_offset['z']
        
        # 计算砖块表面高度 = 砖块中心Z + 砖块高度/2
        brick_surface_z = brick_z + BRICK_SIZE['height'] / 2
        
        # 目标高度 = 砖块表面
        target_z = brick_surface_z
        
        print(f"\n[下降] 移动到砖块表面")
        print(f"  当前高度:     {current_pos[2]:.4f}m")
        print(f"  砖块中心Z:    {brick_z:.4f}m")
        print(f"  砖块表面Z:    {brick_surface_z:.4f}m (中心+{BRICK_SIZE['height']/2:.4f}m)")
        print(f"  目标高度:     {target_z:.4f}m")
        
        # 获取另一只臂位姿
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
            print(f"[下降] 指令已发送，下降 {current_pos[2] - target_z:.4f}m")
        except Exception as e:
            print(f"[下降] 失败: {e}")

    def _reset(self):
        print("\n[重置] 回到初始位置...")
        from mmk2_types.grpc_msgs import JointState, TrajectoryParams, GoalStatus
        
        arm_action = {
            MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
            MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.SPINE: JointState(position=[0.15]),
        }
        
        try:
            result = self.robot.mmk2.set_goal(arm_action, TrajectoryParams())
            if result.value == GoalStatus.Status.SUCCESS:
                print("[重置] 完成")
            else:
                print(f"[重置] 状态: {result}")
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
            
            self._tf = self.tf_client.get_transform('base_link', 'head_camera_link')
            
            # 头部显示
            if time.time() < self._head_until and self._head_frame is not None:
                disp = self._draw_results(self._head_frame, self._head_results)
            else:
                disp = rgb.copy()
                cv2.putText(disp, f"'s':detect '{self.prompt}'", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow("Head", disp)
            
            # 手眼显示
            he_img = self._get_handeye_img()
            if he_img is not None:
                if time.time() < self._handeye_until and self._handeye_frame is not None:
                    he_disp = self._draw_results(self._handeye_frame, self._handeye_results, True)
                else:
                    he_disp = he_img.copy()
                    cv2.putText(he_disp, "'e':segment 'f':fine-move", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                
                # 显示当前补偿值
                offset_text = f"Offset: X={self.handeye_offset['x']:+.3f} Y={self.handeye_offset['y']:+.3f} | 1/2:Y 3/4:X 0:reset"
                cv2.putText(he_disp, offset_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow("HandEye", he_disp)
            
            # 按键
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s') and depth is not None:
                self._head_detect(rgb, depth)
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
            # 手眼补偿调试按键
            elif k == ord('1'):  # Y+
                self.handeye_offset['y'] += self.offset_step
                print(f"[补偿] Y轴: {self.handeye_offset['y']:+.4f}m  (全部: X={self.handeye_offset['x']:+.4f}, Y={self.handeye_offset['y']:+.4f})")
            elif k == ord('2'):  # Y-
                self.handeye_offset['y'] -= self.offset_step
                print(f"[补偿] Y轴: {self.handeye_offset['y']:+.4f}m  (全部: X={self.handeye_offset['x']:+.4f}, Y={self.handeye_offset['y']:+.4f})")
            elif k == ord('3'):  # X+
                self.handeye_offset['x'] += self.offset_step
                print(f"[补偿] X轴: {self.handeye_offset['x']:+.4f}m  (全部: X={self.handeye_offset['x']:+.4f}, Y={self.handeye_offset['y']:+.4f})")
            elif k == ord('4'):  # X-
                self.handeye_offset['x'] -= self.offset_step
                print(f"[补偿] X轴: {self.handeye_offset['x']:+.4f}m  (全部: X={self.handeye_offset['x']:+.4f}, Y={self.handeye_offset['y']:+.4f}, Z={self.handeye_offset['z']:+.4f})")
            elif k == ord('5'):  # Z+ (夹爪上升)
                self._move_gripper_z(self.offset_step)
            elif k == ord('6'):  # Z- (夹爪下降)
                self._move_gripper_z(-self.offset_step)
            elif k == ord('c'):  # 记录当前 Z 差值作为补偿
                self._record_z_offset()                
            elif k == ord('0'):  # 重置补偿
                self.handeye_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                print("[补偿] 已重置为零")
            elif k == ord('r'):
                self._reset()                
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