"""
砖块检测器 - 分割 + 3D 位置计算 + 机械臂控制
- 按 's' 触发 SAM3 分割并计算 3D 位置（相对于 base_link）
- 按 'g' 控制机械臂移动到砖块正上方
- 按 'p' 打印当前机械臂末端位姿
- 按 't' 打印 TF 变换详情
- 按 'q' 退出
"""

import sys
SAM3_PROJECT_PATH = "/home/ypf/sam3-main"
if SAM3_PROJECT_PATH not in sys.path:
    sys.path.insert(0, SAM3_PROJECT_PATH)

import cv2
import argparse
import time
import numpy as np
from PIL import Image
import torch
import gc
from typing import Optional, List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

from airbot_sdk import MMK2RealRobot
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tf_client import TFClient
from mmk2_types.types import MMK2Components
from mmk2_types.grpc_msgs import Pose, Position, Orientation


# D435i 相机内参（640x480）
CAMERA_INTRINSICS = {
    'fx': 607.15,
    'fy': 607.02,
    'cx': 324.25,
    'cy': 248.46,
}

# 砖块尺寸 (m)
BRICK_SIZE = {
    'length': 0.11,
    'width': 0.05,
    'height': 0.025,
}

# 机械臂控制参数
ARM_CONTROL = {
    'hover_height': 0.25,  # 悬停高度：砖块上方 25cm
    'use_left_arm': True,  # 使用左臂
}

# 可视化颜色
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
]


class SAM3Segmenter:
    """SAM3 分割器"""
    _model = None
    _processor = None
    
    def __init__(self, checkpoint_path: str, confidence_threshold: float = 0.5):
        if SAM3Segmenter._model is None:
            print("[SAM3] 加载模型中...")
            SAM3Segmenter._model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            SAM3Segmenter._processor = Sam3Processor(
                SAM3Segmenter._model, resolution=1008, confidence_threshold=confidence_threshold
            )
            print("[SAM3] 模型加载完成!")
            torch.cuda.empty_cache()
            gc.collect()
        
        self.processor = SAM3Segmenter._processor
    
    def segment(self, image_bgr: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """分割图像，返回 masks (N, H, W)"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        masks = output["masks"].cpu().numpy() if output["masks"] is not None else None
        torch.cuda.empty_cache()
        
        return masks


class BrickPositionCalculator:
    """砖块 3D 位置计算器"""
    
    def __init__(self, intrinsics: dict = CAMERA_INTRINSICS, brick_height: float = BRICK_SIZE['height']):
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.brick_height = brick_height
    
    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """像素坐标 + 深度 -> 相机坐标系 3D 点"""
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])
    
    def camera_to_base(self, point_camera: np.ndarray, tf_matrix: np.ndarray) -> np.ndarray:
        """将相机坐标系下的点转换到 base_link 坐标系"""
        point_homogeneous = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        point_base = tf_matrix @ point_homogeneous
        return point_base[:3]
    
    def transform_yaw_to_base(self, yaw_camera: float, tf_matrix: np.ndarray) -> float:
        """将相机坐标系下的 yaw 角度转换到 base_link 坐标系"""
        R_mat = tf_matrix[:3, :3]
        camera_yaw_offset = np.arctan2(R_mat[1, 0], R_mat[0, 0])
        yaw_base = yaw_camera + camera_yaw_offset
        
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        return yaw_base
    
    def compute_positions(self, masks: np.ndarray, depth: np.ndarray, 
                          tf_matrix: Optional[np.ndarray] = None) -> List[Dict]:
        """从分割掩码计算砖块 3D 位置"""
        results = []
        h, w = depth.shape
        
        for i, mask in enumerate(masks):
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            ys, xs = np.where(mask_bool)
            cx, cy = np.mean(xs), np.mean(ys)
            
            yaw_camera, rect_info = self._estimate_orientation(mask_bool)
            
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
            valid_depths = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
            valid_depths = valid_depths[valid_depths > 0]
            
            if len(valid_depths) == 0:
                continue
            
            avg_depth = np.median(valid_depths) / 1000.0
            pos_camera = self.pixel_to_camera(cx, cy, avg_depth)
            pos_camera[2] = max(avg_depth - self.brick_height / 2.0, self.brick_height / 2.0)
            
            if tf_matrix is not None:
                pos_base = self.camera_to_base(pos_camera, tf_matrix)
                yaw_base = self.transform_yaw_to_base(yaw_camera, tf_matrix)
            else:
                pos_base = pos_camera
                yaw_base = yaw_camera
            
            results.append({
                'id': i + 1,
                'position': pos_base,
                'position_camera': pos_camera,
                'yaw': yaw_base,
                'yaw_camera': yaw_camera,
                'yaw_deg': np.degrees(yaw_base),
                'pixel_center': (int(cx), int(cy)),
                'depth_m': avg_depth,
                'mask_area': np.sum(mask_bool),
                'rect_info': rect_info,
            })
        
        results.sort(key=lambda x: x['depth_m'])
        return results
    
    def _estimate_orientation(self, mask_bool: np.ndarray) -> Tuple[float, Optional[Dict]]:
        """从掩码形状估算砖块朝向"""
        contours, _ = cv2.findContours(
            (mask_bool * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0, None
        
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return 0.0, None
        
        rect = cv2.minAreaRect(largest)
        (rect_cx, rect_cy), (rect_w, rect_h), angle = rect
        
        if rect_w < rect_h:
            long_edge_angle = angle + 90
        else:
            long_edge_angle = angle
        
        yaw = -np.radians(long_edge_angle)
        while yaw > np.pi / 2:
            yaw -= np.pi
        while yaw < -np.pi / 2:
            yaw += np.pi
        
        rect_info = {
            'center': (rect_cx, rect_cy),
            'size': (rect_w, rect_h),
            'angle': angle,
            'box_points': cv2.boxPoints(rect),
        }
        
        return float(yaw), rect_info


class ArmController:
    """机械臂控制器 - 末端 X 轴为夹爪指向方向"""
    
    def __init__(self, robot: MMK2RealRobot, use_left_arm: bool = True):
        self.robot = robot
        self.use_left_arm = use_left_arm
        self.arm_component = MMK2Components.LEFT_ARM if use_left_arm else MMK2Components.RIGHT_ARM
        self.arm_name = "左臂" if use_left_arm else "右臂"
    
    def compute_grasp_orientation(self, yaw: float) -> Tuple[float, float, float, float]:
        """
        计算抓取姿态四元数
        
        末端坐标系定义：
        - X 轴：夹爪指向方向（抓取时朝下）
        - Y/Z 轴：夹爪开合方向
        
        目标：
        - X 轴朝下（指向 -Z 方向）
        - 绕 X 轴旋转 yaw，调整夹爪开合方向对齐砖块
        
        Args:
            yaw: 砖块在 base_link 坐标系下的 yaw 角度（弧度）
        
        Returns:
            四元数 (x, y, z, w)
        """
        # 步骤1：构造基础姿态（X轴朝下）
        # 从单位姿态（X朝前）旋转到 X 朝下，需要绕 Y 轴旋转 +90°
        r_base = R.from_euler('y', np.pi / 2)
        
        # 步骤2：绕末端 X 轴旋转 yaw（调整夹爪开合方向）
        adjusted_yaw = yaw
        r_yaw = R.from_euler('z', adjusted_yaw)
        
        # 组合：先基础姿态，再 yaw 调整
        # 注意顺序：世界坐标系下的旋转要放在前面
        r_final = r_yaw * r_base
        
        # 返回四元数
        quat = r_final.as_quat()  # [x, y, z, w]
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    
    def move_to_hover(self, brick_position: np.ndarray, brick_yaw: float, 
                      hover_height: float = 0.20) -> bool:
        """
        移动机械臂到砖块正上方
        
        Args:
            brick_position: 砖块在 base_link 下的位置 [x, y, z]
            brick_yaw: 砖块的 yaw 角度（弧度）
            hover_height: 悬停高度（米）
        
        Returns:
            是否成功
        """
        # 目标位置：砖块正上方
        target_x = float(brick_position[0])
        target_y = float(brick_position[1])
        target_z = float(brick_position[2]) + hover_height
        
        # 计算姿态（根据 yaw 调整）
        ox, oy, oz, ow = self.compute_grasp_orientation(brick_yaw)
        
        print(f"\n[{self.arm_name}] 移动到砖块上方")
        print(f"  目标位置: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]m")
        print(f"  砖块 yaw: {np.degrees(brick_yaw):.1f}°")
        print(f"  姿态四元数: [{ox:.4f}, {oy:.4f}, {oz:.4f}, {ow:.4f}]")
        
        # 验证姿态：打印末端 X 轴方向
        r = R.from_quat([ox, oy, oz, ow])
        rot_mat = r.as_matrix()
        end_x = rot_mat[:, 0]  # 末端 X 轴在世界坐标系下的方向
        end_y = rot_mat[:, 1]  # 末端 Y 轴
        print(f"  末端X轴方向: [{end_x[0]:.4f}, {end_x[1]:.4f}, {end_x[2]:.4f}] (应接近 [0, 0, -1])")
        print(f"  末端Y轴方向: [{end_y[0]:.4f}, {end_y[1]:.4f}, {end_y[2]:.4f}]")
        
        # 构建目标位姿
        target_pose = {
            self.arm_component: Pose(
                position=Position(x=target_x, y=target_y, z=target_z),
                orientation=Orientation(x=ox, y=oy, z=oz, w=ow),
            ),
        }
        
        # 执行移动
        try:
            self.robot.control_arm_poses(target_pose)
            print(f"[{self.arm_name}] 移动指令已发送")
            return True
        except Exception as e:
            print(f"[{self.arm_name}] 移动失败: {e}")
            return False
    
    def print_current_pose(self):
        """打印当前机械臂末端位姿"""
        poses = self.robot.get_arm_end_poses()
        if poses is None:
            print(f"\n[{self.arm_name}] 获取位姿失败")
            return
        
        arm_key = 'left_arm' if self.use_left_arm else 'right_arm'
        if arm_key not in poses:
            print(f"\n[{self.arm_name}] 未找到位姿数据")
            return
        
        pose = poses[arm_key]
        pos = pose['position']
        quat = pose['orientation']
        
        print(f"\n{'='*60}")
        print(f"[{self.arm_name}] 当前末端位姿")
        print(f"{'='*60}")
        print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m")
        print(f"  四元数: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=True)
        print(f"  欧拉角 (xyz): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]°")
        
        rot_mat = r.as_matrix()
        end_x = rot_mat[:, 0]
        end_y = rot_mat[:, 1]
        end_z = rot_mat[:, 2]
        
        angle_x_down = np.degrees(np.arccos(np.clip(np.dot(end_x, [0, 0, -1]), -1, 1)))
        print(f"  末端X轴: [{end_x[0]:.4f}, {end_x[1]:.4f}, {end_x[2]:.4f}], 与朝下夹角: {angle_x_down:.1f}°")
        print(f"  末端Y轴: [{end_y[0]:.4f}, {end_y[1]:.4f}, {end_y[2]:.4f}]")
        print(f"  末端Z轴: [{end_z[0]:.4f}, {end_z[1]:.4f}, {end_z[2]:.4f}]")
        print(f"{'='*60}")


class BrickDetector:
    """砖块检测主程序"""
    
    def __init__(self, ip: str, checkpoint_path: str, text_prompt: str,
                 tf_host: str = "127.0.0.1", tf_port: int = 9999,
                 use_left_arm: bool = True):
        
        self.text_prompt = text_prompt
        
        # 初始化机器人
        print(f"[Init] 连接机器人 {ip}...")
        self.robot = MMK2RealRobot(ip=ip)
        self.robot.set_robot_head_pose(0, -1.08)
        
        # 初始化模块
        self.segmenter = SAM3Segmenter(checkpoint_path)
        self.calculator = BrickPositionCalculator()
        self.tf_client = TFClient(host=tf_host, port=tf_port, auto_connect=True)
        self.arm_controller = ArmController(self.robot, use_left_arm=use_left_arm)
        
        # 缓存
        self._results: List[Dict] = []
        self._cached_frame: Optional[np.ndarray] = None
        self._current_tf: Optional[dict] = None
        self._display_until: float = 0
        
        print("[Init] 初始化完成")
        print("  按 's' - 分割并计算位置（相对于 base_link）")
        print("  按 'g' - 移动机械臂到砖块上方（需先按 's' 检测）")
        print("  按 'p' - 打印当前机械臂末端位姿")
        print("  按 't' - 打印 TF 详情")
        print("  按 'r' - 重置机械臂到初始位置")
        print("  按 'q' - 退出")
    
    def _draw_results(self, frame: np.ndarray) -> np.ndarray:
        """绘制检测结果"""
        result = frame.copy()
        
        for i, brick in enumerate(self._results):
            color = COLORS[i % len(COLORS)]
            cx, cy = brick['pixel_center']
            pos = brick['position']
            yaw_deg = brick['yaw_deg']
            
            cv2.circle(result, (cx, cy), 6, color, -1)
            cv2.circle(result, (cx, cy), 9, (255, 255, 255), 2)
            
            if brick['rect_info']:
                box = brick['rect_info']['box_points'].astype(np.int32)
                cv2.drawContours(result, [box], 0, color, 2)
            
            arrow_len = 40
            yaw_camera = brick['yaw_camera']
            end_x = int(cx + arrow_len * np.cos(yaw_camera))
            end_y = int(cy - arrow_len * np.sin(yaw_camera))
            cv2.arrowedLine(result, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)
            
            info_text = f"#{brick['id']}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m, yaw:{yaw_deg:.1f}°"
            cv2.putText(result, info_text, (cx + 15, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def _draw_tf_info(self, frame: np.ndarray) -> np.ndarray:
        """绘制 TF 信息"""
        h = frame.shape[0]
        
        if self._current_tf is None:
            cv2.putText(frame, "TF: Not connected", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            t = self._current_tf['translation']
            rpy = self._current_tf['rpy_deg']
            cv2.putText(frame, f"Camera T: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"Camera RPY: [{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]deg", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        return frame
    
    def _trigger_detection(self, img_rgb: np.ndarray, img_depth: np.ndarray):
        """触发检测"""
        print(f"\n[检测] 开始，提示词: '{self.text_prompt}'")
        t0 = time.time()
        
        masks = self.segmenter.segment(img_rgb, self.text_prompt)
        t1 = time.time()
        
        if masks is None or len(masks) == 0:
            print(f"[检测] 未检测到目标 ({t1-t0:.2f}s)")
            self._results = []
            return
        
        tf_matrix = None
        if self._current_tf is not None:
            tf_matrix = self._current_tf['matrix']
        else:
            print("[警告] TF 未连接，输出相机坐标系位置")
        
        self._results = self.calculator.compute_positions(masks, img_depth, tf_matrix)
        t2 = time.time()
        
        print(f"[检测] 完成: {len(self._results)} 个目标 (分割:{t1-t0:.2f}s, 计算:{t2-t1:.3f}s)")
        
        coord_frame = "base_link" if tf_matrix is not None else "camera"
        print(f"[坐标系: {coord_frame}]")
        for brick in self._results:
            pos = brick['position']
            print(f"  #{brick['id']}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m, "
                  f"yaw={brick['yaw_deg']:.1f}°, depth={brick['depth_m']:.3f}m")
        
        self._cached_frame = img_rgb.copy()
        self._display_until = time.time() + 5.0
    
    def _trigger_arm_move(self):
        """触发机械臂移动到砖块上方"""
        if not self._results:
            print("\n[机械臂] 请先按 's' 检测砖块")
            return
        
        if self._current_tf is None:
            print("\n[机械臂] TF 未连接，无法移动")
            return
        
        # 使用第一个检测到的砖块（最近的）
        brick = self._results[0]
        pos = brick['position']  # base_link 坐标系
        yaw = brick['yaw']       # base_link 坐标系
        
        print(f"\n[机械臂] 目标砖块 #{brick['id']}")
        print(f"  位置 (base): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m")
        print(f"  yaw (base): {np.degrees(yaw):.1f}°")
        
        # 移动到上方
        self.arm_controller.move_to_hover(
            brick_position=pos,
            brick_yaw=yaw,
            hover_height=ARM_CONTROL['hover_height']
        )

    def _reset_arms(self):
        """重置机械臂到初始位置（不包括头部）"""
        from mmk2_types.types import MMK2Components
        from mmk2_types.grpc_msgs import JointState, TrajectoryParams, GoalStatus
        
        print("\n[重置] 机械臂回到初始位置...")
        arm_action = {
            MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
            MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
            MMK2Components.SPINE: JointState(position=[0.0]),
        }
        
        try:
            result = self.robot.mmk2.set_goal(arm_action, TrajectoryParams())
            if result.value == GoalStatus.Status.SUCCESS:
                print("[重置] 机械臂已回到初始位置")
            else:
                print(f"[重置] 失败: {result}")
        except Exception as e:
            print(f"[重置] 异常: {e}")

    def _print_tf_details(self):
        """打印 TF 详情"""
        if self._current_tf is None:
            print("\n[TF] 未连接")
            return
        
        t = self._current_tf['translation']
        rpy_deg = self._current_tf['rpy_deg']
        matrix = self._current_tf['matrix']
        
        print("\n" + "=" * 60)
        print(f"TF: {self._current_tf['target_frame']} <- {self._current_tf['source_frame']}")
        print("=" * 60)
        print(f"Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
        print(f"RPY (deg):   [{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}]")
        print(f"Matrix:\n{matrix}")
        print("=" * 60 + "\n")
    
    def run(self):
        """主循环"""
        print("\n开始运行，等待相机...")
        cv2.namedWindow("Brick Detector", cv2.WINDOW_NORMAL)
        
        for img_head, img_depth, _, _ in self.robot.camera:
            if img_head is None or img_head.size == 0:
                continue
            
            self._current_tf = self.tf_client.get_transform('base_link', 'head_camera_link')
            
            if time.time() < self._display_until and self._cached_frame is not None:
                display = self._draw_results(self._cached_frame)
            else:
                display = img_head.copy()
                cv2.putText(display, f"'s':detect 'g':move 'p':pose 'r':reset (prompt: {self.text_prompt})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            
            display = self._draw_tf_info(display)
            cv2.imshow("Brick Detector", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and img_depth is not None:
                self._trigger_detection(img_head, img_depth)
            elif key == ord('g'):
                self._trigger_arm_move()
            elif key == ord('p'):
                self.arm_controller.print_current_pose()
            elif key == ord('t'):
                self._print_tf_details()
            elif key == ord('r'):
                self._reset_arms()
            elif key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        self.tf_client.disconnect()
        print("程序结束")


def main():
    parser = argparse.ArgumentParser(description='砖块检测器')
    parser.add_argument("--ip", type=str, default="192.168.11.200")
    parser.add_argument("--prompt", type=str, default="block")
    parser.add_argument("--checkpoint", type=str, default="/home/ypf/sam3-main/checkpoint/sam3.pt")
    parser.add_argument("--tf-host", type=str, default="127.0.0.1")
    parser.add_argument("--tf-port", type=int, default=9999)
    parser.add_argument("--right-arm", action="store_true", help="使用右臂（默认左臂）")
    args = parser.parse_args()
    
    detector = BrickDetector(
        ip=args.ip,
        checkpoint_path=args.checkpoint,
        text_prompt=args.prompt,
        tf_host=args.tf_host,
        tf_port=args.tf_port,
        use_left_arm=not args.right_arm
    )
    detector.run()


if __name__ == '__main__':
    main()