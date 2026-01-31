"""
砖块异常检测测试脚本
- 实时显示 D435i 头部相机 RGB 图像
- 按 's' 键触发 SAM3 分割
- 显示所有砖块的 Z 轴位置和面积
- 用于检测砖块堆叠异常

按键:
  s - 触发分割，显示所有砖块信息
  t - 打印 TF 变换详情
  c - 清除分割结果
  q - 退出
"""

import sys
from pathlib import Path

# Add parent directory to path for importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/ypf/sam3-main")

import cv2
import argparse
import time
import numpy as np
from PIL import Image
import torch
import gc
import threading
from typing import Optional, List, Dict

from airbot_sdk import MMK2RealRobot
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tf_client import TFClient


# 配置
HEAD_INTRINSICS = {'fx': 607.15, 'fy': 607.02, 'cx': 324.25, 'cy': 248.46}
BRICK_HEIGHT = 0.025  # 砖块高度 25mm

# 可视化颜色
COLORS = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 255, 255),  # 黄色
    (255, 0, 255),  # 紫色
    (0, 165, 255),  # 橙色
    (255, 255, 0),  # 青色
]


class SAM3Segmenter:
    """SAM3 分割器 - 单例模式"""
    _instance = None
    _model = None
    _processor = None
    
    def __init__(self,
                 checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
                 text_prompt: str = "brick",
                 sam_resolution: int = 1008,
                 confidence_threshold: float = 0.5):
        
        self.text_prompt = text_prompt
        self.sam_resolution = sam_resolution
        self.confidence_threshold = confidence_threshold
        
        if SAM3Segmenter._model is None:
            print(f"[SAM3] 加载模型中...")
            SAM3Segmenter._model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            SAM3Segmenter._processor = Sam3Processor(
                SAM3Segmenter._model, 
                resolution=sam_resolution, 
                confidence_threshold=confidence_threshold
            )
            print(f"[SAM3] 模型加载完成!")
            torch.cuda.empty_cache()
            gc.collect()
        
        self.model = SAM3Segmenter._model
        self.processor = SAM3Segmenter._processor
        self._lock = threading.Lock()
    
    def segment(self, image_bgr: np.ndarray, prompt: Optional[str] = None) -> Optional[np.ndarray]:
        """对图像进行分割"""
        with self._lock:
            prompt = prompt or self.text_prompt
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            masks = output["masks"].cpu().numpy() if output["masks"] is not None else None
            
            torch.cuda.empty_cache()
            
            return masks


class BrickAnalyzer:
    """砖块分析器 - 计算每个砖块的 Z 位置和面积"""
    
    def __init__(self):
        self.fx = HEAD_INTRINSICS['fx']
        self.fy = HEAD_INTRINSICS['fy']
        self.cx = HEAD_INTRINSICS['cx']
        self.cy = HEAD_INTRINSICS['cy']
    
    def analyze(self, masks: np.ndarray, depth: np.ndarray, 
                tf_matrix: Optional[np.ndarray] = None) -> List[Dict]:
        """
        分析所有砖块的位置和面积
        
        Args:
            masks: SAM3 分割掩码 (N, H, W) 或 (N, 1, H, W)
            depth: 深度图 (H, W)，单位毫米
            tf_matrix: 相机到 base_link 的变换矩阵
            
        Returns:
            砖块信息列表，按 Z 从高到低排序
        """
        results = []
        h, w = depth.shape
        
        for i, mask in enumerate(masks):
            # 处理掩码维度
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), 
                                 interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            # 计算像素中心
            ys, xs = np.where(mask_bool)
            px, py = np.mean(xs), np.mean(ys)
            
            # 计算掩码面积（像素数）
            area_pixels = np.sum(mask_bool)
            
            # 获取深度值
            # 使用腐蚀后的掩码获取更稳定的深度
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
            
            if np.any(eroded):
                valid_depth = depth[eroded > 0]
            else:
                valid_depth = depth[mask_bool]
            
            valid_depth = valid_depth[valid_depth > 0]
            if len(valid_depth) == 0:
                continue
            
            # 深度统计
            depth_median = np.median(valid_depth) / 1000.0  # 转为米
            depth_min = np.min(valid_depth) / 1000.0
            depth_max = np.max(valid_depth) / 1000.0
            depth_std = np.std(valid_depth) / 1000.0
            
            # 计算相机坐标系下的 3D 位置
            z_cam = depth_median
            x_cam = (px - self.cx) * z_cam / self.fx
            y_cam = (py - self.cy) * z_cam / self.fy
            pos_cam = np.array([x_cam, y_cam, z_cam])
            
            # 转换到 base_link 坐标系
            if tf_matrix is not None:
                pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3]
                z_base = pos_base[2]
            else:
                pos_base = pos_cam
                z_base = z_cam
            
            # 估算实际面积（根据深度和像素面积）
            # 每个像素在物体表面的实际尺寸约为 z/f
            pixel_size = z_cam / self.fx  # 米/像素
            area_m2 = area_pixels * (pixel_size ** 2)  # 平方米
            area_cm2 = area_m2 * 10000  # 平方厘米
            
            results.append({
                'id': i + 1,
                'pixel_center': (int(px), int(py)),
                'area_pixels': area_pixels,
                'area_cm2': area_cm2,
                'depth_median_m': depth_median,
                'depth_min_m': depth_min,
                'depth_max_m': depth_max,
                'depth_std_m': depth_std,
                'z_base_m': z_base,
                'position_base': pos_base,
                'position_cam': pos_cam,
                'mask': mask_bool,
            })
        
        # 按 Z 从高到低排序（Z 值大的在上面）
        results = sorted(results, key=lambda x: x['z_base_m'], reverse=True)
        
        # 重新编号
        for i, r in enumerate(results):
            r['rank'] = i + 1  # 1 = 最高
        
        return results
    
    def detect_stacking(self, bricks: List[Dict], 
                        xy_threshold: float = 0.08,
                        z_threshold: float = 0.015) -> List[Dict]:
        """
        检测砖块堆叠关系
        
        Args:
            bricks: 砖块信息列表
            xy_threshold: XY 距离阈值（米），小于此值认为可能堆叠
            z_threshold: Z 高度差阈值（米），大于此值认为是堆叠
            
        Returns:
            添加堆叠信息后的砖块列表
        """
        for brick in bricks:
            brick['is_stacked_on'] = None
            brick['stacking_type'] = 'normal'
        
        # 检测堆叠关系
        for i, upper in enumerate(bricks):
            for j, lower in enumerate(bricks):
                if i == j:
                    continue
                
                # 计算 XY 距离
                xy_dist = np.sqrt(
                    (upper['position_base'][0] - lower['position_base'][0])**2 +
                    (upper['position_base'][1] - lower['position_base'][1])**2
                )
                
                # 计算 Z 差值
                z_diff = upper['z_base_m'] - lower['z_base_m']
                
                # 判断堆叠：XY 距离近且 Z 差值约为一个砖块高度
                if xy_dist < xy_threshold and z_threshold < z_diff < 0.04:
                    upper['is_stacked_on'] = lower['id']
                    upper['stacking_type'] = 'stacked_upper'
                    lower['stacking_type'] = 'stacked_lower'
        
        return bricks


class BrickAnomalyDetector:
    """砖块异常检测系统"""
    
    def __init__(self,
                 ip: str = "192.168.11.200",
                 checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
                 text_prompt: str = "block, brick",
                 display_duration: float = 10.0,
                 tf_server_host: str = "127.0.0.1",
                 tf_server_port: int = 9999):
        
        self.ip = ip
        self.text_prompt = text_prompt
        self.display_duration = display_duration
        
        # 初始化机器人
        print(f"[Camera] 连接机器人 {ip}...")
        self.robot = MMK2RealRobot(ip=ip)
        self.robot.set_robot_head_pose(0, -1.08)
        self.robot.set_spine(0.05)
        
        # 初始化分割器和分析器
        self.segmenter = SAM3Segmenter(checkpoint_path=checkpoint_path, text_prompt=text_prompt)
        self.analyzer = BrickAnalyzer()
        
        # 初始化 TF 客户端
        print(f"[TF] 连接 TF 服务器 {tf_server_host}:{tf_server_port}...")
        self.tf_client = TFClient(host=tf_server_host, port=tf_server_port, auto_connect=True)
        
        # 缓存
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_depth: Optional[np.ndarray] = None
        self._cached_bricks: List[Dict] = []
        self._last_segment_time: float = 0
        self._segment_pending: bool = False
        self._current_tf: Optional[dict] = None
        
        print(f"\n[BrickAnomalyDetector] 初始化完成")
        print("=" * 60)
        print("按键说明:")
        print("  s - 触发 SAM3 分割，分析所有砖块")
        print("  t - 打印 TF 变换详情")
        print("  c - 清除分割结果")
        print("  q - 退出")
        print("=" * 60)
    
    def _draw_brick_info(self, frame: np.ndarray, bricks: List[Dict]) -> np.ndarray:
        """在画面上绘制砖块信息"""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        for brick in bricks:
            color = COLORS[(brick['id'] - 1) % len(COLORS)]
            px, py = brick['pixel_center']
            
            # 根据堆叠状态调整颜色
            if brick['stacking_type'] == 'stacked_upper':
                color = (0, 0, 255)  # 红色 - 上层砖块（需要优先处理）
            elif brick['stacking_type'] == 'stacked_lower':
                color = (128, 128, 128)  # 灰色 - 被压住的砖块
            
            # 绘制掩码轮廓
            mask = brick['mask']
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
            
            # 绘制中心点
            cv2.circle(result, (px, py), 6, color, -1)
            cv2.circle(result, (px, py), 8, (255, 255, 255), 2)
            
            # 绘制编号和信息
            z_m = brick['z_base_m']
            area = brick['area_cm2']
            
            # 第一行：编号和 Z 高度
            label1 = f"#{brick['id']} Z:{z_m:.3f}m"
            cv2.putText(result, label1, (px + 12, py - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(result, label1, (px + 12, py - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 第二行：面积
            label2 = f"Area:{area:.1f}cm2"
            cv2.putText(result, label2, (px + 12, py + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(result, label2, (px + 12, py + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # 如果是堆叠的上层，显示标记
            if brick['stacking_type'] == 'stacked_upper':
                cv2.putText(result, "[STACKED]", (px + 12, py + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        return result
    
    def _draw_summary(self, frame: np.ndarray, bricks: List[Dict]) -> np.ndarray:
        """在画面底部绘制汇总信息"""
        result = frame.copy()
        height = frame.shape[0]
        
        if not bricks:
            return result
        
        # 绘制半透明背景
        overlay = result.copy()
        cv2.rectangle(overlay, (0, height - 120), (450, height), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)
        
        # 汇总信息
        y_offset = height - 100
        cv2.putText(result, f"Detected: {len(bricks)} bricks", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示每个砖块的 Z 和面积
        y_offset += 25
        for brick in bricks:
            stacking_mark = " [S]" if brick['stacking_type'] == 'stacked_upper' else ""
            text = f"#{brick['id']}: Z={brick['z_base_m']:.3f}m, Area={brick['area_cm2']:.1f}cm2{stacking_mark}"
            
            color = (0, 255, 0)
            if brick['stacking_type'] == 'stacked_upper':
                color = (0, 0, 255)
            elif brick['stacking_type'] == 'stacked_lower':
                color = (128, 128, 128)
            
            cv2.putText(result, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 20
        
        return result
    
    def _print_brick_analysis(self, bricks: List[Dict]):
        """打印详细的砖块分析结果"""
        print("\n" + "=" * 80)
        print(f"[砖块分析结果] 检测到 {len(bricks)} 个砖块")
        print("-" * 80)
        print(f"{'#':<3} {'Z(m)':>8} {'面积(cm²)':>10} {'像素':>8} {'深度范围(m)':>15} {'状态':<15}")
        print("-" * 80)
        
        for brick in bricks:
            stacking = ""
            if brick['stacking_type'] == 'stacked_upper':
                stacking = f"[叠在#{brick['is_stacked_on']}上]"
            elif brick['stacking_type'] == 'stacked_lower':
                stacking = "[被压住]"
            
            depth_range = f"{brick['depth_min_m']:.3f}-{brick['depth_max_m']:.3f}"
            
            print(f"{brick['id']:<3} {brick['z_base_m']:>8.4f} {brick['area_cm2']:>10.1f} "
                  f"{brick['area_pixels']:>8} {depth_range:>15} {stacking:<15}")
        
        print("-" * 80)
        
        # 检测堆叠异常
        stacked = [b for b in bricks if b['stacking_type'] == 'stacked_upper']
        if stacked:
            print(f"\n[!] 检测到 {len(stacked)} 个堆叠异常:")
            for b in stacked:
                print(f"    砖块 #{b['id']} 叠在 #{b['is_stacked_on']} 上方 "
                      f"(Z差: {b['z_base_m'] - next(x['z_base_m'] for x in bricks if x['id'] == b['is_stacked_on']):.3f}m)")
        else:
            print("\n[✓] 未检测到堆叠异常")
        
        print("=" * 80 + "\n")
    
    def _trigger_segment(self, frame: np.ndarray, depth: np.ndarray):
        """触发分割和分析"""
        if self._segment_pending:
            print("[SAM3] 分割正在进行中，请稍候...")
            return
        
        self._segment_pending = True
        
        print(f"\n[SAM3] 正在分割，提示词: '{self.text_prompt}'...")
        t0 = time.time()
        
        masks = self.segmenter.segment(frame, self.text_prompt)
        
        if masks is None or len(masks) == 0:
            print(f"[SAM3] 未检测到目标 ({time.time()-t0:.2f}s)")
            self._cached_bricks = []
        else:
            print(f"[SAM3] 检测到 {len(masks)} 个目标 ({time.time()-t0:.2f}s)")
            
            # 获取 TF 矩阵
            tf_matrix = None
            if self._current_tf is not None:
                tf_matrix = self._current_tf['matrix']
            
            # 分析砖块
            bricks = self.analyzer.analyze(masks, depth, tf_matrix)
            
            # 检测堆叠
            bricks = self.analyzer.detect_stacking(bricks)
            
            self._cached_bricks = bricks
            
            # 打印分析结果
            self._print_brick_analysis(bricks)
        
        self._cached_frame = frame.copy()
        self._cached_depth = depth.copy()
        self._last_segment_time = time.time()
        self._segment_pending = False
    
    def _get_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """获取显示帧"""
        current_time = time.time()
        
        # 检查是否显示分割结果
        show_results = (
            self._cached_bricks and
            self._cached_frame is not None and
            (current_time - self._last_segment_time) < self.display_duration
        )
        
        if show_results:
            display = self._draw_brick_info(self._cached_frame, self._cached_bricks)
            display = self._draw_summary(display, self._cached_bricks)
            
            remaining = self.display_duration - (current_time - self._last_segment_time)
            cv2.putText(display, f"Analysis result ({remaining:.1f}s)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display = frame.copy()
            cv2.putText(display, "Live View - Press 's' to analyze bricks", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(display, f"Prompt: '{self.text_prompt}'", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 添加 TF 状态
        if self._current_tf is not None:
            cv2.putText(display, "TF: OK", 
                       (display.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "TF: N/A", 
                       (display.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return display
    
    def _print_tf_details(self):
        """打印 TF 变换详情"""
        if self._current_tf is None:
            print("\n[TF] 未连接到 TF 服务器")
            return
        
        t = self._current_tf['translation']
        rpy_deg = self._current_tf['rpy_deg']
        
        print("\n" + "=" * 60)
        print(f"TF: base_link <- head_camera_link")
        print("-" * 60)
        print(f"Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
        print(f"RPY (deg):   [{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}]")
        print("=" * 60 + "\n")
    
    def run(self):
        """主循环"""
        print("\n开始运行，等待相机图像...\n")
        
        window_name = "Brick Anomaly Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        for img_head, img_depth, _, _ in self.robot.camera:
            if img_head is None or img_head.size == 0:
                continue
            
            if img_depth is None or img_depth.size == 0:
                continue
            
            # 更新 TF
            self._current_tf = self.tf_client.get_transform('base_link', 'head_camera_link')
            
            # 获取显示帧
            display = self._get_display_frame(img_head)
            cv2.imshow(window_name, display)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self._trigger_segment(img_head, img_depth)
            
            elif key == ord('t'):
                self._print_tf_details()
            
            elif key == ord('c'):
                self._cached_bricks = []
                self._cached_frame = None
                print("[清除分割结果]")
            
            elif key == ord('q') or key == 27:
                print("\n[退出]")
                break
        
        cv2.destroyAllWindows()
        self.tf_client.disconnect()
        print("程序结束")


def main():
    parser = argparse.ArgumentParser(description='砖块异常检测测试')
    parser.add_argument("--ip", type=str, default="192.168.11.200")
    parser.add_argument("--prompt", type=str, default="block, brick, rectangular object")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/ypf/sam3-main/checkpoint/sam3.pt")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--tf-host", type=str, default="127.0.0.1")
    parser.add_argument("--tf-port", type=int, default=9999)
    args = parser.parse_args()
    
    detector = BrickAnomalyDetector(
        ip=args.ip,
        checkpoint_path=args.checkpoint,
        text_prompt=args.prompt,
        display_duration=args.duration,
        tf_server_host=args.tf_host,
        tf_server_port=args.tf_port
    )
    detector.run()


if __name__ == '__main__':
    main()