"""
真实相机 + SAM3 分割测试脚本
- 实时显示 D435i 头部相机 RGB 图像
- 按 's' 键触发 SAM3 分割
- 按 't' 键打印 TF 变换详情
- 按 'q' 键退出
"""

import sys
SAM3_PROJECT_PATH = "/home/ypf/sam3-main"
if SAM3_PROJECT_PATH not in sys.path:
    sys.path.insert(0, SAM3_PROJECT_PATH)

import cv2
import os
import argparse
import time
import numpy as np
from PIL import Image
import torch
import gc
import threading
from typing import Optional, List, Tuple

from airbot_sdk import MMK2RealRobot
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tf_client import TFClient


# 可视化颜色
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
]


class SAM3Segmenter:
    """SAM3 分割器 - 单例模式"""
    _instance = None
    _model = None
    _processor = None
    
    def __init__(self,
                 checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
                 text_prompt: str = "red brick",
                 sam_resolution: int = 1008,
                 confidence_threshold: float = 0.5):
        
        self.text_prompt = text_prompt
        self.sam_resolution = sam_resolution
        self.confidence_threshold = confidence_threshold
        
        # 单例模式：避免重复加载模型
        if SAM3Segmenter._model is None:
            print(f"[SAM3] 加载模型中...")
            print(f"  - 模型路径: {checkpoint_path}")
            print(f"  - 提示词: {text_prompt}")
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
            
            print(f"[SAM3] 正在分割，提示词: '{prompt}'...")
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            masks = output["masks"].cpu().numpy() if output["masks"] is not None else None
            
            torch.cuda.empty_cache()
            
            if masks is not None:
                print(f"[SAM3] 检测到 {len(masks)} 个目标")
            else:
                print(f"[SAM3] 未检测到目标")
            
            return masks


class RealCameraSAM3:
    """真实相机 + SAM3 分割系统"""
    
    def __init__(self,
                 ip: str = "192.168.11.200",
                 checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
                 text_prompt: str = "red brick",
                 display_duration: float = 5.0,
                 tf_server_host: str = "127.0.0.1",
                 tf_server_port: int = 9999):
        """初始化"""
        self.ip = ip
        self.text_prompt = text_prompt
        self.display_duration = display_duration
        
        # 初始化机器人
        print(f"[Camera] 连接机器人 {ip}...")
        self.robot = MMK2RealRobot(ip=ip)
        self.robot.set_robot_head_pose(0, -1.08)
        
        # 初始化 SAM3 分割器
        self.segmenter = SAM3Segmenter(
            checkpoint_path=checkpoint_path,
            text_prompt=text_prompt
        )
        
        # 初始化 TF 客户端
        print(f"[TF] 连接 TF 服务器 {tf_server_host}:{tf_server_port}...")
        self.tf_client = TFClient(host=tf_server_host, port=tf_server_port, auto_connect=True)
        
        # 缓存
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_masks: Optional[np.ndarray] = None
        self._last_segment_time: float = 0
        self._segment_pending: bool = False
        self._current_tf: Optional[dict] = None
        
        print(f"[Camera] 初始化完成")
        print(f"  - 按 's' 触发 SAM3 分割")
        print(f"  - 按 't' 打印 TF 变换详情")
        print(f"  - 按 'c' 清除分割结果")
        print(f"  - 按 'q' 或 ESC 退出")
    
    def _draw_segmentation(self, frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """绘制分割结果"""
        if masks is None or len(masks) == 0:
            return frame
        
        result = frame.copy()
        height, width = frame.shape[:2]
        
        for i, mask in enumerate(masks):
            if len(mask.shape) == 3:
                mask = mask[0]
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height),
                                 interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            color = COLORS[i % len(COLORS)]
            
            overlay = result.copy()
            overlay[mask_bool] = color
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            contours, _ = cv2.findContours(
                (mask_bool * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
            
            ys, xs = np.where(mask_bool)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.circle(result, (cx, cy), 5, color, -1)
                cv2.circle(result, (cx, cy), 8, (255, 255, 255), 2)
                cv2.putText(result, f"#{i+1}", (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def _draw_tf_info(self, frame: np.ndarray) -> np.ndarray:
        """在画面上绘制 TF 信息"""
        result = frame.copy()
        height = frame.shape[0]
        
        if self._current_tf is None:
            cv2.putText(result, "TF: Not connected", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return result
        
        t = self._current_tf['translation']
        rpy = self._current_tf['rpy_deg']  # Roll, Pitch, Yaw (角度)
        
        # 显示位移
        tf_text = f"T: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
        cv2.putText(result, tf_text, 
                   (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        # 显示 RPY 角度
        rpy_text = f"RPY: [{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg"
        cv2.putText(result, rpy_text, 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        return result
    
    def _get_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """获取显示帧"""
        current_time = time.time()
        
        show_segmentation = (
            self._cached_masks is not None and
            self._cached_frame is not None and
            (current_time - self._last_segment_time) < self.display_duration
        )
        
        if show_segmentation:
            display = self._draw_segmentation(self._cached_frame, self._cached_masks)
            num_detected = len(self._cached_masks) if self._cached_masks is not None else 0
            remaining = self.display_duration - (current_time - self._last_segment_time)
            cv2.putText(display, f"SAM3: {num_detected} detected ({remaining:.1f}s)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Prompt: '{self.text_prompt}'", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            display = frame.copy()
            cv2.putText(display, "Live View - Press 's' to segment, 't' for TF", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(display, f"Prompt: '{self.text_prompt}'", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 添加 TF 信息
        display = self._draw_tf_info(display)
        
        return display
    
    def _trigger_segment(self, frame: np.ndarray):
        """触发分割"""
        if self._segment_pending:
            print("[SAM3] 分割正在进行中，请稍候...")
            return
        
        self._segment_pending = True
        masks = self.segmenter.segment(frame, self.text_prompt)
        
        self._cached_frame = frame.copy()
        self._cached_masks = masks
        self._last_segment_time = time.time()
        self._segment_pending = False
    
    def _print_tf_details(self):
        """打印 TF 变换详情"""
        if self._current_tf is None:
            print("\n[TF] 未连接到 TF 服务器")
            return
        
        t = self._current_tf['translation']
        r = self._current_tf['rotation']
        rpy_rad = self._current_tf['rpy_rad']
        rpy_deg = self._current_tf['rpy_deg']
        matrix = self._current_tf['matrix']
        
        print("\n" + "=" * 60)
        print(f"TF: {self._current_tf['target_frame']} <- {self._current_tf['source_frame']}")
        print("=" * 60)
        print(f"Translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        print(f"Quaternion:  [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f}]")
        print(f"RPY (rad):   [{rpy_rad[0]:.6f}, {rpy_rad[1]:.6f}, {rpy_rad[2]:.6f}]")
        print(f"RPY (deg):   [{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}]")
        print(f"Matrix:\n{matrix}")
        print("=" * 60 + "\n")
    
    def run(self):
        """主循环"""
        print("\n" + "=" * 50)
        print("开始运行，等待相机图像...")
        print("=" * 50 + "\n")
        
        window_name = "SAM3 Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        for img_head, img_depth, img_left, img_right in self.robot.camera:
            if img_head is None or img_head.size == 0:
                print("等待头部相机图像...")
                time.sleep(0.1)
                continue
            
            # 更新 TF 信息
            self._current_tf = self.tf_client.get_transform('base_link', 'head_camera_link')
            
            # 获取显示帧
            display = self._get_display_frame(img_head)
            cv2.imshow(window_name, display)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                print("\n[触发分割]")
                self._trigger_segment(img_head)
            
            elif key == ord('t'):
                self._print_tf_details()
            
            elif key == ord('q') or key == 27:
                print("\n[退出]")
                break
            
            elif key == ord('c'):
                self._cached_masks = None
                self._cached_frame = None
                print("[清除分割结果]")
        
        cv2.destroyAllWindows()
        self.tf_client.disconnect()
        print("程序结束")


def main():
    parser = argparse.ArgumentParser(description='真实相机 + SAM3 分割测试')
    parser.add_argument("--ip", type=str, default="192.168.11.200", 
                        help="机器人 IP 地址")
    parser.add_argument("--prompt", type=str, default="block", 
                        help="SAM3 分割提示词")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/ypf/sam3-main/checkpoint/sam3.pt",
                        help="SAM3 模型路径")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="分割结果显示时长（秒）")
    parser.add_argument("--tf-host", type=str, default="127.0.0.1",
                        help="TF 服务器地址")
    parser.add_argument("--tf-port", type=int, default=9999,
                        help="TF 服务器端口")
    args = parser.parse_args()
    
    app = RealCameraSAM3(
        ip=args.ip,
        checkpoint_path=args.checkpoint,
        text_prompt=args.prompt,
        display_duration=args.duration,
        tf_server_host=args.tf_host,
        tf_server_port=args.tf_port
    )
    app.run()


if __name__ == '__main__':
    main()