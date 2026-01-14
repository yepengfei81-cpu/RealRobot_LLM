#!/usr/bin/env python3
"""
TF 客户端 - 在本地 conda 环境中运行
从 Docker 中的 TF 服务器获取变换数据
"""

import socket
import json
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional
import threading


class TFClient:
    """TF 变换客户端"""
    
    def __init__(self, host='127.0.0.1', port=9999, auto_connect=True):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self._lock = threading.Lock()
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """连接到 TF 服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[TFClient] 已连接到 TF 服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[TFClient] 连接失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
    
    def get_transform(self, 
                      target_frame='base_link', 
                      source_frame='head_camera_link') -> Optional[dict]:
        """获取 TF 变换"""
        if not self.connected:
            if not self.connect():
                return None
        
        with self._lock:
            try:
                request = json.dumps({
                    'target_frame': target_frame,
                    'source_frame': source_frame
                }) + '\n'
                self.socket.send(request.encode('utf-8'))
                
                response = self.socket.recv(4096).decode('utf-8').strip()
                data = json.loads(response)
                
                if not data.get('success', False):
                    return None
                
                translation = np.array(data['translation'])
                quaternion = np.array(data['rotation'])  # [x, y, z, w]
                
                # 计算旋转矩阵和 RPY
                rotation = Rotation.from_quat(quaternion)
                rotation_matrix = rotation.as_matrix()
                rpy_rad = rotation.as_euler('xyz')  # Roll, Pitch, Yaw (弧度)
                rpy_deg = np.degrees(rpy_rad)       # 转换为角度
                
                # 构建 4x4 变换矩阵
                matrix = np.eye(4)
                matrix[:3, :3] = rotation_matrix
                matrix[:3, 3] = translation
                
                return {
                    'translation': translation,
                    'rotation': quaternion,
                    'rpy_rad': rpy_rad,
                    'rpy_deg': rpy_deg,
                    'matrix': matrix,
                    'target_frame': target_frame,
                    'source_frame': source_frame
                }
                
            except Exception as e:
                print(f"[TFClient] 获取 TF 失败: {e}")
                self.connected = False
                return None
    
    def get_camera_to_base_matrix(self) -> Optional[np.ndarray]:
        """获取相机到 base_link 的 4x4 变换矩阵"""
        tf = self.get_transform('base_link', 'head_camera_link')
        return tf['matrix'] if tf else None
    
    def transform_point_to_base(self, point_camera: np.ndarray) -> Optional[np.ndarray]:
        """将相机坐标系下的点转换到 base_link 坐标系"""
        matrix = self.get_camera_to_base_matrix()
        if matrix is None:
            return None
        point_homogeneous = np.array([*point_camera, 1.0])
        return (matrix @ point_homogeneous)[:3]