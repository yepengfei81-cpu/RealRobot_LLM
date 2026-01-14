#!/usr/bin/env python3
"""
TF 数据服务器 - 在 Docker 中运行
通过 socket 将 TF 变换数据发送给本地程序
"""

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
import socket
import json
import threading
import time


class TFServer(Node):
    def __init__(self, host='0.0.0.0', port=9999):
        super().__init__('tf_server')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Socket 服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.get_logger().info(f'TF Server listening on {host}:{port}')
        
        # 启动服务器线程
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
    
    def get_transform(self, target_frame='base_link', source_frame='head_camera_link'):
        """获取 TF 变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            
            t = transform.transform.translation
            r = transform.transform.rotation
            
            return {
                'success': True,
                'translation': [t.x, t.y, t.z],
                'rotation': [r.x, r.y, r.z, r.w],
                'target_frame': target_frame,
                'source_frame': source_frame,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _server_loop(self):
        """处理客户端连接"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                self.get_logger().info(f'Client connected: {addr}')
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Server error: {e}')
    
    def _handle_client(self, client_socket):
        """处理单个客户端请求"""
        try:
            while self.running:
                data = client_socket.recv(1024).decode('utf-8').strip()
                if not data:
                    break
                
                try:
                    request = json.loads(data)
                    target = request.get('target_frame', 'base_link')
                    source = request.get('source_frame', 'head_camera_link')
                except:
                    target = 'base_link'
                    source = 'head_camera_link'
                
                result = self.get_transform(target, source)
                response = json.dumps(result) + '\n'
                client_socket.send(response.encode('utf-8'))
                
        except Exception as e:
            self.get_logger().error(f'Client error: {e}')
        finally:
            client_socket.close()
    
    def shutdown(self):
        self.running = False
        self.server_socket.close()


def main():
    rclpy.init()
    node = TFServer(host='0.0.0.0', port=9999)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()