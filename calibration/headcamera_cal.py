"""
头部相机外参校正工具
利用棋盘格在已知位置，计算外参偏移补偿

原理：
1. 头部相机检测棋盘格中心位置（检测值）
2. 用机械臂末端触碰棋盘格中心（真实值 = 末端位置）
3. 计算偏移 = 真实值 - 检测值
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from airbot_sdk import MMK2RealRobot
from tf_client import TFClient
from mmk2_types.types import MMK2Components
from mmk2_types.grpc_msgs import Pose, Position, Orientation

# 头部相机内参（RealSense 自带）
HEAD_INTRINSICS = {'fx': 607.15, 'fy': 607.02, 'cx': 324.25, 'cy': 248.46}

# 棋盘格参数
CHECKERBOARD = {
    'rows': 6,
    'cols': 9,
    'square_size': 0.022  # 22mm
}

CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")

# 增量控制参数
DELTA_POS = 0.005     # 平移增量：5mm（精细调整）
DELTA_ROT = 2.0       # 旋转增量：2度


class ArmController:
    """机械臂增量控制器"""
    
    def __init__(self, robot: MMK2RealRobot, use_left_arm: bool = True):
        self.robot = robot
        self.use_left_arm = use_left_arm
        self.arm_component = MMK2Components.LEFT_ARM if use_left_arm else MMK2Components.RIGHT_ARM
        self.arm_key = 'left_arm' if use_left_arm else 'right_arm'
        self.arm_name = "左臂" if use_left_arm else "右臂"
        
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_euler = np.array([0.0, 0.0, 0.0])
        
        self._update_current_pose()
    
    def _update_current_pose(self) -> bool:
        try:
            poses = self.robot.get_arm_end_poses()
            if poses is None or self.arm_key not in poses:
                return False
            
            pose = poses[self.arm_key]
            self.current_pos = np.array(pose['position'])
            quat = pose['orientation']
            self.current_euler = R.from_quat(quat).as_euler('xyz', degrees=True)
            return True
        except Exception as e:
            print(f"[错误] 读取位姿失败: {e}")
            return False
    
    def get_current_position(self) -> np.ndarray:
        """获取当前末端位置"""
        self._update_current_pose()
        return self.current_pos.copy()
    
    def move_delta(self, dx=0, dy=0, dz=0) -> bool:
        """增量移动（只移动位置，保持姿态不变）"""
        if not self._update_current_pose():
            return False
        
        new_pos = self.current_pos + np.array([dx, dy, dz])
        new_quat = R.from_euler('xyz', self.current_euler, degrees=True).as_quat()
        
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
            return True
        except Exception as e:
            print(f"[错误] 移动失败: {e}")
            return False
    
    def move_to_position_down(self, position: np.ndarray) -> bool:
        """移动到指定位置，末端朝下"""
        # X轴朝下的姿态：绕Y轴旋转90度
        r_down = R.from_euler('y', np.pi / 2)
        down_quat = r_down.as_quat()
        
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
            print(f"[错误] 移动失败: {e}")
            return False


class HeadCameraExtrinsicsCalibrator:
    """头部相机外参校正器"""
    
    def __init__(self, robot: MMK2RealRobot, tf_client: TFClient, use_left_arm: bool = True):
        self.robot = robot
        self.tf_client = tf_client
        self.arm_controller = ArmController(robot, use_left_arm)
        
        self.fx, self.fy = HEAD_INTRINSICS['fx'], HEAD_INTRINSICS['fy']
        self.cx, self.cy = HEAD_INTRINSICS['cx'], HEAD_INTRINSICS['cy']
        
        # 棋盘格 3D 点
        self.rows = CHECKERBOARD['rows']
        self.cols = CHECKERBOARD['cols']
        self.square_size = CHECKERBOARD['square_size']
        
        # 棋盘格中心相对于原点的偏移
        self.center_offset_x = (self.cols - 1) * self.square_size / 2
        self.center_offset_y = (self.rows - 1) * self.square_size / 2
        
        # 存储采样数据
        self.samples = []
        
        # 当前检测结果缓存
        self.current_detected_pos = None
        self.current_board_found = False
    
    def detect_board_center(self, rgb: np.ndarray, depth: np.ndarray) -> tuple:
        """
        检测棋盘格中心在 base_link 下的位置
        返回: (found, center_pos_base, vis_image)
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        vis = rgb.copy()
        
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), flags)
        
        if not ret:
            return False, None, vis
        
        # 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis, (self.cols, self.rows), corners, ret)
        
        # 计算棋盘格的几何中心（所有角点的平均值）
        # 这才是真正的棋盘格中心
        all_corners = corners.reshape(-1, 2)
        center_px = np.mean(all_corners, axis=0)
        px, py = int(center_px[0]), int(center_px[1])
        
        # 在图像上标记中心
        cv2.circle(vis, (px, py), 10, (0, 0, 255), -1)
        cv2.putText(vis, "CENTER", (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 从深度图获取深度
        h, w = depth.shape
        px_clamped = max(0, min(px, w-1))
        py_clamped = max(0, min(py, h-1))
        
        # 取中心区域的中值深度
        r = 5
        y1, y2 = max(0, py_clamped-r), min(h, py_clamped+r)
        x1, x2 = max(0, px_clamped-r), min(w, px_clamped+r)
        depth_region = depth[y1:y2, x1:x2]
        valid_depth = depth_region[depth_region > 0]
        
        if len(valid_depth) == 0:
            cv2.putText(vis, "No depth!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return False, None, vis
        
        z = np.median(valid_depth) / 1000.0  # mm -> m
        
        # 相机坐标系下的 3D 位置
        pos_cam = np.array([
            (px - self.cx) * z / self.fx,
            (py - self.cy) * z / self.fy,
            z
        ])
        
        # 获取 TF 变换
        tf_data = self.tf_client.get_transform('base_link', 'head_camera_link')
        if tf_data is None:
            cv2.putText(vis, "No TF!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return False, None, vis
        
        tf_matrix = tf_data['matrix']
        pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3]
        
        # 在图像上显示检测结果
        cv2.putText(vis, f"Detected (base): [{pos_base[0]:.4f}, {pos_base[1]:.4f}, {pos_base[2]:.4f}]",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"Depth: {z:.3f}m", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return True, pos_base, vis
    
    def add_sample(self, detected_pos: np.ndarray, ground_truth_pos: np.ndarray):
        """添加一组采样数据"""
        diff = ground_truth_pos - detected_pos
        self.samples.append({
            'detected': detected_pos.copy(),
            'ground_truth': ground_truth_pos.copy(),
            'diff': diff.copy()
        })
        
        print(f"\n[采样 #{len(self.samples)}]")
        print(f"  检测值 (头部相机): [{detected_pos[0]:.4f}, {detected_pos[1]:.4f}, {detected_pos[2]:.4f}]")
        print(f"  真实值 (机械臂):   [{ground_truth_pos[0]:.4f}, {ground_truth_pos[1]:.4f}, {ground_truth_pos[2]:.4f}]")
        print(f"  偏差:              [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}]")
    
    def calculate_offset(self) -> dict:
        """计算平均偏移量"""
        if len(self.samples) < 1:
            print("[错误] 需要至少 1 个采样点")
            return None
        
        diffs = np.array([s['diff'] for s in self.samples])
        mean_offset = np.mean(diffs, axis=0)
        std_offset = np.std(diffs, axis=0)
        
        result = {
            'offset_xyz': mean_offset.tolist(),
            'std_xyz': std_offset.tolist(),
            'num_samples': len(self.samples),
            'samples': [
                {
                    'detected': s['detected'].tolist(),
                    'ground_truth': s['ground_truth'].tolist(),
                    'diff': s['diff'].tolist()
                }
                for s in self.samples
            ]
        }
        
        print("\n" + "=" * 60)
        print("头部相机外参偏移计算结果")
        print("=" * 60)
        print(f"  平均偏移: X={mean_offset[0]:+.4f}, Y={mean_offset[1]:+.4f}, Z={mean_offset[2]:+.4f} m")
        print(f"  标准差:   X={std_offset[0]:.4f}, Y={std_offset[1]:.4f}, Z={std_offset[2]:.4f} m")
        print(f"  采样数:   {len(self.samples)}")
        print("=" * 60)
        
        return result
    
    def save_offset(self, result: dict):
        """保存偏移量"""
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIB_DIR / "head_camera_offset.json"
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[保存] 头部相机偏移已保存到 {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.11.200")
    parser.add_argument("--tf-host", default="127.0.0.1")
    parser.add_argument("--tf-port", type=int, default=9999)
    parser.add_argument("--right-arm", action="store_true", help="使用右臂")
    args = parser.parse_args()
    
    use_left = not args.right_arm
    arm_name = "左臂" if use_left else "右臂"
    
    print("=" * 70)
    print("头部相机外参校正工具")
    print("=" * 70)
    print(f"使用 {arm_name} 进行校正")
    print()
    print("校正原理:")
    print("  1. 将棋盘格放在桌面固定位置")
    print("  2. 头部相机检测棋盘格中心位置 (检测值)")
    print("  3. 用机械臂末端触碰棋盘格中心 (真实值)")
    print("  4. 计算偏移 = 真实值 - 检测值")
    print()
    print("-" * 70)
    print("机械臂控制 (步进: 5mm):")
    print("  I/K : X 轴 前/后")
    print("  J/L : Y 轴 左/右")
    print("  U/O : Z 轴 上/下")
    print("-" * 70)
    print("功能键:")
    print("  Space - 锁定当前检测结果")
    print("  Enter - 添加采样 (锁定后)")
    print("  c     - 计算偏移")
    print("  s     - 保存结果")
    print("  r     - 重置采样数据")
    print("  p     - 打印当前末端位置")
    print("  x     - 解锁")
    print("  q     - 退出")
    print("=" * 70)
    
    # 初始化
    print("\n[初始化] 连接机器人...")
    robot = MMK2RealRobot(ip=args.ip)
    robot.set_robot_head_pose(0, -1.08)  # 头部朝下看
    robot.set_spine(0.08)
    
    print("[初始化] 连接 TF 服务...")
    tf_client = TFClient(host=args.tf_host, port=args.tf_port, auto_connect=True)
    
    calibrator = HeadCameraExtrinsicsCalibrator(robot, tf_client, use_left)
    arm = calibrator.arm_controller
    
    cv2.namedWindow("Head Camera Calibration", cv2.WINDOW_NORMAL)
    
    locked_detected_pos = None  # 锁定的检测位置
    current_result = None
    
    for rgb, depth, _, _ in robot.camera:
        if rgb is None or depth is None:
            continue
        
        # 检测棋盘格
        found, detected_pos, vis = calibrator.detect_board_center(rgb, depth)
        
        # 获取当前末端位置
        arm_pos = arm.get_current_position()
        
        # 显示状态
        status_y = 80
        if found:
            cv2.putText(vis, "Chessboard: DETECTED", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "Chessboard: NOT DETECTED", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        status_y += 25
        cv2.putText(vis, f"Arm ({arm_name}): [{arm_pos[0]:.4f}, {arm_pos[1]:.4f}, {arm_pos[2]:.4f}]",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        status_y += 25
        cv2.putText(vis, f"Samples: {len(calibrator.samples)}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示锁定的检测位置
        if locked_detected_pos is not None:
            status_y += 25
            cv2.putText(vis, f"LOCKED: [{locked_detected_pos[0]:.4f}, {locked_detected_pos[1]:.4f}, {locked_detected_pos[2]:.4f}]",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示控制说明
        cv2.putText(vis, "Arm: I/K(X) J/L(Y) U/O(Z) | Space:lock Enter:add c:calc s:save q:quit",
                   (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Head Camera Calibration", vis)
        
        k = cv2.waitKey(1) & 0xFF
        
        # 机械臂控制 (I/K/J/L/U/O)
        if k == ord('i'):
            arm.move_delta(dx=DELTA_POS)
        elif k == ord('k'):
            arm.move_delta(dx=-DELTA_POS)
        elif k == ord('j'):
            arm.move_delta(dy=DELTA_POS)
        elif k == ord('l'):
            arm.move_delta(dy=-DELTA_POS)
        elif k == ord('u'):
            arm.move_delta(dz=DELTA_POS)
        elif k == ord('o'):
            arm.move_delta(dz=-DELTA_POS)
        
        # 功能键
        elif k == ord(' '):  # 空格键锁定
            if found:
                locked_detected_pos = detected_pos.copy()
                print(f"\n[锁定] 检测位置: [{locked_detected_pos[0]:.4f}, {locked_detected_pos[1]:.4f}, {locked_detected_pos[2]:.4f}]")
                print("  现在用 I/K/J/L/U/O 控制机械臂触碰棋盘格中心，然后按 Enter 添加采样")
            else:
                print("[错误] 未检测到棋盘格，无法锁定")
        
        elif k == 13:  # Enter 键添加采样
            if locked_detected_pos is not None:
                ground_truth = arm.get_current_position()
                calibrator.add_sample(locked_detected_pos, ground_truth)
                locked_detected_pos = None  # 解锁，准备下一次采样
            else:
                print("[错误] 请先按空格键锁定检测结果")
        
        elif k == ord('c'):
            current_result = calibrator.calculate_offset()
        
        elif k == ord('s'):
            if current_result:
                calibrator.save_offset(current_result)
            else:
                print("[错误] 请先按 'c' 计算偏移")
        
        elif k == ord('r'):
            calibrator.samples = []
            locked_detected_pos = None
            current_result = None
            print("[重置] 已清除所有采样数据")
        
        elif k == ord('p'):
            pos = arm.get_current_position()
            euler = arm.current_euler
            print(f"\n[{arm_name}] 位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m")
            print(f"         欧拉角: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")
        
        elif k == ord('x'):
            locked_detected_pos = None
            print("[解锁] 已解除锁定")
        
        elif k in (ord('q'), 27):
            break
    
    cv2.destroyAllWindows()
    tf_client.disconnect()


if __name__ == '__main__':
    main()