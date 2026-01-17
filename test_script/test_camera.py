import cv2, os
import argparse
import datetime
import time

from airbot_sdk import MMK2RealRobot

def show_all_cameras(ip_address, calib_dir):
    """显示所有相机图像"""
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    mmk2 = MMK2RealRobot(ip=ip_address)
    mmk2.set_robot_head_pose(0, -1.08)
    
    print("等待相机初始化...")
    time.sleep(2)

    for img_head, img_depth, img_left, img_right in mmk2.camera:
        
        # 显示头部RGB相机
        if img_head is not None and img_head.size > 0:
            cv2.imshow("Head Camera (RGB)", img_head)
        else:
            print("等待头部相机图像...")
        
        # 显示头部深度相机
        if img_depth is not None and img_depth.size > 0:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(img_depth, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            cv2.imshow("Head Camera (Depth)", depth_colormap)
        
        # 显示左手眼相机
        if img_left is not None and img_left.size > 0:
            cv2.imshow("Left Hand Camera", img_left)
        else:
            print("等待左手眼相机图像...")
        
        # 显示右手眼相机
        if img_right is not None and img_right.size > 0:
            cv2.imshow("Right Hand Camera", img_right)
        else:
            print("等待右手眼相机图像...")

        key = cv2.waitKey(1) & 0xFF

        # 按's'保存所有相机截图
        if key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            if img_head is not None:
                cv2.imwrite(f"{calib_dir}/{timestamp}_head_rgb.jpg", img_head)
                print(f"已保存头部RGB：{calib_dir}/{timestamp}_head_rgb.jpg")
            if img_depth is not None:
                cv2.imwrite(f"{calib_dir}/{timestamp}_head_depth.png", img_depth)
                print(f"已保存头部深度：{calib_dir}/{timestamp}_head_depth.png")
            if img_left is not None:
                cv2.imwrite(f"{calib_dir}/{timestamp}_left.jpg", img_left)
                print(f"已保存左手眼：{calib_dir}/{timestamp}_left.jpg")
            if img_right is not None:
                cv2.imwrite(f"{calib_dir}/{timestamp}_right.jpg", img_right)
                print(f"已保存右手眼：{calib_dir}/{timestamp}_right.jpg")

        # 按q或ESC退出
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='显示所有相机图像')
    parser.add_argument("--ip", type=str, default="192.168.11.200", help="mmk2 ip address")
    parser.add_argument("--dir", type=str, default="./calib", help="image store dir")
    args = parser.parse_args()

    show_all_cameras(args.ip, args.dir)