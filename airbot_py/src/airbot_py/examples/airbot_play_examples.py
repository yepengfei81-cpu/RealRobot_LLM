from __future__ import annotations

import argparse
import logging
import time
from enum import Enum

import numpy as np
from pynput import keyboard
from transforms3d.euler import euler2quat, quat2euler, euler2mat
from transforms3d.quaternions import quat2mat, mat2quat


from airbot_py.airbot_grpc import airbot_play_pb2
from airbot_py.airbot_play import AirbotPlay

logging.basicConfig(level=logging.INFO)


class ServeMode(Enum):
    JOINT = 1
    TWIST = 2
    POSE = 3


class Transform(Enum):
    X = 1
    Y = 2
    Z = 3
    ROLL = 4
    PITCH = 5
    YAW = 6


SPEDD_GEAR: tuple[float, float, float] = (0.1, 0.2, 0.5)


none_eef_pose1 = [
    [0.33802613615989685, 0.004722504410892725, 0.18526741862297058],
    [
        0.5869339108467102,
        0.29231885075569153,
        -0.2734294533729553,
        0.7037716507911682,
    ],
]
none_eef_pose2 = [
    [0.2930537164211273, -0.005615944042801857, 0.3310807943344116],
    [
        0.629936695098877,
        0.14865931868553162,
        -0.17927676439285278,
        0.7409048676490784,
    ],
]

with_eef_pose1 = [
    [0.38957375288009644, -0.013864623382687569, 0.12052314728498459],
    [0.5926767587661743, 0.36419424414634705, -0.3897015452384949, 0.6035143136978149],
]
with_eef_pose2 = [
    [0.30854591727256775, -0.03529889136552811, 0.048451367765665054],
    [0.5361365675926208, 0.391762375831604, -0.4440719783306122, 0.6015645265579224],
]

joint_q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
joint_q1 = [0.5, 0.0, 0.0, 0.0, -1.0, -1.0]
joint_q2 = [0.7, 0.0, 0.0, 0.5, -1.57, -1.57]
joint_q3 = [1.0, 0.0, 0.0, 0.9, 0.0, 0.0]
joint_home = [0.0, -1.1615, 1.1617, 1.5863, -1.2401, 0.0]


class KeyboardControl:
    """
    A class for controlling the system using keyboard events.

    This class sets up a keyboard listener to capture key press and release events,
    allowing for custom handling of these events to control the system behavior.
    """

    PLANNING_FRAME_ID = "base_link"
    EE_FRAME_ID = "flange"

    def __init__(self, port: int = 50051, robot=None):
        self.port = port
        if robot is None:
            logging.info("Creating new robot object")
            self.robot = AirbotPlay(port=port)
        else:
            logging.info("Using provided robot object")
            self.robot = robot
        self.command_frame_id = None
        self.gripper_joint: bool = True
        self.serve_mode = None
        self.vel_cmd_: float = SPEDD_GEAR[0]
        self.speed_gear_index = 0
        self.vel_up_limit: float = 0.8
        self.listener = None
        self.ctrl_pressed = False

    def on_press(self, key: keyboard.KeyCode):
        """
        Handles the key press event.

        Adjusts the robot's velocity, mode, or sends movement commands based on the key pressed.
        """
        if hasattr(key, "char"):
            match key.char:
                case "a":
                    self.vel_cmd_ *= 1.1
                    self.vel_cmd_ = max(
                        -self.vel_up_limit, min(self.vel_cmd_, self.vel_up_limit)
                    )
                    logging.info(f"Velocity set to {self.vel_cmd_}")
                case "z":
                    self.vel_cmd_ /= 1.1
                    self.vel_cmd_ = max(
                        -self.vel_up_limit, min(self.vel_cmd_, self.vel_up_limit)
                    )
                    logging.info(f"Velocity set to {self.vel_cmd_}")
                case "s":
                    self.speed_gear_index = (self.speed_gear_index + 1) % len(
                        SPEDD_GEAR
                    )
                    self.vel_cmd_ = (
                        SPEDD_GEAR[self.speed_gear_index]
                        if self.vel_cmd_ > 0
                        else -SPEDD_GEAR[self.speed_gear_index]
                    )
                    logging.info(f"Velocity set to {self.vel_cmd_}")
                case "r":
                    self.vel_cmd_ *= -1
                    self.vel_cmd_ = max(
                        -self.vel_up_limit, min(self.vel_cmd_, self.vel_up_limit)
                    )
                    logging.info(f"Velocity set to {self.vel_cmd_}")
                case "y":
                    if not self.robot.online_mode():
                        logging.error("failed to switch to ONLINE_IDLE")
                    if not self.robot.set_robot_mode(
                        mode=airbot_play_pb2.AirbotMode.ONLINE_SERVO
                    ):
                        logging.error("failed to switch to ONLINE_SERVO")
                    self.serve_mode = ServeMode.JOINT
                    logging.info("Switched to joint mode")
                case "t":
                    if not self.robot.online_mode():
                        logging.error("failed to switch to ONLINE_IDLE")
                    if not self.robot.set_robot_mode(
                        mode=airbot_play_pb2.AirbotMode.ONLINE_SERVO
                    ):
                        logging.error("failed to switch to ONLINE_SERVO")
                    self.serve_mode = ServeMode.TWIST
                    logging.info("Switched to twist mode")
                case "w":
                    self.command_frame_id = self.PLANNING_FRAME_ID
                    logging.info("Switched to planning frame")
                case "e":
                    self.command_frame_id = self.EE_FRAME_ID
                    logging.info("Switched to end effector frame")
                case "i":
                    # roll
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.ROLL, value=self.vel_cmd_)
                case "k":
                    # roll
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.ROLL, value=-self.vel_cmd_)
                case "j":
                    # pitch
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.PITCH, value=self.vel_cmd_)
                case "l":
                    # pitch
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.PITCH, value=-self.vel_cmd_)
                case "u":
                    # yaw
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.YAW, value=self.vel_cmd_)
                case "o":
                    # yaw
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.YAW, value=-self.vel_cmd_)
                case ";":
                    logging.debug("PERIOD")
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.Z, value=self.vel_cmd_)
                case ".":
                    logging.debug("PERIOD")
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.Z, value=-self.vel_cmd_)
                case "1" | "2" | "3" | "4" | "5" | "6":
                    if self.mode_check(ServeMode.JOINT):
                        joint = self.robot.get_current_joint_q()
                        index = int(key.char) - 1
                        joint[index] += self.vel_cmd_ * 0.5
                        self.robot.set_target_joint_q(
                            joint, blocking=False, use_planning=False
                        )
                        del joint
                case "7":
                    self.gripper_joint = not self.gripper_joint
                    self.robot.set_target_end(target_end=float(self.gripper_joint))
                case "q":
                    return False  # stop listening
        else:
            match key:
                case keyboard.Key.esc:
                    return False  # stop listening
                case keyboard.Key.up:
                    logging.debug("up")
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.X, value=self.vel_cmd_)
                case keyboard.Key.down:
                    logging.debug("down")
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.X, value=-self.vel_cmd_)
                case keyboard.Key.left:
                    logging.debug("left")
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.Y, value=-self.vel_cmd_)
                case keyboard.Key.right:
                    logging.debug("right")
                    if self.mode_check(ServeMode.TWIST):
                        self.move_one_direction(Transform.Y, value=self.vel_cmd_)
                case _:
                    logging.info("please input legal key")

    def on_release(self, key: keyboard.KeyCode):
        # This function can be used to handle key release events, if necessary
        pass

    def run(self):
        """
        Start the keyboard listener and begin monitoring keyboard events.

        This method first calls info_message to display startup information,
        then creates a keyboard listener that invokes on_press and on_release methods
        when keys are pressed and released, respectively. If the listener is not already running,
        it starts the listener and waits for it to complete.
        """
        self.info_message()
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release, suppress=True
        )
        if not self.listener.running:
            self.listener.start()
        self.listener.join()

    def mode_check(self, mode: ServeMode) -> bool:
        """
        Check if the current serve mode matches the specified mode.

        This method verifies whether the current instance's serve mode matches the given mode.
        If it matches, it returns True.
        If the current serve mode is not set (None), it logs a warning and returns False.
        If the serve mode does not match, it logs an appropriate warning and returns False.

        Parameters:
        mode (ServeMode): The serve mode enum instance to check against.

        Returns:
        bool: True if the current serve mode matches the specified mode; otherwise, False.
        """
        if self.serve_mode is mode:
            return True
        elif self.serve_mode is None:
            logging.warning("Command type is not set, cannot accept input")
            return False
        else:
            logging.warning(f"Command type is not {mode.name}, cannot accept input")
            return False

    def info_message(self) -> None:
        """
        Prints information about the keyboard listener.
        """
        logging.info("Reading from keyboard")
        logging.info("---------------------------")
        logging.info("All commands are in the planning frame")
        logging.info(
            "Use arrow keys and ikjl and the '.' and ';' keys to Cartesian jog"
        )
        logging.info(
            "Use 1|2|3|4|5|6 keys to joint jog. 'r' to reverse the direction of jogging."
        )
        logging.info("Use '7' to open/close gripper")
        logging.info("Use 'a' and 'z' to increase/decrease speed.")
        logging.info("Use 's' to switch speed.")
        logging.info("Use 'y' to select joint jog. ")
        logging.info("Use 't' to select twist ")
        logging.info(
            "Use 'w' and 'e' to switch between sending command in planning frame or end effector frame"
        )
        logging.info("'Q' to quit.")

    def move_one_direction(
        self,
        direction: Transform,
        value: float = 0.05,
    ):
        """
        Moves the robot in a specific direction by a specified amount.

        This function adjusts the robot's position or orientation based on the specified direction and magnitude.
        If the current command frame ID matches the planning frame ID, it directly calculates the new
          target position or orientation based on thedirection. If the frame ID does not match,
          it calls another function to move the robot relatively in the end-effector's coordinate system.

        Parameters:
        - direction (Transform): The direction in which the robot should move, such as X, Y, Z, roll, pitch, yaw.
        - value (float): The amount to move in the specified direction, default is 0.05.

        Returns:
        None
        """
        if self.command_frame_id is self.PLANNING_FRAME_ID:
            x, y, z = self.robot.get_current_translation()
            roll, pitch, yaw = quat2euler(
                reorder_quaternion(self.robot.get_current_rotation())
            )
            match direction:
                case Transform.ROLL:
                    roll += 2 * value
                case Transform.PITCH:
                    pitch += 2 * value
                case Transform.YAW:
                    yaw += 2 * value
                case Transform.X:
                    x += value
                case Transform.Y:
                    y += value
                case Transform.Z:
                    z += value
                case _:
                    logging.error("Invalid direction")
            if direction.value >= 4:
                self.robot.set_target_rotation(
                    target_rotation=order_quaternion(euler2quat(roll, pitch, yaw)),
                    use_planning=False,
                    blocking=False,
                )
            else:
                self.robot.set_target_translation(
                    target_translation=[x, y, z], use_planning=False, blocking=False
                )
        else:
            self.relative_move_in_eef_coordinate(direction, value)

    def relative_move_in_eef_coordinate(self, index: Transform, value: float = 0.1):
        """
        Perform a relative move in the End Effector (EEF) coordinate system.

        This function calculates a new target position for the robot's end effector based
        on the specified transform index and displacement value, then executes the move.

        Parameters:
        - index (Transform): Enum value indicating the axis to transform.
        - value (float): Displacement value along the specified axis, default is 0.1.

        Returns:
        None

        """
        pose = self.robot.get_current_pose()
        pose_mat = np.eye(4)
        pose_mat[:3, 3] = pose[0]
        pose_mat[:3, :3] = quat2mat(reorder_quaternion(pose[1]))

        relative: list = [0 for i in range(6)]
        relative[index.value - 1] = value
        rel_mat = np.eye(4)
        rel_mat[:3, 3] = relative[:3]
        rel_mat[:3, :3] = euler2mat(*relative[3:])

        target_mat = np.dot(pose_mat, rel_mat)
        target_pose = [
            target_mat[:3, 3].tolist(),
            list(order_quaternion(mat2quat(target_mat[:3, :3]).tolist())),
        ]
        self.robot.set_target_pose(target_pose, use_planning=False, blocking=False)


def reorder_quaternion(quaternion):
    """
    Reorder quaternion to have w term first.
    transforms3d use order w, x, y, z
    """
    x, y, z, w = quaternion
    return w, x, y, z


def order_quaternion(quaternion):
    """
    order quaternion to have w term last.
    robot use order x, y, z, w
    """
    w, x, y, z = quaternion
    return x, y, z, w


def gripper_control_example():
    """
    Example of controlling the end effector of the AIRBOT Play using the AirbotPlay class.
    """
    # Notice that the end_effector is only supported for G2
    robot = AirbotPlay()

    while True:
        robot.set_target_end(0)
        time.sleep(1)
        robot.set_target_end(1)
        time.sleep(1)
        robot.set_target_end(0.5)
        time.sleep(1)
        robot.set_target_end(0, blocking=True)
        robot.set_target_end(1, blocking=True)
        robot.set_target_end(0.5, blocking=True)


def joint_servo_example():
    """
    Example of controlling the joints of the AIRBOT Play using the AirbotPlay class in servo mode.
    """
    robot = AirbotPlay()

    # Joint position servo
    for _ in range(500):
        target_joint_q = joint_q0
        robot.set_target_joint_q(target_joint_q, use_planning=False, blocking=False)
        time.sleep(0.01)


def joint_trajectory_example():
    """
    Example of controlling the joints of the AIRBOT Play using the AirbotPlay class in trajectory mode.
    """
    robot = AirbotPlay()

    # Joint position trajectory
    target_joint_q = joint_q1
    robot.set_target_joint_q(target_joint_q, use_planning=True, blocking=False)
    time.sleep(3)

    target_joint_q = joint_q0
    robot.set_target_joint_q(target_joint_q, use_planning=True, blocking=True)
    time.sleep(0.01)


def joint_waypoints_example():
    """
    Example of controlling the joints of the AIRBOT Play using the AirbotPlay class in waypoints mode.
    """
    robot = AirbotPlay()
    robot.set_target_joint_q_waypoints(waypoints=[joint_q1, joint_q2, joint_q3])


def pose_servo_example():
    """
    Example of controlling the end effector of the AIRBOT Play using the AirbotPlay class in servo mode.
    """
    robot = AirbotPlay()

    print("Move to safe pose")
    robot.set_target_joint_q(joint_home, use_planning=True, blocking=True)

    if robot.params["eef_type"] == "none":
        pose1 = none_eef_pose1
        pose2 = none_eef_pose2
    else:
        pose1 = with_eef_pose1
        pose2 = with_eef_pose2

    # Use servo to move to zero position
    for _ in range(500):
        robot.set_target_pose(pose1, use_planning=False)
        time.sleep(0.01)
    print("Arrived at pose1")
    # Use trajectory to move to home position
    for _ in range(500):
        robot.set_target_pose(pose2, use_planning=False)
        time.sleep(0.01)
    print("Arrived at pose2")


def pose_trajectory_example():
    """
    Example of controlling the end effector of the AIRBOT Play using the AirbotPlay class in trajectory mode.
    """
    robot = AirbotPlay()
    if robot.params["eef_type"] == "none":
        pose1 = none_eef_pose1
        pose2 = none_eef_pose2
    else:
        pose1 = with_eef_pose1
        pose2 = with_eef_pose2
    robot.set_target_pose(pose1)
    time.sleep(3)
    robot.set_target_pose(pose2, use_planning=True, blocking=True)


def pose_waypoints_example():
    """
    Example of controlling the end effector of the AIRBOT Play using the AirbotPlay class in waypoints mode.
    """
    robot = AirbotPlay()
    if robot.params["eef_type"] == "none":
        pose1 = none_eef_pose1
        pose2 = none_eef_pose2
    else:
        pose1 = with_eef_pose1
        pose2 = with_eef_pose2
    robot.set_target_pose_waypoints(waypoints=[pose1, pose2])


def status_request_example():
    """
    Example of requesting the status of the AIRBOT Play using the AirbotPlay class.
    """
    robot = AirbotPlay()

    while True:
        print("Current end:", robot.get_current_end())
        print("Current joint error code:", robot.get_current_joint_error_code())
        print("Current joint q:", robot.get_current_joint_q())
        print("Current joint t:", robot.get_current_joint_t())
        print("Current joint temperature:", robot.get_current_joint_temperature())
        print("Current joint v:", robot.get_current_joint_v())
        print("Current pose:", robot.get_current_pose())
        print("Current rotation:", robot.get_current_rotation())
        print("Current translation:", robot.get_current_translation())
        print("Current state:", robot.get_current_state())
        time.sleep(1)


def joint_follow_example():
    """
    Example of controlling AIRBOT Play following another robot's joint position.
    """
    master_robot = AirbotPlay()
    slave_robot = AirbotPlay(port=50000)
    slave_robot.slave_mode(int(master_robot.params["ros_domain_id"]))
    slave_robot.follow_start()
    time.sleep(10)
    slave_robot.follow_stop()


def keyboard_control_example():
    """
    Example of controlling the AIRBOT Play using the keyboard.
    Press Q or ESC to exit the program
    """
    robot = AirbotPlay()
    keyboard_control = KeyboardControl(robot=robot)
    try:
        keyboard_control.run()
    except KeyboardInterrupt:
        keyboard_control.listener.stop()


def get_system_params_example():
    """
    Example of getting AIRBOT Play parameters.
    """
    robot = AirbotPlay()
    params_name: list[str] = [
        "ArmElement",
        "FirmwareVersion",
        "HardwareVersion",
        "SN",
        "ArmSN",
        "ProductFlag",
        "ErrorList",
        "arm_type",
        "end_effector",
        "arm_sn_code",
    ]
    param_dict = robot.get_system_params(params_name)
    for key, value in param_dict.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # gripper_control_example()
    # joint_servo_example()
    # joint_trajectory_example()
    # joint_waypoints_example()
    # pose_servo_example()
    # pose_trajectory_example()
    # pose_waypoints_example()
    # get_system_params_example()
    # keyboard_control_example()
    status_request_example()
