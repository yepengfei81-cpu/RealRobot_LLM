"""
Position Calculators for Brick Detection

Provides geometric calculations for:
- Head camera: Convert depth + mask to 3D position + area
- Hand-eye camera: Fine positioning with known reference
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


# ==================== Configuration ====================
CALIB_DIR = Path("/home/ypf/qiuzhiarm_LLM/calibration")

HEAD_INTRINSICS = {
    'fx': 607.15, 
    'fy': 607.02, 
    'cx': 324.25, 
    'cy': 248.46
}


# ==================== Dynamic Z Compensator ====================

class DynamicZCompensator:
    """Dynamic Z-axis Compensator - same as in single_grasp.py"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.model_type = "linear"
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
        """Load from config file"""
        if not self.config_path.exists():
            print(f"[Z-Comp] Config file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path) as f:
                data = json.load(f)
            
            dz = data.get('dynamic_z_compensation', {})
            self.enabled = dz.get('enabled', False)
            self.model_type = dz.get('model', 'linear')
            self.coefficients = dz.get('coefficients', self.coefficients)
            
            if self.enabled:
                print(f"[Z-Comp] Loaded dynamic compensation model: {self.model_type}")
                print(f"[Z-Comp] Coefficients: intercept={self.coefficients.get('intercept', 0):.4f}, "
                      f"x={self.coefficients.get('x_coeff', 0):.4f}, y={self.coefficients.get('y_coeff', 0):.4f}")
            else:
                print("[Z-Comp] Dynamic compensation disabled in config")
        except Exception as e:
            print(f"[Z-Comp] Failed to load config: {e}")
    
    def compute_compensation(self, x: float, y: float) -> float:
        """Compute Z compensation value based on X, Y coordinates"""
        if not self.enabled:
            return 0.0
        
        c = self.coefficients
        
        if self.model_type == "linear":
            return c.get('intercept', 0.0) + c.get('x_coeff', 0.0) * x + c.get('y_coeff', 0.0) * y
        
        elif self.model_type == "quadratic":
            return (c.get('intercept', 0.0) + 
                    c.get('x_coeff', 0.0) * x + 
                    c.get('y_coeff', 0.0) * y +
                    c.get('xy_coeff', 0.0) * x * y +
                    c.get('x2_coeff', 0.0) * x * x +
                    c.get('y2_coeff', 0.0) * y * y)
        
        return 0.0


# ==================== Utility Functions ====================

def estimate_orientation(mask: np.ndarray) -> float:
    """
    Estimate orientation angle (yaw) from binary mask.
    
    Uses minimum area rectangle fitting to find the major axis.
    
    Args:
        mask: binary mask (H, W)
        
    Returns:
        yaw angle in radians
    """
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    
    rect = cv2.minAreaRect(largest)
    (_, _), (w, h), angle = rect
    
    # Determine long axis angle
    long_angle = angle + 90 if w < h else angle
    yaw = -np.radians(long_angle)
    
    # Normalize to [-pi/2, pi/2]
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    
    return float(np.clip(yaw, -np.pi / 2, np.pi / 2))


def calculate_real_area(
    mask: np.ndarray, 
    depth: np.ndarray, 
    fx: float, 
    fy: float
) -> Tuple[float, int]:
    """
    Calculate real-world area of a segmented region.
    
    Uses depth and camera intrinsics to estimate actual area.
    
    Args:
        mask: binary mask (H, W) or boolean mask
        depth: depth image in mm
        fx: camera focal length x
        fy: camera focal length y
        
    Returns:
        Tuple of (area_cm2, area_pixels)
    """
    # Ensure mask is boolean
    mask_bool = mask > 0.5 if mask.dtype != bool else mask
    
    # Get pixel count
    pixel_count = int(np.sum(mask_bool))
    
    if pixel_count == 0:
        return 0.0, 0
    
    # Get valid depth values in mask region
    valid_depth = depth[mask_bool]
    valid_depth = valid_depth[(valid_depth > 100) & (valid_depth < 2000)]  # 10cm - 2m in mm
    
    if len(valid_depth) == 0:
        return 0.0, pixel_count
    
    # Use median depth for robustness
    median_depth_m = np.median(valid_depth) / 1000.0  # Convert to meters
    
    # Calculate pixel size at this depth
    # At depth z, each pixel covers (z/fx) × (z/fy) meters
    pixel_size_m2 = (median_depth_m / fx) * (median_depth_m / fy)
    
    # Calculate area
    area_m2 = pixel_count * pixel_size_m2
    area_cm2 = area_m2 * 10000  # Convert to cm²
    
    return area_cm2, pixel_count


# ==================== Head Camera Calculator ====================

class HeadCameraCalculator:
    """
    Head Camera Position Calculator
    
    Computes brick information in base_link frame from:
    - Segmentation mask
    - Depth image
    - Camera TF transform
    - Dynamic Z compensation
    
    Returns:
    - position: [x, y, z] in base_link frame
    - yaw: orientation angle
    - area_cm2: estimated real area in cm²
    - area_pixels: pixel count
    - z_compensation: applied Z offset
    - depth_median_m: median depth value
    """
    
    def __init__(
        self, 
        intrinsics: Optional[Dict] = None,
        z_compensator: Optional[DynamicZCompensator] = None,
    ):
        """
        Initialize calculator.
        
        Args:
            intrinsics: camera intrinsics dict with fx, fy, cx, cy
            z_compensator: DynamicZCompensator for Z-axis correction
        """
        if intrinsics is None:
            intrinsics = HEAD_INTRINSICS
        
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        # Z compensator
        self.z_compensator = z_compensator
        
        # Load static offset
        self.offset = np.zeros(3)
        offset_path = CALIB_DIR / "head_camera_offset.json"
        if offset_path.exists():
            try:
                with open(offset_path) as f:
                    data = json.load(f)
                    self.offset = np.array(data.get('offset_xyz', [0, 0, 0]))
                    print(f"[HeadCalc] Static offset: X={self.offset[0]:+.4f}, Y={self.offset[1]:+.4f}, Z={self.offset[2]:+.4f}")
            except Exception as e:
                print(f"[HeadCalc] Failed to load offset: {e}")
        
        # Print Z compensator status
        if self.z_compensator is not None and self.z_compensator.enabled:
            print(f"[HeadCalc] DynamicZCompensator: ENABLED ({self.z_compensator.model_type})")
        else:
            print("[HeadCalc] DynamicZCompensator: DISABLED")
    
    def compute(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray, 
        tf_matrix: np.ndarray
    ) -> Optional[Dict]:
        """
        Compute brick information in base_link frame.
        
        Args:
            mask: segmentation mask
            depth: depth image (mm)
            tf_matrix: 4x4 transform from camera to base_link
            
        Returns:
            Dictionary with:
            - 'position': [x, y, z] in base_link frame
            - 'yaw': orientation angle (radians)
            - 'area_cm2': estimated real area in cm²
            - 'area_pixels': pixel count of mask
            - 'z_compensation': applied Z offset
            - 'depth_median_m': median depth value in meters
            
            Returns None if computation failed
        """
        h, w = depth.shape
        
        # Handle mask dimensions
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.float32), (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        # Get mask centroid
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        
        # Estimate orientation
        yaw_cam = estimate_orientation(mask_bool)
        
        # Get depth value (erode mask for stability)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=2)
        valid = depth[eroded > 0] if np.any(eroded) else depth[mask_bool]
        valid = valid[valid > 0]
        
        if len(valid) == 0:
            return None
        
        z = np.median(valid) / 1000.0  # Convert to meters
        depth_median_m = z  # Save for return
        z = np.clip(z, 0.1, 2.0)
        
        # ========== Calculate Real Area ==========
        area_cm2, area_pixels = calculate_real_area(
            mask_bool, depth, self.fx, self.fy
        )
        
        # Compute 3D position in camera frame
        pos_cam = np.array([
            (px - self.cx) * z / self.fx,
            (py - self.cy) * z / self.fy,
            z
        ])
        
        # Transform to base_link frame
        pos_base = (tf_matrix @ np.append(pos_cam, 1))[:3] + self.offset
        
        # Transform yaw to base frame
        yaw_base = yaw_cam + np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0]) + np.pi
        
        # ========== Apply Dynamic Z Compensation ==========
        z_compensation = 0.0
        if self.z_compensator is not None and self.z_compensator.enabled:
            z_compensation = self.z_compensator.compute_compensation(pos_base[0], pos_base[1])
            pos_base[2] += z_compensation
            print(f"[HeadCalc] Z compensation: {z_compensation*1000:+.1f} mm at X={pos_base[0]:.3f}, Y={pos_base[1]:.3f}")
        
        # Normalize yaw
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        return {
            'position': pos_base, 
            'yaw': yaw_base,
            'area_cm2': area_cm2,
            'area_pixels': area_pixels,
            'z_compensation': z_compensation,
            'depth_median_m': depth_median_m,
        }


# ==================== Hand-Eye Camera Calculator ====================

class HandEyeCalculator:
    """
    Hand-Eye Camera Position Calculator
    
    Provides fine positioning using hand-mounted camera
    with known reference Z from head camera.
    """
    
    def __init__(self, intrinsics: Dict, extrinsics: Dict):
        """
        Initialize calculator.
        
        Args:
            intrinsics: camera intrinsics with fx, fy, cx, cy
            extrinsics: camera-to-gripper transform with rotation_matrix and translation
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        # Build camera-to-gripper transform
        self.T_cam2gripper = np.eye(4)
        self.T_cam2gripper[:3, :3] = np.array(extrinsics['rotation_matrix'])
        self.T_cam2gripper[:3, 3] = np.array(extrinsics['translation'])
    
    def compute(
        self, 
        mask: np.ndarray, 
        shape: Tuple[int, int],
        R_g2b: np.ndarray, 
        t_g2b: np.ndarray,
        reference_z: float, 
        reference_yaw: float
    ) -> Optional[Dict]:
        """
        Compute refined brick position.
        
        Args:
            mask: segmentation mask
            shape: image shape (h, w)
            R_g2b: gripper-to-base rotation matrix
            t_g2b: gripper-to-base translation
            reference_z: Z coordinate from head camera (already compensated)
            reference_yaw: yaw from head camera
            
        Returns:
            Dictionary with 'position' and 'yaw' or None if failed
        """
        h, w = shape
        
        # Handle mask dimensions
        mask = mask[0] if len(mask.shape) == 3 else mask
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.float32), (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
        
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return None
        
        # Get mask centroid
        ys, xs = np.where(mask_bool)
        px, py = np.mean(xs), np.mean(ys)
        
        # Estimate orientation in camera frame
        yaw_cam = estimate_orientation(mask_bool)
        
        # Build gripper-to-base transform
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_g2b
        T_g2b[:3, 3] = t_g2b.flatten()
        
        # Camera-to-base transform
        T_cam2base = T_g2b @ self.T_cam2gripper
        
        # Compute Z in camera frame using reference_z
        R_mat = T_cam2base[:3, :3]
        t_vec = T_cam2base[:3, 3]
        
        nx = (px - self.cx) / self.fx
        ny = (py - self.cy) / self.fy
        coeff = R_mat[2, 0] * nx + R_mat[2, 1] * ny + R_mat[2, 2]
        
        if abs(coeff) > 1e-6:
            z_cam = (reference_z - t_vec[2]) / coeff
        else:
            z_cam = 0.3
        
        z_cam = max(0.05, min(1.0, z_cam))
        
        # Compute 3D position in camera frame
        pos_cam = np.array([
            (px - self.cx) * z_cam / self.fx,
            (py - self.cy) * z_cam / self.fy,
            z_cam
        ])
        
        # Transform to base frame
        pos_base = (T_cam2base @ np.append(pos_cam, 1))[:3]
        
        # Transform yaw to base frame
        yaw_base = yaw_cam + np.arctan2(T_cam2base[1, 0], T_cam2base[0, 0])
        
        # Normalize yaw
        while yaw_base > np.pi:
            yaw_base -= 2 * np.pi
        while yaw_base < -np.pi:
            yaw_base += 2 * np.pi
        
        # Fix yaw jump relative to reference
        diff = yaw_base - reference_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        if abs(diff) > np.pi / 2:
            yaw_base = yaw_base - np.pi if yaw_base > 0 else yaw_base + np.pi
        
        return {'position': pos_base, 'yaw': yaw_base}


# ==================== Factory Functions ====================

def create_head_calculator(
    intrinsics: Optional[Dict] = None,
    calib_dir: Optional[Path] = None,
    enable_z_compensation: bool = True,
) -> HeadCameraCalculator:
    """
    Create head camera calculator with optional Z compensation.
    
    Args:
        intrinsics: camera intrinsics (optional)
        calib_dir: calibration directory path
        enable_z_compensation: whether to enable dynamic Z compensation
        
    Returns:
        HeadCameraCalculator instance
    """
    if calib_dir is None:
        calib_dir = CALIB_DIR
    
    # Create Z compensator if enabled
    z_compensator = None
    if enable_z_compensation:
        offset_path = calib_dir / "head_camera_offset.json"
        z_compensator = DynamicZCompensator(offset_path)
    
    return HeadCameraCalculator(intrinsics=intrinsics, z_compensator=z_compensator)


def create_handeye_calculator(
    side: str = "left",
    calib_dir: Optional[Path] = None
) -> HandEyeCalculator:
    """
    Create hand-eye camera calculator.
    
    Args:
        side: "left" or "right"
        calib_dir: calibration directory path
        
    Returns:
        HandEyeCalculator instance
    """
    if calib_dir is None:
        calib_dir = CALIB_DIR
    
    intr_path = calib_dir / f"hand_eye_intrinsics_{side}.json"
    extr_path = calib_dir / f"hand_eye_extrinsics_{side}.json"
    
    with open(intr_path) as f:
        intrinsics = json.load(f)
    with open(extr_path) as f:
        extrinsics = json.load(f)
    
    return HandEyeCalculator(intrinsics, extrinsics)