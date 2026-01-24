"""
SAM3 Segmentation Module

Singleton pattern for efficient model loading and thread-safe segmentation.
"""

# ===== Setup SAM3 path FIRST (before any SAM3 imports) =====
import sys
sys.path.insert(0, "/home/ypf/sam3-main")

# ===== Suppress warnings =====
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import logging
logging.getLogger("root").setLevel(logging.WARNING)

# ===== Standard imports =====
import cv2
import gc
import torch
import threading
import numpy as np
from PIL import Image
from typing import Optional

# ===== SAM3 imports (after path setup) =====
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Segmenter:
    """
    SAM3 Segmenter (Singleton, Thread-safe)
    
    Usage:
        segmenter = SAM3Segmenter("/path/to/checkpoint.pt")
        masks = segmenter.segment(image, "brick")
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, checkpoint_path: str, confidence: float = 0.5):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_path: str, confidence: float = 0.5):
        if self._initialized:
            return
        
        print("[SAM3] Loading model...")
        
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.processor = Sam3Processor(
            self.model, 
            resolution=1008, 
            confidence_threshold=confidence
        )
        self._segment_lock = threading.Lock()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        self._initialized = True
        print("[SAM3] Model loaded")
    
    def segment(self, img_bgr: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """
        Segment image with text prompt.
        
        Args:
            img_bgr: BGR image (OpenCV format)
            prompt: text prompt for segmentation
            
        Returns:
            Segmentation masks or None if no detection
        """
        with self._segment_lock:
            # Convert BGR to RGB PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            # Process
            state = self.processor.set_image(pil_img)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            
            # Extract masks
            masks = None
            if out["masks"] is not None:
                masks = out["masks"].cpu().numpy()
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            return masks
    
    def segment_rgb(self, img_rgb: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """
        Segment RGB image with text prompt.
        
        Args:
            img_rgb: RGB image
            prompt: text prompt
            
        Returns:
            Segmentation masks or None
        """
        with self._segment_lock:
            pil_img = Image.fromarray(img_rgb)
            state = self.processor.set_image(pil_img)
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
            
            masks = None
            if out["masks"] is not None:
                masks = out["masks"].cpu().numpy()
            
            torch.cuda.empty_cache()
            return masks

    def draw_detection(
        self, 
        frame: np.ndarray, 
        mask: Optional[np.ndarray],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        selected_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw detection bounding box on frame.
        
        Args:
            frame: BGR image to draw on
            mask: segmentation mask(s) from segment() - can be single or multiple
            color: box color (BGR) for selected/single detection
            thickness: line thickness
            selected_index: if provided, highlight this mask specially (others drawn dimmer)
            
        Returns:
            Frame with detection visualization
        """
        if mask is None:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Color palette for multiple detections
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
            (255, 255, 0),  # Cyan
        ]
        
        # Handle single vs multiple masks
        if len(mask.shape) == 2:
            # Single 2D mask
            masks_list = [mask]
        elif len(mask.shape) == 3:
            # Multiple masks: (N, H, W) or single (1, H, W)
            if mask.shape[0] == 1:
                masks_list = [mask[0]]
            else:
                masks_list = [mask[i] for i in range(mask.shape[0])]
        elif len(mask.shape) == 4:
            # Shape like (N, 1, H, W)
            masks_list = [mask[i, 0] for i in range(mask.shape[0])]
        else:
            return result
        
        # Draw each mask
        for i, m in enumerate(masks_list):
            # Resize if needed
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = (m > 0.5).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            largest = max(contours, key=cv2.contourArea)
            
            # Determine color and thickness for this mask
            if selected_index is not None:
                if i == selected_index:
                    # Selected brick: bright color, thick line
                    draw_color = (0, 255, 0)  # Bright green
                    draw_thickness = 3
                else:
                    # Non-selected: dimmer color, thin line
                    draw_color = (100, 100, 100)  # Gray
                    draw_thickness = 1
            else:
                # No selection: show all with different colors
                draw_color = colors[i % len(colors)]
                draw_thickness = thickness
            
            # Draw oriented bounding box
            if len(largest) >= 5:
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(result, [box], 0, draw_color, draw_thickness)
            
            # Draw center point and index
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Center point
                if selected_index is None or i == selected_index:
                    cv2.circle(result, (cx, cy), 5, draw_color, -1)
                    # Draw index number
                    cv2.putText(result, f"#{i}", (cx + 10, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                else:
                    cv2.circle(result, (cx, cy), 3, draw_color, -1)
                    cv2.putText(result, f"#{i}", (cx + 10, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
        
        return result

def create_segmenter(
    checkpoint_path: str = "/home/ypf/sam3-main/checkpoint/sam3.pt",
    confidence: float = 0.5
) -> SAM3Segmenter:
    """
    Create or get SAM3 segmenter instance.
    
    Args:
        checkpoint_path: path to SAM3 checkpoint
        confidence: confidence threshold
        
    Returns:
        SAM3Segmenter instance
    """
    return SAM3Segmenter(checkpoint_path, confidence)