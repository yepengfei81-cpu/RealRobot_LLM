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
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection bounding box on frame.
        
        Args:
            frame: BGR image to draw on
            mask: segmentation mask from segment()
            color: box color (BGR)
            thickness: line thickness
            
        Returns:
            Frame with detection visualization
        """
        if mask is None:
            return frame
        
        result = frame.copy()
        
        # Handle mask dimensions
        m = mask[0] if len(mask.shape) == 3 else mask
        h, w = result.shape[:2]
        
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = (m > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result
        
        largest = max(contours, key=cv2.contourArea)
        
        # Draw oriented bounding box only
        if len(largest) >= 5:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(result, [box], 0, color, thickness)
        
        # Draw center point
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)
        
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