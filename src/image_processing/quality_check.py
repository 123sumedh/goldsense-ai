"""Image quality assessment — pre-flight check before AI processing."""
import cv2
import numpy as np
from typing import Tuple


class ImageQualityChecker:
    """Pre-flight image quality checks before running AI models."""
    
    def __init__(self, blur_threshold: float = 100, 
                 min_brightness: float = 50, 
                 max_brightness: float = 220):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
    
    def check_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """Laplacian variance — higher = sharper."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return score >= self.blur_threshold, score
    
    def check_brightness(self, image: np.ndarray) -> Tuple[bool, float]:
        """Check if image is too dark or overexposed."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        is_ok = self.min_brightness <= brightness <= self.max_brightness
        return is_ok, brightness
    
    def check_resolution(self, image: np.ndarray, min_size: int = 200) -> Tuple[bool, tuple]:
        """Ensure minimum resolution for downstream processing."""
        h, w = image.shape[:2]
        is_ok = h >= min_size and w >= min_size
        return is_ok, (h, w)
    
    def assess(self, image: np.ndarray) -> dict:
        """Run all checks and return comprehensive report."""
        blur_ok, blur_score = self.check_blur(image)
        bright_ok, brightness = self.check_brightness(image)
        res_ok, (h, w) = self.check_resolution(image)
        
        issues = []
        if not blur_ok:
            issues.append(f"Image too blurry (score: {blur_score:.0f}, need >{self.blur_threshold})")
        if not bright_ok:
            if brightness < self.min_brightness:
                issues.append(f"Image too dark (brightness: {brightness:.0f})")
            else:
                issues.append(f"Image overexposed (brightness: {brightness:.0f})")
        if not res_ok:
            issues.append(f"Resolution too low: {w}x{h}")
        
        # Quality score 0-100
        quality_score = 0
        if blur_ok: quality_score += 40
        if bright_ok: quality_score += 30
        if res_ok: quality_score += 30
        
        return {
            'is_acceptable': len(issues) == 0,
            'quality_score': quality_score,
            'blur_score': blur_score,
            'brightness': brightness,
            'resolution': (h, w),
            'issues': issues,
        }