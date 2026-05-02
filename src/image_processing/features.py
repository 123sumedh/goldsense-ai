import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class ColorFeatureExtractor:
    """Extract color & texture features for gold purity classification.
    
    KEY FIX: Auto-filters white/dark backgrounds and analyzes only the 
    saturated jewelry pixels. Critical for catalog-style images with 
    white backgrounds (like Tanishq dataset).
    """
    
    def __init__(self, min_saturation=40, min_value=50):
        # Pixels below these thresholds are considered background
        self.min_saturation = min_saturation
        self.min_value = min_value
    
    def auto_mask(self, image):
        """Auto-create mask: include only colored (saturated) jewelry pixels."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Saturated AND not pure white AND not pure black
        mask = (hsv[:, :, 1] > self.min_saturation) & \
               (hsv[:, :, 2] > self.min_value) & \
               (hsv[:, :, 2] < 250)  # Not pure white
        return mask.astype(np.uint8) * 255
    
    def extract(self, image, mask=None):
        if mask is None:
            mask = self.auto_mask(image)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        mask_bool = mask > 0
        
        # If no pixels found, return zeros (avoid crash)
        if not mask_bool.any() or mask_bool.sum() < 100:
            return self._empty_features()
        
        hsv_pixels = hsv[mask_bool]
        lab_pixels = lab[mask_bool]
        
        features = {
            'hue_mean': float(np.mean(hsv_pixels[:, 0])),
            'hue_std': float(np.std(hsv_pixels[:, 0])),
            'saturation_mean': float(np.mean(hsv_pixels[:, 1])),
            'saturation_std': float(np.std(hsv_pixels[:, 1])),
            'value_mean': float(np.mean(hsv_pixels[:, 2])),
            'lab_l_mean': float(np.mean(lab_pixels[:, 0])),
            'lab_a_mean': float(np.mean(lab_pixels[:, 1])),
            'lab_b_mean': float(np.mean(lab_pixels[:, 2])),
            'highlight_ratio': float(np.sum((gray > 200) & mask_bool) / max(mask_bool.sum(), 1)),
            'mask_coverage': float(mask_bool.sum() / mask_bool.size),
        }
        
        # GLCM texture features
        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)
            features['contrast'] = float(graycoprops(glcm, 'contrast').mean())
            features['homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
            features['energy'] = float(graycoprops(glcm, 'energy').mean())
        except Exception:
            features['contrast'] = features['homogeneity'] = features['energy'] = 0.0
        
        return features
    
    def _empty_features(self):
        return {k: 0.0 for k in [
            'hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean',
            'lab_l_mean', 'lab_a_mean', 'lab_b_mean', 'highlight_ratio', 'mask_coverage',
            'contrast', 'homogeneity', 'energy'
        ]}