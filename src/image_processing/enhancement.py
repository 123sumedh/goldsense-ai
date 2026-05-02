"""Image enhancement pipeline — clean noisy real-world phone photos."""
import cv2
import numpy as np


class ImageEnhancer:
    """DIP pipeline to clean phone photos before AI analysis."""
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Bilateral filter — edge-preserving denoising."""
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    def white_balance(self, image: np.ndarray) -> np.ndarray:
        """Gray-world assumption white balance — fix yellow/blue tints."""
        result = image.copy().astype(np.float32)
        for i in range(3):
            avg = np.mean(result[:, :, i])
            if avg > 0:
                result[:, :, i] *= 128.0 / avg
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on L channel of LAB — fix uneven lighting."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    
    def remove_glare(self, image: np.ndarray) -> np.ndarray:
        """Inpaint specular highlights from glossy gold."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 240)).astype(np.uint8) * 255
        if np.sum(mask) > 0:
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return image
    
    def enhance_pipeline(self, image: np.ndarray) -> dict:
        """Run full enhancement pipeline; return all intermediate stages."""
        stages = {'original': image.copy()}
        denoised = self.denoise(image)
        stages['denoised'] = denoised
        wb = self.white_balance(denoised)
        stages['white_balanced'] = wb
        normalized = self.normalize_lighting(wb)
        stages['normalized'] = normalized
        final = self.remove_glare(normalized)
        stages['final'] = final
        return stages