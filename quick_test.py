"""Quick test to verify everything works without launching Streamlit."""
import cv2
import numpy as np
from src.image_processing.quality_check import ImageQualityChecker
from src.image_processing.enhancement import ImageEnhancer
from src.image_processing.features import ColorFeatureExtractor
from src.computer_vision.purity_classifier import PurityClassifier

def quick_test(image_path):
    print(f"Testing on: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found, using synthetic test")
        image = np.ones((400, 400, 3), dtype=np.uint8) * 30
        cv2.circle(image, (200, 200), 80, (40, 175, 215), -1)
    
    # Run mini pipeline
    quality = ImageQualityChecker().assess(image)
    print(f"Quality: {quality['quality_score']}/100")
    
    enhanced = ImageEnhancer().enhance_pipeline(image)['final']
    print("✓ Enhancement complete")
    
    features = ColorFeatureExtractor().extract(enhanced)
    print(f"Hue: {features['hue_mean']:.1f}°, b: {features['lab_b_mean']:.0f}")
    
    result = PurityClassifier().classify(features)
    print(f"Purity: {result['purity_band']} (conf: {result['confidence']:.0%})")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/test.jpg"
    quick_test(path)