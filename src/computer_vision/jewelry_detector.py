"""Jewelry type detection using YOLOv8 or fallback heuristics."""
import numpy as np


class JewelryDetector:
    """Detect jewelry type (ring, chain, bangle, etc.) from image.
    
    For prototype: Uses YOLOv8 pretrained on COCO (or fallback heuristics).
    For production: Fine-tune YOLOv8 on jewelry-specific labeled dataset.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", lazy_load: bool = True):
        self.model_path = model_path
        self.model = None
        self._initialized = False
        self.lazy_load = lazy_load
        
        if not lazy_load:
            self._initialize_model()
    
    def _initialize_model(self):
        """Lazy-load YOLOv8 (downloads ~6MB on first use)."""
        if self._initialized:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._initialized = True
        except Exception as e:
            print(f"⚠️  YOLOv8 not available: {e}")
            self.model = None
            self._initialized = True
    
    def detect(self, image: np.ndarray) -> dict:
        """Detect jewelry type. Falls back to heuristic if YOLO unavailable."""
        # Lazy-init on first use
        if self.lazy_load and not self._initialized:
            self._initialize_model()
        
        # Try YOLOv8 first
        if self.model is not None:
            try:
                results = self.model(image, verbose=False)
                detections = []
                for r in results:
                    for box in r.boxes:
                        detections.append({
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': float(box.conf[0]),
                            'class': self.model.names[int(box.cls[0])]
                        })
                
                if detections:
                    best = max(detections, key=lambda x: x['confidence'])
                    return {
                        'detected': True,
                        'jewelry_type': best['class'],
                        'confidence': best['confidence'],
                        'bbox': best['bbox'],
                        'all_detections': detections,
                        'method': 'yolov8'
                    }
            except Exception as e:
                print(f"⚠️  YOLO detection failed: {e}")
        
        # Fallback: aspect-ratio based heuristic
        h, w = image.shape[:2]
        ratio = w / max(h, 1)
        if 0.85 < ratio < 1.15:
            jewelry_type = "ring"
        elif ratio > 2.5:
            jewelry_type = "chain"
        elif ratio > 1.5:
            jewelry_type = "necklace"
        else:
            jewelry_type = "pendant"
        
        return {
            'detected': False,
            'jewelry_type': jewelry_type,
            'confidence': 0.5,
            'method': 'aspect_ratio_fallback',
            'reason': "Object detection unavailable — using image dimensions"
        }