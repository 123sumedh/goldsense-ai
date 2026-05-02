"""Hallmark detection using EasyOCR for gold purity stamps (916, BIS, 22K)."""
import re
import numpy as np


class HallmarkDetector:
    """Detect hallmarks like '916', 'BIS', '22K' in jewelry images.
    
    Uses EasyOCR if available; gracefully falls back to mock results otherwise.
    Hallmark detection is the strongest signal for purity classification.
    """
    
    def __init__(self, lazy_load=True):
        self.reader = None
        self._initialized = False
        self.lazy_load = lazy_load
        
        # Mapping from purity numbers to karat values
        self.purity_map = {
            '999': '24K',
            '995': '24K',
            '958': '23K',
            '916': '22K',
            '875': '21K',
            '750': '18K',
            '585': '14K',
            '375': '9K',
        }
        
        # Regex patterns
        self.bis_pattern = re.compile(r'BIS|BS', re.IGNORECASE)
        self.karat_pattern = re.compile(r'(\d{1,2})\s*K', re.IGNORECASE)
        self.purity_number_pattern = re.compile(r'\b(999|995|958|916|875|750|585|375)\b')
        
        if not lazy_load:
            self._initialize_reader()
    
    def _initialize_reader(self):
        """Lazy-initialize EasyOCR (slow first time, ~50MB download)."""
        if self._initialized:
            return
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self._initialized = True
        except Exception as e:
            print(f"⚠️  EasyOCR initialization failed: {e}")
            print("   Hallmark detection will return empty results.")
            self.reader = None
            self._initialized = True
    
    def detect(self, image: np.ndarray) -> dict:
        """Detect hallmarks in the image.
        
        Args:
            image: BGR image as numpy array
        
        Returns:
            dict with keys: detected, hallmarks, declared_purity, bis_certified, all_text_found
        """
        # Lazy-init OCR on first use
        if self.lazy_load and not self._initialized:
            self._initialize_reader()
        
        if self.reader is None:
            return {
                'detected': False,
                'hallmarks': [],
                'declared_purity': None,
                'bis_certified': False,
                'all_text_found': [],
                'error': 'EasyOCR not available'
            }
        
        try:
            results = self.reader.readtext(image, detail=1)
        except Exception as e:
            return {
                'detected': False,
                'hallmarks': [],
                'declared_purity': None,
                'bis_certified': False,
                'all_text_found': [],
                'error': str(e)
            }
        
        hallmarks = []
        purity = None
        bis_certified = False
        
        for (bbox, text, conf) in results:
            text_clean = text.strip().upper()
            if conf < 0.3:
                continue
            
            # Check for purity number (916, 750, 999, etc.)
            num_match = self.purity_number_pattern.search(text_clean)
            if num_match:
                num = num_match.group(1)
                purity = self.purity_map.get(num)
                hallmarks.append({
                    'type': 'purity_number',
                    'value': num,
                    'karat': purity,
                    'confidence': float(conf),
                    'text': text_clean
                })
            
            # Check for karat text (22K, 18K, etc.)
            karat_match = self.karat_pattern.search(text_clean)
            if karat_match:
                karat = f"{karat_match.group(1)}K"
                hallmarks.append({
                    'type': 'karat',
                    'value': karat,
                    'confidence': float(conf),
                    'text': text_clean
                })
                if not purity:
                    purity = karat
            
            # Check for BIS certification mark
            if self.bis_pattern.search(text_clean):
                bis_certified = True
                hallmarks.append({
                    'type': 'bis',
                    'value': 'BIS',
                    'confidence': float(conf),
                    'text': text_clean
                })
        
        return {
            'detected': len(hallmarks) > 0,
            'hallmarks': hallmarks,
            'declared_purity': purity,
            'bis_certified': bis_certified,
            'all_text_found': [r[1] for r in results]
        }