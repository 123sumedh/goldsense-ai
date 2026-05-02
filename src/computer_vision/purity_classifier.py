import numpy as np

class PurityClassifier:
    """Estimate gold purity band from extracted features.
    
    NOTE: This is a heuristic-based classifier for the prototype.
    In production, this would be a trained ML model on labeled gold images.
    
    Hue thresholds are calibrated for OpenCV HSV scale (0-180).
    Real gold ranges:
    - 22K-24K (Pure):  Hue 18-28°, b_lab > 150 (rich yellow)
    - 18K-22K:         Hue 22-32°, b_lab > 130 (slightly lighter)
    - 14K-18K:         Hue 25-38°, b_lab > 110 (pale yellow)
    - Plated/Fake:     Outside these ranges or very low saturation
    """
    
    def classify(self, features: dict, hallmark_data: dict = None) -> dict:
        # Hallmark is the strongest signal
        if hallmark_data and hallmark_data.get('declared_purity'):
            return {
                'purity_band': hallmark_data['declared_purity'],
                'confidence': 0.85 if hallmark_data['bis_certified'] else 0.70,
                'method': 'hallmark_detection',
                'reasoning': f"Hallmark detected: {hallmark_data['declared_purity']}" + 
                            (" (BIS certified)" if hallmark_data['bis_certified'] else "")
            }
        
        # Fall back to color analysis
        hue = features.get('hue_mean', 0)
        b_yellow = features.get('lab_b_mean', 0)
        saturation = features.get('saturation_mean', 0)
        
        score_22k = 0
        score_18k = 0
        score_14k = 0
        score_plated = 0
        
        # Hue scoring (OpenCV scale 0-180; gold is 18-38°)
        if 18 <= hue <= 28:
            score_22k += 0.4
        elif 22 <= hue <= 32:
            score_18k += 0.4
        elif 25 <= hue <= 38:
            score_14k += 0.4
        elif 10 <= hue <= 18 or 38 <= hue <= 45:
            # Adjacent ranges - lower confidence gold
            score_18k += 0.2
            score_14k += 0.2
        else:
            score_plated += 0.3
        
        # b_lab scoring (yellow intensity, scale 0-255 with 128 = neutral)
        if b_yellow > 150:
            score_22k += 0.3
        elif b_yellow > 130:
            score_18k += 0.3
        elif b_yellow > 110:
            score_14k += 0.3
        else:
            score_plated += 0.3
        
        # Saturation scoring (gold should be moderately saturated)
        if saturation > 100:
            score_22k += 0.3
            score_18k += 0.2
        elif saturation > 70:
            score_18k += 0.3
            score_14k += 0.2
        elif saturation > 40:
            score_14k += 0.3
        else:
            score_plated += 0.2
        
        # Pick winner
        scores = {
            '22K-24K': score_22k,
            '18K-22K': score_18k,
            '14K-18K': score_14k,
            'Suspicious/Plated': score_plated
        }
        winner = max(scores, key=scores.get)
        confidence = scores[winner] / sum(scores.values()) if sum(scores.values()) > 0 else 0.4
        
        return {
            'purity_band': winner,
            'confidence': float(confidence),
            'method': 'color_analysis',
            'all_scores': scores,
            'reasoning': f"Color analysis: hue={hue:.1f}°, b={b_yellow:.0f}, sat={saturation:.0f}"
        }