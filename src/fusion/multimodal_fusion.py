"""Multimodal fusion combining vision, audio, and customer data."""
import numpy as np


class MultimodalFusion:
    """Quality-aware fusion of vision + audio + hallmark + customer signals."""
    
    def __init__(self):
        self.weights = {
            'vision': 0.5,
            'audio': 0.25,
            'hallmark': 0.25,
        }
    
    def adjust_weights_for_quality(self, vision_quality, audio_quality, has_hallmark):
        """Down-weight modalities with poor quality or missing signals."""
        weights = self.weights.copy()
        
        # If hallmark not detected, redistribute its weight
        if not has_hallmark:
            redistribute = weights['hallmark']
            weights['vision'] += redistribute * 0.7
            weights['audio'] += redistribute * 0.3
            weights['hallmark'] = 0.05  # Tiny remaining weight
        
        # If audio is poor quality, down-weight it
        if audio_quality < 0.4:
            weights['vision'] += weights['audio'] * 0.7
            if has_hallmark:
                weights['hallmark'] += weights['audio'] * 0.3
            weights['audio'] = 0.05
        
        # If vision quality is poor, increase others
        if vision_quality < 0.4:
            if has_hallmark:
                weights['hallmark'] *= 1.5
            weights['vision'] *= 0.5
        
        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def fuse(self, vision_result, audio_result, hallmark_result, quality_score):
        """Combine all modality outputs into final decision."""
        vision_conf = vision_result.get('confidence', 0) if vision_result else 0
        audio_conf = audio_result.get('confidence', 0) if audio_result else 0
        has_hallmark = bool(hallmark_result and hallmark_result.get('detected'))
        hallmark_conf = 0.9 if has_hallmark else 0
        
        weights = self.adjust_weights_for_quality(
            vision_quality=quality_score / 100,
            audio_quality=audio_conf,
            has_hallmark=has_hallmark
        )
        
        # Weighted final confidence
        final_confidence = (
            weights['vision'] * vision_conf +
            weights['audio'] * audio_conf +
            weights['hallmark'] * hallmark_conf
        )
        
        # Boost confidence if vision alone is strong (e.g., 70%+)
        if vision_conf > 0.6 and not audio_result:
            final_confidence = max(final_confidence, vision_conf * 0.85)
        
        # Determine final purity decision
        if has_hallmark and hallmark_result.get('declared_purity'):
            final_purity = hallmark_result['declared_purity']
            purity_source = 'hallmark'
        elif vision_result:
            final_purity = vision_result.get('purity_band', 'Unknown')
            purity_source = 'color_analysis'
        else:
            final_purity = 'Unknown'
            purity_source = 'none'
        
        # Cross-validation
        cross_validation_passed = True
        if audio_result and audio_result.get('classification') == 'alloy-like':
            if 'Plated' not in final_purity and 'Suspicious' not in final_purity:
                final_confidence *= 0.7
                cross_validation_passed = False
        
        return {
            'final_purity': final_purity,
            'final_confidence': float(min(final_confidence, 0.99)),
            'purity_source': purity_source,
            'modality_weights': weights,
            'modality_confidences': {
                'vision': vision_conf,
                'audio': audio_conf,
                'hallmark': hallmark_conf
            },
            'cross_validation_passed': cross_validation_passed,
        }