"""Final decision engine — converts fusion results to NBFC actionable decision."""


class DecisionEngine:
    """Convert fusion results to PRE_APPROVE / VERIFY / REJECT decision.
    
    Decision logic priorities (in order):
    1. HIGH risk + suspicious purity → REJECT (clear fraud)
    2. LOW risk + HIGH confidence + good purity → PRE_APPROVE
    3. Anything ambiguous → VERIFY (human check)
    
    Key principle: Low confidence ≠ Reject. It means "needs human verification."
    """
    
    def __init__(self):
        self.thresholds = {
            'pre_approve': {'min_confidence': 0.65, 'max_risk': 0.30},
            'verify':      {'min_confidence': 0.30, 'max_risk': 0.65},
        }
    
    def compute_risk_score(self, fusion_result, fraud_signals):
        """Compute risk score in [0, 1]. Higher = more risky."""
        risk = 0.0
        
        # Low confidence increases risk slightly (not heavily)
        risk += (1 - fusion_result['final_confidence']) * 0.2
        
        # Cross-validation failure is significant
        if not fusion_result.get('cross_validation_passed', True):
            risk += 0.3
        
        # Suspicious/Plated classification is the strongest risk signal
        purity = fusion_result.get('final_purity', '')
        if 'Suspicious' in purity or 'Plated' in purity:
            risk += 0.5
        
        # Each fraud signal adds risk
        if fraud_signals:
            risk += min(0.4, len(fraud_signals) * 0.15)
        
        return min(risk, 1.0)
    
    def decide(self, fusion_result, fraud_signals=None, declared_weight=None, estimated_weight=None):
        """Final decision: PRE_APPROVE / VERIFY / REJECT.
        
        Decision priority:
        1. Clear fraud signals → REJECT
        2. Strong consistent gold detection → PRE_APPROVE  
        3. Everything ambiguous → VERIFY
        """
        if fraud_signals is None:
            fraud_signals = []
        
        # Weight mismatch fraud check
        if declared_weight and estimated_weight:
            mismatch_ratio = abs(declared_weight - estimated_weight) / max(declared_weight, 1)
            if mismatch_ratio > 0.5:
                fraud_signals.append('weight_mismatch')
        
        risk_score = self.compute_risk_score(fusion_result, fraud_signals)
        confidence = fusion_result['final_confidence']
        purity = fusion_result.get('final_purity', '')
        
        # ========== DECISION LOGIC ==========
        
        # REJECT only if HIGH risk OR confirmed plated/suspicious
        if risk_score >= 0.65:
            decision = 'REJECT'
            risk_level = 'HIGH'
            color = 'red'
            message = "Strong fraud or quality concerns detected. Recommend decline."
        
        # PRE_APPROVE only if LOW risk AND decent confidence AND gold-like
        elif risk_score < 0.30 and confidence >= 0.65 and 'Suspicious' not in purity and 'Plated' not in purity:
            decision = 'PRE_APPROVE'
            risk_level = 'LOW'
            color = 'green'
            message = "Strong consistent signals. Eligible for instant pre-approval."
        
        # Everything else → VERIFY (this is the SAFE default)
        else:
            decision = 'VERIFY'
            risk_level = 'MEDIUM'
            color = 'orange'
            
            # Provide specific reason
            if confidence < 0.5:
                message = "Confidence is moderate. Physical verification recommended."
            elif 'Suspicious' in purity:
                message = "Color analysis inconclusive. Visual inspection recommended."
            else:
                message = "Some signals require physical verification before disbursement."
        
        return {
            'decision': decision,
            'risk_level': risk_level,
            'risk_score': float(risk_score),
            'confidence': float(confidence),
            'message': message,
            'color': color,
            'fraud_signals': fraud_signals,
        }