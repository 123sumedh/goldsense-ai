
import numpy as np
class AudioFeatureExtractor:
    """Extract acoustic features from clean tap audio for purity classification.
    
    Designed to work on:
    - Real recordings (smartphone microphone)
    - Synthetic audio (physics-based generated samples)
    - Noisy audio (with quality-aware fusion)
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def detect_taps(self, audio: np.ndarray, max_taps: int = 5) -> list:
        """Find tap events using onset envelope peak detection.
        
        Uses librosa's onset_strength + peak_pick for reliable tap detection.
        Filters out silence false-positives by checking signal energy.
        """
        try:
            import librosa
            
            # Compute onset strength envelope
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            
            # Find peaks in onset envelope
            # wait=25 frames ≈ 290ms gap minimum (prevents double-detection of one tap)
            peaks_frames = librosa.util.peak_pick(
                onset_env,
                pre_max=10, post_max=10,
                pre_avg=20, post_avg=20,
                delta=1.0,
                wait=25
            )
            
            # Convert frame indices to sample indices
            window_size = int(0.25 * self.sr)  # 250ms window per tap
            tap_segments = []
            
            for peak_frame in peaks_frames[:max_taps + 2]:  # Get a few extra in case of false positives
                peak_sample = int(librosa.frames_to_samples(peak_frame))
                
                # Capture 30ms before peak for clean attack
                start = max(0, peak_sample - int(0.03 * self.sr))
                end = min(len(audio), start + window_size)
                segment = audio[start:end]
                
                # Quality filter: only keep segments with actual signal energy
                if len(segment) > 100 and np.max(np.abs(segment)) > 0.1:
                    tap_segments.append(segment)
                
                if len(tap_segments) >= max_taps:
                    break
            
            # Fallback if no taps detected
            if not tap_segments:
                return self._fallback_segment(audio)
            
            return tap_segments
            
        except Exception as e:
            print(f"⚠️  Tap detection error: {e}")
            return self._fallback_segment(audio)
    
    def _fallback_segment(self, audio: np.ndarray) -> list:
        """Extract loudest 250ms region as fallback."""
        window_samples = int(0.25 * self.sr)
        if len(audio) > window_samples:
            energy = np.abs(audio)
            peak_idx = int(np.argmax(energy))
            start = max(0, peak_idx - window_samples // 2)
            end = min(len(audio), start + window_samples)
            return [audio[start:end]]
        return [audio]
    
    def compute_decay_rate(self, audio: np.ndarray) -> float:
        """Fit exponential decay to envelope.
        
        Pure gold = high decay rate (fast fade)
        Plated/alloy = low decay rate (slow ring)
        """
        try:
            from scipy.signal import savgol_filter
            
            envelope = np.abs(audio)
            
            # Smooth the envelope
            if len(envelope) > 51:
                envelope = savgol_filter(envelope, 51, 3)
            
            envelope = np.maximum(envelope, 1e-10)
            
            # Find peak — decay starts after peak
            peak_idx = np.argmax(envelope)
            post_peak = envelope[peak_idx:]
            
            if len(post_peak) > 100:
                # Fit log-linear
                t = np.arange(len(post_peak))
                log_env = np.log(post_peak + 1e-10)
                coef = np.polyfit(t, log_env, 1)
                return float(-coef[0])  # Positive value
        except Exception as e:
            print(f"⚠️  Decay rate computation failed: {e}")
        
        return 0.0
    
    def average_taps(self, tap_segments: list) -> np.ndarray:
        """Coherent averaging of multiple taps for SNR boost (~√N)."""
        if not tap_segments:
            return np.zeros(int(0.25 * self.sr))
        
        if len(tap_segments) == 1:
            return tap_segments[0]
        
        try:
            max_len = max(len(t) for t in tap_segments)
            target_peak = max_len // 2
            
            aligned = []
            for tap in tap_segments:
                if len(tap) < 10:
                    continue
                
                peak_idx = np.argmax(np.abs(tap))
                offset = target_peak - peak_idx
                
                aligned_segment = np.zeros(max_len)
                if offset >= 0:
                    end = min(max_len, len(tap) + offset)
                    aligned_segment[offset:end] = tap[:end - offset]
                else:
                    start = -offset
                    end = min(max_len, len(tap) - start)
                    aligned_segment[:end] = tap[start:start + end]
                aligned.append(aligned_segment)
            
            if aligned:
                return np.mean(aligned, axis=0)
            return tap_segments[0]
        except Exception as e:
            print(f"⚠️  Averaging failed: {e}")
            return tap_segments[0]
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """Extract all relevant acoustic features from audio."""
        if len(audio) < 100:
            return self._empty_features()
        
        try:
            import librosa
            features = {}
            
            # ====== Pitch / Fundamental Frequency ======
            try:
                f0, voiced, _ = librosa.pyin(
                    audio,
                    fmin=1500,
                    fmax=10000,
                    sr=self.sr
                )
                features['f0_mean'] = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
                features['f0_std'] = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
            except Exception:
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
            
            # ====== Spectral Features ======
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
            features['brightness'] = float(np.mean(spec_centroid))
            
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
            features['rolloff'] = float(np.mean(rolloff))
            
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr'] = float(np.mean(zcr))
            
            # ====== MFCCs ======
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            
            # ====== Harmonic vs Percussive ======
            try:
                harmonic, percussive = librosa.effects.hpss(audio)
                audio_energy = float(np.sum(audio ** 2)) + 1e-10
                features['harmonic_ratio'] = float(np.sum(harmonic ** 2) / audio_energy)
                features['percussive_ratio'] = float(np.sum(percussive ** 2) / audio_energy)
            except Exception:
                features['harmonic_ratio'] = 0.5
                features['percussive_ratio'] = 0.5
            
            # ====== Temporal Features ======
            features['decay_rate'] = self.compute_decay_rate(audio)
            features['duration_ms'] = len(audio) / self.sr * 1000
            features['peak_amplitude'] = float(np.max(np.abs(audio)))
            features['rms_energy'] = float(np.sqrt(np.mean(audio ** 2)))
            
            return features
            
        except Exception as e:
            print(f"⚠️  Feature extraction failed: {e}")
            return self._empty_features()
    
    def classify_audio(self, features: dict) -> dict:
        """Classify audio as gold-like vs alloy-like based on physics."""
        f0 = features.get('f0_mean', 0)
        brightness = features.get('brightness', 0)
        decay = features.get('decay_rate', 0)
        harmonic_ratio = features.get('harmonic_ratio', 0)
        rms = features.get('rms_energy', 0)
        
        # Quality check
        if rms < 0.005:
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'reason': 'Audio signal too weak'
            }
        
        gold_score = 0
        alloy_score = 0
        reasons = []
        
        # ====== Fundamental Frequency Scoring ======
        if 1800 < f0 < 4500:
            gold_score += 0.30
            reasons.append(f"f0 in gold range ({f0:.0f}Hz)")
        elif f0 > 4500:
            alloy_score += 0.30
            reasons.append(f"f0 too high ({f0:.0f}Hz, alloy-like)")
        elif f0 == 0:
            pass  # Couldn't detect — neutral
        else:
            alloy_score += 0.15
            reasons.append(f"f0 unusually low ({f0:.0f}Hz)")
        
        # ====== Brightness Scoring ======
        if brightness < 4500:
            gold_score += 0.20
            reasons.append("low spectral brightness (gold-like)")
        elif brightness < 6000:
            gold_score += 0.10
        else:
            alloy_score += 0.25
            reasons.append("high spectral brightness (alloy-like)")
        
        # ====== Decay Rate Scoring ======
        if decay > 0.0005:
            gold_score += 0.30
            reasons.append(f"fast decay ({decay:.4f})")
        elif decay > 0.0002:
            gold_score += 0.15
            reasons.append("moderate decay")
        else:
            alloy_score += 0.25
            reasons.append("slow decay (alloy ringing)")
        
        # ====== Harmonic Ratio Scoring ======
        if harmonic_ratio < 0.40:
            gold_score += 0.20
            reasons.append("clean harmonic profile")
        elif harmonic_ratio < 0.60:
            gold_score += 0.10
            alloy_score += 0.10
        else:
            alloy_score += 0.25
            reasons.append("rich harmonics (alloy ringing)")
        
        # ====== Final Decision ======
        total = gold_score + alloy_score
        if total == 0:
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'gold_score': 0,
                'alloy_score': 0,
                'reasons': ['No discriminative features detected']
            }
        
        if gold_score > alloy_score:
            return {
                'classification': 'gold-like',
                'confidence': gold_score / total,
                'gold_score': gold_score,
                'alloy_score': alloy_score,
                'reasons': reasons
            }
        else:
            return {
                'classification': 'alloy-like',
                'confidence': alloy_score / total,
                'gold_score': gold_score,
                'alloy_score': alloy_score,
                'reasons': reasons
            }
    
    def _empty_features(self):
        """Return zero-valued features when extraction fails."""
        features = {
            'f0_mean': 0.0,
            'f0_std': 0.0,
            'brightness': 0.0,
            'rolloff': 0.0,
            'zcr': 0.0,
            'decay_rate': 0.0,
            'harmonic_ratio': 0.0,
            'percussive_ratio': 0.0,
            'duration_ms': 0.0,
            'peak_amplitude': 0.0,
            'rms_energy': 0.0,
        }
        for i in range(13):
            features[f'mfcc_{i}'] = 0.0
        return features
 