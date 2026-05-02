"""Audio noise reduction pipeline for tap-test recordings."""
import numpy as np
from scipy.signal import butter, filtfilt


class AudioDenoiser:
    """Multi-stage audio cleaning for noisy tap-test recordings.
    
    Stages:
    1. Spectral subtraction (remove fan/AC stationary noise)
    2. Bandpass filtering (1-10 kHz, metal-tap range)
    3. Wavelet denoising (preserve sharp transients)
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def spectral_subtraction(self, audio: np.ndarray, noise_sample: np.ndarray = None) -> np.ndarray:
        """Remove stationary background noise using spectral subtraction.
        
        Args:
            audio: Input noisy audio
            noise_sample: Optional clean noise sample (e.g., 2s of silence)
        """
        try:
            import noisereduce as nr
            if noise_sample is not None and len(noise_sample) > 0:
                return nr.reduce_noise(
                    y=audio, sr=self.sr, y_noise=noise_sample,
                    stationary=False, prop_decrease=0.9
                )
            return nr.reduce_noise(
                y=audio, sr=self.sr,
                stationary=True, prop_decrease=0.85
            )
        except ImportError:
            # Fallback if noisereduce not installed
            return audio
        except Exception as e:
            print(f"⚠️  Spectral subtraction failed: {e}")
            return audio
    
    def bandpass_filter(self, audio: np.ndarray, low: float = 1000, 
                        high: float = 10000, order: int = 4) -> np.ndarray:
        """Butterworth bandpass — keep only metal-tap frequency range."""
        nyquist = self.sr / 2
        low_norm = low / nyquist
        high_norm = min(high / nyquist, 0.99)
        try:
            b, a = butter(order, [low_norm, high_norm], btype='band')
            return filtfilt(b, a, audio)
        except Exception as e:
            print(f"⚠️  Bandpass filter failed: {e}")
            return audio
    
    def wavelet_denoise(self, audio: np.ndarray, wavelet: str = 'db8', level: int = 6) -> np.ndarray:
        """Wavelet denoising — best for sharp transients like taps."""
        try:
            import pywt
            coeffs = pywt.wavedec(audio, wavelet, level=level)
            # Estimate noise from finest detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(audio)))
            # Soft thresholding (preserves transients)
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, wavelet)
            return denoised[:len(audio)]
        except ImportError:
            return audio
        except Exception as e:
            print(f"⚠️  Wavelet denoising failed: {e}")
            return audio
    
    def clean_pipeline(self, audio: np.ndarray, noise_sample: np.ndarray = None) -> dict:
        """Run full denoising pipeline, return all stages.
        
        Returns:
            dict with keys: original, spectral_subtracted, bandpass_filtered, wavelet_denoised
        """
        stages = {'original': audio.copy()}
        
        # Stage 1: Spectral subtraction
        spec_clean = self.spectral_subtraction(audio, noise_sample)
        stages['spectral_subtracted'] = spec_clean
        
        # Stage 2: Bandpass filter
        filtered = self.bandpass_filter(spec_clean)
        stages['bandpass_filtered'] = filtered
        
        # Stage 3: Wavelet denoising
        wavelet_clean = self.wavelet_denoise(filtered)
        stages['wavelet_denoised'] = wavelet_clean
        
        return stages