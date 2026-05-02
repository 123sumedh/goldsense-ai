"""
Synthetic Audio Generator for GoldSense AI Tap-Test Demo
 
Generates realistic acoustic signatures for different metal types based on
published research on material density and acoustic properties.
 
Physics basis:
- Pure gold (Au, density 19.3 g/cm³) → low fundamental freq, fast decay, few harmonics
- Plated/alloy → higher pitch, slow ringing decay, many harmonics
- Hollow items → echo, longer ring, complex harmonics
 
Usage:
    python generate_test_audio.py
    
Outputs to: data/samples/audio/
"""
 
import numpy as np
import os
import sys
 
try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile...")
    os.system(f"{sys.executable} -m pip install soundfile --quiet")
    import soundfile as sf
 
 
SAMPLE_RATE = 44100
 
 
def generate_tap(fundamental_hz, decay_rate, duration_ms=250, harmonics=None, sr=SAMPLE_RATE):
    """Generate a single CLEAN tap with physics-based parameters."""
    n_samples = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)
    
    signal = np.sin(2 * np.pi * fundamental_hz * t) * np.exp(-decay_rate * t)
    
    if harmonics:
        for multiplier, amplitude in harmonics:
            harmonic_freq = fundamental_hz * multiplier
            harmonic_decay = decay_rate * (0.7 + 0.3 * multiplier)
            signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t) * np.exp(-harmonic_decay * t)
    
    # 3ms attack
    attack_samples = int(0.003 * sr)
    if attack_samples < len(signal):
        attack = np.linspace(0, 1, attack_samples)
        signal[:attack_samples] *= attack
    
    # Normalize to peak 0.7
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    
    return signal.astype(np.float32)
 
 
def quiet_silence(duration_s, sr=SAMPLE_RATE):
    """Quiet silence with very subtle ambience."""
    return (np.random.randn(int(sr * duration_s)) * 0.0008).astype(np.float32)
 
 
def room_noise(duration_s, sr=SAMPLE_RATE):
    """Realistic room noise: fan + ambient."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n)
    noise = 0.025 * np.sin(2 * np.pi * 60 * t)
    noise += 0.018 * np.sin(2 * np.pi * 120 * t)
    noise += 0.010 * np.sin(2 * np.pi * 200 * t)
    noise += np.random.randn(n) * 0.012
    return noise.astype(np.float32)
 
 
def create_recording(tap_func, num_taps=4, gap_s=0.7, pre_s=0.6, post_s=0.4, sr=SAMPLE_RATE, add_noise=False):
    """Realistic multi-tap recording: [silence][tap][gap][tap]...[silence]"""
    segments = [quiet_silence(pre_s, sr)]
    for i in range(num_taps):
        segments.append(tap_func())
        if i < num_taps - 1:
            segments.append(quiet_silence(gap_s, sr))
    segments.append(quiet_silence(post_s, sr))
    
    audio = np.concatenate(segments)
    
    if add_noise:
        audio = audio + room_noise(len(audio) / sr, sr)
    
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = audio / max_val * 0.85
    
    return audio.astype(np.float32)
 
 
# ============ METAL SIGNATURES ============
 
def gold_22k_tap():
    """22K Gold: low pitch, fast decay, few harmonics."""
    return generate_tap(
        fundamental_hz=2800, decay_rate=22, duration_ms=180,
        harmonics=[(2.1, 0.20), (3.0, 0.08)]
    )
 
 
def gold_18k_tap():
    """18K Gold: slightly higher pitch."""
    return generate_tap(
        fundamental_hz=3400, decay_rate=17, duration_ms=220,
        harmonics=[(2.0, 0.25), (3.1, 0.12), (4.0, 0.05)]
    )
 
 
def silver_tap():
    """Sterling Silver: brighter ring."""
    return generate_tap(
        fundamental_hz=4200, decay_rate=11, duration_ms=300,
        harmonics=[(2.0, 0.35), (3.0, 0.20), (4.1, 0.10)]
    )
 
 
def plated_fake_tap():
    """Plated brass/copper: HIGH pitch, SLOW decay, MANY harmonics (ringy)."""
    return generate_tap(
        fundamental_hz=6800, decay_rate=6, duration_ms=420,
        harmonics=[(1.5, 0.30), (2.0, 0.40), (2.7, 0.28), (3.3, 0.20), (4.1, 0.15)]
    )
 
 
def hollow_gold_tap():
    """Hollow gold-plated: cavity resonance."""
    return generate_tap(
        fundamental_hz=5500, decay_rate=5, duration_ms=480,
        harmonics=[(1.3, 0.30), (1.8, 0.40), (2.5, 0.30), (3.4, 0.22)]
    )
 
 
# ============ MAIN ============
 
def main():
    output_dir = os.path.join(os.getcwd(), "data", "samples", "audio")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GOLDSENSE AI — SYNTHETIC AUDIO GENERATOR")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    print("Generating physics-based acoustic signatures:")
    print()
    print("  Material      | Pitch (Hz) | Decay      | Character")
    print("  --------------|------------|------------|---------------")
    print("  22K Gold      | 2800 (low) | FAST       | Clean, few harmonics")
    print("  18K Gold      | 3400       | FAST       | Slight overtones")
    print("  Silver        | 4200       | Medium     | Brighter ring")
    print("  Plated/Fake   | 6800 (high)| SLOW       | RINGY (many harmonics)")
    print("  Hollow gold   | 5500       | Very slow  | Cavity resonance")
    print()
    
    files = [
        ("gold_22k_tap.wav",     gold_22k_tap,     False, "22K Gold (clean tap)"),
        ("gold_18k_tap.wav",     gold_18k_tap,     False, "18K Gold"),
        ("silver_tap.wav",       silver_tap,       False, "Sterling Silver"),
        ("plated_fake_tap.wav",  plated_fake_tap,  False, "Plated/Fake (key fraud signal)"),
        ("hollow_gold_tap.wav",  hollow_gold_tap,  False, "Hollow Gold (fraud)"),
        ("noisy_gold_tap.wav",   gold_22k_tap,     True,  "22K Gold + Room Noise"),
    ]
    
    for filename, generator, with_noise, description in files:
        filepath = os.path.join(output_dir, filename)
        audio = create_recording(generator, num_taps=4, add_noise=with_noise)
        sf.write(filepath, audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filename:25s} ({duration:.1f}s, {size_kb:.0f} KB) — {description}")
    
    print()
    print("=" * 70)
    print("✓ All audio files generated!")
    print("=" * 70)
    print()
    print("DEMO COMBINATIONS:")
    print()
    print("  1. Gold ring + gold_22k_tap.wav  → VERIFY/PRE_APPROVE")
    print("  2. Silver ring + plated_fake_tap.wav  → VERIFY/REJECT")
    print("  3. Gold chain + noisy_gold_tap.wav  → robustness demo")
    print()
    print("HONEST DISCLOSURE FOR JUDGES:")
    print('  "Audio samples synthesized based on published acoustic properties')
    print('   of metals. DSP pipeline processes them identically to real')
    print('   recordings. Production needs 1000+ real recordings."')
    print()
 
 
if __name__ == "__main__":
    main()