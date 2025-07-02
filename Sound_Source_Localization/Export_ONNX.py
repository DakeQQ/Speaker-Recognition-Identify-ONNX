#!/usr/bin/env python3
import argparse
import math
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxslim import slim

from STFT_Process import STFT_Process

# ============================================================
# GLOBAL CONSTANTS - Pre-calculated for optimization
# ============================================================

# Audio processing constants
SAMPLE_RATE = 16000                     # Audio sampling rate (Hz)
SPEED_OF_SOUND = 343.0                  # Speed of sound (m/s) for TDOA calculations
MIC_DISTANCE = 0.02                     # Distance between microphones (m)
PRE_EMPHASIZE = 0.97                    # Pre-emphasis filter coefficient
NFFT = 1024                             # Number of FFT points for frequency resolution
HOP_LENGTH = 512                        # Hop length between successive frames
WINDOW_LENGTH = 1024                    # Length of each analysis window
MAX_SIGNAL_LENGTH = 4096                # Maximum signal length to process
WINDOW_TYPE = 'kaiser'                  # Window function type for STFT
ALPHA = 0.3                             # Algorithm factor for geometric mean score
BETA = 0.6                              # Algorithm factor for arithmetic mean score
GAMMA = 0.1                             # Algorithm factor for peak-weighted score, ALPHA + BETA + GAMMA = 1.0

# Model export settings
DYNAMIC_AXES = True                     # Enable dynamic axes for ONNX export
ONNX_MODEL_PATH = "./SoundSourceLocalize.onnx"

# Pre-calculated constants for optimization
TWO_PI = 2.0 * math.pi                  # Pre-calculated 2π

# Angle grid for DOA estimation (0° = left, 90° = front, 180° = right)
ANGLE_GRID = torch.arange(0, 181, step=1, dtype=torch.uint8)

# Sector definitions for direction classification

"""
                             Cabin
========================= L ⊙-MIC-⊙ R ========================
|                              |                             |
|     Front-Left (0°-45°)      |    Front-Right (135°-180°)  |
|                              |                             |
==============================================================
|                              |                             |
|     Rear-Left (45°-90°)      |    Rear-Right (90°-135°)    |
|                              |                             |
==============================================================
"""

SECTORS = {
    0: (0.0, 45.0),     # Front-Left
    1: (45.0, 90.0),    # Rear-Left
    2: (90.0, 135.0),   # Rear-Right
    3: (135.0, 180.0)   # Front-Right
}

SECTOR_NAMES = {
    0: "Front-Left",
    1: "Rear-Left",
    2: "Rear-Right",
    3: "Front-Right"
}

# ============================================================
# SOUND TYPES AND VEHICLE ENVIRONMENTS
# ============================================================

# Extended sound types for comprehensive testing
SOUND_TYPES = {
    'tone':        {'freq_range': (500,  1500), 'energy_profile': 'tone'},
    'speech':      {'freq_range': (300,  3400), 'energy_profile': 'speech'},
    'noise':       {'freq_range': (100,  4000), 'energy_profile': 'white'},
    'chirp':       {'freq_range': (200,  2000), 'energy_profile': 'sweep'},
    'music':       {'freq_range': (100,  5000), 'energy_profile': 'poly'},
    'alarm':       {'freq_range': (800,  2000), 'energy_profile': 'beeps'},
    'impulse':     {'freq_range': (20,   8000), 'energy_profile': 'spikes'},
    'engine_rev':  {'freq_range': (60,   800),  'energy_profile': 'harm'},
    'pink_noise':  {'freq_range': (20,   8000), 'energy_profile': 'pink'},
}

# Vehicle environment noise levels (dBFS - negative values below full-scale)
VEHICLE_ENVIRONMENTS = {
    'quiet':       {'road_noise': -45, 'wind_noise': -50, 'engine_level': -40},
    'normal':      {'road_noise': -35, 'wind_noise': -40, 'engine_level': -30},
    'noisy':       {'road_noise': -25, 'wind_noise': -28, 'engine_level': -25},
    'sportscar':   {'road_noise': -30, 'wind_noise': -25, 'engine_level': -18},
    'truck':       {'road_noise': -20, 'wind_noise': -22, 'engine_level': -15},
    'convertible': {'road_noise': -28, 'wind_noise': -18, 'engine_level': -25},
    'rainy':       {'road_noise': -32, 'wind_noise': -35, 'engine_level': -28},
    'offroad':     {'road_noise': -23, 'wind_noise': -30, 'engine_level': -20},
}

ENVIRONMENT_DESCRIPTIONS = {
    'quiet':       'Quiet highway, windows up',
    'normal':      'City driving, moderate traffic',
    'noisy':       'Heavy traffic, windows down',
    'sportscar':   'High-rev sports car, spirited drive',
    'truck':       'Loaded truck, diesel rumble',
    'convertible': 'Roof down, open cabin',
    'rainy':       'Rain storm, poor asphalt',
    'offroad':     'Dirt road, bumps & gravel'
}


# ============================================================
# UTILITY FUNCTIONS - Optimized and cleaned
# ============================================================


def normalize_to_int16(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio tensor to int16 with proper scaling"""
    max_val = torch.max(torch.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * scaling_factor).to(torch.int16)


def sector_of(angle: torch.Tensor) -> torch.Tensor:
    """Determine sector ID from angle using vectorized operations"""
    angle = torch.clamp(angle, 0.0, 180.0)
    # Vectorized sector assignment - more efficient than loops
    sector_boundaries = torch.tensor([45.0, 90.0, 135.0, 180.0])
    return torch.searchsorted(sector_boundaries, angle, right=False)


def fractional_delay(sig: torch.Tensor, delay: float) -> torch.Tensor:
    """Apply fractional sample delay using frequency domain method"""
    if abs(delay) < 1e-6:
        return sig.clone()

    N = sig.numel()
    # Sanity check to prevent extreme, unphysical delays
    if abs(delay) >= N / 2:
        delay = np.sign(delay) * N / 4

    # Pre-calculate FFT frequencies and apply phase shift
    X = torch.fft.fft(sig)
    freqs = torch.fft.fftfreq(N, device=sig.device)
    shift = torch.exp(-1j * TWO_PI * freqs * delay)
    return torch.fft.ifft(X * shift).real


# ============================================================
# VEHICLE NOISE GENERATION - Optimized
# ============================================================

def generate_vehicle_noise(environment: str, duration: float, sample_rate: int) -> torch.Tensor:
    """Generate realistic vehicle environment noise with optimized calculations"""
    if environment not in VEHICLE_ENVIRONMENTS:
        raise ValueError(f"Unknown environment '{environment}'. Available: {list(VEHICLE_ENVIRONMENTS.keys())}")

    env_params = VEHICLE_ENVIRONMENTS[environment]
    n_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, n_samples)

    # Initialize stereo noise channels
    noise_left = torch.zeros(n_samples)
    noise_right = torch.zeros(n_samples)

    # Road noise generation - pre-calculated amplitudes
    road_level_db = env_params['road_noise']
    road_amplitude = 10 ** (road_level_db / 20.0)
    road_freqs = [25, 40, 60, 80, 120, 160]

    for f in road_freqs:
        if f < sample_rate / 2:
            # Add frequency variation and random phases
            f_actual = f * (1 + random.uniform(-0.1, 0.1))
            phase_l = random.uniform(0, TWO_PI)
            phase_r = random.uniform(0, TWO_PI)
            amp = road_amplitude * random.uniform(0.5, 1.0) / len(road_freqs)

            # Generate sinusoidal components
            noise_left += amp * torch.sin(TWO_PI * f_actual * t + phase_l)
            noise_right += amp * torch.sin(TWO_PI * f_actual * t + phase_r)

    # Wind/rain noise - broadband with band-pass filtering
    wind_level_db = env_params['wind_noise']
    wind_amplitude = 10 ** (wind_level_db / 20.0)
    broadband = wind_amplitude * torch.randn(n_samples)

    # Apply crude band-pass filter (200-2000 Hz) if sufficient samples
    if n_samples > 200:
        kernel_size = min(101, n_samples // 10)
        kernel_size += 1 - (kernel_size % 2)  # Ensure odd kernel size
        mid = kernel_size // 2

        # Create band-pass filter kernel using sinc functions
        tk = (torch.arange(kernel_size) - mid) / sample_rate
        f_high, f_low = 2000, 200
        kernel = f_high * torch.sinc(f_high * tk) - f_low * torch.sinc(f_low * tk)
        kernel *= torch.hann_window(kernel_size) # Apply a window to the filter
        kernel /= torch.sum(torch.abs(kernel))

        pad = kernel_size // 2
        wind_filtered = F.conv1d(broadband.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=pad).squeeze()

        # Add filtered wind noise with random scaling
        noise_left += wind_filtered * random.uniform(0.8, 1.2)
        noise_right += wind_filtered * random.uniform(0.8, 1.2)

    # Engine noise - harmonic series with modulation
    engine_db = env_params['engine_level']
    engine_amplitude = 10 ** (engine_db / 20.0)
    f0 = random.uniform(80, 120)  # Base engine frequency

    for harmonic in [1, 2, 3, 4, 6, 8]:
        f = f0 * harmonic
        if f < sample_rate / 2:
            amp = engine_amplitude / (harmonic ** 0.5) / 6
            # Add amplitude modulation for realism
            modulation = 1 + 0.2 * torch.sin(TWO_PI * random.uniform(2, 8) * t)
            phase_l = random.uniform(0, TWO_PI)
            phase_r = phase_l + random.uniform(-0.2, 0.2)

            noise_left += amp * modulation * torch.sin(TWO_PI * f * t + phase_l)
            noise_right += amp * modulation * torch.sin(TWO_PI * f * t + phase_r)

    return torch.stack([noise_left, noise_right], 0)


def apply_vehicle_environment(stereo: torch.Tensor, environment: str, snr_db: float = 15.0) -> torch.Tensor:
    """Apply vehicle environment noise to stereo signal with specified SNR"""
    if stereo.shape[0] != 2:
        raise ValueError("Stereo signal expected (2, N)")

    # Generate matching duration vehicle noise
    duration_sec = stereo.shape[1] / SAMPLE_RATE
    vehicle_noise = generate_vehicle_noise(environment, duration_sec, SAMPLE_RATE)

    # Align signal lengths
    N = min(stereo.shape[1], vehicle_noise.shape[1])
    stereo = stereo[:, :N]
    vehicle_noise = vehicle_noise[:, :N]

    # Calculate power levels and apply SNR
    signal_power = torch.mean(stereo ** 2)
    noise_power = torch.mean(vehicle_noise ** 2)

    if signal_power < 1e-10 or noise_power < 1e-10:
        return stereo

    snr_linear = 10 ** (snr_db / 10.0)
    noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))

    # Mix signal and noise, then normalize
    noisy_signal = stereo + noise_scale * vehicle_noise
    max_amplitude = torch.max(torch.abs(noisy_signal))

    return noisy_signal * 0.95 / max_amplitude if max_amplitude > 0.95 else noisy_signal


# ============================================================
# MAIN LOCALIZATION MODEL - Optimized with pre-calculations
# ============================================================

class SoundSourceLocalize(nn.Module):
    """Optimized sound source localization using enhanced MVDR beamforming"""
    def __init__(self, sample_rate: int, d_mic: float, angle_grid: torch.Tensor, nfft: int, pre_emphasis: float, alpha, beta, gamma, max_signal_len, custom_stft):
        super().__init__()
        self.custom_stft = custom_stft

        # Pre-calculated constants for efficiency
        self.inv_int16 = float(1.0 / 32768.0)
        self.pre_emphasis = float(pre_emphasis)

        # Pre-calculate steering vectors for all frequencies and angles
        freqs = torch.fft.rfftfreq(nfft, 1.0 / sample_rate)  # Frequency bins
        tau = -(d_mic * torch.cos(torch.deg2rad(angle_grid))) / SPEED_OF_SOUND  # Time delays
        phase_matrix = TWO_PI * tau.unsqueeze(1) * freqs.unsqueeze(0)  # [Angles, Freqs]

        # Register pre-calculated steering vectors (real/imaginary parts)
        self.register_buffer('steer_real', torch.cos(phase_matrix))
        self.register_buffer('steer_imag', torch.sin(phase_matrix))
        self.register_buffer('angle_grid', angle_grid)

        # Enhanced frequency weighting - combines multiple strategies
        self._setup_frequency_weighting(freqs, d_mic, sample_rate)

        # MVDR combination weights (could be made learnable)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # Pre-calculated exponential weights for temporal smoothing
        weights = torch.exp(torch.linspace(-2, 0, max_signal_len, dtype=torch.float32))
        self.register_buffer('temporal_weights', (weights / weights.sum()).view(1, 1, -1))

    def _setup_frequency_weighting(self, freqs: torch.Tensor, d_mic: float, sample_rate: int):
        """Pre-calculate optimal frequency weighting for MVDR"""
        # Strategy 1: Spatial resolution weighting
        wavelengths = SPEED_OF_SOUND / (freqs + 1e-6)
        spatial_resolution = d_mic / wavelengths
        spatial_weight = torch.sigmoid(4 * (spatial_resolution - 0.1))

        # Strategy 2: Mid-frequency emphasis
        freq_weight = 1.0 - torch.exp(-((freqs - sample_rate / 6) ** 2) / (2 * (sample_rate / 4) ** 2))

        # Strategy 3: Anti-aliasing weighting
        aliasing_weight = torch.where(spatial_resolution < 0.5, torch.ones_like(freqs), torch.exp(-2 * (spatial_resolution - 0.5) ** 2))

        # Combine and normalize weighting strategies
        combined_weight = (spatial_weight * freq_weight * aliasing_weight).unsqueeze(0)
        self.register_buffer('freq_weight', combined_weight / (combined_weight.sum() + 1e-6))

    def _enhanced_mvdr_spectrum(self, p_L: torch.Tensor, p_R: torch.Tensor, r_LR: torch.Tensor, i_LR: torch.Tensor) -> torch.Tensor:
        """Enhanced MVDR spectrum calculation with multiple integration strategies"""
        # Temporal smoothing with pre-calculated exponential weights
        weights = self.temporal_weights[..., :p_L.shape[-1]]

        # Compute smoothed covariance matrix elements
        R_LL = (p_L * weights).sum(dim=2)
        R_RR = (p_R * weights).sum(dim=2)
        R_LR_r = (r_LR * weights).sum(dim=2)
        R_LR_i = (i_LR * weights).sum(dim=2)

        # Diagonal loading for numerical stability
        trace = (R_LL + R_RR) * 0.5

        # Robust matrix inversion with conditioning
        determinant = R_LL * R_RR - (R_LR_r ** 2 + R_LR_i ** 2)
        determinant = determinant + 1e-6 * trace  # Diagonal loading

        inv_det = 1.0 / determinant
        iR_LL = R_RR * inv_det
        iR_RR = R_LL * inv_det
        iR_LR_r = -R_LR_r * inv_det  # Note the negative sign
        iR_LR_i = -R_LR_i * inv_det

        # MVDR beamformer power calculation for all angles
        quadratic_form = iR_LR_r * self.steer_real + iR_LR_i * self.steer_imag
        quadratic_form = iR_LL + iR_RR + quadratic_form + quadratic_form

        mvdr_power = 1.0 / (quadratic_form + 1e-6)  # Avoid division by zero

        # Multiple integration strategies for robust estimation
        log_power = torch.log(mvdr_power + 1e-6)
        weighted_log = (log_power * self.freq_weight).sum(-1)
        geometric_score = torch.exp(weighted_log)

        arithmetic_score = (mvdr_power * self.freq_weight).sum(-1)

        peak_weight = torch.softmax(mvdr_power + mvdr_power, dim=-1)
        peak_score = (mvdr_power * peak_weight * self.freq_weight).sum(-1)

        # Weighted combination of strategies
        final_score = self.alpha * geometric_score + self.beta * arithmetic_score + self.gamma * peak_score

        return final_score

    def forward(self, mic_wav_L: torch.ShortTensor, mic_wav_R: torch.ShortTensor) -> torch.Tensor:
        """Forward pass: convert audio to DOA estimate"""
        # Fused preprocessing: normalization, DC removal, pre-emphasis
        mic_wav_L = mic_wav_L * self.inv_int16
        mic_wav_R = mic_wav_R * self.inv_int16

        # Remove DC offset
        mic_wav_L = mic_wav_L - torch.mean(mic_wav_L, dim=-1, keepdim=True)
        mic_wav_R = mic_wav_R - torch.mean(mic_wav_R, dim=-1, keepdim=True)

        # Apply pre-emphasis filter
        mic_wav_L = torch.cat([mic_wav_L[:, :, :1], mic_wav_L[:, :, 1:] - self.pre_emphasis * mic_wav_L[:, :, :-1]], dim=-1)
        mic_wav_R = torch.cat([mic_wav_R[:, :, :1], mic_wav_R[:, :, 1:] - self.pre_emphasis * mic_wav_R[:, :, :-1]], dim=-1)

        # STFT computation
        r_L, i_L = self.custom_stft(mic_wav_L, 'constant')
        r_R, i_R = self.custom_stft(mic_wav_R, 'constant')

        # Compute power and cross-spectra
        p_L = r_L * r_L + i_L * i_L
        p_R = r_R * r_R + i_R * i_R
        r_LR = r_L * r_R + i_L * i_R
        i_LR = i_L * r_R - r_L * i_R

        # Enhanced MVDR processing
        mvdr_scores = self._enhanced_mvdr_spectrum(p_L, p_R, r_LR, i_LR)

        # Return estimated angle
        max_indices = torch.argmax(mvdr_scores, dim=-1)
        return self.angle_grid[max_indices]


# ============================================================
# SIGNAL GENERATION - Optimized and extended
# ============================================================
def _apply_fade(signal: torch.Tensor, fade_ms: float = 6.0, sr: int = SAMPLE_RATE):
    fade_len = int(sr * fade_ms * 1e-3)
    if fade_len == 0 or fade_len * 2 >= signal.numel():
        return signal
    window = torch.sin(torch.linspace(0, math.pi / 2, fade_len)) ** 2
    signal[:fade_len] *= window            # fade-in
    signal[-fade_len:] *= window.flip(0)   # fade-out
    return signal


def generate_test_signal(signal_type: str,
                         duration: float,
                         sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Produce a 1-D torch tensor in the range [-1,1] that mimics a variety
    of acoustic scenes.  All signals are returned at −6 dBFS head-room and
    include a short fade-in/out to avoid spectral splatter.
    """
    n = int(duration * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    two_pi_t = TWO_PI * t

    if signal_type == 'tone':
        f = random.uniform(500, 1500)
        signal = torch.sin(two_pi_t * f)

    elif signal_type == 'speech':
        # 1. Voiced excitation: impulse train with jitter (same as before)
        f0 = random.uniform(100, 230)      # base pitch
        jitter = 0.03                      # 3 % jitter
        pulse_period = int(sample_rate / f0)
        excitation = torch.zeros_like(t)
        idx = torch.arange(0, n, pulse_period, dtype=torch.long)
        jitter_samples = (torch.randn_like(idx, dtype=torch.float32) * jitter * pulse_period).long()
        idx = (idx + jitter_samples).clamp(0, n - 1).unique(sorted=True)
        excitation[idx] = 1.0

        # 2. Vocal tract filter: Sum of formant resonators in freq domain
        n_fft = 2 * (n // 2 + 1) # Ensure FFT size is appropriate
        freqs = torch.fft.rfftfreq(n_fft, 1./sample_rate)
        
        # Typical formant frequencies (Hz) and bandwidths (Hz) for a vowel
        formants = [(500, 60), (1500, 90), (2500, 120), (3500, 150)]
        vocal_tract_response = torch.zeros_like(freqs, dtype=torch.complex64)

        for fc, bw in formants:
            # Create a simple resonant peak for each formant (Lorentzian shape)
            resonance = (bw / 2)**2 / ((freqs - fc)**2 + (bw / 2)**2)
            vocal_tract_response += resonance
            
        # 3. Apply filter to excitation via frequency domain convolution
        excitation_fft = torch.fft.rfft(excitation, n=n_fft)
        signal_fft = excitation_fft * vocal_tract_response
        signal = torch.fft.irfft(signal_fft, n=n_fft)[:n] # inverse FFT and trim to original length
        
        signal = signal / (signal.abs().max() + 1e-6)

    elif signal_type == 'chirp':
        f0, f1 = 200.0, min(sample_rate / 2 - 200, 4000.0)
        k = (f1 - f0) / duration
        signal = torch.sin(two_pi_t * (f0 + 0.5 * k * t))     # analytic formula

    elif signal_type == 'noise':
        signal = torch.randn(n)

    elif signal_type == 'pink_noise':
        # --- Voss–McCartney 1/f noise algorithm ---
        # This algorithm generates pink noise by summing several octaves of
        # up-sampled white noise.
        num_rows = int(math.ceil(math.log2(n)))  # enough octaves to cover N
        noise = torch.zeros(n)
        for r in range(num_rows):
            step = 1 << r  # 1, 2, 4, 8, ...
            # Create ⌈N / step⌉ random values and up-sample by repeating
            rand = torch.randn((n + step - 1) // step)
            noise += rand.repeat_interleave(step)[:n]
        signal = noise / num_rows  # normalise power

    elif signal_type == 'alarm':
        beep_len = int(sample_rate * 0.25)                     # 250 ms beeps
        gap_len  = beep_len
        f = 1400
        pattern = torch.cat([torch.ones(beep_len), torch.zeros(gap_len)])
        pattern = pattern.repeat(math.ceil(n / pattern.numel()))[:n]
        signal = torch.sin(two_pi_t * f) * pattern

    elif signal_type == 'impulse':
        signal = torch.zeros(n)
        step = HOP_LENGTH                                      # place on STFT frame
        indices = torch.arange(step // 2, n, step)
        signal[indices] = torch.sign(torch.randn(indices.size()))

    elif signal_type == 'music':
        freqs = [261.63, 329.63, 392.00]                       # C-major triad
        trem = 0.6 + 0.4 * torch.sin(two_pi_t * 3)
        signal = sum(torch.sin(two_pi_t * f) for f in freqs) * trem / len(freqs)

    elif signal_type == 'engine_rev':
        f0 = random.uniform(50, 120)
        rev_curve = torch.cat([
            torch.linspace(1.0, 3.0, int(n * 0.30)),
            3.0 * torch.ones(n - int(n * 0.30))  # Hold revs
        ])
        signal = sum(torch.sin(two_pi_t * h * f0 * rev_curve) / h
                     for h in range(1, 6))

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    # --- tidy-up --------------------------------------------------------
    signal = _apply_fade(signal)                      # kill hard edges
    signal = signal / (signal.abs().max() + 1e-9)     # full-scale
    signal *= 0.5                                     # −6 dBFS head-room
    return signal


def create_stereo_scene(source: torch.Tensor, doa: float, d_mic: float, sample_rate: int, vehicle_env: Optional[str] = None, snr_db: float = 15) -> torch.Tensor:
    """Create stereo scene with proper TDOA and optional vehicle noise"""
    doa = np.clip(doa, 0, 180)

    # Calculate time delay and apply to appropriate channel
    tau = d_mic * math.cos(math.radians(doa)) / SPEED_OF_SOUND
    delay_samples = tau * sample_rate

    if delay_samples >= 0:
        left_channel = fractional_delay(source, delay_samples)
        right_channel = source.clone()
    else:
        left_channel = source.clone()
        right_channel = fractional_delay(source, -delay_samples)

    # Create stereo signal
    stereo = torch.stack([left_channel, right_channel], 0)

    # Add environment or basic noise
    if vehicle_env:
        stereo = apply_vehicle_environment(stereo, vehicle_env, snr_db)
    else:
        stereo += 0.01 * torch.randn_like(stereo)

    # Final normalization
    max_amplitude = torch.max(torch.abs(stereo))
    return stereo * 0.95 / max_amplitude if max_amplitude > 0.95 else stereo


# ============================================================
# COMPREHENSIVE TEST SUITE
# ============================================================

def run_systematic_tests(model: SoundSourceLocalize, tests_per_type: int = 10, verbose: bool = False) -> Dict:
    """Run comprehensive systematic tests across all sound types and environments"""
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE SOUND SOURCE LOCALIZATION TEST SUITE")
    print("=" * 80)

    # Initialize random seeds for reproducibility
    random.seed(9527)
    torch.manual_seed(9527)

    sound_types = list(SOUND_TYPES.keys())
    environments = list(VEHICLE_ENVIRONMENTS.keys()) + [None]

    # Initialize results tracking
    results = {
        'total_tests': 0, 'correct_sector': 0, 'total_error': 0.0, 'failed_tests': 0,
        'by_env': {str(env): {'correct': 0, 'total': 0, 'failed': 0, 'errors': []} for env in environments},
        'confusion_matrix': torch.zeros(len(SECTORS), len(SECTORS), dtype=torch.long)
    }

    test_id = 0

    # Run tests for each sound type
    for sound_type in sound_types:
        for test_idx in range(tests_per_type):
            environment = environments[test_idx % len(environments)]
            env_key = str(environment)
            test_id += 1

            try:
                # Generate test parameters
                sector_id = test_idx % len(SECTORS)
                angle_min, angle_max = SECTORS[sector_id]
                true_angle = random.uniform(angle_min + 0.1 * (angle_max - angle_min), angle_max - 0.1 * (angle_max - angle_min))

                # Generate test signal and scene
                test_signal = generate_test_signal(sound_type, 2.0, SAMPLE_RATE)
                snr_db = random.uniform(5, 15)
                stereo_scene = create_stereo_scene(test_signal, true_angle, MIC_DISTANCE, SAMPLE_RATE, vehicle_env=environment, snr_db=snr_db)

                # Convert to int16 and run inference
                stereo_scene = normalize_to_int16(stereo_scene.unsqueeze(0))
                estimated_doa = float(model(stereo_scene[:, [0]], stereo_scene[:, [1]]).item())

                # Calculate metrics
                angular_error = abs(estimated_doa - true_angle)
                true_sector = sector_of(torch.tensor([true_angle]))[0].item()
                estimated_sector = sector_of(torch.tensor([estimated_doa]))[0].item()
                sector_correct = (true_sector == estimated_sector)

                # Update statistics
                results['total_tests'] += 1
                results['correct_sector'] += int(sector_correct)
                results['total_error'] += angular_error

                env_stats = results['by_env'][env_key]
                env_stats['total'] += 1
                env_stats['correct'] += int(sector_correct)
                env_stats['errors'].append(angular_error)
                results['confusion_matrix'][true_sector, estimated_sector] += 1

                # Verbose output
                if verbose:
                    true_sector_name = SECTOR_NAMES[true_sector]
                    est_sector_name = SECTOR_NAMES[estimated_sector]
                    status = "✓" if sector_correct else "✗"
                    env_label = environment if environment else "clean"
                    snr_info = f"(SNR {snr_db:.1f}dB)" if environment else ""

                    print(f"[{test_id:3d}] {sound_type:<12} {env_label:<12} {status} "
                          f"{true_angle:6.1f}° → {estimated_doa:6.1f}° "
                          f"(err {angular_error:4.1f}°) [{true_sector_name} → {est_sector_name}] {snr_info}")

            except Exception as e:
                results['failed_tests'] += 1
                results['by_env'][env_key]['failed'] += 1
                if verbose:
                    print(f"[{test_id:3d}] {sound_type:<12} {env_key:<12} ✗ FAILED: {e}")

    # Print comprehensive summary
    _print_test_summary(results)
    return results


def _print_test_summary(results: Dict):
    """Print detailed test results summary"""
    successful_tests = results['total_tests']

    if successful_tests > 0:
        accuracy = 100 * results['correct_sector'] / successful_tests
        mean_error = results['total_error'] / successful_tests

        print("\n" + "=" * 80)
        print(f"TOTAL TESTS: {successful_tests}   FAILED: {results['failed_tests']}")
        print(f"SECTOR ACCURACY: {accuracy:.1f}%   MEAN ANGULAR ERROR: {mean_error:.1f}°")
        print("\nPERFORMANCE BY ENVIRONMENT:")
        print("Environment      Accuracy%    Mean Error°    Tests")
        print("-" * 50)

        for env_key, stats in results['by_env'].items():
            if stats['total'] > 0:
                env_accuracy = 100 * stats['correct'] / stats['total']
                env_mean_error = sum(stats['errors']) / len(stats['errors']) if stats['errors'] else 0.0
                env_label = env_key if env_key != "None" else "clean"
                print(f"{env_label:<15} {env_accuracy:8.1f}  {env_mean_error:11.1f}   {stats['total']:5d}")

        # Print confusion matrix
        print("\nCONFUSION MATRIX (True Sector \\ Estimated Sector):")
        col_width = max(len(name) for name in SECTOR_NAMES.values()) + 2

        # Header row
        print(" " * col_width + "".join(f"{SECTOR_NAMES[j]:>{col_width}}" for j in range(len(SECTORS))))
        print("-" * (col_width * (len(SECTORS) + 1)))

        # Data rows
        for i in range(len(SECTORS)):
            print(f"{SECTOR_NAMES[i]:<{col_width}}", end="")
            for j in range(len(SECTORS)):
                print(f"{results['confusion_matrix'][i, j].item():>{col_width}}", end="")
            print()
    else:
        print("\nNO SUCCESSFUL TESTS COMPLETED!")


# ============================================================
# MAIN FUNCTION - CLI and model export
# ============================================================

def main() -> int:
    """Main function with argument parsing and model testing/export"""
    parser = argparse.ArgumentParser(
        description="Advanced 2-microphone sound source localization system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tests-per-type', type=int, default=20, help='Number of tests per sound type')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose test output')
    args = parser.parse_args()

    # Initialize STFT processor and localization model
    print("Initializing STFT processor and localization model...")
    custom_stft = STFT_Process(
        model_type='stft_B',
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        max_frames=0,
        window_type=WINDOW_TYPE
    ).eval()

    model = SoundSourceLocalize(
        SAMPLE_RATE,
        MIC_DISTANCE,
        ANGLE_GRID,
        NFFT,
        PRE_EMPHASIZE,
        ALPHA,
        BETA,
        GAMMA,
        MAX_SIGNAL_LENGTH,
        custom_stft
    ).to('cpu')

    # Run comprehensive test suite
    print("Running comprehensive test suite...")
    test_results = run_systematic_tests(
        model,
        tests_per_type=args.tests_per_type,
        verbose=not args.quiet
    )

    # Export model to ONNX format
    print(f"\nExporting model to ONNX format.")
    dummy_mic_L = torch.ones((1, 1, MAX_SIGNAL_LENGTH), dtype=torch.int16)
    dummy_mic_R = torch.ones((1, 1, MAX_SIGNAL_LENGTH), dtype=torch.int16)

    torch.onnx.export(
        model,
        (dummy_mic_L, dummy_mic_R),
        ONNX_MODEL_PATH,
        input_names=['audio_mic_L', 'audio_mic_R'],
        output_names=['estimated_angle_degrees'],
        dynamic_axes={
            'audio_mic_L': {2: 'audio_length'},
            'audio_mic_R': {2: 'audio_length'},
        } if DYNAMIC_AXES else None,
        opset_version=17,
        do_constant_folding=True
    )
    slim(
        model=ONNX_MODEL_PATH,
        output_model=ONNX_MODEL_PATH,
        no_shape_infer=False,            # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=False,
        verbose=False
    )
    print(f"\nModel successfully exported to {ONNX_MODEL_PATH}")

    # Determine exit code based on test performance
    if test_results['total_tests'] > 0:
        accuracy = 100 * test_results['correct_sector'] / test_results['total_tests']

        if test_results['failed_tests'] == 0 and accuracy > 80:
            exit_code = 0  # Excellent performance
        elif accuracy > 65:
            exit_code = 0  # Acceptable performance
        else:
            exit_code = 1  # Poor performance

        print(f"\nTest suite completed with {accuracy:.1f}% accuracy")
    else:
        exit_code = 1
        print("\nTest suite failed - no successful tests!")

    return exit_code


if __name__ == '__main__':
    exit(main())
