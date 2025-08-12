#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxslim import slim

# Assuming STFT_Process is in a file named STFT_Process.py
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
ANGLE_GRID = torch.arange(0, 181, step=1, dtype=torch.float32)

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

# Boundary tensor for sector lookup (constructed lazily on device)
_SECTOR_BOUNDS_CPU = torch.tensor([45.0, 90.0, 135.0, 180.1], dtype=torch.float32)

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

# ============================================================
# UTILITY FUNCTIONS - Optimized and cleaned
# ============================================================


def normalize_to_int16(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio tensor to int16 with proper scaling."""
    max_val = torch.max(torch.abs(audio))
    scaling_factor = 32767.0 / (max_val + 1e-9)
    return (audio * scaling_factor).to(torch.int16)


def _sector_boundaries_on_device(device: torch.device) -> torch.Tensor:
    """Get sector boundary tensor on the correct device without reallocating each call."""
    return _SECTOR_BOUNDS_CPU.to(device=device)


def sector_of(angle: torch.Tensor) -> torch.Tensor:
    """Determine sector ID from angle using vectorized operations."""
    angle = torch.clamp(angle, 0.0, 180.0)
    bounds = _sector_boundaries_on_device(angle.device)
    return torch.searchsorted(bounds, angle, right=False)


def sector_is_correct(true_angle: float, est_angle: float, tol_deg: float = 0.5) -> bool:
    """
    Boundary-tolerant sector comparison.
    If the true angle is within tol of a sector boundary, accept either adjacent sector.
    """
    bounds = [45.0, 90.0, 135.0]
    for b in bounds:
        if abs(true_angle - b) <= tol_deg:
            # Accept either sector around the boundary
            left_sector = int(sector_of(torch.tensor([b - 1e-3]))[0].item())
            right_sector = int(sector_of(torch.tensor([b + 1e-3]))[0].item())
            est_sector = int(sector_of(torch.tensor([est_angle]))[0].item())
            return est_sector in (left_sector, right_sector)
    # Non-boundary case: exact sector match
    s_true = int(sector_of(torch.tensor([true_angle]))[0].item())
    s_est = int(sector_of(torch.tensor([est_angle]))[0].item())
    return s_true == s_est


def fractional_delay(sig: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    """
    Apply fractional sample delay using frequency domain method.
    Delay can be a scalar or a tensor of the same size as the signal for time-varying delay.
    """
    if torch.all(torch.abs(delay) < 1e-6):
        return sig.clone()

    N = sig.numel()
    # Heuristic to prevent extreme phase wrapping from overly large delays
    delay = torch.clamp(delay, -N / 4.0, N / 4.0)

    X = torch.fft.fft(sig)
    freqs = torch.fft.fftfreq(N, device=sig.device)
    
    # If delay is a tensor, we need to compute the shift for each time step,
    # which is not directly possible this way. This function assumes a constant delay.
    # For time-varying delay, a more complex time-frequency approach would be needed.
    # Here, we will use the mean delay if a tensor is passed, as a simplification.
    if delay.numel() > 1:
        delay = torch.mean(delay) # Simplified for this implementation

    shift = torch.exp(torch.tensor(-1j, device=sig.device) * TWO_PI * freqs * delay)
    return torch.fft.ifft(X * shift).real


# ============================================================
# VEHICLE NOISE GENERATION - Optimized
# ============================================================

def generate_vehicle_noise(environment: str, duration: float, sample_rate: int) -> torch.Tensor:
    """Generate realistic vehicle environment noise with optimized calculations."""
    if environment not in VEHICLE_ENVIRONMENTS:
        raise ValueError(f"Unknown environment '{environment}'. Available: {list(VEHICLE_ENVIRONMENTS.keys())}")

    env_params = VEHICLE_ENVIRONMENTS[environment]
    n_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, n_samples, dtype=torch.float32)

    # Initialize stereo noise channels
    noise_left = torch.zeros(n_samples, dtype=torch.float32)
    noise_right = torch.zeros(n_samples, dtype=torch.float32)

    # Road noise generation - pre-calculated amplitudes
    road_level_db = env_params['road_noise']
    road_amplitude = 10 ** (road_level_db / 20.0)
    road_freqs = [25, 40, 60, 80, 120, 160]

    for f in road_freqs:
        if f < sample_rate / 2:
            f_actual = f * (1 + random.uniform(-0.1, 0.1))
            phase_l = random.uniform(0, TWO_PI)
            phase_r = random.uniform(0, TWO_PI)
            amp = road_amplitude * random.uniform(0.5, 1.0) / len(road_freqs)
            omega_t = TWO_PI * f_actual * t
            noise_left += amp * torch.sin(omega_t + phase_l)
            noise_right += amp * torch.sin(omega_t + phase_r)

    # Wind/rain noise - broadband with band-pass filtering
    wind_level_db = env_params['wind_noise']
    wind_amplitude = 10 ** (wind_level_db / 20.0)
    broadband = wind_amplitude * torch.randn(n_samples, dtype=torch.float32)

    if n_samples > 200:
        kernel_size = min(101, n_samples // 10)
        kernel_size += 1 - (kernel_size % 2)  # ensure odd
        mid = kernel_size // 2
        tk = (torch.arange(kernel_size, dtype=torch.float32) - mid) / sample_rate
        f_high, f_low = 2000.0, 200.0
        kernel = f_high * torch.sinc(f_high * tk) - f_low * torch.sinc(f_low * tk)
        kernel *= torch.hann_window(kernel_size)
        kernel /= torch.sum(torch.abs(kernel)) + 1e-9
        pad = kernel_size // 2
        wind_filtered = F.conv1d(
            broadband.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=pad
        ).squeeze()
        scale_l = random.uniform(0.8, 1.2)
        scale_r = random.uniform(0.8, 1.2)
        noise_left += wind_filtered * scale_l
        noise_right += wind_filtered * scale_r

    # Engine noise - harmonic series with modulation
    engine_db = env_params['engine_level']
    engine_amplitude = 10 ** (engine_db / 20.0)
    f0 = random.uniform(80, 120)

    for harmonic in [1, 2, 3, 4, 6, 8]:
        f = f0 * harmonic
        if f < sample_rate / 2:
            amp = engine_amplitude / (harmonic ** 0.5) / 6.0
            modulation = 1 + 0.2 * torch.sin(TWO_PI * random.uniform(2, 8) * t)
            phase_l = random.uniform(0, TWO_PI)
            phase_r = phase_l + random.uniform(-0.2, 0.2)
            omega_t = TWO_PI * f * t
            noise_left += amp * modulation * torch.sin(omega_t + phase_l)
            noise_right += amp * modulation * torch.sin(omega_t + phase_r)

    return torch.stack([noise_left, noise_right], 0)


def apply_vehicle_environment(stereo: torch.Tensor, environment: str, snr_db: float = 15.0) -> torch.Tensor:
    """Apply vehicle environment noise to stereo signal with specified SNR."""
    if stereo.shape[0] != 2:
        raise ValueError("Stereo signal expected (2, N)")

    duration_sec = stereo.shape[1] / SAMPLE_RATE
    vehicle_noise = generate_vehicle_noise(environment, duration_sec, SAMPLE_RATE)

    N = min(stereo.shape[1], vehicle_noise.shape[1])
    stereo = stereo[:, :N]
    vehicle_noise = vehicle_noise[:, :N]

    signal_power = torch.mean(stereo ** 2)
    noise_power = torch.mean(vehicle_noise ** 2)

    if signal_power < 1e-10 or noise_power < 1e-10:
        return stereo

    snr_linear = 10 ** (snr_db / 10.0)
    noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))

    noisy_signal = stereo + noise_scale * vehicle_noise
    max_amplitude = torch.max(torch.abs(noisy_signal))
    return noisy_signal * 0.95 / (max_amplitude + 1e-9) if max_amplitude > 0 else noisy_signal

# ============================================================
# MAIN LOCALIZATION MODEL - Optimized with pre-calculations
# ============================================================


class SoundSourceLocalize(nn.Module):
    """Optimized sound source localization using enhanced MVDR beamforming."""
    def __init__(
        self,
        sample_rate: int,
        d_mic: float,
        angle_grid: torch.Tensor,
        nfft: int,
        pre_emphasis: float,
        alpha: float,
        beta: float,
        gamma: float,
        max_signal_len: int,
        custom_stft: nn.Module
    ):
        super().__init__()
        self.custom_stft = custom_stft

        self.inv_int16 = float(1.0 / 32768.0)
        self.pre_emphasis = float(pre_emphasis)

        # Precompute steering vectors on the angle x frequency grid
        freqs = torch.fft.rfftfreq(nfft, 1.0 / sample_rate)
        tau = -(d_mic * torch.cos(torch.deg2rad(angle_grid))) / SPEED_OF_SOUND
        phase_matrix = TWO_PI * tau.unsqueeze(1) * freqs.unsqueeze(0)

        self.register_buffer('steer_real', torch.cos(phase_matrix))
        self.register_buffer('steer_imag', torch.sin(phase_matrix))
        self.register_buffer('angle_grid', angle_grid)

        self._setup_frequency_weighting(freqs, d_mic, sample_rate)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # Temporal weights for frame integration (decaying)
        weights = torch.exp(torch.linspace(-2, 0, max_signal_len, dtype=torch.float32))
        self.register_buffer('temporal_weights', (weights / weights.sum()).view(1, 1, -1))

    def _setup_frequency_weighting(self, freqs: torch.Tensor, d_mic: float, sample_rate: int):
        """Pre-calculate optimal frequency weighting for MVDR."""
        wavelengths = SPEED_OF_SOUND / (freqs + 1e-6)
        spatial_resolution = d_mic / wavelengths
        spatial_weight = torch.sigmoid(4 * (spatial_resolution - 0.1))

        freq_weight = 1.0 - torch.exp(-((freqs - sample_rate / 6) ** 2) / (2 * (sample_rate / 4) ** 2))

        aliasing_weight = torch.where(spatial_resolution < 0.5, torch.ones_like(freqs), torch.exp(-2 * (spatial_resolution - 0.5) ** 2))

        combined_weight = (spatial_weight * freq_weight * aliasing_weight).unsqueeze(0)
        self.register_buffer('freq_weight', combined_weight / (combined_weight.sum() + 1e-6))

    def _enhanced_mvdr_spectrum(self, p_L: torch.Tensor, p_R: torch.Tensor, r_LR: torch.Tensor, i_LR: torch.Tensor) -> torch.Tensor:
        """Enhanced MVDR spectrum calculation with multiple integration strategies."""
        num_frames = p_L.shape[-1]
        weights = self.temporal_weights[..., :num_frames]
        weights = weights / (weights.sum() + 1e-9)  # Re-normalize for current frames

        # Frame-averaged entries of the 2x2 spatial covariance
        R_LL = (p_L * weights).sum(dim=2)
        R_RR = (p_R * weights).sum(dim=2)
        R_LR_r = (r_LR * weights).sum(dim=2)
        R_LR_i = (i_LR * weights).sum(dim=2)

        trace = (R_LL + R_RR) * 0.5
        determinant = R_LL * R_RR - (R_LR_r ** 2 + R_LR_i ** 2)
        # Diagonal loading for stability
        determinant = determinant + 1e-6 * trace

        inv_det = 1.0 / (determinant + 1e-12)
        iR_LL = R_RR * inv_det
        iR_RR = R_LL * inv_det
        iR_LR_r = -R_LR_r * inv_det
        iR_LR_i = -R_LR_i * inv_det

        # Denominator ~ v^H R^-1 v for each steering vector v
        quadratic_form = 2.0 * (iR_LR_r * self.steer_real + iR_LR_i * self.steer_imag)
        denominator = iR_LL + iR_RR + quadratic_form

        mvdr_power = 1.0 / (denominator + 1e-6)

        # Three integration strategies across frequency with prior weights
        log_power = torch.log(mvdr_power + 1e-6)
        weighted_log = (log_power * self.freq_weight).sum(-1)
        geometric_score = torch.exp(weighted_log)

        arithmetic_score = (mvdr_power * self.freq_weight).sum(-1)

        peak_weight = torch.softmax(mvdr_power + mvdr_power, dim=-1)
        peak_score = (mvdr_power * peak_weight * self.freq_weight).sum(-1)

        final_score = self.alpha * geometric_score + self.beta * arithmetic_score + self.gamma * peak_score
        return final_score  # shape: (batch, angles)

    def forward(self, mic_wav_L: torch.ShortTensor, mic_wav_R: torch.ShortTensor) -> torch.Tensor:
        """Forward pass: convert audio to DOA estimate."""
        # Normalization to float
        mic_wav_L = mic_wav_L.to(torch.float32) * self.inv_int16
        mic_wav_R = mic_wav_R.to(torch.float32) * self.inv_int16

        # DC removal
        mic_wav_L = mic_wav_L - torch.mean(mic_wav_L, dim=-1, keepdim=True)
        mic_wav_R = mic_wav_R - torch.mean(mic_wav_R, dim=-1, keepdim=True)

        # Pre-emphasis filter (keep L/R separate)
        mic_wav_L = torch.cat([mic_wav_L[..., :1], mic_wav_L[..., 1:] - self.pre_emphasis * mic_wav_L[..., :-1]], dim=-1)
        mic_wav_R = torch.cat([mic_wav_R[..., :1], mic_wav_R[..., 1:] - self.pre_emphasis * mic_wav_R[..., :-1]], dim=-1)

        # Custom STFT (preserved and separate for L/R)
        r_L, i_L = self.custom_stft(mic_wav_L, 'constant')
        r_R, i_R = self.custom_stft(mic_wav_R, 'constant')

        # Power and cross terms
        p_L = r_L * r_L + i_L * i_L
        p_R = r_R * r_R + i_R * i_R
        r_LR = r_L * r_R + i_L * i_R
        i_LR = i_L * r_R - r_L * i_R

        mvdr_scores = self._enhanced_mvdr_spectrum(p_L, p_R, r_LR, i_LR)

        max_indices = torch.argmax(mvdr_scores, dim=-1)
        return self.angle_grid[max_indices]


# ============================================================
# SIGNAL GENERATION - Optimized and extended
# ============================================================
def _apply_fade(signal: torch.Tensor, fade_ms: float = 6.0, sr: int = SAMPLE_RATE):
    fade_len = int(sr * fade_ms * 1e-3)
    if fade_len == 0 or fade_len * 2 >= signal.numel():
        return signal
    window = torch.hann_window(fade_len * 2, periodic=False)[:fade_len]
    signal[:fade_len] *= window
    signal[-fade_len:] *= window.flip(0)
    return signal


def generate_test_signal(signal_type: str, duration: float, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Generates a 1D test signal of a specific type."""
    n = int(duration * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    two_pi_t = TWO_PI * t

    if signal_type == 'tone':
        f = random.uniform(500, 1500)
        signal = torch.sin(two_pi_t * f)
    elif signal_type == 'speech':
        f0 = random.uniform(100, 230)
        jitter = 0.03
        pulse_period = max(1, int(sample_rate / f0))
        excitation = torch.zeros_like(t)
        idx = torch.arange(0, n, pulse_period, dtype=torch.long)
        jitter_samples = (torch.randn_like(idx, dtype=torch.float32) * jitter * pulse_period).long()
        idx = (idx + jitter_samples).clamp(0, n - 1).unique(sorted=True)
        excitation[idx] = 1.0
        n_fft = 2 * (n // 2 + 1)
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        formants = [(500, 60), (1500, 90), (2500, 120), (3500, 150)]
        vocal_tract_response = torch.zeros_like(freqs, dtype=torch.complex64)
        for fc, bw in formants:
            resonance = (bw / 2) ** 2 / ((freqs - fc) ** 2 + (bw / 2) ** 2)
            vocal_tract_response += resonance.to(torch.complex64)
        excitation_fft = torch.fft.rfft(excitation, n=n_fft)
        signal_fft = excitation_fft * vocal_tract_response
        signal = torch.fft.irfft(signal_fft, n=n_fft)[:n]
    elif signal_type == 'chirp':
        f0, f1 = 200.0, min(sample_rate / 2 - 200, 4000.0)
        k = (f1 - f0) / duration
        signal = torch.sin(two_pi_t * (f0 + 0.5 * k * t))
    elif signal_type == 'noise':
        signal = torch.randn(n)
    elif signal_type == 'pink_noise':
        # Efficient vectorized implementation of the Voss-McCartney algorithm
        num_rows = int(math.ceil(math.log2(n))) if n > 0 else 1
        white = torch.randn(num_rows, n)
        pink = torch.zeros(n)
        for r in range(num_rows):
            # Each row is a random sequence updated at half the frequency of the previous one
            pink += white[r, ::(1 << r)].repeat_interleave(1 << r)[:n]
        signal = pink / num_rows
    elif signal_type == 'alarm':
        beep_len = int(sample_rate * 0.25)
        gap_len = beep_len
        f = 1400.0
        pattern = torch.cat([torch.ones(beep_len), torch.zeros(gap_len)])
        pattern = pattern.repeat(math.ceil(n / pattern.numel()))[:n]
        signal = torch.sin(two_pi_t * f) * pattern
    elif signal_type == 'impulse':
        signal = torch.zeros(n)
        step = HOP_LENGTH
        indices = torch.arange(step // 2, n, step)
        signal[indices] = torch.sign(torch.randn(indices.size()))
    elif signal_type == 'music':
        freqs = [261.63, 329.63, 392.00]
        trem = 0.6 + 0.4 * torch.sin(two_pi_t * 3.0)
        signal = sum(torch.sin(two_pi_t * f) for f in freqs) * trem
    elif signal_type == 'engine_rev':
        f0 = random.uniform(50, 120)
        ramp_len = int(n * 0.30)
        rev_curve = torch.cat([torch.linspace(1.0, 3.0, ramp_len), 3.0 * torch.ones(n - ramp_len)])
        signal = sum(torch.sin(two_pi_t * h * f0 * rev_curve) / h for h in range(1, 6))
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    signal = _apply_fade(signal)
    signal = signal / (signal.abs().max() + 1e-9)
    signal *= 0.5
    return signal


def create_stereo_scene(
    source: torch.Tensor,
    doa: torch.Tensor, # Can be a scalar or a tensor for moving sources
    d_mic: float,
    sample_rate: int,
    vehicle_env: Optional[str] = None,
    snr_db: float = 15,
) -> torch.Tensor:
    """Create stereo scene with proper TDOA and optional vehicle noise."""
    doa_rad = torch.deg2rad(torch.as_tensor(doa, dtype=torch.float32).clamp(0.0, 180.0))
    tau = d_mic * torch.cos(doa_rad) / SPEED_OF_SOUND
    delay_samples = tau * sample_rate

    # The fractional_delay function handles both scalar and tensor delays,
    # but the FFT-based approach is only exact for a constant delay.
    # For a true time-varying filter, a different implementation would be needed.
    # We use this as an approximation for moving sources.
    if torch.mean(delay_samples) >= 0:
        left_channel = fractional_delay(source, delay_samples)
        right_channel = source.clone()
    else:
        left_channel = source.clone()
        right_channel = fractional_delay(source, -delay_samples)

    stereo = torch.stack([left_channel, right_channel], 0)

    if vehicle_env:
        stereo = apply_vehicle_environment(stereo, vehicle_env, snr_db)
    else:
        # Light sensor noise
        stereo += 0.01 * torch.randn_like(stereo)

    max_amplitude = torch.max(torch.abs(stereo))
    return stereo * 0.95 / (max_amplitude + 1e-9) if max_amplitude > 0 else stereo


# --- Adversarial scene augmentations (for difficult tests) ---

def apply_simple_reverb(stereo: torch.Tensor, rt60: float = 0.25, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Apply a lightweight early-reflection style reverb via a sparse FIR.
    Not a room simulator; just a few comb-like taps.
    """
    n = stereo.shape[-1]
    # Tap delays in samples (early reflections)
    delays_ms = [7.0, 13.0, 23.0, 37.0]
    delays = [int(sr * d * 1e-3) for d in delays_ms if int(sr * d * 1e-3) < n // 2]
    if not delays:
        return stereo
    # Exponential decay factor from rough RT60
    decay = math.exp(-3.0 / max(1e-3, rt60))
    taps = [1.0] + [decay ** i * random.uniform(0.4, 0.9) for i in range(1, len(delays) + 1)]
    # Build FIR kernel
    kernel_len = delays[-1] + 1
    h = torch.zeros(kernel_len, dtype=stereo.dtype)
    h[0] = taps[0]
    for i, d in enumerate(delays, start=1):
        h[d] = taps[i]
    h = h / (h.abs().sum() + 1e-9)

    # Convolve each channel separately with same IR (stationary room)
    h_ = h.view(1, 1, -1)
    out_L = F.conv1d(stereo[0:1, :].unsqueeze(0), h_, padding=h_.shape[-1] - 1).squeeze(0)[..., :n]
    out_R = F.conv1d(stereo[1:2, :].unsqueeze(0), h_, padding=h_.shape[-1] - 1).squeeze(0)[..., :n]
    out = torch.stack([out_L.squeeze(), out_R.squeeze()], dim=0)
    # Re-normalize to avoid clipping
    return out / (out.abs().max() + 1e-9)


def apply_mic_mismatch(stereo: torch.Tensor, gain_db: float = 1.0, frac_delay_samples: float = 0.2) -> torch.Tensor:
    """
    Apply small gain imbalance and fractional delay mismatch between mics.
    Positive gain_db increases Right channel level slightly.
    """
    gain = 10 ** (gain_db / 20.0)
    L = stereo[0].clone()
    R = fractional_delay(stereo[1], torch.tensor(frac_delay_samples)) * gain
    out = torch.stack([L, R], dim=0)
    max_amp = out.abs().max()
    return out * 0.98 / (max_amp + 1e-9) if max_amp > 0 else out


def mix_interferer(
    base_stereo: torch.Tensor,
    interferer: torch.Tensor,
    doa_interferer: float,
    d_mic: float,
    sr: int,
    interferer_rel_db: float = -6.0
) -> torch.Tensor:
    """
    Mix a second interfering source at a different DOA.
    interferer_rel_db: level of interferer relative to base in dB (negative = quieter).
    """
    inter_scene = create_stereo_scene(interferer, torch.tensor(doa_interferer), d_mic, sr, vehicle_env=None, snr_db=15)
    rel = 10 ** (interferer_rel_db / 20.0)
    n = min(base_stereo.shape[-1], inter_scene.shape[-1])
    out = base_stereo[:, :n] + rel * inter_scene[:, :n]
    # Normalize
    return out / (out.abs().max() + 1e-9)


# ============================================================
# COMPREHENSIVE TEST SUITE - Enhanced for correctness
# ============================================================

def run_systematic_tests(model: SoundSourceLocalize, tests_per_type: int = 10, verbose: bool = False) -> Dict:
    """Run comprehensive systematic tests across all sound types and environments."""
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE SOUND SOURCE LOCALIZATION TEST SUITE")
    print("=" * 80)

    random.seed(9527)
    np.random.seed(9527)
    torch.manual_seed(9527)

    sound_types = list(SOUND_TYPES.keys())
    environments = list(VEHICLE_ENVIRONMENTS.keys()) + [None]

    test_cases: List[Dict] = []

    # 1. Generate standard, randomized tests within each sector
    for sound_type in sound_types:
        for i in range(tests_per_type):
            env = environments[i % len(environments)]
            sector_id = i % len(SECTORS)
            angle_min, angle_max = SECTORS[sector_id]
            # Avoid exact boundaries for random tests
            true_angle = random.uniform(angle_min + 1, angle_max - 1)
            test_cases.append({'type': sound_type, 'env': env, 'angle': true_angle, 'snr': random.uniform(5, 15), 'desc': 'random'})

    # 2. Generate critical boundary condition tests (plus near-boundary)
    boundary_angles = [45.0, 90.0, 135.0]
    for angle in boundary_angles:
        for delta in [0.0, -0.4, 0.4]:
            for sound_type in ['speech', 'tone', 'noise']:  # Representative sounds
                for env in environments:
                    test_cases.append({'type': sound_type, 'env': env, 'angle': angle + delta, 'snr': 10.0, 'desc': 'boundary'})

    results = {
        'total_tests': 0, 'correct_sector': 0, 'total_error': 0.0, 'failed_tests': 0,
        'by_env': {str(env): {'correct': 0, 'total': 0, 'failed': 0, 'errors': []} for env in environments},
        'confusion_matrix': torch.zeros(len(SECTORS), len(SECTORS), dtype=torch.long)
    }

    t0 = time.time()
    with torch.inference_mode():
        for test_id, case in enumerate(test_cases):
            env_key = str(case['env'])
            try:
                test_signal = generate_test_signal(case['type'], 2.0, SAMPLE_RATE)
                stereo_scene = create_stereo_scene(test_signal, torch.tensor(case['angle']), MIC_DISTANCE, SAMPLE_RATE, vehicle_env=case['env'], snr_db=case['snr'])

                stereo_scene_int16 = normalize_to_int16(stereo_scene).unsqueeze(0)

                # The model expects two inputs, each of shape (batch, 1, length)
                estimated_doa = float(model(stereo_scene_int16[:, 0:1, :], stereo_scene_int16[:, 1:2, :]).item())

                angular_error = abs(estimated_doa - case['angle'])
                true_sector = sector_of(torch.tensor([case['angle']]))[0].item()
                estimated_sector = sector_of(torch.tensor([estimated_doa]))[0].item()
                sector_correct = sector_is_correct(case['angle'], estimated_doa, tol_deg=0.5)

                results['total_tests'] += 1
                results['correct_sector'] += int(sector_correct)
                results['total_error'] += angular_error

                env_stats = results['by_env'][env_key]
                env_stats['total'] += 1
                env_stats['correct'] += int(sector_correct)
                env_stats['errors'].append(angular_error)
                results['confusion_matrix'][int(true_sector), int(estimated_sector)] += 1

                if verbose:
                    status = "✓" if sector_correct else "✗"
                    env_label = case['env'] if case['env'] else "clean"
                    snr_info = f"(SNR {case['snr']:.1f}dB)" if case['env'] else ""
                    test_desc = f"({case['desc']})" if case['desc'] == 'boundary' else ''
                    print(f"[{test_id+1:4d}] {case['type']:<12} {env_label:<12} {status} "
                          f"{case['angle']:6.1f}° → {estimated_doa:6.1f}° "
                          f"(err {angular_error:4.1f}°) [{SECTOR_NAMES[int(true_sector)]} → {SECTOR_NAMES[int(estimated_sector)]}] {snr_info} {test_desc}")

            except Exception as e:
                results['failed_tests'] += 1
                results['by_env'][env_key]['failed'] += 1
                if verbose:
                    print(f"[{test_id+1:4d}] {case['type']:<12} {env_key:<12} ✗ FAILED: {e}")
    t1 = time.time()
    results['elapsed_sec'] = t1 - t0
    _print_test_summary(results, title="STANDARD TESTS")
    return results


def run_adversarial_tests(model: SoundSourceLocalize, verbose: bool = False) -> Dict:
    """
    Hard tests: low SNR, interferers, reverberation, mic mismatch, moving sources.
    """
    print("\n" + "=" * 80)
    print(" ADVERSARIAL / DIFFICULT TESTS")
    print("=" * 80)

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Define adversarial cases
    adversarial_cases: List[Dict] = []
    
    # 1) Extremely low SNR
    for angle in [30.0, 90.0, 150.0]:
        adversarial_cases.append({'type': 'speech', 'env': 'truck', 'angle': angle, 'snr': random.uniform(0, 3), 'mods': [], 'desc': 'low_snr'})

    # 2) Coherent Interferer (Speech on Speech)
    for angle in [20.0, 160.0]:
        adversarial_cases.append({'type': 'speech', 'env': 'normal', 'angle': angle, 'snr': 12.0, 'mods': ['interferer_coherent'], 'desc': 'interferer_coherent'})

    # 3) Mild reverberation
    for angle in [45.0, 90.0, 135.0]:
        adversarial_cases.append({'type': 'impulse', 'env': 'quiet', 'angle': angle, 'snr': 15.0, 'mods': ['reverb'], 'desc': 'reverb'})

    # 4) Mic mismatch
    for angle in [10.0, 170.0]:
        adversarial_cases.append({'type': 'music', 'env': None, 'angle': angle, 'snr': 15.0, 'mods': ['mismatch'], 'desc': 'mismatch'})

    # 5) Moving Source (e.g., car passing)
    for start_angle, end_angle in [(20.0, 70.0), (160.0, 110.0)]:
        adversarial_cases.append({'type': 'chirp', 'env': 'normal', 'angle': (start_angle, end_angle), 'snr': 15.0, 'mods': ['moving'], 'desc': 'moving_source'})
    
    # 6) Exact boundaries with multiple corruptions
    for angle in [45.0, 135.0]:
        adversarial_cases.append({'type': 'speech', 'env': 'offroad', 'angle': angle, 'snr': 8.0, 'mods': ['reverb', 'mismatch'], 'desc': 'boundary_combo'})

    results = {
        'total_tests': 0, 'correct_sector': 0, 'total_error': 0.0, 'failed_tests': 0,
        'confusion_matrix': torch.zeros(len(SECTORS), len(SECTORS), dtype=torch.long)
    }

    t0 = time.time()
    with torch.inference_mode():
        for idx, case in enumerate(adversarial_cases):
            try:
                # Handle both static (float) and moving (tuple) angle cases
                is_moving = 'moving' in case['mods']
                true_angle_repr = case['angle']
                if is_moving:
                    start_angle, end_angle = case['angle']
                    n_samples = int(2.0 * SAMPLE_RATE)
                    # The "true" angle for error calculation is the average position
                    true_angle_for_eval = (start_angle + end_angle) / 2.0
                    doa_trajectory = torch.linspace(start_angle, end_angle, n_samples)
                else:
                    true_angle_for_eval = case['angle']
                    doa_trajectory = torch.tensor(float(case['angle']))

                src = generate_test_signal(case['type'], 2.0, SAMPLE_RATE)
                base = create_stereo_scene(src, doa_trajectory, MIC_DISTANCE, SAMPLE_RATE, vehicle_env=case['env'], snr_db=case['snr'])

                # Apply adversarial modifications
                if 'interferer_coherent' in case['mods']:
                    inter = generate_test_signal('speech', 2.0, SAMPLE_RATE) # Same type
                    doa_int = (true_angle_for_eval + random.choice([-60, 60, 80])) % 180.0
                    base = mix_interferer(base, inter, doa_int, MIC_DISTANCE, SAMPLE_RATE, interferer_rel_db=random.uniform(-9.0, -3.0))
                if 'reverb' in case['mods']:
                    base = apply_simple_reverb(base, rt60=random.uniform(0.15, 0.35), sr=SAMPLE_RATE)
                if 'mismatch' in case['mods']:
                    base = apply_mic_mismatch(base, gain_db=random.uniform(-1.5, 1.5), frac_delay_samples=random.uniform(-0.3, 0.3))

                base_i16 = normalize_to_int16(base).unsqueeze(0)
                est_angle = float(model(base_i16[:, 0:1, :], base_i16[:, 1:2, :]).item())
                err = abs(est_angle - true_angle_for_eval)

                s_true = int(sector_of(torch.tensor([true_angle_for_eval]))[0].item())
                s_est = int(sector_of(torch.tensor([est_angle]))[0].item())
                correct = sector_is_correct(true_angle_for_eval, est_angle, tol_deg=0.5)

                results['total_tests'] += 1
                results['correct_sector'] += int(correct)
                results['total_error'] += err
                results['confusion_matrix'][s_true, s_est] += 1

                if verbose:
                    status = "✓" if correct else "✗"
                    env_label = case['env'] if case['env'] else "clean"
                    mods = ",".join(case['mods']) if case['mods'] else "none"
                    angle_str = f"{true_angle_repr[0]:.0f}°→{true_angle_repr[1]:.0f}°" if is_moving else f"{true_angle_for_eval:6.1f}°"
                    print(f"[{idx+1:4d}] {case['desc']:<20} {status} "
                          f"{angle_str:<12} → {est_angle:6.1f}° "
                          f"(err {err:4.1f}°) mods=[{mods}] [{SECTOR_NAMES[s_true]}→{SECTOR_NAMES[s_est]}]")
            except Exception as e:
                results['failed_tests'] += 1
                if verbose:
                    print(f"[{idx+1:4d}] ✗ FAILED: {e}")
    t1 = time.time()
    results['elapsed_sec'] = t1 - t0
    _print_test_summary(results, title="ADVERSARIAL TESTS")
    return results


def _print_test_summary(results: Dict, title: str = "RESULTS SUMMARY"):
    """Print detailed test results summary."""
    successful_tests = results.get('total_tests', 0)
    elapsed = results.get('elapsed_sec', None)

    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    if successful_tests > 0:
        accuracy = 100.0 * results['correct_sector'] / successful_tests
        mean_error = results['total_error'] / successful_tests
        print(f"TOTAL TESTS: {successful_tests}   FAILED: {results['failed_tests']}")
        print(f"SECTOR ACCURACY: {accuracy:.1f}%   MEAN ANGULAR ERROR: {mean_error:.1f}°")
        if elapsed is not None:
            print(f"ELAPSED: {elapsed:.2f}s   AVG PER TEST: {elapsed / successful_tests:.3f}s")
        if 'by_env' in results and results['by_env']:
            print("\nPERFORMANCE BY ENVIRONMENT:")
            print("Environment      Accuracy%    Mean Error°    Tests")
            print("-" * 50)
            for env_key, stats in sorted(results['by_env'].items(), key=lambda item: str(item[0])):
                if stats['total'] > 0:
                    env_accuracy = 100.0 * stats['correct'] / stats['total']
                    env_mean_error = sum(stats['errors']) / len(stats['errors']) if stats['errors'] else 0.0
                    env_label = env_key if env_key != "None" else "clean"
                    print(f"{env_label:<15} {env_accuracy:8.1f}  {env_mean_error:11.1f}   {stats['total']:5d}")

        print("\nCONFUSION MATRIX (True Sector \\ Estimated Sector):")
        col_width = max(len(name) for name in SECTOR_NAMES.values()) + 2
        print(" " * col_width + "".join(f"{SECTOR_NAMES[j]:>{col_width}}" for j in range(len(SECTORS))))
        print("-" * (col_width * (len(SECTORS) + 1)))
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
    """Main function with argument parsing and model testing/export."""
    parser = argparse.ArgumentParser(
        description="Advanced 2-microphone sound source localization system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tests-per-type', type=int, default=50, help='Number of random tests per sound type')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose test output')
    parser.add_argument('--skip-adv', action='store_true', help='Skip adversarial tests')
    args = parser.parse_args()

    print("Initializing STFT processor and localization model...")
    # This assumes STFT_Process is a valid nn.Module that can be scripted or exported
    custom_stft = STFT_Process(
        model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH,
        win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE
    ).eval()

    model = SoundSourceLocalize(
        SAMPLE_RATE, MIC_DISTANCE, ANGLE_GRID, NFFT, PRE_EMPHASIZE,
        ALPHA, BETA, GAMMA, MAX_SIGNAL_LENGTH, custom_stft
    ).eval().to('cpu')

    # Standard tests
    std_results = run_systematic_tests(
        model, tests_per_type=args.tests_per_type, verbose=args.verbose
    )

    # Adversarial tests
    adv_results = {'total_tests': 0, 'correct_sector': 0, 'failed_tests': 0}
    if not args.skip_adv:
        adv_results = run_adversarial_tests(model, verbose=args.verbose)

    print(f"\nExporting model to ONNX format at {ONNX_MODEL_PATH}")
    dummy_mic_L = torch.ones((1, 1, MAX_SIGNAL_LENGTH), dtype=torch.int16)
    dummy_mic_R = torch.ones((1, 1, MAX_SIGNAL_LENGTH), dtype=torch.int16)

    try:
        with torch.inference_mode():
            torch.onnx.export(
                model, (dummy_mic_L, dummy_mic_R), ONNX_MODEL_PATH,
                input_names=['audio_mic_L', 'audio_mic_R'],
                output_names=['estimated_angle_degrees'],
                dynamic_axes={'audio_mic_L': {2: 'audio_length'}, 'audio_mic_R': {2: 'audio_length'}} if DYNAMIC_AXES else None,
                opset_version=17, do_constant_folding=True
            )
        slim(
            model=ONNX_MODEL_PATH, output_model=ONNX_MODEL_PATH,
            no_shape_infer=False, skip_fusion_patterns=False,
            no_constant_folding=False, save_as_external_data=False, verbose=False
        )
        print("Model successfully exported and optimized.")
    except Exception as e:
        print(f"ERROR: Failed to export ONNX model: {e}")

    # Exit code based on standard tests primarily, but also reflect adversarial stability
    exit_ok_std = (std_results.get('total_tests', 0) > 0 and
                   (100 * std_results.get('correct_sector', 0) / std_results['total_tests']) > 70 and
                   std_results.get('failed_tests', 0) == 0)

    exit_ok_adv = True
    if not args.skip_adv and adv_results.get('total_tests', 0) > 0:
        adv_acc = 100 * adv_results.get('correct_sector', 0) / adv_results['total_tests']
        # Adversarial acceptance: lower threshold, just a sanity baseline
        exit_ok_adv = adv_acc > 45 and adv_results.get('failed_tests', 0) == 0

    exit_code = 0 if (exit_ok_std and exit_ok_adv) else 1
    overall_acc = 100 * std_results.get('correct_sector', 0) / max(1, std_results.get('total_tests', 1))
    print(f"\nTest suite completed with {overall_acc:.1f}% sector accuracy on standard tests. Exit code: {exit_code}")
    return exit_code


if __name__ == '__main__':
    # A simple try-except block to avoid crashing if STFT_Process is not found
    try:
        from STFT_Process import STFT_Process
        raise SystemExit(main())
    except ImportError:
        print("\nERROR: The 'STFT_Process.py' file could not be found.")
        print("Please ensure the custom STFT implementation is in the same directory.")
        raise SystemExit(1)
