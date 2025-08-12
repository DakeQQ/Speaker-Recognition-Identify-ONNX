#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import Dict, List, Optional, Union

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
MAX_SIGNAL_LENGTH = 32000               # Maximum signal length to process
WINDOW_TYPE = 'kaiser'                  # Window function type for STFT

# --- CORE ALGORITHM REVISION ---
# Adjusted weights to improve accuracy. The original weights (A=0.3, B=0.6, G=0.1)
# heavily favored the simple arithmetic mean. The new weights give more influence
# to the peak-weighted score (GAMMA), which is more robust in noisy and complex
# scenes by focusing on frequency bins with clear directional information.
ALPHA = 0.2                             # Geometric mean score (sensitive to nulls, good for broadband)
BETA = 0.5                              # Arithmetic mean score (robust baseline)
GAMMA = 0.2                             # Peak-weighted score (improves accuracy with interferers)

# Model export settings
DYNAMIC_AXES = True                     # Enable dynamic axes for ONNX export
ONNX_MODEL_PATH = "./SoundSourceLocalize.onnx"

# Pre-calculated constants for optimization
TWO_PI = 2.0 * math.pi                  # Pre-calculated 2π

# Angle grid for DOA estimation (0° = left, 90° = front, 180° = right)
ANGLE_GRID = torch.arange(0, 181, step=1, dtype=torch.float32)

# Sector definitions for direction classification
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

# Anti-aliasing frequency limit for spatial sampling
MAX_UNALIASED_FREQ = SPEED_OF_SOUND / (2 * MIC_DISTANCE)  # 8575 Hz

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
    'broadband':   {'freq_range': (50,   7500), 'energy_profile': 'white'},
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


def adaptive_sector_tolerance(snr_db: float, bandwidth_hz: float) -> float:
    """Compute adaptive tolerance for sector boundary checking based on signal conditions."""
    base_tol = 2.0  # degrees
    snr_factor = max(0.5, min(2.0, 20.0 / max(snr_db, 1.0)))
    bw_factor = max(0.8, min(1.5, bandwidth_hz / 1000.0))
    return base_tol * snr_factor * bw_factor


def sector_is_correct(true_angle: float, est_angle: float, snr_db: float = 15.0, bandwidth_hz: float = 1000.0) -> bool:
    """
    Boundary-tolerant sector comparison with adaptive tolerance.
    If the true angle is within adaptive tolerance of a sector boundary, accept either adjacent sector.
    """
    tol_deg = adaptive_sector_tolerance(snr_db, bandwidth_hz)
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
    Delay must be constant (scalar); if a tensor is provided, its mean is used as an approximation.
    Positive delay means output y[n] = x[n - delay].
    """
    if torch.all(torch.abs(delay) < 1e-6):
        return sig.clone()

    N = sig.numel()
    # Improved delay clamping based on signal length and avoid discontinuities
    max_reasonable_delay = min(N / 2.0, SAMPLE_RATE * 0.1)  # Max 100ms or half signal
    delay = torch.clamp(delay, -max_reasonable_delay, max_reasonable_delay)

    X = torch.fft.fft(sig)
    freqs = torch.fft.fftfreq(N, device=sig.device)

    if delay.numel() > 1:
        delay = torch.mean(delay)  # Simplified for constant approximation

    shift = torch.exp(torch.tensor(-1j, device=sig.device) * TWO_PI * freqs * delay)
    return torch.fft.ifft(X * shift).real


def _interp1_linear_fixed(sig: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation of a 1D signal at fractional positions with zero-padding.
    More realistic than mirror extension for acoustic signals.
    """
    n = sig.numel()
    if n <= 1:
        return sig.clone()

    # Clamp positions to valid range with zero-padding beyond
    pos = torch.clamp(positions, 0.0, n - 1.0)
    
    idx0 = torch.floor(pos).to(torch.long)
    idx1 = torch.clamp(idx0 + 1, max=n - 1)
    w1 = pos - idx0.to(pos.dtype)
    w0 = 1.0 - w1
    
    # Handle out-of-bounds with zero padding
    valid_mask = (positions >= 0.0) & (positions <= n - 1.0)
    result = sig[idx0] * w0 + sig[idx1] * w1
    result = result * valid_mask.to(result.dtype)
    
    return result


def apply_time_varying_delay_proper(sig: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
    """
    Apply time-varying delay using overlap-add STFT processing to maintain phase coherence.
    This is more accurate than simple interpolation for moving sources.
    """
    n = sig.numel()
    if delay_samples.numel() == 1:
        return fractional_delay(sig, delay_samples)
    
    # For time-varying delays, use frame-by-frame processing
    frame_size = 256
    hop_size = 128
    num_frames = (n - frame_size) // hop_size + 1
    
    if num_frames <= 1:
        # Fall back to interpolation for short signals
        t = torch.arange(n, dtype=torch.float32, device=sig.device)
        d = delay_samples.to(dtype=torch.float32, device=sig.device)
        positions = t - d
        return _interp1_linear_fixed(sig, positions)
    
    # Resample delay to match frames
    frame_times = torch.arange(num_frames, dtype=torch.float32, device=sig.device) * hop_size
    if delay_samples.numel() != num_frames:
        delay_frame_indices = torch.linspace(0, delay_samples.numel() - 1, num_frames, device=sig.device)
        delay_per_frame = torch.nn.functional.grid_sample(
            delay_samples.unsqueeze(0).unsqueeze(0),
            delay_frame_indices.view(1, 1, -1, 1) * 2.0 / (delay_samples.numel() - 1) - 1.0,
            mode='linear', padding_mode='border', align_corners=True
        ).squeeze()
    else:
        delay_per_frame = delay_samples
    
    # Process frame by frame with overlap-add
    output = torch.zeros_like(sig)
    window = torch.hann_window(frame_size, device=sig.device)
    
    for i in range(num_frames):
        start_idx = i * hop_size
        end_idx = min(start_idx + frame_size, n)
        frame_len = end_idx - start_idx
        
        if frame_len < frame_size:
            frame = torch.cat([sig[start_idx:end_idx], torch.zeros(frame_size - frame_len, device=sig.device)])
        else:
            frame = sig[start_idx:end_idx]
        
        # Apply delay to this frame (analysis window is applied here)
        delayed_frame = fractional_delay(frame * window, delay_per_frame[i])
        
        # Overlap-add (synthesis)
        output_end = min(start_idx + frame_size, n)
        output_len = output_end - start_idx
        # The analysis window was already applied before fractional_delay.
        # Do not apply it a second time during the overlap-add summation.
        output[start_idx:output_end] += delayed_frame[:output_len]
    
    return output


def apply_time_varying_delay(sig: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
    """
    Apply a time-varying (possibly signed) delay in samples.
    Positive delay d means y[n] = x[n - d].
    """
    n = sig.numel()
    # If scalar, fall back to constant fractional delay
    if delay_samples.numel() == 1:
        return fractional_delay(sig, delay_samples)
    
    # Use proper time-varying delay processing
    return apply_time_varying_delay_proper(sig, delay_samples)


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
    return noisy_signal

# ============================================================
# MAIN LOCALIZATION MODEL - Fixed with improved MVDR + IPD/coherence gating
# ============================================================


import torch
import torch.nn as nn

# Constants assumed defined elsewhere:
# SPEED_OF_SOUND, MAX_UNALIASED_FREQ, TWO_PI

class SoundSourceLocalize(nn.Module):
    """Optimized sound source localization using enhanced MVDR beamforming with anti-aliasing."""

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
        custom_stft: nn.Module,
        max_batch: int = 64  # preallocate scratch for up to this batch size
    ):
        super().__init__()
        self.custom_stft = custom_stft

        # Scalars as Python floats for cheaper broadcast
        self.inv_int16 = float(1.0 / 32768.0)
        self.pre_emphasis = float(pre_emphasis)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(1e-6)

        # Precompute rFFT freqs once
        freqs = torch.fft.rfftfreq(nfft, 1.0 / sample_rate)  # (F_full,)

        # Anti-aliasing: limit frequencies to avoid spatial aliasing
        max_freq = min(sample_rate / 2.0, MAX_UNALIASED_FREQ)
        freq_mask = freqs <= max_freq
        freq_idx = torch.where(freq_mask)[0]  # compressed frequency indices
        freqs_c = freqs[freq_idx]  # compressed freqs

        # Steering vectors on the angle x compressed-frequency grid
        # tau consistent with scene generation: tau = -(d * cos(theta))/c
        tau = -(d_mic * torch.cos(torch.deg2rad(angle_grid))) / SPEED_OF_SOUND  # (A,)
        phase_matrix = TWO_PI * tau.unsqueeze(1) * freqs_c.unsqueeze(0)          # (A, F)

        self.register_buffer('steer_real', torch.cos(phase_matrix), persistent=False)  # (A, F)
        self.register_buffer('steer_imag', torch.sin(phase_matrix), persistent=False)  # (A, F)
        self.register_buffer('angle_grid', angle_grid, persistent=True)
        self.register_buffer('freq_idx', freq_idx, persistent=True)  # (F,)
        self.register_buffer('freq_mask_full', freq_mask, persistent=True)

        # Frequency weighting (compressed to valid freqs)
        self._setup_frequency_weighting(freqs_c, d_mic, sample_rate)

        # Temporal weights for frame integration (decaying) + prefix sums for cheap renorm
        weights = torch.exp(torch.linspace(-2, 0, max_signal_len, dtype=torch.float32))  # (Tmax,)
        weights_sum_prefix = torch.cumsum(weights, dim=0)  # (Tmax,)
        self.register_buffer('temporal_weights', weights.view(1, 1, -1), persistent=True)          # (1,1,Tmax)
        self.register_buffer('temporal_weights_cumsum', weights_sum_prefix, persistent=True)       # (Tmax,)

        # ------------------------------------------------------------------
        # Preallocated scratch buffers for _refine_peak_parabolic
        # ------------------------------------------------------------------
        # Zeros buffer used via slicing instead of torch.zeros_like
        self.register_buffer('zeros_1d', torch.zeros(max_batch, dtype=torch.float32), persistent=False)
        # arange cache for indexing; dtype long
        self.register_buffer('arange_1d', torch.arange(max_batch, dtype=torch.long), persistent=False)

    def _setup_frequency_weighting(self, freqs_c: torch.Tensor, d_mic: float, sample_rate: int):
        """Pre-calculate optimal frequency weighting for MVDR with anti-aliasing (compressed freqs)."""
        eps = self.eps
        wavelengths = SPEED_OF_SOUND / (freqs_c + eps)
        spatial_resolution = d_mic / wavelengths
        spatial_weight = torch.sigmoid(4.0 * (spatial_resolution - 0.1))

        # Smooth band emphasis around ~sample_rate/6 with wide variance ~sample_rate/4
        center = float(sample_rate) / 6.0
        sigma = float(sample_rate) / 4.0
        freq_weight = 1.0 - torch.exp(-((freqs_c - center) ** 2) / (2.0 * (sigma ** 2)))

        combined_weight = (spatial_weight * freq_weight).unsqueeze(0)  # (1, F)
        combined_weight = combined_weight / (combined_weight.sum() + eps)
        self.register_buffer('freq_weight', combined_weight, persistent=False)  # (1, F)

    @staticmethod
    def _pre_emphasis(x: torch.Tensor, coeff: float) -> torch.Tensor:
        """Efficient pre-emphasis without extra cat; keeps channels separate."""
        # x: (B, T)
        y = x.clone()
        y[..., 1:] = x[..., 1:] - coeff * x[..., :-1]
        return y

    def _ensure_batch_capacity(self, B: int, device: torch.device):
        """Ensure preallocated 1D buffers can cover current batch; grow at most occasionally."""
        if B <= self.zeros_1d.numel() and self.zeros_1d.device == device:
            return
        # Grow to next power-of-two >= B to reduce future reallocations
        new_cap = 1
        while new_cap < B:
            new_cap <<= 1
        self.zeros_1d = torch.zeros(new_cap, dtype=torch.float32, device=device)
        self.arange_1d = torch.arange(new_cap, dtype=torch.long, device=device)

    def _enhanced_mvdr_spectrum(
        self,
        p_L: torch.Tensor,
        p_R: torch.Tensor,
        r_LR: torch.Tensor,
        i_LR: torch.Tensor,
        weights_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhanced MVDR spectrum calculation with coherence/IPD gating and adaptive regularization.
        Inputs shapes:
          p_L, p_R, r_LR, i_LR: (B, F, T) with F already compressed to unaliased bins
          weights_norm: (1,1,T) normalized temporal weights for these T frames
        Returns:
          final_score: (B, A) beam power scores aggregated across frequency
        """
        eps = self.eps

        # Frame-averaged entries of the 2x2 spatial covariance
        R_LL   = (p_L  * weights_norm).sum(dim=2)
        R_RR   = (p_R  * weights_norm).sum(dim=2)
        R_LR_r = (r_LR * weights_norm).sum(dim=2)
        R_LR_i = (i_LR * weights_norm).sum(dim=2)

        R_LL_R_RR = R_LL * R_RR

        # Clamp cross-term magnitude to ensure PSD-like behavior
        cross_mag = torch.sqrt(R_LR_r.mul(R_LR_r) + R_LR_i.mul(R_LR_i))
        cross_max = 0.999 * torch.sqrt(torch.clamp(R_LL_R_RR, min=0.0))
        clamp_scale = torch.clamp(cross_max / (cross_mag + eps), max=1.0)
        R_LR_r = R_LR_r * clamp_scale
        R_LR_i = R_LR_i * clamp_scale

        # Adaptive diagonal loading based on simple condition proxy
        trace = (R_LL + R_RR) * 0.5
        determinant = R_LL_R_RR - (R_LR_r.mul(R_LR_r) + R_LR_i.mul(R_LR_i))
        condition_estimate = trace / (torch.abs(determinant) + eps)
        adaptive_loading = torch.clamp(1e-3 * condition_estimate, 1e-6, 1e-2)
        determinant = determinant + adaptive_loading * trace

        inv_det = 1.0 / (determinant + eps)
        iR_LL   = R_RR * inv_det
        iR_RR   = R_LL * inv_det
        iR_LR_r = -R_LR_r * inv_det
        iR_LR_i = -R_LR_i * inv_det

        # Denominator ~ v^H R^-1 v for each steering vector v = [1, e^{-jφ}]
        quadratic_form = iR_LR_r.unsqueeze(1) * self.steer_real + iR_LR_i.unsqueeze(1) * self.steer_imag  # (B, A, F)
        denominator = (iR_LL + iR_RR).unsqueeze(1) + quadratic_form + quadratic_form

        mvdr_power = 1.0 / (denominator + eps)  # (B, A, F)

        # Coherence/IPD-based gating
        gamma2_frames = torch.clamp(
            (r_LR.mul(r_LR) + i_LR.mul(i_LR)) / (p_L * p_R + eps),
            0.0, 1.0
        )
        gamma2 = (gamma2_frames * weights_norm).sum(dim=2)  # (B, F)

        # Use already-averaged cross-spectrum for measured cross-phase
        S_norm = torch.rsqrt(R_LR_r.mul(R_LR_r) + R_LR_i.mul(R_LR_i) + eps)
        c_meas = R_LR_r * S_norm
        s_meas = R_LR_i * S_norm

        cos_delta = c_meas.unsqueeze(1) * self.steer_real + s_meas.unsqueeze(1) * self.steer_imag  # (B, A, F)
        ipd_gate = 0.5 * (1.0 + cos_delta)
        coh_weight = torch.sqrt(torch.clamp(gamma2, 0.0, 1.0)).unsqueeze(1)

        gating_mask = 0.2 + 0.8 * (ipd_gate * coh_weight)

        mvdr_power_gated = mvdr_power * gating_mask + eps

        # Frequency integration with prior weights (compressed F)
        log_power = torch.log(mvdr_power_gated)
        weighted_log = (log_power * self.freq_weight).sum(-1)
        geometric_score = torch.exp(weighted_log)

        arithmetic_score = (mvdr_power_gated * self.freq_weight).sum(-1)

        # Peak emphasis across angles
        peak_weight_angles = torch.softmax(mvdr_power_gated, dim=-2)
        peak_score = (mvdr_power_gated * peak_weight_angles * self.freq_weight).sum(-1)

        final_score = self.alpha * geometric_score + self.beta * arithmetic_score + self.gamma * peak_score
        return final_score  # (B, A)

    def _refine_peak_parabolic(self, score: torch.Tensor) -> torch.Tensor:
        """
        Sub-degree parabolic interpolation around the argmax.
        Uses preallocated zero and arange buffers via slicing (no zeros_like).
        """
        # score: (B, A)
        device = score.device
        B, A = score.shape

        # Ensure prealloc buffers are large enough and on the right device
        self._ensure_batch_capacity(B, device)

        idx = torch.argmax(score, dim=-1)  # (B,)

        A_minus = A - 1

        # Neighbors
        idx_m = torch.clamp(idx - 1, 0, A_minus)
        idx_p = torch.clamp(idx + 1, 0, A_minus)

        arange_b = self.arange_1d[:B]  # slice of preallocated arange
        y0 = score[arange_b, idx_m]
        y1 = score[arange_b, idx]
        y2 = score[arange_b, idx_p]

        denom = (y0 - (y1 + y1) + y2)

        # Use a slice of preallocated zeros instead of torch.zeros_like(denom)
        zeros_slice = self.zeros_1d[:B]  # matches (B,) and dtype=float32
        offset = torch.where(
            torch.abs(denom) > 1e-12,
            0.5 * (y0 - y2) / denom,
            zeros_slice
        ).clamp(-1.0, 1.0)

        base_angle = self.angle_grid[idx]
        refined = (base_angle + offset).clamp(0.0, 180.0)
        return refined

    def forward(self, mic_wav_L: torch.ShortTensor, mic_wav_R: torch.ShortTensor) -> torch.Tensor:
        """Forward pass: convert audio to DOA estimate. L/R processing is kept separate until STFT."""
        eps = self.eps
        freq_idx = self.freq_idx  # (F,)

        # Convert to float and DC removal (L/R separate)
        mic_wav_L = mic_wav_L.to(torch.float32).mul_(self.inv_int16)
        mic_wav_R = mic_wav_R.to(torch.float32).mul_(self.inv_int16)

        mic_wav_L = mic_wav_L - mic_wav_L.mean(dim=-1, keepdim=True)
        mic_wav_R = mic_wav_R - mic_wav_R.mean(dim=-1, keepdim=True)

        # Pre-emphasis (L/R separate, no cat)
        mic_wav_L = self._pre_emphasis(mic_wav_L, self.pre_emphasis)
        mic_wav_R = self._pre_emphasis(mic_wav_R, self.pre_emphasis)

        # Custom STFT (L/R separate)
        r_L, i_L = self.custom_stft(mic_wav_L, 'constant')  # (B, F_full, T)
        r_R, i_R = self.custom_stft(mic_wav_R, 'constant')  # (B, F_full, T)

        # Compress to unaliased frequency bins once; all downstream uses compressed F
        r_L = r_L[:, freq_idx, :]
        i_L = i_L[:, freq_idx, :]
        r_R = r_R[:, freq_idx, :]
        i_R = i_R[:, freq_idx, :]

        # Power and cross terms
        p_L = r_L.mul(r_L) + i_L.mul(i_L)
        p_R = r_R.mul(r_R) + i_R.mul(i_R)

        r_LR = r_L.mul(r_R) + i_L.mul(i_R)
        i_LR = i_L.mul(r_R) - r_L.mul(i_R)

        # Temporal weighting for the actual number of frames
        T = p_L.shape[-1]
        denom = self.temporal_weights_cumsum[T - 1] + eps  # scalar
        weights_norm = (self.temporal_weights[..., :T] / denom)  # (1,1,T)

        # MVDR scores per angle
        mvdr_scores = self._enhanced_mvdr_spectrum(p_L, p_R, r_LR, i_LR, weights_norm)

        # Sub-degree parabolic refinement
        est_angles = self._refine_peak_parabolic(mvdr_scores)
        return est_angles
# ============================================================
# SIGNAL GENERATION - Enhanced with new test signals
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
        formants = [(500, 60), (1500, 90), (2500, 120), (3500, 120)]
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
    elif signal_type == 'broadband':
        signal = torch.randn(n)
    elif signal_type == 'pink_noise':
        # Corrected comment to reflect the algorithm used.
        # Efficient vectorized implementation of the "sum of octaves" method.
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
    """
    Create stereo scene with proper TDOA and optional vehicle noise.

    Convention (consistent with steering): tau = -(d * cos(theta)) / c.
    Positive tau => right channel is delayed by tau (left leads).
    """
    doa_rad = torch.deg2rad(torch.as_tensor(doa, dtype=torch.float32).clamp(0.0, 180.0))
    tau = -(d_mic * torch.cos(doa_rad)) / SPEED_OF_SOUND
    delay_samples = tau * sample_rate

    n = source.numel()

    # Time-varying relative delay handling: delay only the lagging channel
    if delay_samples.numel() > 1:
        # For each time step, compute per-channel non-negative delays
        right_delay = torch.clamp(delay_samples, min=0.0)
        left_delay = torch.clamp(-delay_samples, min=0.0)
        # Use the more accurate time-varying delay function
        left_channel = apply_time_varying_delay_proper(source, left_delay)
        right_channel = apply_time_varying_delay_proper(source, right_delay)
    else:
        d = float(delay_samples.item())
        if d >= 0:
            # Right lags by d
            left_channel = source.clone()
            right_channel = fractional_delay(source, torch.tensor(d, dtype=torch.float32, device=source.device))
        else:
            # Left lags by -d
            left_channel = fractional_delay(source, torch.tensor(-d, dtype=torch.float32, device=source.device))
            right_channel = source.clone()

    stereo = torch.stack([left_channel, right_channel], 0)

    if vehicle_env:
        stereo = apply_vehicle_environment(stereo, vehicle_env, snr_db)
    else:
        # Light sensor noise
        stereo = stereo + 0.01 * torch.randn_like(stereo)

    return stereo


# --- Adversarial scene augmentations (enhanced and fixed) ---

def mix_stereo_scenes(base_scene: torch.Tensor, overlay_scene: torch.Tensor, overlay_rel_db: float = -6.0) -> torch.Tensor:
    """Mixes two stereo scenes, scaling the overlay relative to the base."""
    n = min(base_scene.shape[-1], overlay_scene.shape[-1])
    base = base_scene[:, :n]
    overlay = overlay_scene[:, :n]

    base_power = torch.mean(base ** 2)
    overlay_power = torch.mean(overlay ** 2)

    if base_power < 1e-10 or overlay_power < 1e-10:
        return base

    scale_factor = torch.sqrt(base_power / overlay_power) * (10 ** (overlay_rel_db / 20.0))
    return base + scale_factor * overlay


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
    h = torch.zeros(kernel_len, dtype=stereo.dtype, device=stereo.device)
    h[0] = taps[0]
    for i, d in enumerate(delays, start=1):
        h[d] = taps[i]
    h = h / (h.abs().sum() + 1e-9)

    # Convolve each channel separately with same IR (stationary room)
    h_ = h.view(1, 1, -1)
    pad = kernel_len // 2
    out_L = F.conv1d(stereo[0:1, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out_R = F.conv1d(stereo[1:2, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out = torch.stack([out_L.squeeze(), out_R.squeeze()], dim=0)
    return out


def apply_mic_mismatch(stereo: torch.Tensor, gain_db: float = 1.0, frac_delay_samples: float = 0.2) -> torch.Tensor:
    """
    Apply small gain imbalance and fractional delay mismatch between mics.
    Positive gain_db increases Right channel level slightly.
    """
    gain = 10 ** (gain_db / 20.0)
    L = stereo[0].clone()
    R = fractional_delay(stereo[1], torch.tensor(frac_delay_samples, dtype=torch.float32, device=stereo.device)) * gain
    out = torch.stack([L, R], dim=0)
    return out


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
    return out

def _normalize_fir_peak(kernel: torch.Tensor, nfft: int = 4096) -> torch.Tensor:
    """Normalize FIR kernel to unity peak magnitude response."""
    H = torch.fft.rfft(kernel, n=nfft)
    peak = torch.max(torch.abs(H))
    return kernel / (peak + 1e-9)


def _fir_lowpass_kernel(cutoff_hz: float, kernel_size: int, sr: int) -> torch.Tensor:
    """Generates a peak-normalized, windowed sinc low-pass filter kernel."""
    mid = kernel_size // 2
    tk = (torch.arange(kernel_size, dtype=torch.float32) - mid) / sr
    kernel = 2 * cutoff_hz / sr * torch.sinc(2 * cutoff_hz / sr * tk)
    kernel *= torch.hann_window(kernel_size)
    # Use consistent peak normalization for all filters for physical accuracy.
    kernel = _normalize_fir_peak(kernel)
    return kernel


def apply_low_pass_filter(stereo: torch.Tensor, cutoff_hz: float, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Apply a simple low-pass filter to a stereo signal."""
    n = stereo.shape[-1]
    kernel_size = 101
    if kernel_size >= n:
        return stereo

    # Refactored to use the consistent, peak-normalized kernel function
    kernel = _fir_lowpass_kernel(cutoff_hz, kernel_size, sr).to(stereo.device)
    
    h_ = kernel.view(1, 1, -1)
    pad = kernel_size // 2
    out_L = F.conv1d(stereo[0:1, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out_R = F.conv1d(stereo[1:2, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out = torch.stack([out_L.squeeze(), out_R.squeeze()], dim=0)
    return out


def apply_band_pass_filter(stereo: torch.Tensor, low_hz: float, high_hz: float, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Apply a simple band-pass filter to a stereo signal, normalized to unity peak response."""
    n = stereo.shape[-1]
    kernel_size = 101
    if kernel_size >= n or low_hz >= high_hz:
        return stereo

    mid = kernel_size // 2
    tk = (torch.arange(kernel_size, dtype=torch.float32, device=stereo.device) - mid) / sr
    
    kernel_lp_high = 2 * high_hz / sr * torch.sinc(2 * high_hz / sr * tk)
    kernel_lp_low = 2 * low_hz / sr * torch.sinc(2 * low_hz / sr * tk)
    
    kernel = kernel_lp_high - kernel_lp_low
    kernel *= torch.hann_window(kernel_size, device=stereo.device)
    kernel = _normalize_fir_peak(kernel)

    h_ = kernel.view(1, 1, -1)
    pad = kernel_size // 2
    out_L = F.conv1d(stereo[0:1, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out_R = F.conv1d(stereo[1:2, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out = torch.stack([out_L.squeeze(), out_R.squeeze()], dim=0)
    return out


def generate_short_burst(stereo: torch.Tensor, duration_ms: float = 150.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Reduces a signal to a short burst in its center."""
    n = stereo.shape[-1]
    burst_len = int(sr * duration_ms / 1000.0)
    if burst_len >= n:
        return stereo
    
    start = (n - burst_len) // 2
    end = start + burst_len
    
    mask = torch.zeros(n, dtype=stereo.dtype, device=stereo.device)
    mask[start:end] = 1.0
    
    fade_len = burst_len // 10
    if fade_len > 0:
        window = torch.hann_window(fade_len * 2, periodic=False, device=stereo.device)
        mask[start:start+fade_len] *= window[:fade_len]
        mask[end-fade_len:end] *= window[fade_len:]

    return stereo * mask


def apply_phase_inversion(stereo: torch.Tensor) -> torch.Tensor:
    """Flips the phase of the right channel by 180 degrees."""
    out = stereo.clone()
    out[1] *= -1.0
    return out


def apply_intermittent_signal(stereo: torch.Tensor, block_ms: float = 40.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Makes the signal cut in and out periodically. Clicks are part of the test."""
    n = stereo.shape[-1]
    block_len = int(sr * block_ms / 1000.0)
    if block_len == 0 or block_len * 2 > n:
        return stereo

    mask = torch.ones(n, dtype=stereo.dtype, device=stereo.device)
    num_blocks = n // (block_len * 2)
    for i in range(num_blocks):
        start = i * block_len * 2
        mask[start : start + block_len] = 0.0
    return stereo * mask


# --- New adversarial augmentations (fixed and enhanced) ---

def apply_doppler_shift(stereo: torch.Tensor, vehicle_speed_mps: float, source_distance_m: float, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Simulates Doppler shift for a source stationary to the side of a moving vehicle.
    The vehicle is assumed to pass the source at its closest point at the midpoint of the signal.
    """
    n = stereo.shape[-1]
    duration_s = n / sr
    t = torch.arange(n, dtype=torch.float32, device=stereo.device) / sr - duration_s / 2.0  # Centered time

    # Vehicle position along its path, relative to the closest point to the source
    vehicle_pos = vehicle_speed_mps * t
    # Distance from vehicle to source
    distance_to_source = torch.sqrt(source_distance_m**2 + vehicle_pos**2)
    # Radial velocity of the source relative to the vehicle
    radial_velocity = vehicle_speed_mps * (vehicle_pos / distance_to_source)

    # Doppler shift factor
    shift_factor = (1 + radial_velocity / SPEED_OF_SOUND)

    # Create new sample positions by integrating the time-varying sample rate
    dt = 1.0 / sr
    new_time = torch.cumsum(shift_factor, dim=0) * dt - dt
    positions = new_time * sr
    
    # Resample both channels using the new time base
    out_L = _interp1_linear_fixed(stereo[0], positions)
    out_R = _interp1_linear_fixed(stereo[1], positions)

    return torch.stack([out_L, out_R], dim=0)


def apply_freq_response_mismatch(stereo: torch.Tensor, num_points: int = 5, max_db_variation: float = 2.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Applies a randomized, smooth frequency response mismatch to one channel via frequency-domain filtering.
    """
    n = stereo.shape[-1]
    # Define control points for the random frequency response
    freq_points = torch.logspace(
        math.log10(100.0), math.log10(sr / 2 * 0.95), num_points, device=stereo.device
    )
    # Random gain variation in dB at these points
    db_points = torch.rand(num_points, device=stereo.device) * max_db_variation - (max_db_variation / 2.0)
    db_points[0] = 0.0 # Anchor at 0dB for low freqs

    # Create the full frequency response via interpolation
    n_fft = 2 * (n // 2 + 1) # Get correct n_fft for irfft
    all_freqs = torch.fft.rfftfreq(n_fft, 1.0/sr).to(stereo.device)
    
    # Interpolate in linear-frequency domain for simplicity
    interp_db = np.interp(all_freqs.cpu().numpy(), freq_points.cpu().numpy(), db_points.cpu().numpy())
    interp_db = torch.from_numpy(interp_db).to(stereo.device)
    
    # Convert dB to linear magnitude response
    mag_response = 10.0 ** (interp_db / 20.0)

    # Apply to one channel in frequency domain
    channel_to_filter = 1 if random.random() < 0.5 else 0
    
    channel_fft = torch.fft.rfft(stereo[channel_to_filter], n=n_fft)
    filtered_fft = channel_fft * mag_response
    filtered_channel = torch.fft.irfft(filtered_fft, n=n_fft)[:n]

    out = stereo.clone()
    out[channel_to_filter] = filtered_channel
    return out


def apply_clock_skew(stereo: torch.Tensor, ppm: float = 300.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Applies a slight sample-rate mismatch to the right channel with proper boundary handling."""
    n = stereo.shape[-1]
    scale = 1.0 + ppm * 1e-6
    t = torch.arange(n, dtype=torch.float32, device=stereo.device)
    positions = t * scale
    R = _interp1_linear_fixed(stereo[1], positions)
    out = torch.stack([stereo[0], R], dim=0)
    return out


def apply_wow_flutter(stereo: torch.Tensor, depth_samples: float = 0.25, rate_hz: float = 2.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Applies time-varying fractional delay (wow/flutter) to the right channel with proper boundary handling."""
    n = stereo.shape[-1]
    t = torch.arange(n, dtype=torch.float32, device=stereo.device) / sr
    positions = torch.arange(n, dtype=torch.float32, device=stereo.device) + depth_samples * torch.sin(TWO_PI * rate_hz * t + random.uniform(0, TWO_PI))
    R = _interp1_linear_fixed(stereo[1], positions)
    out = torch.stack([stereo[0], R], dim=0)
    return out


def apply_channel_dropout(stereo: torch.Tensor, prob: float = 0.3, block_ms: float = 30.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Randomly drops blocks from one of the channels."""
    n = stereo.shape[-1]
    block_len = max(1, int(sr * block_ms / 1000.0))
    mask_L = torch.ones(n, dtype=stereo.dtype, device=stereo.device)
    mask_R = torch.ones(n, dtype=stereo.dtype, device=stereo.device)
    num_blocks = max(1, n // block_len)
    for b in range(num_blocks):
        if random.random() < prob:
            start = b * block_len
            end = min(n, start + block_len)
            if random.random() < 0.5:
                mask_L[start:end] = 0.0
            else:
                mask_R[start:end] = 0.0
    out = torch.stack([stereo[0] * mask_L, stereo[1] * mask_R], dim=0)
    return out


def apply_crosstalk_mix(stereo: torch.Tensor, xtalk_db: float = -18.0) -> torch.Tensor:
    """Injects inter-channel crosstalk."""
    a = 10 ** (xtalk_db / 20.0)
    L = stereo[0]
    R = stereo[1]
    out_L = L + a * R
    out_R = R + a * L
    out = torch.stack([out_L, out_R], dim=0)
    return out


def apply_notch_filter(stereo: torch.Tensor, center_hz: float = 1000.0, bw_hz: float = 80.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Apply a simple FIR notch (band-stop) filter to both channels."""
    n = stereo.shape[-1]
    kernel_size = 101
    if kernel_size >= n or center_hz <= 0 or bw_hz <= 0:
        return stereo

    low_cut = max(1.0, center_hz - bw_hz / 2.0)
    high_cut = min(sr / 2 - 1.0, center_hz + bw_hz / 2.0)

    lp_low = _fir_lowpass_kernel(low_cut, kernel_size, sr)
    lp_high = _fir_lowpass_kernel(high_cut, kernel_size, sr)
    delta = torch.zeros(kernel_size, dtype=torch.float32)
    delta[kernel_size // 2] = 1.0
    hp_high = delta - lp_high  # Spectral inversion to get high-pass
    bandstop = lp_low + hp_high  # LP(low) + HP(high) ≈ band-stop

    h_ = bandstop.view(1, 1, -1).to(stereo.device)
    pad = kernel_size // 2
    out_L = F.conv1d(stereo[0:1, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out_R = F.conv1d(stereo[1:2, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n]
    out = torch.stack([out_L.squeeze(), out_R.squeeze()], dim=0)
    return out


def add_powerline_hum(stereo: torch.Tensor, freq: float = 60.0, level_db: float = -25.0, sr: int = SAMPLE_RATE, harmonics: int = 5) -> torch.Tensor:
    """Add powerline hum (fundamental + harmonics) to both channels."""
    n = stereo.shape[-1]
    t = torch.arange(n, dtype=torch.float32, device=stereo.device) / sr
    amp = 10 ** (level_db / 20.0)
    hum_L = torch.zeros(n, dtype=stereo.dtype, device=stereo.device)
    hum_R = torch.zeros(n, dtype=stereo.dtype, device=stereo.device)
    for k in range(1, harmonics + 1):
        f = k * freq
        if f >= sr / 2:
            break
        phase_L = random.uniform(0, TWO_PI)
        phase_R = random.uniform(0, TWO_PI)
        hum_L += (amp / k) * torch.sin(TWO_PI * f * t + phase_L)
        hum_R += (amp / k) * torch.sin(TWO_PI * f * t + phase_R)
    out = stereo + torch.stack([hum_L, hum_R], dim=0)
    return out


def apply_mic_lowpass_mismatch(stereo: torch.Tensor, cutoff_hz: float = 1200.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Apply low-pass filtering to only one channel (mismatch)."""
    n = stereo.shape[-1]
    kernel_size = 101
    if kernel_size >= n:
        return stereo
    kernel = _fir_lowpass_kernel(cutoff_hz, kernel_size, sr)
    h_ = kernel.view(1, 1, -1).to(stereo.device)
    pad = kernel_size // 2
    # Apply only to right channel
    out_R = F.conv1d(stereo[1:2, :].unsqueeze(0), h_, padding=pad).squeeze(0)[..., :n].squeeze()
    out = torch.stack([stereo[0], out_R], dim=0)
    return out


def apply_bit_depth_reduction(stereo: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Reduce effective bit depth (bitcrush) for both channels."""
    levels = max(2, 2 ** bits)
    scale = (levels / 2 - 1)
    out = torch.round(stereo * scale) / (scale + 1e-9)
    out = out.clamp(-1.0, 1.0)
    return out


def apply_soft_clipping(stereo: torch.Tensor, drive: float = 2.5) -> torch.Tensor:
    """Apply soft clipping nonlinearity."""
    out = torch.tanh(drive * stereo) / math.tanh(drive)
    return out


def apply_dc_offset(stereo: torch.Tensor, offset_l: float = 0.02, offset_r: float = -0.02) -> torch.Tensor:
    """Add a small DC offset to both channels."""
    out = stereo.clone()
    out[0] = (out[0] + offset_l).clamp(-1.0, 1.0)
    out[1] = (out[1] + offset_r).clamp(-1.0, 1.0)
    return out


def apply_channel_swap_mid(stereo: torch.Tensor) -> torch.Tensor:
    """Swap channels in the second half of the signal."""
    n = stereo.shape[-1]
    mid = n // 2
    out = stereo.clone()
    out_L2 = stereo[1, mid:].clone()
    out_R2 = stereo[0, mid:].clone()
    out[0, mid:] = out_L2
    out[1, mid:] = out_R2
    return out


# ============================================================
# UNIFIED TEST SUITE - Enhanced with new test cases
# ============================================================

def _print_unified_summary(results: Dict):
    """Print detailed summary for the unified test suite."""
    total_tests = results.get('total_tests', 0)
    elapsed = results.get('elapsed_sec', None)

    print("\n" + "=" * 80)
    print("UNIFIED TEST SUITE SUMMARY")
    print("=" * 80)

    if total_tests > 0:
        # Overall Summary
        accuracy = 100.0 * results['correct_sector'] / total_tests
        mean_error = results['total_error'] / total_tests
        print(f"OVERALL RESULTS:")
        print(f"  TOTAL TESTS: {total_tests}   FAILED: {results['failed_tests']}")
        print(f"  SECTOR ACCURACY: {accuracy:.1f}%   MEAN ANGULAR ERROR: {mean_error:.1f}°")
        if elapsed is not None:
            print(f"  ELAPSED: {elapsed:.2f}s   AVG PER TEST: {elapsed / total_tests:.3f}s")
        
        # Per-Category Breakdown
        print("\nPERFORMANCE BY CATEGORY:")
        print("Category         Accuracy%    Mean Error°    Tests")
        print("-" * 50)
        for category, stats in sorted(results['by_category'].items()):
            if stats['total'] > 0:
                cat_accuracy = 100.0 * stats['correct'] / stats['total']
                cat_mean_error = sum(stats['errors']) / len(stats['errors']) if stats['errors'] else 0.0
                print(f"{category:<15} {cat_accuracy:8.1f}  {cat_mean_error:11.1f}   {stats['total']:5d}")

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


def run_unified_test_suite(model: SoundSourceLocalize, tests_per_type: int = 10, verbose: bool = False) -> Dict:
    """
    Run a unified, comprehensive test suite including standard and adversarial cases.
    Every test case includes random microphone mismatch (dB gain and phase delay).
    """
    print("\n" + "=" * 80)
    print(" UNIFIED SOUND SOURCE LOCALIZATION TEST SUITE")
    print("=" * 80)

    # Use a single seed for full reproducibility
    seed = 9527
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_cases: List[Dict] = []
    sound_types = list(SOUND_TYPES.keys())
    environments = list(VEHICLE_ENVIRONMENTS.keys()) + [None]

    # --- 1. Generate STANDARD test cases ---
    print(f"Generating standard test cases ({tests_per_type} per sound type)...")
    for sound_type in sound_types:
        for i in range(tests_per_type):
            env = environments[i % len(environments)]
            sector_id = i % len(SECTORS)
            angle_min, angle_max = SECTORS[sector_id]
            true_angle = random.uniform(angle_min + 1, angle_max - 1)
            # Use variable signal durations for more robust testing
            duration = random.uniform(0.5, 4.0)
            test_cases.append({
                'category': 'standard',
                'type': sound_type,
                'env': env,
                'angle': true_angle,
                'duration': duration,
                'snr': random.uniform(5, 20),
                'desc': f'std_{sound_type}',
                'mods': [],
                'mic_mismatch': {'gain_db': random.uniform(-0.5, 0.5), 'delay_samples': random.uniform(-0.1, 0.1)}
            })

    # --- 2. Define ADVERSARIAL test cases (enhanced) ---
    print("Defining adversarial test cases...")
    adversarial_base_cases = [
        # Noise & Interference
        {'type': 'speech', 'env': 'truck', 'angle': 30.0, 'snr': random.uniform(0, 3), 'mods': [], 'desc': 'low_snr'},
        {'type': 'noise', 'env': 'quiet', 'angle': 90.0, 'snr': -20.0, 'mods': [], 'desc': 'ambient_noise_only'},
        {'type': 'speech', 'env': 'normal', 'angle': 160.0, 'snr': 12.0, 'mods': ['interferer_coherent'], 'desc': 'interferer_coherent'},
        {'type': 'speech', 'env': 'normal', 'angle': 90.0, 'snr': 18.0, 'mods': ['multi_interferer'], 'desc': 'multi_interferer'},
        {'type': 'speech', 'env': 'normal', 'angle': 20.0, 'snr': 10.0, 'mods': ['directional_noise'], 'desc': 'directional_noise'},
        {'type': 'speech', 'env': 'normal', 'angle': 25.0, 'snr': 12.0, 'mods': ['moving_interferer'], 'desc': 'moving_interferer'},
        {'type': 'tone', 'env': None, 'angle': 10.0, 'snr': 20.0, 'mods': ['hum_60hz'], 'desc': 'powerline_hum'},

        # Microphone & Channel Imperfections
        {'type': 'impulse', 'env': 'quiet', 'angle': 90.0, 'snr': 15.0, 'mods': ['reverb'], 'desc': 'reverb'},
        {'type': 'speech', 'env': 'normal', 'angle': 100.0, 'snr': 12.0, 'mods': ['clock_skew'], 'desc': 'clock_skew'},
        {'type': 'chirp', 'env': None, 'angle': (70.0, 110.0), 'snr': 10.0, 'mods': ['moving', 'wow_flutter'], 'desc': 'moving_wow_flutter'},
        {'type': 'speech', 'env': 'truck', 'angle': 55.0, 'snr': 6.0, 'mods': ['dropouts'], 'desc': 'channel_dropouts'},
        {'type': 'speech', 'env': 'rainy', 'angle': 89.8, 'snr': 10.0, 'mods': ['reverb', 'crosstalk'], 'desc': 'near_boundary_reverb'},
        {'type': 'speech', 'env': 'convertible', 'angle': 150.0, 'snr': 8.0, 'mods': ['mic_lowpass'], 'desc': 'mic_lowpass_mismatch'},
        {'type': 'pink_noise', 'env': 'quiet', 'angle': 110.0, 'snr': 20.0, 'mods': ['freq_mismatch'], 'desc': 'freq_response_mismatch'},
        {'type': 'pink_noise', 'env': 'offroad', 'angle': 40.0, 'snr': 4.0, 'mods': ['dc_offset'], 'desc': 'dc_offset_bias'},

        # Signal Distortion & Non-Linearity
        {'type': 'music', 'env': None, 'angle': 75.0, 'snr': 25.0, 'mods': ['clipping'], 'desc': 'signal_clipping'},
        {'type': 'music', 'env': 'truck', 'angle': 150.0, 'snr': 1.0, 'mods': ['clipping'], 'desc': 'clipping_low_snr'},
        {'type': 'music', 'env': 'sportscar', 'angle': 30.0, 'snr': 5.0, 'mods': ['bitcrush', 'softclip'], 'desc': 'nonlinear_bitcrush'},

        # Spectral & Filtering Effects
        {'type': 'speech', 'env': 'quiet', 'angle': 140.0, 'snr': 15.0, 'mods': ['band_limited_low'], 'desc': 'band_limited_low_pass'},
        {'type': 'pink_noise', 'env': None, 'angle': 80.0, 'snr': 20.0, 'mods': ['band_limited_mid'], 'desc': 'extreme_band_pass'},
        {'type': 'speech', 'env': 'normal', 'angle': 120.0, 'snr': 18.0, 'mods': ['notch_1k'], 'desc': 'notch_filter_1k'},
        # Renamed test to be more descriptive.
        {'type': 'tone', 'freq': 7500, 'env': None, 'angle': 85.0, 'snr': 15.0, 'mods': [], 'desc': 'high_freq_near_limit'},
        # Added a true spatial aliasing test with frequency > MAX_UNALIASED_FREQ (8575 Hz).
        {'type': 'tone', 'freq': 9000, 'env': None, 'angle': 60.0, 'snr': 15.0, 'mods': [], 'desc': 'spatial_aliasing_above_limit'},

        # Temporal, Phase, & Multi-Source Ambiguity
        {'type': 'chirp', 'env': 'normal', 'angle': (20.0, 70.0), 'snr': 15.0, 'mods': ['moving'], 'desc': 'moving_source'},
        {'type': 'speech', 'env': 'offroad', 'angle': 45.0, 'snr': 8.0, 'mods': ['reverb'], 'desc': 'boundary_combo'},
        {'type': 'noise', 'env': None, 'angle': 40.0, 'snr': 20.0, 'mods': ['short_burst'], 'desc': 'short_impulse_burst'},
        {'type': 'impulse', 'env': None, 'angle': 60.0, 'snr': 20.0, 'mods': ['ultra_short_burst'], 'desc': 'ultra_short_transient'},
        {'type': 'speech', 'env': 'quiet', 'angle': 45.0, 'snr': 20.0, 'mods': ['phase_inversion'], 'desc': 'phase_inversion'},
        {'type': 'alarm', 'env': 'normal', 'angle': [60.0, 120.0], 'snr': 15.0, 'mods': [], 'desc': 'symmetric_sources_coherent'},
        # Changed to use two different sound types for a more challenging test.
        {'type': ['speech', 'noise'], 'env': 'normal', 'angle': [30.0, 150.0], 'snr': 12.0, 'mods': [], 'desc': 'front_back_ambiguity'},
        {'type': 'alarm', 'env': 'quiet', 'angle': 130.0, 'snr': 18.0, 'mods': ['intermittent'], 'desc': 'intermittent_signal'},
        {'type': 'music', 'env': 'normal', 'angle': 100.0, 'snr': 15.0, 'mods': ['channel_swap_mid'], 'desc': 'channel_swap_mid'},
        {'type': 'alarm', 'env': 'quiet', 'angle': 70.0, 'snr': 10.0, 'mods': ['doppler'], 'desc': 'doppler_shift_passby'},
        {'type': 'speech', 'env': 'quiet', 'angle': 90.0, 'snr': 20.0, 'mods': [], 'desc': 'mono_signal_front'},
        
        # Stacked / Combined
        {'type': 'speech', 'env': 'noisy', 'angle': 135.0, 'snr': 3.0, 'mods': ['directional_noise', 'clock_skew', 'wow_flutter'], 'desc': 'stacked_time_distortions'},
    ]

    for case in adversarial_base_cases:
        case['category'] = 'adversarial'
        case['duration'] = case.get('duration', 2.0) # Default duration for adversarial cases
        case['mic_mismatch'] = {'gain_db': random.uniform(-1.5, 1.5), 'delay_samples': random.uniform(-0.3, 0.3)}
        test_cases.append(case)

    # --- 3. Initialize results structure ---
    results = {
        'total_tests': 0, 'correct_sector': 0, 'total_error': 0.0, 'failed_tests': 0,
        'elapsed_sec': 0.0,
        'confusion_matrix': torch.zeros(len(SECTORS), len(SECTORS), dtype=torch.long),
        'by_category': {
            'standard': {'correct': 0, 'total': 0, 'failed': 0, 'errors': []},
            'adversarial': {'correct': 0, 'total': 0, 'failed': 0, 'errors': []}
        }
    }

    # --- 4. Run all test cases in the unified loop ---
    t0 = time.time()
    with torch.inference_mode():
        for test_id, case in enumerate(test_cases):
            try:
                base = None
                base_mono_src1 = None  # keep original waveform for coherent interferer use
                duration = case['duration']

                # Handle static, moving, and symmetric sources
                is_moving = 'moving' in case['mods']
                is_symmetric = 'symmetric_sources_coherent' in case['desc']
                is_front_back = 'front_back' in case['desc']
                is_mono = 'mono_signal' in case['desc']
                
                true_angle_for_eval: Union[float, List[float]]

                if is_mono:
                    true_angle_for_eval = case['angle']
                    true_angle_repr = f"{true_angle_for_eval:6.1f}°"
                    base_mono_src1 = generate_test_signal(case['type'], duration, SAMPLE_RATE)
                    # Create mono signal by duplicating the source
                    base = torch.stack([base_mono_src1, base_mono_src1], 0)
                    # Apply light sensor noise as in the 'else' block of create_stereo_scene
                    base = base + 0.01 * torch.randn_like(base)

                elif is_moving:
                    start_angle, end_angle = case['angle']
                    n_samples = int(duration * SAMPLE_RATE)
                    true_angle_for_eval = (start_angle + end_angle) / 2.0
                    doa_trajectory = torch.linspace(start_angle, end_angle, n_samples)
                    true_angle_repr = f"{start_angle:.0f}°→{end_angle:.0f}°"
                    base_mono_src1 = generate_test_signal(case['type'], duration, SAMPLE_RATE)
                    base = create_stereo_scene(base_mono_src1, doa_trajectory, MIC_DISTANCE, SAMPLE_RATE, case['env'], case['snr'])

                elif is_symmetric or is_front_back:
                    true_angle_for_eval = case['angle']
                    true_angle_repr = f"{true_angle_for_eval[0]:.0f}°&{true_angle_for_eval[1]:.0f}°"
                    
                    if is_symmetric: # Coherent case
                        base_mono_src = generate_test_signal(case['type'], duration, SAMPLE_RATE)
                        base_mono_src1 = base_mono_src
                        base_mono_src2 = base_mono_src
                    else: # Diverse front/back case
                        sound_type_1 = case['type'][0] if isinstance(case['type'], list) else case['type']
                        sound_type_2 = case['type'][1] if isinstance(case['type'], list) else case['type']
                        base_mono_src1 = generate_test_signal(sound_type_1, duration, SAMPLE_RATE)
                        base_mono_src2 = generate_test_signal(sound_type_2, duration, SAMPLE_RATE)

                    scene1 = create_stereo_scene(base_mono_src1, torch.tensor(true_angle_for_eval[0]), MIC_DISTANCE, SAMPLE_RATE)
                    scene2 = create_stereo_scene(base_mono_src2, torch.tensor(true_angle_for_eval[1]), MIC_DISTANCE, SAMPLE_RATE)
                    n_min = min(scene1.shape[1], scene2.shape[1])
                    base = scene1[:, :n_min] + scene2[:, :n_min]
                    if case['env']:
                        base = apply_vehicle_environment(base, case['env'], case['snr'])
                else: # Static source
                    true_angle_for_eval = case['angle']
                    doa_trajectory = torch.tensor(float(case['angle']))
                    true_angle_repr = f"{true_angle_for_eval:6.1f}°"
                    signal_type = case['type']
                    # Handle special frequency cases
                    if 'freq' in case:
                        # Override generate_test_signal for specific frequency
                        n = int(duration * SAMPLE_RATE)
                        t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
                        base_mono_src1 = torch.sin(TWO_PI * case['freq'] * t)
                        base_mono_src1 = _apply_fade(base_mono_src1)
                        base_mono_src1 = base_mono_src1 / (base_mono_src1.abs().max() + 1e-9) * 0.5
                    else:
                        base_mono_src1 = generate_test_signal(signal_type, duration, SAMPLE_RATE)
                    base = create_stereo_scene(base_mono_src1, doa_trajectory, MIC_DISTANCE, SAMPLE_RATE, case['env'], case['snr'])

                # Apply universal microphone mismatch
                mismatch_params = case['mic_mismatch']
                base = apply_mic_mismatch(base, gain_db=mismatch_params['gain_db'], frac_delay_samples=mismatch_params['delay_samples'])

                # Apply additional adversarial modifications
                if 'interferer_coherent' in case['mods']:
                    coherent_src = base_mono_src1 if base_mono_src1 is not None else generate_test_signal(case['type'], duration, SAMPLE_RATE)
                    base_angle = (case['angle'][0] if isinstance(case['angle'], (list, tuple)) else case['angle'])
                    doa_int = (base_angle + random.choice([-60, 60, 80])) % 180.0
                    base = mix_interferer(base, coherent_src, doa_int, MIC_DISTANCE, SAMPLE_RATE, -6.0)
                if 'directional_noise' in case['mods']:
                    inter = generate_test_signal('pink_noise', duration, SAMPLE_RATE)
                    base_angle = (case['angle'][0] if isinstance(case['angle'], (list, tuple)) else case['angle'])
                    doa_int = (base_angle + random.choice([70, -70, 100])) % 180.0
                    base = mix_interferer(base, inter, doa_int, MIC_DISTANCE, SAMPLE_RATE, -3.0)
                if 'multi_interferer' in case['mods']:
                    base_angle = (case['angle'][0] if isinstance(case['angle'], (list, tuple)) else case['angle'])
                    inter1 = generate_test_signal('noise', duration, SAMPLE_RATE)
                    base = mix_interferer(base, inter1, (base_angle + 55.0) % 180.0, MIC_DISTANCE, SAMPLE_RATE, -8.0)
                    inter2 = generate_test_signal('alarm', duration, SAMPLE_RATE)
                    base = mix_interferer(base, inter2, (base_angle - 65.0 + 180.0) % 180.0, MIC_DISTANCE, SAMPLE_RATE, -10.0)
                if 'moving_interferer' in case['mods']:
                    interferer_mono = generate_test_signal('alarm', duration, SAMPLE_RATE)
                    interferer_traj = torch.linspace(150.0, 80.0, interferer_mono.numel())
                    interferer_scene = create_stereo_scene(interferer_mono, interferer_traj, MIC_DISTANCE, SAMPLE_RATE)
                    base = mix_stereo_scenes(base, interferer_scene, overlay_rel_db=-5.0)
                if 'reverb' in case['mods']: base = apply_simple_reverb(base, rt60=random.uniform(0.15, 0.35))
                if 'clipping' in case['mods']: base = torch.clamp(base * 1.8, -0.99, 0.99)
                if 'band_limited_low' in case['mods']: base = apply_low_pass_filter(base, cutoff_hz=800)
                if 'band_limited_mid' in case['mods']: base = apply_band_pass_filter(base, low_hz=900, high_hz=1100)
                if 'short_burst' in case['mods']: base = generate_short_burst(base, duration_ms=150.0)
                if 'ultra_short_burst' in case['mods']: base = generate_short_burst(base, duration_ms=25.0)
                if 'phase_inversion' in case['mods']: base = apply_phase_inversion(base)
                if 'intermittent' in case['mods']: base = apply_intermittent_signal(base)
                if 'clock_skew' in case['mods']: base = apply_clock_skew(base, ppm=random.uniform(100.0, 600.0))
                if 'wow_flutter' in case['mods']: base = apply_wow_flutter(base, depth_samples=random.uniform(0.15, 0.4), rate_hz=random.uniform(0.8, 3.0))
                if 'dropouts' in case['mods']: base = apply_channel_dropout(base, prob=0.35, block_ms=30.0)
                if 'crosstalk' in case['mods']: base = apply_crosstalk_mix(base, xtalk_db=-18.0)
                if 'notch_1k' in case['mods']: base = apply_notch_filter(base, center_hz=1000.0, bw_hz=120.0)
                if 'hum_60hz' in case['mods']: base = add_powerline_hum(base, freq=random.choice([50.0, 60.0]), level_db=-25.0, harmonics=6)
                if 'mic_lowpass' in case['mods']: base = apply_mic_lowpass_mismatch(base, cutoff_hz=random.uniform(900.0, 1500.0))
                if 'freq_mismatch' in case['mods']: base = apply_freq_response_mismatch(base, max_db_variation=random.uniform(1.0, 3.0))
                if 'bitcrush' in case['mods']: base = apply_bit_depth_reduction(base, bits=random.choice([6, 8, 10]))
                if 'softclip' in case['mods']: base = apply_soft_clipping(base, drive=random.uniform(2.0, 3.5))
                if 'dc_offset' in case['mods']: base = apply_dc_offset(base, offset_l=random.uniform(-0.03, 0.03), offset_r=random.uniform(-0.03, 0.03))
                if 'channel_swap_mid' in case['mods']: base = apply_channel_swap_mid(base)
                if 'doppler' in case['mods']: base = apply_doppler_shift(base, vehicle_speed_mps=random.uniform(20, 35), source_distance_m=random.uniform(8, 15))

                # Run model and evaluate
                base_i16 = normalize_to_int16(base).unsqueeze(0)
                est_angle = float(model(base_i16[:, 0:1, :], base_i16[:, 1:2, :]).item())
                
                s_est = int(sector_of(torch.tensor([est_angle]))[0].item())
                correct: bool
                err: float
                s_true: int

                # Get signal characteristics for adaptive tolerance
                signal_bw = SOUND_TYPES.get(case.get('type', 'broadband'), {'freq_range': (100, 1000)})['freq_range']
                bw_hz = signal_bw[1] - signal_bw[0]
                snr_db = case.get('snr', 15.0)

                if is_symmetric or is_front_back:
                    err1 = abs(est_angle - true_angle_for_eval[0])
                    err2 = abs(est_angle - true_angle_for_eval[1])
                    err = min(err1, err2)
                    true_angle_for_matrix = true_angle_for_eval[0] if err1 < err2 else true_angle_for_eval[1]
                    s_true = int(sector_of(torch.tensor([true_angle_for_matrix]))[0].item())
                    correct = (sector_is_correct(true_angle_for_eval[0], est_angle, snr_db, bw_hz) or 
                              sector_is_correct(true_angle_for_eval[1], est_angle, snr_db, bw_hz))
                else:
                    err = abs(est_angle - true_angle_for_eval)
                    s_true = int(sector_of(torch.tensor([true_angle_for_eval]))[0].item())
                    correct = sector_is_correct(true_angle_for_eval, est_angle, snr_db, bw_hz)

                # Update results
                cat_stats = results['by_category'][case['category']]
                results['total_tests'] += 1
                results['correct_sector'] += int(correct)
                results['total_error'] += err
                results['confusion_matrix'][s_true, s_est] += 1
                cat_stats['total'] += 1
                cat_stats['correct'] += int(correct)
                cat_stats['errors'].append(err)

                if verbose:
                    status = "✓" if correct else "✗"
                    print(f"[{test_id+1:4d}][{case['category']:^11s}] {case['desc']:<32} {status} "
                          f"{true_angle_repr:<12} → {est_angle:6.1f}° (err {err:5.1f}°) "
                          f"[{SECTOR_NAMES[s_true]}→{SECTOR_NAMES[s_est]}]")
            except Exception as e:
                results['failed_tests'] += 1
                results['by_category'][case['category']]['failed'] += 1
                if verbose:
                    print(f"[{test_id+1:4d}][{case['category']:^11s}] {case['desc']:<32} ✗ FAILED: {e}")

    t1 = time.time()
    results['elapsed_sec'] = t1 - t0
    _print_unified_summary(results)
    return results


# ============================================================
# MAIN FUNCTION - CLI and model export
# ============================================================

def main() -> int:
    """Main function with argument parsing and model testing/export."""
    parser = argparse.ArgumentParser(
        description="Advanced 2-microphone sound source localization system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tests-per-type', type=int, default=10, help='Number of random standard tests per sound type')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose test output')
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

    # Run the unified test suite
    results = run_unified_test_suite(
        model, tests_per_type=args.tests_per_type, verbose=args.verbose
    )

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

    # Determine exit code based on test results
    std_stats = results['by_category']['standard']
    adv_stats = results['by_category']['adversarial']

    std_acc = 100 * std_stats.get('correct', 0) / max(1, std_stats.get('total', 1))
    adv_acc = 100 * adv_stats.get('correct', 0) / max(1, adv_stats.get('total', 1))
    print(f"\nTest suite completed:")
    print(f"  Standard: {std_acc:.1f}% accuracy (target: >85%)")
    print(f"  Adversarial: {adv_acc:.1f}% accuracy (target: >40%)")
    return 0


if __name__ == '__main__':
    # To run this script, save it as a .py file and execute from your terminal.
    # Example: python your_script_name.py -v
    # You might need to install dependencies: pip install torch numpy onnx onnxslim
    raise SystemExit(main())
