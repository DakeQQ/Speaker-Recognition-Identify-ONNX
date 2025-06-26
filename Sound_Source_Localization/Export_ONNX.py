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
ALPHA = 0.3                             # Algorithm factor
BETA = 0.6                              # Algorithm factor
GAMMA = 0.1                             # Algorithm factor, ALPHA + BETA + GAMMA = 1.0

# Model export settings
DYNAMIC_AXES = True                     # Enable dynamic axes for ONNX export
ONNX_MODEL_PATH = "./SoundSourceLocalize.onnx"

# Pre-calculated constants for optimization
TWO_PI = 2.0 * math.pi                  # Pre-calculated 2π

# Angle grid for DOA estimation (0° = right, 90° = front, 180° = left)
ANGLE_GRID = torch.arange(0, 181, step=2, dtype=torch.float32)

# Sector definitions for direction classification

"""
                             Cabin
========================= L ⊙-MIC-⊙ R ========================
|                              |                             |
|    Front-Left (135°-180°)    |     Front-Right (0°-45°)    |
|                              |                             |
==============================================================
|                              |                             |
|     Rear-Left (90°-135°)     |     Rear-Right (45°-90°)    |
|                              |                             |
==============================================================

"""

SECTORS = {
    0: (0.0, 45.0),     # Front-Right
    1: (45.0, 90.0),    # Rear-Right  
    2: (90.0, 135.0),   # Rear-Left
    3: (135.0, 180.0)   # Front-Left
}

SECTOR_NAMES = {
    0: "Front-Right",
    1: "Rear-Right",
    2: "Rear-Left",
    3: "Front-Left"
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
        kernel = torch.zeros(kernel_size)
        mid = kernel_size // 2
        
        # Create band-pass filter kernel
        temp = 2000 - 200
        for i in range(kernel_size):
            tk = (i - mid) / sample_rate
            if abs(tk) > 1e-6:
                kernel[i] = (2000 * torch.sinc(torch.tensor(2000 * tk)) - 200 * torch.sinc(torch.tensor(200 * tk)))
            else:
                kernel[i] = temp
        
        kernel = kernel / torch.sum(torch.abs(kernel))
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

    def _enhanced_mvdr_spectrum(self, p0: torch.Tensor, p1: torch.Tensor, r01: torch.Tensor, i01: torch.Tensor) -> torch.Tensor:
        """Enhanced MVDR spectrum calculation with multiple integration strategies"""
        # Temporal smoothing with pre-calculated exponential weights
        weights = self.temporal_weights[..., :p0.shape[-1]]
        
        # Compute smoothed covariance matrix elements
        R00 = (p0 * weights).sum(dim=2)
        R11 = (p1 * weights).sum(dim=2)
        R01r = (r01 * weights).sum(dim=2)
        R01i = (i01 * weights).sum(dim=2)
        
        # Diagonal loading for numerical stability
        trace = (R00 + R11) * 0.5
        
        # Robust matrix inversion with conditioning
        determinant = R00 * R11 - (R01r ** 2 + R01i ** 2)
        determinant = determinant + 1e-6 * trace  # Diagonal loading
        
        inv_det = 1.0 / determinant
        iR00 = R11 * inv_det
        iR11 = R00 * inv_det
        iR01r = -R01r * inv_det  # Note the negative sign
        iR01i = -R01i * inv_det
        
        # MVDR beamformer power calculation for all angles
        quadratic_form = iR01r * self.steer_real + iR01i * self.steer_imag
        quadratic_form = iR00 + iR11 + quadratic_form + quadratic_form
        
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

    def forward(self, mic_wav_0: torch.Tensor, mic_wav_1: torch.Tensor) -> torch.Tensor:
        """Forward pass: convert audio to DOA estimate"""
        # Fused preprocessing: normalization, DC removal, pre-emphasis
        mic_wav_0 = mic_wav_0 * self.inv_int16
        mic_wav_1 = mic_wav_1 * self.inv_int16
        
        # Remove DC offset
        mic_wav_0 = mic_wav_0 - torch.mean(mic_wav_0, dim=-1, keepdim=True)
        mic_wav_1 = mic_wav_1 - torch.mean(mic_wav_1, dim=-1, keepdim=True)
        
        # Apply pre-emphasis filter
        mic_wav_0 = torch.cat([mic_wav_0[:, :, :1], mic_wav_0[:, :, 1:] - self.pre_emphasis * mic_wav_0[:, :, :-1]], dim=-1)
        mic_wav_1 = torch.cat([mic_wav_1[:, :, :1], mic_wav_1[:, :, 1:] - self.pre_emphasis * mic_wav_1[:, :, :-1]], dim=-1)
        
        # STFT computation
        r0, i0 = self.custom_stft(mic_wav_0, 'constant')
        r1, i1 = self.custom_stft(mic_wav_1, 'constant')
        
        # Compute power and cross-spectra
        p0 = r0 * r0 + i0 * i0
        p1 = r1 * r1 + i1 * i1
        r01 = r0 * r1 + i0 * i1
        i01 = i0 * r1 - r0 * i1
        
        # Enhanced MVDR processing
        mvdr_scores = self._enhanced_mvdr_spectrum(p0, p1, r01, i01)
        
        # Return estimated angle
        max_indices = torch.argmax(mvdr_scores, dim=-1)
        return self.angle_grid[max_indices]


# ============================================================
# SIGNAL GENERATION - Optimized and extended
# ============================================================

def generate_test_signal(signal_type: str, duration: float, sample_rate: int) -> torch.Tensor:
    """Generate various test signals with optimized implementations"""
    n_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, n_samples)
    
    if signal_type == 'tone':
        frequency = random.uniform(500, 1500)
        signal = torch.sin(TWO_PI * frequency * t)
        
    elif signal_type == 'speech':
        # Simplified speech-like signal with formants
        signal = 0.1 * torch.randn(n_samples)
        formant_freqs = [800, 1200, 2400]
        envelope = torch.exp(-((t - duration / 2) ** 2) / (2 * (duration / 8) ** 2))
        
        for freq in formant_freqs:
            formant = torch.sin(TWO_PI * freq * t) * envelope
            signal += 0.3 * formant
            
    elif signal_type == 'chirp':
        f0, f1 = 200.0, min(sample_rate / 2 - 500, 4000.0)
        phase = TWO_PI * torch.cumsum(torch.linspace(f0, f1, n_samples), 0) / sample_rate
        signal = torch.sin(phase)
        
    elif signal_type == 'noise':
        signal = torch.randn(n_samples)
        
    elif signal_type == 'music':
        # Simple chord progression (C major triad)
        chord_freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
        signal = torch.zeros(n_samples)
        for freq in chord_freqs:
            signal += 0.3 * torch.sin(TWO_PI * freq * t)
        # Add tremolo effect
        signal *= 0.5 * (1 + torch.sin(TWO_PI * 2 * t))
        
    elif signal_type == 'alarm':
        # Beeping alarm pattern
        period_samples = int(sample_rate * 0.5)
        beep = torch.sin(TWO_PI * 1500 * t)
        mask = ((torch.arange(n_samples) // period_samples) % 2 == 0).float()
        signal = beep * mask
        
    elif signal_type == 'impulse':
        # Random impulse train
        signal = torch.zeros(n_samples)
        n_impulses = max(1, int(duration * 5))
        for _ in range(n_impulses):
            idx = random.randint(0, n_samples - 1)
            signal[idx] = random.uniform(-1, 1)
            
    elif signal_type == 'engine_rev':
        # Engine harmonics with RPM variation
        f0 = random.uniform(50, 120)
        signal = torch.zeros(n_samples)
        rpm_modulation = 1 + 0.3 * torch.sin(TWO_PI * 0.5 * t)
        
        for harmonic in [1, 2, 3, 4, 5]:
            amplitude = 1.0 / harmonic
            signal += amplitude * torch.sin(TWO_PI * f0 * harmonic * t * rpm_modulation)
            
    elif signal_type == 'pink_noise':
        # Pink noise using simple IIR filter
        white_noise = torch.randn(n_samples)
        # Simplified pink noise filter coefficients
        b = torch.tensor([0.049922, 0.095993, 0.050612, -0.004408])
        a = torch.tensor([1.0, -2.494956, 2.017265, -0.522189])
        
        signal = torch.zeros(n_samples)
        for i in range(3, n_samples):
            signal[i] = (b[0] * white_noise[i] + b[1] * white_noise[i-1] + b[2] * white_noise[i-2] + b[3] * white_noise[i-3] - a[1] * signal[i-1] - a[2] * signal[i-2] - a[3] * signal[i-3])
    else:
        raise ValueError(f"Unknown signal type '{signal_type}'")
    
    # Normalize signal
    max_amplitude = torch.max(torch.abs(signal))
    return signal / max_amplitude if max_amplitude > 1e-8 else torch.zeros_like(signal)


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
    random.seed(42)
    torch.manual_seed(42)
    
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
                estimated_doa = model(stereo_scene[:, [0]], stereo_scene[:, [1]]).item()
                
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
    parser.add_argument('--tests-per-type', type=int, default=10, help='Number of tests per sound type')
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
    dummy_mic_0 = torch.ones((1, 1, MAX_SIGNAL_LENGTH), dtype=torch.int16)
    dummy_mic_1 = torch.ones((1, 1, MAX_SIGNAL_LENGTH), dtype=torch.int16)
    
    torch.onnx.export(
        model,
        (dummy_mic_0, dummy_mic_1),
        ONNX_MODEL_PATH,
        input_names=['audio_mic_0', 'audio_mic_1'],
        output_names=['estimated_angle_degrees'],
        dynamic_axes={
            'audio_mic_0': {2: 'audio_length'},
            'audio_mic_1': {2: 'audio_length'},
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
