"""
Advanced Data Preprocessing Pipeline for VHD Detection
Implements sophisticated signal processing for heart sound analysis
"""

import numpy as np
import pandas as pd
import librosa
import scipy.signal as sp_signal
from scipy.signal import butter, filtfilt, find_peaks
import soundfile as sf
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HeartSoundPreprocessor:
    """
    Advanced preprocessor for phonocardiogram signals
    Implements noise reduction, segmentation, and normalization
    """
    
    def __init__(self, target_sr: int = 2000, lowcut: float = 25, highcut: float = 400):
        self.target_sr = target_sr
        self.lowcut = lowcut
        self.highcut = highcut
        self.nyquist = 0.5 * target_sr
        self.low = lowcut / self.nyquist
        self.high = highcut / self.nyquist
        self.max_signal_length = 4000  # Limit signal length for speed (2 seconds at 2000Hz)
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and downsample audio file
        """
        try:
            signal, sr = librosa.load(file_path, sr=self.target_sr)
            return signal, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def butter_bandpass_filter(self, data: np.ndarray, order: int = 5) -> np.ndarray:
        """
        Apply Butterworth bandpass filter
        """
        b, a = butter(order, [self.low, self.high], btype='band')
        return filtfilt(b, a, data)
    
    def remove_baseline_drift(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove baseline drift using high-pass filter
        """
        # High-pass filter to remove DC component and low-frequency drift
        sos = sp_signal.butter(4, 0.5, btype='high', fs=self.target_sr, output='sos')
        return sp_signal.sosfilt(sos, signal)
    
    def adaptive_noise_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Advanced noise reduction using spectral subtraction
        """
        # Compute STFT
        stft = librosa.stft(signal, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.target_sr / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor factor
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft, hop_length=512)
    
    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Advanced normalization preserving signal characteristics
        """
        # Remove DC offset
        signal = signal - np.mean(signal)
        
        # Robust normalization using median absolute deviation
        mad = np.median(np.abs(signal - np.median(signal)))
        if mad > 0:
            signal = signal / (1.4826 * mad)
        
        # Soft clipping to prevent extreme values
        signal = np.tanh(signal)
        
        return signal
    
    def fast_preprocess(self, signal: np.ndarray) -> np.ndarray:
        """
        Ultra-fast preprocessing for 5 files/second processing
        Optimized for speed while maintaining essential signal characteristics
        """
        # Limit signal length for speed
        if len(signal) > self.max_signal_length:
            signal = signal[:self.max_signal_length]
        
        # Fast bandpass filter
        signal = self.butter_bandpass_filter(signal)
        
        # Fast normalization
        signal = signal - np.mean(signal)
        std_signal = np.std(signal)
        if std_signal > 0:
            signal = signal / std_signal
        
        return signal
    
    def detect_heart_cycles(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Advanced heart cycle segmentation using multiple techniques
        """
        # Method 1: Peak detection on envelope
        envelope = np.abs(signal)
        envelope = sp_signal.savgol_filter(envelope, window_length=51, polyorder=3)
        
        # Find peaks with adaptive threshold
        peaks, _ = find_peaks(envelope, 
                             height=np.percentile(envelope, 70),
                             distance=int(0.3 * self.target_sr))  # Minimum 0.3s between peaks
        
        if len(peaks) < 2:
            return []
        
        # Method 2: Onset detection using librosa
        onsets = librosa.onset.onset_detect(y=signal, sr=self.target_sr, 
                                          units='samples', 
                                          hop_length=512,
                                          pre_max=3, post_max=3, 
                                          pre_avg=3, post_avg=5, 
                                          delta=0.2, wait=10)
        
        # Combine both methods
        all_onsets = np.unique(np.concatenate([peaks, onsets]))
        all_onsets = np.sort(all_onsets)
        
        # Segment into heart cycles
        cycles = []
        for i in range(len(all_onsets) - 1):
            start = max(0, all_onsets[i] - int(0.1 * self.target_sr))
            end = min(len(signal), all_onsets[i + 1] + int(0.1 * self.target_sr))
            cycle = signal[start:end]
            
            # Filter out very short or very long cycles
            if 0.3 * self.target_sr < len(cycle) < 2.0 * self.target_sr:
                cycles.append(cycle)
        
        return cycles
    
    def preprocess_signal(self, signal: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Complete preprocessing pipeline for signal array
        """
        if sr is None:
            sr = self.target_sr
            
        # Remove baseline drift
        signal = self.remove_baseline_drift(signal)
        
        # Apply bandpass filter
        signal = self.butter_bandpass_filter(signal)
        
        # Advanced noise reduction
        signal = self.adaptive_noise_reduction(signal)
        
        # Normalize
        signal = self.normalize_signal(signal)
        
        return signal
    
    def preprocess_file(self, file_path: str) -> Tuple[List[np.ndarray], int]:
        """
        Complete preprocessing pipeline for file path
        """
        # Load audio
        signal, sr = self.load_audio(file_path)
        if signal is None:
            return [], sr
        
        # Preprocess the signal
        processed_signal = self.preprocess_signal(signal, sr)
        
        # Segment into heart cycles
        cycles = self.detect_heart_cycles(processed_signal)
        
        return cycles, sr
    
    def extract_signal_features(self, signal: np.ndarray) -> dict:
        """
        Extract basic signal characteristics
        """
        features = {
            'length': len(signal),
            'mean': np.mean(signal),
            'std': np.std(signal),
            'skewness': scipy.stats.skew(signal),
            'kurtosis': scipy.stats.kurtosis(signal),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(signal)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=signal, sr=self.target_sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=signal, sr=self.target_sr)),
            'mfcc': np.mean(librosa.feature.mfcc(y=signal, sr=self.target_sr, n_mfcc=13), axis=1)
        }
        return features

import scipy.stats
