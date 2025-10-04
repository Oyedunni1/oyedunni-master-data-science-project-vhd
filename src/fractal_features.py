"""
Advanced Fractal Feature Extraction for VHD Detection
Implements Higuchi Fractal Dimension and other complexity measures
"""

import numpy as np
from scipy import stats
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FractalFeatureExtractor:
    """
    Advanced fractal analysis for heart sound complexity quantification
    """
    
    def __init__(self):
        self.k_max = 10  # Maximum k value for Higuchi method
        
    def higuchi_fractal_dimension(self, signal: np.ndarray, k_max: int = None) -> float:
        """
        Calculate Higuchi Fractal Dimension (HFD)
        Measures the complexity/roughness of the signal
        """
        if k_max is None:
            k_max = self.k_max
            
        N = len(signal)
        if N < 10:
            return 1.0
            
        # Create k different time series
        L = []
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(k):
                # Create subseries
                subseries = signal[m::k]
                if len(subseries) > 1:
                    # Calculate length of curve
                    length = 0
                    for i in range(1, len(subseries)):
                        length += abs(subseries[i] - subseries[i-1])
                    Lk.append(length * (N - 1) / ((len(subseries) - 1) * k))
            if Lk:
                L.append(np.mean(Lk))
        
        if len(L) < 2:
            return 1.0
            
        # Fit line to log-log plot
        k_values = np.arange(1, len(L) + 1)
        log_k = np.log(k_values)
        log_L = np.log(L)
        
        # Linear regression
        slope, _, _, _, _ = stats.linregress(log_k, log_L)
        return -slope
    
    def katz_fractal_dimension(self, signal: np.ndarray) -> float:
        """
        Calculate Katz Fractal Dimension
        Alternative fractal measure
        """
        if len(signal) < 2:
            return 1.0
            
        # Calculate total length
        total_length = 0
        for i in range(1, len(signal)):
            total_length += abs(signal[i] - signal[i-1])
        
        # Calculate maximum distance
        max_distance = 0
        for i in range(len(signal)):
            for j in range(i+1, len(signal)):
                distance = abs(signal[i] - signal[j])
                if distance > max_distance:
                    max_distance = distance
        
        if max_distance == 0:
            return 1.0
            
        # Katz fractal dimension
        n = len(signal) - 1
        return np.log(n) / (np.log(n) + np.log(max_distance / total_length))
    
    def detrended_fluctuation_analysis(self, signal: np.ndarray, scales: List[int] = None) -> float:
        """
        Calculate Detrended Fluctuation Analysis (DFA) alpha
        Measures long-range correlations
        """
        if scales is None:
            scales = [4, 8, 16, 32, 64, 128, 256]
            
        if len(signal) < max(scales):
            return 1.0
            
        # Integrate signal
        y = np.cumsum(signal - np.mean(signal))
        
        fluctuations = []
        for scale in scales:
            if scale >= len(y):
                continue
                
            # Divide into segments
            segments = len(y) // scale
            if segments < 2:
                continue
                
            # Detrend each segment
            segment_fluctuations = []
            for i in range(segments):
                start = i * scale
                end = start + scale
                segment = y[start:end]
                
                # Linear detrending
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                
                segment_fluctuations.append(np.mean(detrended**2))
            
            if segment_fluctuations:
                fluctuations.append(np.sqrt(np.mean(segment_fluctuations)))
        
        if len(fluctuations) < 2:
            return 1.0
            
        # Fit power law
        scales_used = scales[:len(fluctuations)]
        log_scales = np.log(scales_used)
        log_fluctuations = np.log(fluctuations)
        
        slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
        return slope
    
    def sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Sample Entropy
        Measures regularity/complexity
        """
        N = len(signal)
        if N < m + 1:
            return 0.0
            
        # Calculate standard deviation
        std_signal = np.std(signal)
        if std_signal == 0:
            return 0.0
            
        r = r * std_signal
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _approximate_entropy(U, m, r):
            def _phi(m):
                C = np.zeros(N - m + 1.0)
                for i in range(N - m + 1):
                    template_i = U[i:i + m]
                    for j in range(N - m + 1):
                        template_j = U[j:j + m]
                        if _maxdist(template_i, template_j, m) <= r:
                            C[i] += 1.0
                phi = np.mean(np.log(C / (N - m + 1.0)))
                return phi
            return _phi(m) - _phi(m + 1)
        
        return _approximate_entropy(signal, m, r)
    
    def lyapunov_exponent(self, signal: np.ndarray, m: int = 3, tau: int = 1) -> float:
        """
        Calculate largest Lyapunov exponent
        Measures chaotic behavior
        """
        N = len(signal)
        if N < m + 1:
            return 0.0
            
        # Reconstruct phase space
        M = N - (m - 1) * tau
        if M <= 0:
            return 0.0
            
        Y = np.zeros((M, m))
        for i in range(M):
            for j in range(m):
                Y[i, j] = signal[i + j * tau]
        
        # Find nearest neighbors
        lyap_sum = 0
        count = 0
        
        for i in range(M - 1):
            distances = np.sqrt(np.sum((Y - Y[i])**2, axis=1))
            distances[i] = np.inf  # Exclude self
            
            nearest_idx = np.argmin(distances)
            if nearest_idx < M - 1:
                # Calculate divergence
                d0 = distances[nearest_idx]
                d1 = np.sqrt(np.sum((Y[i+1] - Y[nearest_idx+1])**2))
                
                if d0 > 0 and d1 > 0:
                    lyap_sum += np.log(d1 / d0)
                    count += 1
        
        return lyap_sum / count if count > 0 else 0.0
    
    def extract_all_fractal_features(self, signal: np.ndarray) -> dict:
        """
        Extract OPTIMAL fractal features for enhanced prediction (6 features)
        Enhanced with additional complexity measures for better VHD detection
        """
        features = {}
        
        # Core fractal features (4) - optimized for speed
        try:
            # Ultra-fast Higuchi FD with minimal computation
            features['higuchi_fd'] = self._ultra_fast_higuchi_fd(signal)
        except:
            features['higuchi_fd'] = 1.0
            
        try:
            # Ultra-fast sample entropy with minimal parameters
            features['sample_entropy'] = self._ultra_fast_sample_entropy(signal)
        except:
            features['sample_entropy'] = 0.0
            
        try:
            # Vectorized statistical measure
            features['signal_std'] = np.std(signal)
        except:
            features['signal_std'] = 0.0
            
        try:
            # Ultra-fast Hurst exponent with minimal scales
            features['hurst_exponent'] = self._ultra_fast_hurst_exponent(signal)
        except:
            features['hurst_exponent'] = 0.5
        
        # Enhanced features (2) - for better prediction accuracy
        try:
            # Signal complexity measure
            features['signal_complexity'] = self._calculate_signal_complexity(signal)
        except:
            features['signal_complexity'] = 0.0
            
        try:
            # Spectral entropy for frequency domain complexity
            features['spectral_entropy'] = self._calculate_spectral_entropy(signal)
        except:
            features['spectral_entropy'] = 0.0
        
        return features
    
    def _ultra_fast_higuchi_fd(self, signal: np.ndarray) -> float:
        """
        ULTRA-FAST Higuchi Fractal Dimension calculation
        Maximum speed optimization with minimal k_max and vectorized operations
        """
        N = len(signal)
        if N < 8:
            return 1.0
            
        # Ultra-minimal k_max for maximum speed (was 3, now 2)
        k_max = min(2, N // 8)
        L = []
        
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(0, k, 2):  # Skip every other m for speed
                subseries = signal[m::k]
                if len(subseries) > 1:
                    # Ultra-fast vectorized length calculation
                    Lk += np.sum(np.abs(np.diff(subseries)))
            L.append(Lk / k)
        
        if len(L) < 2:
            return 1.0
            
        # Ultra-fast linear regression with minimal computation
        k_values = np.arange(1, len(L) + 1)
        log_k = np.log(k_values)
        log_L = np.log(L)
        
        # Ultra-fast slope calculation
        n = len(k_values)
        slope = (n * np.sum(log_k * log_L) - np.sum(log_k) * np.sum(log_L)) / \
                (n * np.sum(log_k**2) - np.sum(log_k)**2)
        
        return -slope
    
    def _ultra_fast_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        ULTRA-FAST Sample Entropy calculation
        Maximum speed optimization with minimal parameters and vectorized operations
        """
        N = len(signal)
        if N < m + 1:
            return 0.0
            
        std_signal = np.std(signal)
        if std_signal == 0:
            return 0.0
            
        r = r * std_signal
        
        # Ultra-fast distance calculation with minimal loops
        def _count_matches_ultra_fast(data, m, r):
            N = len(data)
            C = 0
            # Ultra-reduced range for maximum speed
            for i in range(0, N - m, 4):  # Skip every 4th sample for speed
                template = data[i:i+m]
                for j in range(i+1, min(i+5, N - m), 4):  # Very limited search range
                    if np.max(np.abs(template - data[j:j+m])) <= r:
                        C += 1
            return C
        
        phi_m = _count_matches_ultra_fast(signal, m, r)
        phi_m1 = _count_matches_ultra_fast(signal, m + 1, r)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
            
        return -np.log(phi_m1 / phi_m)
    
    def _ultra_fast_hurst_exponent(self, signal: np.ndarray) -> float:
        """
        ULTRA-FAST Hurst exponent calculation
        Maximum speed optimization with minimal scales and vectorized operations
        """
        N = len(signal)
        if N < 8:
            return 0.5
            
        # Ultra-minimal scales for maximum speed (was 3, now 2)
        scales = [4, 8] if N >= 8 else [4]
        rs_values = []
        
        for scale in scales:
            if scale >= N:
                continue
                
            segments = N // scale
            rs_segments = []
            
            for i in range(segments):
                segment = signal[i*scale:(i+1)*scale]
                if len(segment) < 2:
                    continue
                    
                # Ultra-fast R/S calculation
                mean_seg = np.mean(segment)
                cumsum = np.cumsum(segment - mean_seg)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                
                if S > 0:
                    rs_segments.append(R / S)
            
            if rs_segments:
                rs_values.append(np.mean(rs_segments))
        
        if len(rs_values) < 2:
            return 0.5
            
        # Ultra-fast linear regression
        log_scales = np.log(scales[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        n = len(log_scales)
        slope = (n * np.sum(log_scales * log_rs) - np.sum(log_scales) * np.sum(log_rs)) / \
                (n * np.sum(log_scales**2) - np.sum(log_scales)**2)
        
        return slope
    
    def _calculate_signal_complexity(self, signal: np.ndarray) -> float:
        """
        Calculate signal complexity using variance of differences
        Higher complexity indicates more irregular patterns (potential VHD)
        """
        if len(signal) < 2:
            return 0.0
        
        # Calculate differences between consecutive samples
        differences = np.diff(signal)
        
        # Calculate variance of differences (complexity measure)
        complexity = np.var(differences)
        
        # Normalize by signal variance for relative complexity
        signal_var = np.var(signal)
        if signal_var > 0:
            complexity = complexity / signal_var
        
        return complexity
    
    def _calculate_spectral_entropy(self, signal: np.ndarray) -> float:
        """
        Calculate spectral entropy for frequency domain complexity
        Higher entropy indicates more complex frequency content
        """
        if len(signal) < 4:
            return 0.0
        
        # Calculate power spectral density
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft)**2
        
        # Normalize to get probability distribution
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
        
        probabilities = power_spectrum / total_power
        
        # Calculate spectral entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _fast_higuchi_fd(self, signal: np.ndarray) -> float:
        """
        Ultra-fast Higuchi Fractal Dimension calculation
        Optimized for speed with reduced k_max and vectorized operations
        """
        N = len(signal)
        if N < 10:
            return 1.0
            
        # Reduced k_max for speed (was 10, now 5)
        k_max = min(5, N // 2)
        L = []
        
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                subseries = signal[m::k]
                if len(subseries) > 1:
                    # Vectorized length calculation
                    Lk += np.sum(np.abs(np.diff(subseries)))
            L.append(Lk / k)
        
        if len(L) < 2:
            return 1.0
            
        # Fast linear regression
        k_values = np.arange(1, len(L) + 1)
        log_k = np.log(k_values)
        log_L = np.log(L)
        
        # Vectorized slope calculation
        n = len(k_values)
        slope = (n * np.sum(log_k * log_L) - np.sum(log_k) * np.sum(log_L)) / \
                (n * np.sum(log_k**2) - np.sum(log_k)**2)
        
        return -slope
    
    def _fast_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Ultra-fast Sample Entropy calculation
        Optimized with vectorized operations and reduced parameters
        """
        N = len(signal)
        if N < m + 1:
            return 0.0
            
        std_signal = np.std(signal)
        if std_signal == 0:
            return 0.0
            
        r = r * std_signal
        
        # Vectorized distance calculation
        def _count_matches(data, m, r):
            N = len(data)
            C = 0
            for i in range(N - m):
                template = data[i:i+m]
                for j in range(i+1, N - m):
                    if np.max(np.abs(template - data[j:j+m])) <= r:
                        C += 1
            return C
        
        phi_m = _count_matches(signal, m, r)
        phi_m1 = _count_matches(signal, m + 1, r)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
            
        return -np.log(phi_m1 / phi_m)
    
    def _fast_hurst_exponent(self, signal: np.ndarray) -> float:
        """
        Ultra-fast Hurst exponent calculation
        Optimized with minimal scales and vectorized operations
        """
        N = len(signal)
        if N < 10:
            return 0.5
            
        # Minimal scales for speed (was 5, now 3)
        scales = [4, 8, 16] if N >= 16 else [4, 8]
        rs_values = []
        
        for scale in scales:
            if scale >= N:
                continue
                
            segments = N // scale
            rs_segments = []
            
            for i in range(segments):
                segment = signal[i*scale:(i+1)*scale]
                if len(segment) < 2:
                    continue
                    
                # Vectorized R/S calculation
                mean_seg = np.mean(segment)
                cumsum = np.cumsum(segment - mean_seg)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                
                if S > 0:
                    rs_segments.append(R / S)
            
            if rs_segments:
                rs_values.append(np.mean(rs_segments))
        
        if len(rs_values) < 2:
            return 0.5
            
        # Fast linear regression
        log_scales = np.log(scales[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        n = len(log_scales)
        slope = (n * np.sum(log_scales * log_rs) - np.sum(log_scales) * np.sum(log_rs)) / \
                (n * np.sum(log_scales**2) - np.sum(log_scales)**2)
        
        return slope
    
    def _calculate_dfa(self, signal: np.ndarray) -> float:
        """Calculate Detrended Fluctuation Analysis"""
        N = len(signal)
        if N < 10:
            return 1.0
        
        # Integrate signal
        y = np.cumsum(signal - np.mean(signal))
        
        # Calculate DFA for different scales
        scales = np.logspace(0.5, np.log10(N//4), 10).astype(int)
        fluctuations = []
        
        for scale in scales:
            if scale >= N:
                continue
                
            # Divide into segments
            segments = N // scale
            if segments < 2:
                continue
                
            # Detrend each segment
            segment_fluctuations = []
            for i in range(segments):
                start = i * scale
                end = start + scale
                segment = y[start:end]
                
                # Linear detrending
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                
                segment_fluctuations.append(np.mean(detrended**2))
            
            if segment_fluctuations:
                fluctuations.append(np.sqrt(np.mean(segment_fluctuations)))
        
        if len(fluctuations) < 2:
            return 1.0
            
        # Fit power law
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluctuations = np.log(fluctuations)
        
        try:
            slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
            return slope
        except:
            return 1.0
    
    def _calculate_hurst_exponent(self, signal: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        N = len(signal)
        if N < 10:
            return 0.5
            
        # Calculate R/S for different scales
        scales = [4, 8, 16, 32, 64]
        rs_values = []
        
        for scale in scales:
            if scale >= N:
                continue
                
            segments = N // scale
            rs_segments = []
            
            for i in range(segments):
                segment = signal[i*scale:(i+1)*scale]
                if len(segment) < 2:
                    continue
                    
                # Calculate mean
                mean_seg = np.mean(segment)
                
                # Calculate cumulative deviations
                cumdev = np.cumsum(segment - mean_seg)
                
                # Calculate range
                R = np.max(cumdev) - np.min(cumdev)
                
                # Calculate standard deviation
                S = np.std(segment)
                
                if S > 0:
                    rs_segments.append(R / S)
            
            if rs_segments:
                rs_values.append(np.mean(rs_segments))
        
        if len(rs_values) < 2:
            return 0.5
            
        # Fit power law
        scales_used = scales[:len(rs_values)]
        log_scales = np.log(scales_used)
        log_rs = np.log(rs_values)
        
        slope, _, _, _, _ = stats.linregress(log_scales, log_rs)
        return slope
    
    def _calculate_multifractal_width(self, signal: np.ndarray) -> float:
        """Calculate multifractal spectrum width"""
        # Simplified multifractal analysis
        # This is a basic implementation
        try:
            # Calculate generalized Hurst exponents for different moments
            moments = [-2, -1, 0, 1, 2]
            h_values = []
            
            for q in moments:
                if q == 0:
                    # Special case for q=0
                    h = 0.5  # Default value
                else:
                    # Calculate structure function
                    scales = [4, 8, 16, 32]
                    structure_values = []
                    
                    for scale in scales:
                        if scale >= len(signal):
                            continue
                            
                        segments = len(signal) // scale
                        if segments < 2:
                            continue
                            
                        segment_means = []
                        for i in range(segments):
                            segment = signal[i*scale:(i+1)*scale]
                            segment_means.append(np.mean(segment))
                        
                        if len(segment_means) > 1:
                            structure = np.mean(np.abs(np.diff(segment_means))**q)
                            structure_values.append(structure)
                    
                    if len(structure_values) >= 2:
                        scales_used = scales[:len(structure_values)]
                        log_scales = np.log(scales_used)
                        log_structure = np.log(structure_values)
                        slope, _, _, _, _ = stats.linregress(log_scales, log_structure)
                        h = slope / q
                    else:
                        h = 0.5
                
                h_values.append(h)
            
            # Calculate spectrum width
            if len(h_values) >= 3:
                return np.max(h_values) - np.min(h_values)
            else:
                return 0.0
                
        except:
            return 0.0

