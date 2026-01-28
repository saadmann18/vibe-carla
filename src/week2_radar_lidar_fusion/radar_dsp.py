# src/week2_radar_lidar_fusion/radar_dsp.py
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq

class RadarDSP:
    """Radar digital signal processing module."""
    
    def __init__(self, fft_size=256, sample_rate=10000):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.clutter_filter_state = np.zeros(5)  # RLS state
    
    # =====================================================================
    # DOPPLER FFT PROCESSING
    # =====================================================================
    
    def compute_range_doppler_map(self, radar_detections, num_doppler_bins=64):
        """
        Compute Range-Doppler map from radar detections.
        
        This mimics the DSP pipeline: IF signal → FFT → Range-Doppler matrix
        
        Args:
            radar_detections: List of CARLA radar detections
            num_doppler_bins: Number of Doppler FFT bins
        
        Returns:
            rd_map: (num_range_bins, num_doppler_bins) power matrix
        """
        if not radar_detections:
            return np.zeros((64, num_doppler_bins))
        
        ranges = np.array([det.depth for det in radar_detections])
        velocities = np.array([det.velocity for det in radar_detections])
        
        # Create synthetic IF signal (simplified)
        # In real radar: received signal after mixer/IF stage
        range_bins = np.linspace(0, 100, 64)
        rd_map = np.zeros((64, num_doppler_bins))
        
        for det in radar_detections:
            range_idx = int((det.depth / 100) * 64)
            # Doppler shift: f_doppler = 2 * v * f_carrier / c
            # Normalized to bin index
            doppler_idx = int((det.velocity / 10) * (num_doppler_bins / 2) + num_doppler_bins / 2)
            
            if 0 <= range_idx < 64 and 0 <= doppler_idx < num_doppler_bins:
                rd_map[range_idx, doppler_idx] += 1.0
        
        return rd_map
    
    def doppler_fft(self, time_series):
        """
        Apply FFT to extract Doppler information.
        
        Args:
            time_series: Nx1 array of time-domain samples
        
        Returns:
            doppler_spectrum: Magnitude spectrum (freq domain)
            freqs: Frequency bins
        """
        N = len(time_series)
        fft_result = fft(time_series, n=self.fft_size)
        freqs = fftfreq(self.fft_size, 1 / self.sample_rate)
        
        magnitude = np.abs(fft_result) / N
        
        return magnitude[:self.fft_size // 2], freqs[:self.fft_size // 2]
    
    # =====================================================================
    # CFAR (CONSTANT FALSE ALARM RATE) DETECTOR
    # =====================================================================
    
    def cfar_detector(self, rd_map, guard_cells=2, train_cells=8, pfa=1e-4):
        """
        Cell Averaging CFAR detector for target detection.
        
        Maintains constant false alarm rate by adapting threshold to noise level.
        
        Args:
            rd_map: Range-Doppler map from compute_range_doppler_map()
            guard_cells: Cells around test cell (protection)
            train_cells: Training cells for noise estimation
            pfa: Probability of false alarm (typical: 1e-3 to 1e-6)
        
        Returns:
            detections: List of (range_idx, doppler_idx, test_power) above threshold
        """
        detections = []
        height, width = rd_map.shape
        
        # Calculate threshold multiplier from PFA
        # For Rayleigh distribution: T = N * ln(1/PFA)
        num_train = 2 * train_cells
        threshold_factor = num_train * np.log(1 / pfa)
        
        for i in range(guard_cells + train_cells, height - train_cells - guard_cells):
            for j in range(guard_cells + train_cells, width - train_cells - guard_cells):
                # Extract training cells (exclude guard cells)
                train_region = rd_map[
                    i - train_cells - guard_cells : i + train_cells + guard_cells + 1,
                    j - train_cells - guard_cells : j + train_cells + guard_cells + 1
                ]
                
                # Remove guard cell region
                guard_region = rd_map[
                    i - guard_cells : i + guard_cells + 1,
                    j - guard_cells : j + guard_cells + 1
                ]
                
                # Estimate noise power (average of training region)
                noise_power = (np.sum(train_region) - np.sum(guard_region)) / num_train
                
                # Test cell power
                test_power = rd_map[i, j]
                
                # Adaptive threshold
                threshold = threshold_factor * noise_power
                
                if test_power > threshold:
                    detections.append((i, j, test_power))
        
        return detections
    
    # =====================================================================
    # ADAPTIVE LMS FILTER FOR CLUTTER REJECTION
    # =====================================================================
    
    def adaptive_lms_clutter_filter(self, radar_signal, desired_signal, mu=0.01, order=5):
        """
        Least Mean Squares (LMS) adaptive filter for clutter rejection.
        
        Learns and cancels clutter (ground reflection) from received signal.
        
        Args:
            radar_signal: Received signal (contaminated by clutter)
            desired_signal: Reference signal (clutter characteristics)
            mu: Step size (0.001-0.1 typical)
            order: Filter order (number of taps)
        
        Returns:
            filtered_signal: Clutter-suppressed signal
            weights: Learned filter taps
        """
        N = len(radar_signal)
        filtered_signal = np.zeros(N)
        weights = np.zeros(order)
        
        for n in range(order, N):
            # Sliding window of desired signal
            x = desired_signal[n-order:n][::-1]
            
            # Estimate clutter using current weights
            clutter_estimate = np.dot(weights, x)
            
            # Output: suppressed signal
            error = radar_signal[n] - clutter_estimate
            filtered_signal[n] = error
            
            # Update weights (LMS update rule)
            weights = weights + mu * error * x
        
        return filtered_signal, weights
    
    # =====================================================================
    # UTILITY FUNCTIONS
    # =====================================================================
    
    def apply_hamming_window(self, signal_data):
        """Apply Hamming window to reduce spectral leakage."""
        window = signal.hamming(len(signal_data))
        return signal_data * window
