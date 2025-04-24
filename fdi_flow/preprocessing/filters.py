import numpy as np
import pandas as pd
from typing import Union, List, Optional

ArrayLike = Union[List[float], np.ndarray, pd.Series, pd.DataFrame]

class BaseFilter:
    """Base class for filters that handle different input formats"""
    
    def _validate_input(self, data: ArrayLike) -> np.ndarray:
        """Converts input data to numpy.ndarray."""
        if isinstance(data, (list, pd.Series)):
            return np.array(data)
        elif isinstance(data, pd.DataFrame):
            return data.values  # Теперь оставляем ориентацию столбцов как есть
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported input type. Use List, np.ndarray, pd.Series, or pd.DataFrame.")

    def _format_output(self, filtered_data: np.ndarray, input_format: ArrayLike) -> ArrayLike:
        """Returns data in its original format."""
        if isinstance(input_format, list):
            return filtered_data.tolist()
        elif isinstance(input_format, pd.Series):
            return pd.Series(filtered_data, index=input_format.index)
        elif isinstance(input_format, pd.DataFrame):
            return pd.DataFrame(filtered_data, index=input_format.index, columns=input_format.columns)
        return filtered_data

    def apply(self, data: ArrayLike, **kwargs) -> ArrayLike:
        """The basic method for applying the filter."""
        input_data = self._validate_input(data)
        filtered_data = self._filter_impl(input_data, **kwargs)
        return self._format_output(filtered_data, data)

    def _filter_impl(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Filter implementation (overridden in child classes)."""
        raise NotImplementedError


class ExponentialSmoothing(BaseFilter):
    """Exponential smoothing with axis parameter.
    
    Parameters:
        alpha: Smoothing factor (0 < alpha < 1)
        axis: Axis along which to filter (0=rows, 1=columns, default=1)
    """

    def __init__(self, alpha: float = 0.3, axis: int = 1):
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be between 0 and 1")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
        self.alpha = alpha
        self.axis = axis

    def _filter_impl(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._smooth_1d(data)
        
        if self.axis == 0:
            # Применяем к каждой строке
            return np.array([self._smooth_1d(row) for row in data])
        else:
            # Применяем к каждому столбцу
            return np.array([self._smooth_1d(col) for col in data.T]).T

    def _smooth_1d(self, series: np.ndarray) -> np.ndarray:
        smoothed = np.zeros_like(series)
        smoothed[0] = series[0]
        for i in range(1, len(series)):
            smoothed[i] = self.alpha * series[i] + (1 - self.alpha) * smoothed[i-1]
        return smoothed


class DoubleExponentialSmoothing(BaseFilter):
    """Double exponential smoothing (Holt method) with axis support.
    
    Parameters:
        alpha: Level coefficient (0 < alpha < 1)
        beta: Trend coefficient (0 < beta < 1)
        axis: Axis along which to filter (0=rows, 1=columns, default=1)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, axis: int = 1):
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("Alpha and beta must be between 0 and 1")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
        self.alpha = alpha
        self.beta = beta
        self.axis = axis

    def _filter_impl(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._smooth_1d(data)
        
        if self.axis == 0:
            # Apply to each row
            return np.array([self._smooth_1d(row) for row in data])
        else:
            # Apply to each column (default)
            return np.array([self._smooth_1d(col) for col in data.T]).T

    def _smooth_1d(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        if n < 2:
            return series.copy()
            
        level, trend = np.zeros(n), np.zeros(n)
        level[0] = series[0]
        trend[0] = series[1] - series[0]  # Initial trend
        
        for i in range(1, n):
            level[i] = self.alpha * series[i] + (1 - self.alpha) * (level[i-1] + trend[i-1])
            trend[i] = self.beta * (level[i] - level[i-1]) + (1 - self.beta) * trend[i-1]
        
        return level + trend


class MovingAverage(BaseFilter):
    """Moving average filter with axis support and edge handling options.

    Parameters:
        window_size: Size of the moving window (must be positive)
        pad_with_zeros: If True, pads edges with zeros when window doesn't fit.
                      If False (default), returns shorter array.
        axis: Axis along which to filter (0=rows, 1=columns, default=1)
    """
    
    def __init__(self, window_size: int = 3, pad_with_zeros: bool = False, axis: int = 1):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
        self.window_size = window_size
        self.pad_with_zeros = pad_with_zeros
        self.axis = axis

    def _filter_impl(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._ma_1d(data)
        
        if self.axis == 0:
            return np.array([self._ma_1d(row) for row in data])
        else:
            return np.array([self._ma_1d(col) for col in data.T]).T

    def _ma_1d(self, series: np.ndarray) -> np.ndarray:
        if len(series) < self.window_size:
            return series.copy()
            
        if self.pad_with_zeros:
            return self._ma_1d_padded(series)
        return self._ma_1d_valid(series)

    def _ma_1d_valid(self, series: np.ndarray) -> np.ndarray:
        return np.convolve(series, np.ones(self.window_size)/self.window_size, mode='valid')
        
    def _ma_1d_padded(self, series: np.ndarray) -> np.ndarray:
        """Ensures output length matches input length"""
        pad_size = (self.window_size - 1) // 2
        if self.window_size % 2 == 0:
            pad_left = pad_size
            pad_right = pad_size + 1
        else:
            pad_left = pad_right = pad_size
            
        padded = np.pad(series, (pad_left, pad_right), mode='edge')
        result = np.convolve(padded, np.ones(self.window_size)/self.window_size, mode='valid')
        
        return result[:len(series)]

    def _format_output(self, filtered_data: np.ndarray, input_format: ArrayLike) -> ArrayLike:
        if isinstance(input_format, pd.DataFrame):
            if filtered_data.size == 0:
                return pd.DataFrame(columns=input_format.columns)
                
            if len(filtered_data) == len(input_format):
                return pd.DataFrame(filtered_data, 
                                 index=input_format.index,
                                 columns=input_format.columns)
            else:
                return pd.DataFrame(filtered_data, 
                                 columns=input_format.columns)
        return super()._format_output(filtered_data, input_format)
        


class MedianFilter(BaseFilter):
    """Median filter.
    
    Parameters:
        window_size: Size of the moving window (must be positive)
        axis: Axis along which to filter (0=rows, 1=columns, default=1)
    """

    def __init__(self, window_size: int = 3, pad_with_zeros: bool = False, axis: int = 1):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
        self.window_size = window_size
        self.pad_with_zeros = pad_with_zeros
        self.axis = axis

    def _filter_impl(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._median_1d(data)
        
        if self.axis == 0:
            return np.array([self._median_1d(row) for row in data])
        else:
            return np.array([self._median_1d(col) for col in data.T]).T

    def _median_1d(self, series: np.ndarray) -> np.ndarray:
        pad_size = self.window_size // 2
        
        if self.pad_with_zeros:
            padded = np.pad(series, (pad_size,), mode='constant', constant_values=0)
        else:
            padded = np.pad(series, (pad_size,), mode='edge')
            
        return np.array([
            np.median(padded[i:i+self.window_size])
            for i in range(len(series))
        ])

    def _format_output(self, filtered_data: np.ndarray, input_format: ArrayLike) -> ArrayLike:
        """Handles index alignment for different padding modes"""
        if isinstance(input_format, pd.DataFrame):
            return pd.DataFrame(filtered_data, 
                             index=input_format.index,
                             columns=input_format.columns)
        return super()._format_output(filtered_data, input_format)


class BandpassFilter(BaseFilter):
    """Bandpass filter (FFT-based) with axis support.
    
    Parameters:
        low_cutoff: Lower cutoff frequency (Hz)
        high_cutoff: Upper cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)
        axis: Axis along which to filter (0=rows, 1=columns, default=1)
    """

    def __init__(self, low_cutoff: float, high_cutoff: float, sample_rate: float, axis: int = 1):
        if low_cutoff >= high_cutoff:
            raise ValueError("Low cutoff must be less than high cutoff")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
        self.low = low_cutoff
        self.high = high_cutoff
        self.sr = sample_rate
        self.axis = axis

    def _filter_impl(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._bandpass_1d(data)
        
        if self.axis == 0:
            # Apply to each row
            return np.array([self._bandpass_1d(row) for row in data])
        else:
            # Apply to each column (default)
            return np.array([self._bandpass_1d(col) for col in data.T]).T

    def _bandpass_1d(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        if n == 0:
            return series.copy()
            
        fft = np.fft.fft(series)
        freqs = np.fft.fftfreq(n, d=1/self.sr)
        
        # Apply bandpass filter
        fft[(np.abs(freqs) < self.low) | (np.abs(freqs) > self.high)] = 0
        return np.real(np.fft.ifft(fft))

    def _format_output(self, filtered_data: np.ndarray, input_format: ArrayLike) -> ArrayLike:
        """Ensures proper index alignment for DataFrames"""
        if isinstance(input_format, pd.DataFrame):
            return pd.DataFrame(filtered_data, 
                             index=input_format.index,
                             columns=input_format.columns)
        return super()._format_output(filtered_data, input_format)
    
class KalmanFilter(BaseFilter):
    """Kalman Filter implementation for time series data.
    
    Parameters:
        q (float): Process noise covariance
        r (float): Measurement noise covariance
        x0 (float): Initial state value
        p0 (float): Initial estimate covariance
        axis (int): Axis along which to filter (0=rows, 1=columns, default=1)
    """
    
    def __init__(self, 
                 q: float = 1.0, 
                 r: float = 1.0, 
                 x0: Optional[float] = None,
                 p0: float = 1.0,
                 axis: int = 1):
        self.q = q  # Process noise
        self.r = r  # Measurement noise
        self.x0 = x0  # Initial state (None for auto-init)
        self.p0 = p0  # Initial covariance
        self.axis = axis
        
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")

    def _filter_impl(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._kalman_1d(data)
        
        if self.axis == 0:
            # Apply to each row
            return np.array([self._kalman_1d(row) for row in data])
        else:
            # Apply to each column
            return np.array([self._kalman_1d(col) for col in data.T]).T

    def _kalman_1d(self, series: np.ndarray) -> np.ndarray:
        """Apply 1D Kalman filter to a single time series."""
        n = len(series)
        if n == 0:
            return series.copy()
            
        # Initialize
        x = series[0] if self.x0 is None else self.x0
        p = self.p0
        filtered = np.zeros(n)
        filtered[0] = x
        
        for i in range(1, n):
            # Prediction
            x_pred = x
            p_pred = p + self.q
            
            # Update
            k_gain = p_pred / (p_pred + self.r)
            x = x_pred + k_gain * (series[i] - x_pred)
            p = (1 - k_gain) * p_pred
            
            filtered[i] = x
            
        return filtered

    def reset(self):
        """Reset filter state (for streaming/repeated use)."""
        self.x = None
        self.p = self.p0

    def _format_output(self, filtered_data: np.ndarray, input_format: ArrayLike) -> ArrayLike:
        """Ensure proper formatting of output data."""
        if isinstance(input_format, pd.DataFrame):
            return pd.DataFrame(filtered_data, 
                             index=input_format.index,
                             columns=input_format.columns)
        return super()._format_output(filtered_data, input_format)