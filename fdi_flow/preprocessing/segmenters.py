import numpy as np
import pandas as pd
from typing import Union, List, Optional
from scipy.interpolate import interp1d, splrep, splev
from collections import Counter

ArrayLike = Union[List[float], np.ndarray, pd.Series, pd.DataFrame]

class TimeSeriesSegmenter:
    """Cutting time series into smaller segments with axis support.
    
    Parameters:
        window_size: Length of each segment
        step: Offset for the next segment (default 1)
        drop_last: Drop the last segment if shorter than window_size (default True)
        axis: Axis along which to segment (0=rows, 1=columns, default=1)
    """
    
    def __init__(self, window_size: int, step: int = 1, drop_last: bool = True, axis: int = 1):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
        self.window_size = window_size
        self.step = step
        self.drop_last = drop_last
        self.axis = axis

    def segment(self, data: ArrayLike) -> List[ArrayLike]:
        """Segment the input data along specified axis."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return self._segment_pandas(data)
        return self._segment_array(np.array(data))

    def _segment_array(self, data: np.ndarray) -> List[np.ndarray]:
        if data.ndim == 1:
            return self._segment_1d(data)
        
        if self.axis == 0:
            # Для axis=0 (ряды как временные ряды)
            n_series = data.shape[0]
            n_samples = data.shape[1]
            segments = []
            
            for start in range(0, n_samples - self.window_size + 1, self.step):
                end = start + self.window_size
                segment = data[:, start:end]  # Берем срез по всем рядам
                segments.append(segment)
            
            if not self.drop_last and (n_samples - start - self.step) > 0:
                last_segment = data[:, -self.window_size:]
                segments.append(last_segment)
                
        else:
            # Для axis=1 (столбцы как временные ряды - по умолчанию)
            n_series = data.shape[1]
            n_samples = data.shape[0]
            segments = []
            
            for start in range(0, n_samples - self.window_size + 1, self.step):
                end = start + self.window_size
                segment = data[start:end, :]  # Берем срез по всем столбцам
                segments.append(segment)
            
            if not self.drop_last and (n_samples - start - self.step) > 0:
                last_segment = data[-self.window_size:, :]
                segments.append(last_segment)
        
        return segments

    def _segment_pandas(self, data: Union[pd.Series, pd.DataFrame]) -> List[Union[pd.Series, pd.DataFrame]]:
        if isinstance(data, pd.Series):
            return self._segment_series(data)
        else:
            return self._segment_dataframe(data)

    def _segment_series(self, series: pd.Series) -> List[pd.Series]:
        segments = []
        n_samples = len(series)
        for start in range(0, n_samples - self.window_size + 1, self.step):
            end = start + self.window_size
            segments.append(series.iloc[start:end])
        
        if not self.drop_last and (n_samples - start - self.step) > 0:
            last_segment = series.iloc[-self.window_size:]
            segments.append(last_segment)
        
        return segments

    def _segment_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        segments = []
        n_samples = len(df)
        
        for start in range(0, n_samples - self.window_size + 1, self.step):
            end = start + self.window_size
            segments.append(df.iloc[start:end])
        
        if not self.drop_last and (n_samples - start - self.step) > 0:
            last_segment = df.iloc[-self.window_size:]
            segments.append(last_segment)
        
        return segments


class TimeSeriesResampler:
    """Change the sampling frequency of time series with axis support.
    
    Parameters:
        target_size: Desired length of the series after resampling
        method: Interpolation method ('linear', 'nearest', 'cubic', default='linear')
        axis: Axis along which to resample (0=rows, 1=columns, default=1)
    """
    
    def __init__(self, target_size: int, method: str = 'linear', axis: int = 1):
        if target_size <= 0:
            raise ValueError("Target size must be positive")
        if method not in ('linear', 'nearest', 'cubic'):
            raise ValueError("Method must be 'linear', 'nearest' or 'cubic'")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
        self.target_size = target_size
        self.method = method
        self.axis = axis

    def resample(self, data: ArrayLike) -> ArrayLike:
        """Resample input data along specified axis."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return self._resample_pandas(data)
        return self._resample_array(np.array(data))

    def _resample_array(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._resample_1d(data)
        
        if self.axis == 0:
            # Resample each row (each row is a time series)
            return np.array([self._resample_1d(row) for row in data])
        else:
            # Resample each column (default - each column is a time series)
            return np.array([self._resample_1d(col) for col in data.T]).T

    def _resample_1d(self, series: np.ndarray) -> np.ndarray:
        """Resample a single time series."""
        if len(series) == 0:
            return series.copy()
            
        n_samples = len(series)
        x_original = np.linspace(0, 1, n_samples)
        x_new = np.linspace(0, 1, self.target_size)
        
        f = interp1d(x_original, series, kind=self.method, fill_value="extrapolate")
        return f(x_new)

    def _resample_pandas(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(data, pd.Series):
            resampled = self._resample_1d(data.values)
            new_index = np.linspace(data.index[0], data.index[-1], self.target_size)
            return pd.Series(resampled, index=new_index)
        else:
            if self.axis == 0:
                # Resample along index (time axis)
                resampled_values = np.array([self._resample_1d(data[col].values) for col in data.columns])
                new_index = np.linspace(data.index[0], data.index[-1], self.target_size)
                return pd.DataFrame(resampled_values.T, index=new_index, columns=data.columns)
            else:
                # Resample each column separately (default)
                resampled = {col: self._resample_1d(data[col].values) for col in data.columns}
                new_index = np.arange(self.target_size)
                return pd.DataFrame(resampled, index=new_index)
            

class SplineUpsampler:
    """Increase the length of a time series using splines.

    Parameters:
        target_size: Desired length after upsampling
        spline_degree: Spline degree (1=linear, 2=quadratic, 3=cubic, default=3)
        smoothing: Smoothing parameter (0=interpolation, >0=smoothing, default=0)
        axis: Axis along which to upsample (0=rows, 1=columns, default=1)
    """
    
    def __init__(self, target_size: int, spline_degree: int = 3, 
                 smoothing: float = 0.0, axis: int = 1):
        if target_size <= 0:
            raise ValueError("Target size must be positive")
        if spline_degree not in (1, 2, 3):
            raise ValueError("Spline degree must be 1, 2 or 3")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
        self.target_size = target_size
        self.spline_degree = spline_degree
        self.smoothing = smoothing
        self.axis = axis

    def upsample(self, data: ArrayLike) -> ArrayLike:
        """Upsample input data along specified axis."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return self._upsample_pandas(data)
        return self._upsample_array(np.array(data))

    def _upsample_array(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return self._upsample_1d(data)
        
        if self.axis == 0:
            # Upsample each row (each row is a time series)
            return np.array([self._upsample_1d(row) for row in data])
        else:
            # Upsample each column (default - each column is a time series)
            return np.array([self._upsample_1d(col) for col in data.T]).T

    def _upsample_1d(self, series: np.ndarray) -> np.ndarray:
        """Upsample a single time series using splines."""
        if len(series) == 0:
            return series.copy()
            
        n_samples = len(series)
        x_original = np.linspace(0, 1, n_samples)
        x_new = np.linspace(0, 1, self.target_size)
        
        tck = splrep(x_original, series, k=self.spline_degree, s=self.smoothing)
        return splev(x_new, tck)

    def _upsample_pandas(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(data, pd.Series):
            upsampled = self._upsample_1d(data.values)
            new_index = np.linspace(data.index[0], data.index[-1], self.target_size)
            return pd.Series(upsampled, index=new_index)
        else:
            if self.axis == 0:
                # Upsample along index (time axis)
                upsampled_values = np.array([self._upsample_1d(data[col].values) for col in data.columns])
                new_index = np.linspace(data.index[0], data.index[-1], self.target_size)
                return pd.DataFrame(upsampled_values.T, index=new_index, columns=data.columns)
            else:
                # Upsample each column separately (default)
                upsampled = {col: self._upsample_1d(data[col].values) for col in data.columns}
                new_index = np.linspace(data.index[0], data.index[-1], self.target_size)
                return pd.DataFrame(upsampled, index=new_index)
            
            
class FailureEncoder:
    """Converts a time series with failure labels to mode (most frequent non-zero code).

    Parameters:
    failure_threshold: Percentage of non-zero values ​​to detect failure (0-100)
    axis: Axis along which to process data (0=rows, 1=columns, default 1)
    """
    
    def __init__(self, failure_threshold: float = 10.0, axis: int = 1):
        if not 0 <= failure_threshold <= 100:
            raise ValueError("failure_threshold должен быть между 0 и 100")
        if axis not in (0, 1):
            raise ValueError("axis должен быть 0 (строки) или 1 (столбцы)")
            
        self.failure_threshold = failure_threshold
        self.axis = axis

    def encode(self, data: ArrayLike) -> ArrayLike:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return self._encode_pandas(data)
        return self._encode_array(np.array(data))

    def _encode_array(self, data: np.ndarray) -> Union[int, np.ndarray]:
        if data.ndim == 1:
            return self._encode_1d(data)
        
        if self.axis == 0:
            # Обработка каждой строки как временного ряда
            return np.array([self._encode_1d(row) for row in data])
        else:
            # Обработка каждого столбца как временного ряда (по умолчанию)
            return np.array([self._encode_1d(col) for col in data.T]).T

    def _encode_1d(self, series: np.ndarray) -> int:
        non_zero = series[series != 0]
        if len(non_zero) == 0:
            return 0
            
        failure_ratio = 100 * len(non_zero) / len(series)
        if failure_ratio >= self.failure_threshold:
            counts = Counter(non_zero)
            return counts.most_common(1)[0][0]  # Возвращаем моду
        return 0

    def _encode_pandas(self, data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
        if isinstance(data, pd.Series):
            return self._encode_1d(data.values)
        else:
            if self.axis == 0:
                return pd.Series([self._encode_1d(row) for row in data.values], 
                               index=data.index)
            else:
                return data.apply(lambda col: self._encode_1d(col.values))