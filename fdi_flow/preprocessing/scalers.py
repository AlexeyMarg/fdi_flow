import numpy as np
import pandas as pd
from typing import Union, List, Optional

ArrayLike = Union[List[float], np.ndarray, pd.Series, pd.DataFrame]

class BaseScaler:
    """Base class for scalers that handle different input formats."""
    
    def _validate_input(self, data: ArrayLike) -> np.ndarray:
        """Converts input data to numpy.ndarray."""
        if isinstance(data, (list, pd.Series)):
            return np.array(data)
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported input type. Use List, np.ndarray, pd.Series, or pd.DataFrame.")

    def _format_output(self, scaled_data: np.ndarray, input_format: ArrayLike) -> ArrayLike:
        """Returns data in its original format."""
        if isinstance(input_format, list):
            return scaled_data.tolist()
        elif isinstance(input_format, pd.Series):
            return pd.Series(scaled_data, index=input_format.index)
        elif isinstance(input_format, pd.DataFrame):
            return pd.DataFrame(scaled_data, index=input_format.index, columns=input_format.columns)
        return scaled_data

    def apply(self, data: ArrayLike, axis: int = 1) -> ArrayLike:
        """Main method for applying scaling."""
        input_data = self._validate_input(data)
        scaled_data = self._scale_impl(input_data, axis)
        return self._format_output(scaled_data, data)

    def _scale_impl(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Scaling implementation (overridden in child classes)."""
        raise NotImplementedError


class StandardScaler(BaseScaler):
    """Standardize features by removing mean and scaling to unit variance.
    
    Parameters:
        with_mean: If True, center the data before scaling
        with_std: If True, scale the data to unit variance
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def _scale_impl(self, data: np.ndarray, axis: int) -> np.ndarray:
        if data.ndim == 1:
            return self._scale_1d(data)
        
        if axis == 0:
            # Scale each row
            return np.array([self._scale_1d(row) for row in data])
        else:
            # Scale each column (default)
            return np.array([self._scale_1d(col) for col in data.T]).T

    def _scale_1d(self, series: np.ndarray) -> np.ndarray:
        """Scale a single time series."""
        if len(series) == 0:
            return series.copy()
            
        scaled = series.astype(np.float64)  
        if self.with_mean:
            scaled -= np.mean(scaled)
        if self.with_std:
            std = np.std(scaled)
            if std > 0:
                scaled /= std
        return scaled


class MinMaxScaler(BaseScaler):
    """Transform features by scaling each to given range.
    
    Parameters:
        feature_range: Desired range of transformed data (min, max)
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.min, self.max = feature_range
        if self.min >= self.max:
            raise ValueError("Minimum of feature range must be less than maximum")

    def _scale_impl(self, data: np.ndarray, axis: int) -> np.ndarray:
        if data.ndim == 1:
            return self._scale_1d(data)
        
        if axis == 0:
            # Scale each row
            return np.array([self._scale_1d(row) for row in data])
        else:
            # Scale each column (default)
            return np.array([self._scale_1d(col) for col in data.T]).T

    def _scale_1d(self, series: np.ndarray) -> np.ndarray:
        """Scale a single time series."""
        if len(series) == 0:
            return series.copy()
            
        series = series.astype(np.float64)  
        data_min = np.min(series)
        data_max = np.max(series)
        
        if data_max - data_min == 0:
            return np.full_like(series, (self.min + self.max) / 2)
            
        scaled = (series - data_min) / (data_max - data_min)
        return scaled * (self.max - self.min) + self.min


class RobustScaler(BaseScaler):
    """Scale features using statistics that are robust to outliers.
    
    Parameters:
        with_centering: If True, center the data before scaling
        with_scaling: If True, scale the data to IQR
    """
    
    def __init__(self, with_centering: bool = True, with_scaling: bool = True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def _scale_impl(self, data: np.ndarray, axis: int) -> np.ndarray:
        if data.ndim == 1:
            return self._scale_1d(data)
        
        if axis == 0:
            # Scale each row
            return np.array([self._scale_1d(row) for row in data])
        else:
            # Scale each column (default)
            return np.array([self._scale_1d(col) for col in data.T]).T

    def _scale_1d(self, series: np.ndarray) -> np.ndarray:
        """Scale a single time series."""
        if len(series) == 0:
            return series.copy()
            
        scaled = series.astype(np.float64)
        
        if self.with_centering:
            median = np.median(scaled)
            scaled -= median
            
        if self.with_scaling:
            q75, q25 = np.percentile(scaled, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                scaled /= iqr
                
        return scaled