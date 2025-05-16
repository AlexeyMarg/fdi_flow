# fdi_flow

`fdi_flow` is a Python library for fault detection, forecasting, and preprocessing of time series data.

## Features

- Preprocessing utilities (filters, scalers, segmenters)
- Linear and nonlinear, continuous and discrete models simulation
- Forecasting and fault detection frameworks
- Example notebooks for quick start

#

## Project Structure

```
fdi_flow/
├── examples/                  # Jupyter notebooks with usage examples
├── fdi_flow/                  # Core library
│   ├── preprocessing/         # Filters, scalers, segmenters
│   ├── models/                # Data generators, linear and nonlinear models
│   ├── forecasting/           # Forecasting utilities
│   ├── detectors/             # Fault detection models
│   └── dynamics_approximation/ # Approximation of unknown dynamics with ML models
└── README.md                  # This file
```

## Core Components

### `preprocessing.filters`
- `ExponentialSmoothing`, `DoubleExponentialSmoothing`: Smooth data with exponential methods.
- `MovingAverage`, `MedianFilter`: Classic window-based filters.
- `BandpassFilter`: Frequency-based filtering using FFT.
- `KalmanFilter`: Recursive estimator for noisy signals.

### `preprocessing.scalers`
- `StandardScaler`: Zero-mean, unit-variance transformation.
- `MinMaxScaler`: Rescales data to a specific range.
- `RobustScaler`: Uses IQR and median for outlier robustness.

### `preprocessing.segmenters`
- `TimeSeriesSegmenter`: Splits data into fixed-size overlapping or non-overlapping windows.
- `TimeSeriesResampler`: Resamples series to a fixed length.
- `SplineUpsampler`: Smooth upsampling with spline interpolation.
- `FailureEncoder`: Encodes fault presence based on most frequent values with thresholds.


### `models.linear_models`
- `LinearStateSpaceModel`: continuous-time model with optional ODE solver support (`solve_ivp`).
- `LinearDiscreteStateSpaceModel`: discrete-time model with fixed time step.

### `models.nonlinear_models.py`  
- `NonlinearStateSpaceModel`: continuous-time simulation defined by user-specified `f(x, u, t)` and `g(x, u, t)`
- `NonlinearDiscreteStateSpaceModel`: discrete-time version using `f(x, u, k)` and `g(x, u, k)`

### `models.utils.py`  
- `plot_simulation_results(result)`: plots time series for system inputs, outputs, and state trajectories based on a standard simulation result dictionary.

### `detectors.catboost_detectors.py`
- `CatBoostFaultDetector`: catboost based classifier for fault detection.
  
### `detectors.lgbm_detectors.py`
- `LGBMFaultDetector`: LGBM based classifier for fault detection.

### `detectors.sklearn_detectors.py`
- `KNNFaultDetector`: k-NN based classifier for fault detection.
- `RandomForestFaultDetector`: random forest based classifier for fault detection.
- `SVMFaultDetector`: SVM based classifier for fault detection.
- `GradientBoostingFaultDetector`: gradient boosted trees based classifier for fault detection.

### `detectors.xgb_detectors.py`
- `XGBFaultDetector`: XGB based classifier for fault detection.




