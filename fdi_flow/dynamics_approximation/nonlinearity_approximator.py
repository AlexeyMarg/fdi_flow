import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error


class MultilevelLinearRegressor:
    """
    Многоуровневая линейная регрессия (Stacked Linear Regression).
    На каждом уровне результат предыдущей модели добавляется к входу следующей.
    """
    def __init__(self, n_levels=3, regressor_type='ridge', alpha=0.1):
        """
        Многоуровневая линейная регрессия для аппроксимации нелинейностей
        
        Parameters:
        -----------
        n_levels : int, default=3
            Количество уровней модели
        regressor_type : str, default='ridge'
            Тип регрессора: 'ridge', 'lasso' или 'linear'
        alpha : float, default=0.1
            Параметр регуляризации
        """
        self.n_levels = n_levels
        self.regressor_type = regressor_type
        self.alpha = alpha
        self.regressors = []
        self.feature_counts = []
    
    def _get_model(self):
        """
        Создает новый регрессор заданного типа.
        
        Returns:
            объект регрессора
        """
        if self.regressor_type == 'ridge':
            return Ridge(alpha=self.alpha)
        elif self.regressor_type == 'lasso':
            return Lasso(alpha=self.alpha)
        elif self.regressor_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown regressor_type: {self.regressor_type}")
    
    def fit(self, X, y):
        """
        Обучает многоуровневую модель регрессии.
        
        Parameters:
            X (array-like): Входные признаки, shape (n_samples, n_features)
            y (array-like): Целевые значения, shape (n_samples,)
        
        Returns:
            self: Возвращает обученную модель
        """
        X_current = X.copy()
        self.regressors = []
        self.feature_counts = []
        for level in range(self.n_levels):
            model = self._get_model()
            model.fit(X_current, y)
            self.regressors.append(model)
            self.feature_counts.append(X_current.shape[1])
            # Добавляем предсказания как новый признак
            y_pred = model.predict(X_current).reshape(-1, 1)
            X_current = np.hstack([X_current, y_pred])
        return self
    
    def predict(self, X):
        """
        Предсказывает значения на основе обученной модели.
        
        Parameters:
            X (array-like): Входные признаки, shape (n_samples, n_features)
        
        Returns:
            array-like: Предсказанные значения, shape (n_samples,)
        """
        X_current = X.copy()
        for model, feat_count in zip(self.regressors, self.feature_counts):
            # Используем только те признаки, которые были на обучении этого уровня
            X_input = X_current[:, :feat_count]
            y_pred = model.predict(X_input).reshape(-1, 1)
            X_current = np.hstack([X_current, y_pred])
        # Итоговое предсказание последнего уровня
        return self.regressors[-1].predict(X_current[:, :self.feature_counts[-1]])