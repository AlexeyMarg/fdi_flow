import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class NonlinearityApproximator:
    """
    Базовый класс для аппроксимации нелинейной динамики на основе временных рядов.
    """
    def fit(self, X, y):
        raise NotImplementedError("fit() должен быть реализован в подклассе")

    def predict(self, X):
        raise NotImplementedError("predict() должен быть реализован в подклассе")

class MultilevelLinearRegressor(NonlinearityApproximator):
    """
    Многоуровневая линейная регрессия (Stacked Linear Regression).
    На каждом уровне результат предыдущей модели добавляется к входу следующей.
    Можно использовать LinearRegression, Ridge или Lasso.
    """
    def __init__(self, levels=2, model_type='ridge', alpha=1.0, random_state=None):
        self.levels = levels
        self.model_type = model_type
        self.alpha = alpha
        self.random_state = random_state
        self.models = []
        self.feature_counts = []

    def _get_model(self):
        if self.model_type == 'ridge':
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.model_type == 'lasso':
            return Lasso(alpha=self.alpha, random_state=self.random_state)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X, y):
        X_current = X.copy()
        self.models = []
        self.feature_counts = []
        for level in range(self.levels):
            model = self._get_model()
            model.fit(X_current, y)
            self.models.append(model)
            self.feature_counts.append(X_current.shape[1])
            # Добавляем предсказания как новый признак
            y_pred = model.predict(X_current).reshape(-1, 1)
            X_current = np.hstack([X_current, y_pred])

    def predict(self, X):
        X_current = X.copy()
        for model, feat_count in zip(self.models, self.feature_counts):
            # Используем только те признаки, которые были на обучении этого уровня
            X_input = X_current[:, :feat_count]
            y_pred = model.predict(X_input).reshape(-1, 1)
            X_current = np.hstack([X_current, y_pred])
        # Итоговое предсказание последнего уровня
        return self.models[-1].predict(X_current[:, :self.feature_counts[-1]])
