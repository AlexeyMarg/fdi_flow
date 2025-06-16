import numpy as np
from fdi_flow.dynamics_approximation.nonlinearity_approximator import MultilevelLinearRegressor
import matplotlib.pyplot as plt

# Генерируем данные
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + 0.2 * np.random.randn(300)

# Обучаем модель
model = MultilevelLinearRegressor(levels=3, model_type='ridge', alpha=0.5)
model.fit(X, y)
y_pred = model.predict(X)

# Визуализация
plt.plot(X, y, 'o', label='Данные', alpha=0.4)
plt.plot(X, y_true, label='Истинная функция')
plt.plot(X, y_pred, label='Аппроксимация (MLR)')
plt.legend()
plt.title('MLR аппроксимация синуса')
plt.show()
