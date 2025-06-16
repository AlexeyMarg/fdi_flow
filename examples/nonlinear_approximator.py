"""
Пример базового использования классов аппроксимации нелинейности.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from fdi_flow.dynamics_approximation.nonlinearity_approximator import MultilevelLinearRegressor

# Создаем синтетические данные с нелинейной зависимостью
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# Разделяем на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создаем и обучаем многоуровневую линейную регрессию
model = MultilevelLinearRegressor(n_levels=3, regressor_type='ridge', alpha=0.1)
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Оцениваем качество модели
mse = np.mean((y_test - y_pred) ** 2)
print(f"Среднеквадратичная ошибка: {mse:.4f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Тренировочные данные')
plt.scatter(X_test, y_test, color='green', label='Тестовые данные')

# Создаем плотную сетку для визуализации предсказаний
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Предсказания модели')

plt.title('Многоуровневая аппроксимация нелинейной функции')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('examples/basic_usage_result.png')
plt.show()

# Информация о структуре модели
print("Структура модели:")
for i, count in enumerate(model.feature_counts):
    print(f"Уровень {i+1}: {count} признаков")