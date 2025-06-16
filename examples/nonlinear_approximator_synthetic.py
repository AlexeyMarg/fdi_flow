"""
Расширенное тестирование классов аппроксимации на различных синтетических данных.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Импортируем нужный класс
from fdi_flow.dynamics_approximation.nonlinearity_approximator import MultilevelLinearRegressor

# Функции для генерации разных типов нелинейных данных
def generate_sine_data(n_samples=100):
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(n_samples)
    return X, y

def generate_quadratic_data(n_samples=100):
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = X.ravel()**2 + 0.1 * np.random.randn(n_samples)
    return X, y

def generate_exp_data(n_samples=100):
    X = np.sort(2 * np.random.rand(n_samples, 1), axis=0)
    y = np.exp(X).ravel() + 0.1 * np.random.randn(n_samples)
    return X, y

# Тестируем модели с разным числом уровней
def test_model(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = []
    for n_levels in [1, 2, 3, 5]:
        model = MultilevelLinearRegressor(n_levels=n_levels, regressor_type='ridge', alpha=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((n_levels, mse, r2))
        
    return results, X_train, X_test, y_train, y_test

# Запускаем тесты и выводим результаты
np.random.seed(42)

datasets = {
    'sine': generate_sine_data(),
    'quadratic': generate_quadratic_data(),
    'exponential': generate_exp_data()
}

for name, (X, y) in datasets.items():
    print(f"\nТестирование на данных типа {name}:")
    results, X_train, X_test, y_train, y_test = test_model(X, y, name)
    
    for n_levels, mse, r2 in results:
        print(f"  Уровней: {n_levels}, MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Визуализируем результаты лучшей модели (с 3 уровнями)
    model = MultilevelLinearRegressor(n_levels=3, regressor_type='ridge', alpha=0.1)
    model.fit(X_train, y_train)
    
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='blue', label='Тренировочные данные', alpha=0.5)
    plt.scatter(X_test, y_test, color='green', label='Тестовые данные', alpha=0.5)
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Предсказания модели')
    plt.title(f'Аппроксимация функции типа {name}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'examples/{name}_approximation.png')