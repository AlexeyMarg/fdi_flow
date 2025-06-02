import numpy as np
from scipy.linalg import place_poles

class SlidingModeObserver:
    """
    Скользящий наблюдатель (Sliding Mode Observer) для линейных систем.
    
    Параметры:
        A (np.ndarray): Матрица состояния системы (n x n)
        B (np.ndarray): Матрица управления (n x m)
        C (np.ndarray): Матрица выхода (p x n)
        desired_poles (list): Желаемые полюса наблюдателя
        eta (float): Параметр запаса устойчивости (по умолчанию 1.0)
        eps (float): Параметр сглаживания sign функции (по умолчанию 0.1)
    """
    
    def __init__(self, A, B, C, desired_poles, eta=1.0, eps=0.1):
        self.A = A
        self.B = B
        self.C = C
        self.eta = eta
        self.eps = eps
        
        # Проверка размерностей
        self.n = A.shape[0]  # Размерность состояния
        self.m = B.shape[1]  # Размерность управления
        self.p = C.shape[0]  # Размерность выхода
        
        # Инициализация параметров
        self.L = None  # Матрица обратной связи
        self.K = None  # Матрица разрывного управления
        self.e_max = None  # Максимальная ошибка оценивания
        if desired_poles is None:
            # По умолчанию делаем полюса наблюдателя устойчивее полюсов системы
            system_poles = np.linalg.eigvals(self.A)
            self.desired_poles = (system_poles - 1)
        else:
            self.desired_poles = desired_poles
        # Расчет параметров наблюдателя
        self._compute_observer_gains()
        
    def _compute_observer_gains(self):
        """Вычисление матриц L и K для наблюдателя"""
        # Расчет L методом размещения полюсов
        result = place_poles(self.A.T, self.C.T, self.desired_poles)
        self.L = result.gain_matrix.T
        
        # Оценка максимальной ошибки e_max
        A_minus_LC = self.A - self.L @ self.C
        eigvals = np.linalg.eigvals(A_minus_LC)
        alpha = -max(eigvals.real)  # Скорость затухания
        M = 1.5  # Оценка для нормы экспоненты
        
        # Начальная ошибка (можно задать или оставить 1.0 по умолчанию)
        initial_error = 1.0
        steady_state_error = (M / alpha) * 1.0
        self.e_max = max(initial_error, steady_state_error) * 1.5  # С запасом
        
        # Расчет K для скользящего режима
        C_norm = np.linalg.norm(self.C, 2)
        A_LC_norm = np.linalg.norm(A_minus_LC, 2)
        self.K = (C_norm * A_LC_norm * self.e_max + self.eta) * np.sign(self.C)
        
    def _smooth_sign(self, S):
        """Сглаженная функция sign для уменьшения чёттеринга"""
        return np.tanh(S / self.eps)
    
    def compute_derivative(self, x_hat, y, u):
        """
        Вычисление производной оценки состояния
        
        Параметры:
            x_hat (np.ndarray): Текущая оценка состояния (n x 1)
            y (np.ndarray): Текущее измерение выхода (p x 1)
            u (np.ndarray): Текущее управление (m x 1)
            
        Возвращает:
            dx_hat (np.ndarray): Производная оценки состояния (n x 1)
        """
        S = y - self.C @ x_hat  # Ошибка выхода (скользящая поверхность)
        dx_hat = (self.A @ x_hat + self.B @ u + 
                  self.L @ S + 
                  self.K @ self._smooth_sign(S))
        return dx_hat
    
    def step(self, x_hat, y, dt):
        """
        Шаг работы наблюдателя
            
        Возвращает:
            x_hat (np.ndarray): Оценка состояния
        """
        dx_hat = self.compute_derivative(x_hat, y, np.zeros((self.m, 1)))
        x_hat += dx_hat * dt
                
        return x_hat
    


