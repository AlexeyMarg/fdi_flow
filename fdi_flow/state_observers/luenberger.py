import numpy as np
from scipy.signal import place_poles

class LuenbergerObserver:
    """
    Класс реализующий наблюдатель Люенбергера для линейных систем
    
    Параметры:
    ----------
    A : numpy.ndarray
        Матрица состояния системы (n x n)
    B : numpy.ndarray
        Матрица управления (n x m)
    C : numpy.ndarray
        Матрица выхода (p x n)
    L : numpy.ndarray, optional
        Матрица коэффициентов усиления наблюдателя (n x p). Если не задана, 
        будет рассчитана автоматически с помощью метода размещения полюсов.
    desired_poles : array_like, optional
        Желаемые полюса наблюдателя. По умолчанию в 5-10 раз быстрее полюсов системы.
    """
    
    def __init__(self, A, B, C, dt, x_hat=None, L=None, desired_poles=None):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)
        self.dt = dt
        
        # Проверка размерностей матриц
        n = self.A.shape[0]
        m = self.B.shape[1]
        p = self.C.shape[0]
        
        assert self.A.shape == (n, n), "Матрица A должна быть квадратной"
        assert self.B.shape == (n, m), "Несоответствие размерностей A и B"
        assert self.C.shape == (p, n), "Несоответствие размерностей A и C"
        
        # Расчет матрицы L (коэффициентов усиления наблюдателя)
        if L is not None:
            self.L = np.array(L, dtype=float)
            assert self.L.shape == (n, p), "Неверная размерность матрицы L"
        else:
            if desired_poles is None:
                # По умолчанию делаем полюса наблюдателя устойчивее полюсов системы
                system_poles = np.linalg.eigvals(self.A)
                desired_poles = (system_poles - 1)
            
            L = self.calculate_observer_gain(desired_poles=desired_poles)
            self.L = L
        
        # Инициализация состояния наблюдателя
        if x_hat is None:
            self.x_hat = np.zeros((n, 1))
        else:
            self.x_hat = np.array(x_hat, dtype=float)
            assert self.x_hat.shape == (n, 1), "Неверная размерность начального состояния"
    
    def calculate_observer_gain(self, desired_poles):
        """
        Расчет матрицы коэффициентов усиления наблюдателя методом размещения полюсов
        
        Параметры:
        ----------
        desired_poles : array_like
            Желаемые полюса наблюдателя (должны быть устойчивы)
            
        Возвращает:
        ----------
        L : numpy.ndarray
            Матрица коэффициентов усиления наблюдателя (n x p)
        """

        L = place_poles(self.A.T, self.C.T, desired_poles)
        
        return L.gain_matrix.T
    
    def update(self, u, y):
        """
        Обновление состояния наблюдателя
        
        Параметры:
        ----------
        u : numpy.ndarray
            Вектор управления (m x 1)
        y : numpy.ndarray
            Вектор измерений (p x 1)
            
        Возвращает:
        ----------
        x_hat : numpy.ndarray
            Оценка состояния системы (n x 1)
        """
        u = np.array(u, dtype=float).reshape(-1, 1)
        y = np.array(y, dtype=float).reshape(-1, 1)
        
        # Уравнение наблюдателя Люенбергера
        dx_hat = self.A @ self.x_hat + self.B @ u - self.L @ (self.C @ self.x_hat - y)
        
        # Простейшая интеграция (на практике следует использовать ODE solver)
        self.x_hat += dx_hat * self.dt
        
        return self.x_hat.copy()
    