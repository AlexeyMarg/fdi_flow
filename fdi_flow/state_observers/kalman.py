import numpy as np

class KalmanFilter:
    """
    Класс, реализующий фильтр Калмана для линейной системы.
    
    Параметры:
    - F: Матрица перехода состояния (state transition matrix)
    - B: Матрица управления (control matrix)
    - H: Матрица наблюдений (observation matrix)
    - Q: Матрица ковариации шума процесса (process noise covariance)
    - R: Матрица ковариации шума измерений (measurement noise covariance)
    - x0: Начальное состояние (initial state estimate)
    - P0: Начальная ковариация ошибки состояния (initial error covariance)
    """
    
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F  # Матрица перехода состояния
        self.B = B  # Матрица управления
        self.H = H  # Матрица наблюдений
        self.Q = Q  # Шум процесса
        self.R = R  # Шум измерений
        self.x = x0  # Текущая оценка состояния
        self.P = P0  # Текущая ковариация ошибки
        
    def predict(self, u=None):
        """
        Этап предсказания (прогноза)
        
        Параметры:
        - u: Вектор управления (control vector), опционально
        
        Возвращает:
        - x: Предсказанное состояние
        - P: Предсказанная ковариация ошибки
        """
        # Прогноз состояния
        if u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
            
        # Прогноз ковариации ошибки
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, z):
        """
        Этап обновления (коррекции) по измерениям
        
        Параметры:
        - z: Вектор измерений (measurement vector)
        
        Возвращает:
        - x: Обновленная оценка состояния
        - P: Обновленная ковариация ошибки
        """
        # Ошибка предсказания
        y = z - self.H @ self.x
        
        # Ковариация ошибки предсказания
        S = self.H @ self.P @ self.H.T + self.R
        
        # Оптимальный коэффициент усиления Калмана
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Обновление оценки состояния
        self.x = self.x + K @ y
        
        # Обновление ковариации ошибки
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        return self.x.copy(), self.P.copy()
    
    def step(self, z, u=None):
        """
        Полный шаг фильтра (предсказание + обновление)
        
        Параметры:
        - z: Вектор измерений (measurement vector)
        - u: Вектор управления (control vector), опционально
        
        Возвращает:
        - x: Обновленная оценка состояния
        - P: Обновленная ковариация ошибки
        """
        self.predict(u)
        return self.update(z)

