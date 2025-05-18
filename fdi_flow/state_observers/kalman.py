import numpy as np

class MatrixKalmanFilter:
    def __init__(self, F, H, Q, R, x_initial=None, P_initial=None):
        """
        Фильтр Калмана.
        
        Параметры:
            F (numpy array): матриц переходов между состояниями
            H (numpy array): матрица наблюдения (измерений)
            Q (numpy array): матрица ковариации процесса
            R (numpy array): матрица ковариации наблюдений
            x_initial (numpy array): начальное состояние
            P_initial (numpy array): начальная ковариация ошибки

        Пример использования:
            Определим параметры системы:
            F = np.array([[1]])               # матрица перехода
            H = np.array([[1]])               # матрица наблюдения
            Q = np.array([[0.01]])            # ковариация процесса
            R = np.array([[0.1]])             # ковариация наблюдения
            x_initial = np.array([0])         # начальное состояние
            P_initial = np.eye(1) * 1         # начальная ковариация ошибки
            
            # Создаем экземпляр фильтра
            kf = MatrixKalmanFilter(F=F, H=H, Q=Q, R=R, x_initial=x_initial, P_initial=P_initial)
            
            # Измерения (пример)
            measurements = np.array([1.0, 2.0, 3.0, 4.0])
            
            estimates = []                    # список оцененных значений
            
            for z in measurements:
                kf.step(z)
                estimates.append(float(kf.x))  # добавляем текущую оценку
        """
        if x_initial is None:
            raise ValueError('Необходимо задать начальное состояние.')
        if P_initial is None:
            raise ValueError('Необходимо задать начальную ковариацию ошибки.')
            
        self.F = F      # матрица перехода
        self.H = H      # матрица наблюдения
        self.Q = Q      # процесс-шум
        self.R = R      # наблюдение-шум
        self.x = x_initial.reshape(-1, 1)  # начальное состояние
        self.P = P_initial  # начальная ковариация

    def predict(self):
        """
        Шаг прогнозирования состояния.
        Возвращает предиктивное состояние и новую ковариацию.
        """
        # Предсказываем новое состояние: x_(k|k-1) = F*x_(k-1|k-1)
        x_pred = np.dot(self.F, self.x)
        
        # Преобразуем ковариацию: P_(k|k-1) = F*P_(k-1|k-1)*F.T + Q
        P_pred = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return x_pred, P_pred

    def update(self, z):
        """
        Шаг коррекции (обновления).
        Аргументы:
            z (numpy array): вектор новых измерений
        """
        # Вычисляем разность между наблюдением и моделью: y = z - H*x_
        innovation = z.reshape(-1, 1) - np.dot(self.H, self.x)
        
        # Ковариация инноваций: S = H*P_*H.T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Коэффициент усиления Калмана: K = P_*H.T*S^(-1)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Корректируем состояние: x_ += K*y
        self.x += np.dot(K, innovation)
        
        # Обновляем ковариацию ошибки: P -= K*H*P_
        self.P -= np.dot(np.dot(K, self.H), self.P)

    def step(self, z):
        """
        Полный шаг фильтра Калмана — предсказание и коррекция.
        Аргументы:
            z (numpy array): новые данные измерения
        """
        x_pred, P_pred = self.predict()
        self.x = x_pred.copy()
        self.P = P_pred.copy()
        self.update(z)




