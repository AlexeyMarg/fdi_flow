import numpy as np
from scipy.linalg import place

class LuenbergerObserver:
    def __init__(self, A, B, C, poles=None):
        """_summary_

        Args:
            A (numpy array): Матрица динамики системы
            B (numpy array): Входная матрица
            C (numpy array): Выходная матрица измерений
            poles (numpy array): Полюсы наблюдателя. None по умолчанию.

        Raises:
            ValueError: Не заданы полюсы наблюдателя
        
        Example:
            # Параметры системы
            A = np.array([[1, 1],
                        [-1, 0]])
            B = np.array([[-1],
                        [1]])
            C = np.array([[1, 0]])
            
            # Желаемые полюсы наблюдателя (корни характеристического уравнения замкнутой системы)
            desired_poles = [-2, -3]
            
            # Создаём экземпляр наблюдателя
            luenberger_observer = LuenbergerObserver(A=A, B=B, C=C, poles=desired_poles)
            
            # Тестируем работу наблюдателя
            initial_state_guess = np.array([[0], [0]])  # начальное предположение состояния
            measured_output = np.array([[1]])           # измеренный выход
            input_signal = 1                            # внешний управляющий сигнал
            
            next_state_estimate = luenberger_observer.update_state_estimate(initial_state_guess, measured_output, input_signal)
        """
        self.A = A          # Матрица динамики системы
        self.B = B          # Входная матрица
        self.C = C          # Выходная матрица измерений
        
        if poles is None:
            raise ValueError("Необходимо задать желаемые полюсы наблюдателя.")
            
        # Вычисляем матрицу коэффициентов усиления наблюдателя L методом pole placement
        self.L = place(self.A.T, self.C.T, poles).T
    
    def update_state_estimate(self, x_hat_prev, y_measured, u_input=0):
        """Обновляет оценку состояния с учётом новых измерений"""
        # Обновление состояния наблюдателя
        x_hat_next = self.A.dot(x_hat_prev) + self.B*u_input + self.L.dot(y_measured - self.C.dot(x_hat_prev))
        return x_hat_next


