import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
import tensorflow.keras as keras  # Для keras.models.load_model
import os  # Для создания фиктивных данных


class TimeSeriesPredictor:
    """
    Класс для загрузки, предобработки, обучения и оценки моделей
    временных рядов (LSTM или Conv1D-LSTM).
    """

    def __init__(
        self,
        time_steps: int = 50,
        output_step: int = 1,
        model_type: str = "convlstm",  # 'lstm' или 'convlstm'
        epochs: int = 50,
        batch_size: int = 32,
        optimizer: str = "adam",
        loss: str = "mse",
        metrics: list = None,
        verbose: int = 1,
    ):
        """
        Инициализирует параметры предсказателя временных рядов.

        Args:
            time_steps (int): Количество предыдущих временных шагов для использования в качестве входных данных (X).
            output_step (int): Количество будущих временных шагов для предсказания (Y).
            model_type (str): Тип архитектуры модели ('lstm' для простой LSTM, 'convlstm' для Conv1D + LSTM).
            epochs (int): Количество эпох обучения.
            batch_size (int): Размер пакета для обучения.
            optimizer (str): Оптимизатор для компиляции модели (например, 'adam').
            loss (str): Функция потерь для компиляции модели (например, 'mse').
            metrics (list): Список метрик для отслеживания во время обучения (например, ['accuracy', 'mae']).
            verbose (int): Режим вывода информации во время обучения (0, 1 или 2).
        """
        self.time_steps = time_steps
        self.output_step = output_step
        self.out_steps = output_step  # Для слоя Dense, обычно совпадает с output_step
        self.model_type = (
            model_type.lower()
        )  # Приводим к нижнему регистру для сравнения
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = (
            metrics
            if metrics is not None
            else ["accuracy", "mean_squared_error", "mean_absolute_error"]
        )
        self.verbose = verbose

        self.model = None
        self.min_value = None
        self.max_value = None
        self.history = None
        self.normalized_test_data = (
            None  # Сохраняем нормализованные тестовые данные для отрисовки
        )

    def _create_dataset(self, data: np.ndarray) -> tuple:
        """
        Преобразует 1D массив во временные последовательности (X) и
        соответствующие будущие значения (Y) для обучения с учителем.

        Args:
            data (np.ndarray): Входной массив данных временного ряда (1D).

        Returns:
            tuple: (X, Y), где X - входные последовательности, Y - целевые выходные данные.
        """
        X, Y = [], []
        for i in range(len(data) - self.time_steps - self.output_step + 1):
            # Входная последовательность (X)
            X.append(data[i : (i + self.time_steps)])
            # Целевой выход (Y)
            Y.append(
                data[(i + self.time_steps) : (i + self.time_steps + self.output_step)]
            )
        return np.array(X), np.array(Y)

    def load_and_preprocess_data(
        self, train_filenames: list, test_filename: str
    ) -> tuple:
        """
        Загружает, предобрабатывает и разделяет данные временных рядов
        для обучения и тестирования.

        Args:
            train_filenames (list): Список путей к CSV-файлам для обучения.
            test_filename (str): Путь к CSV-файлу для тестирования.

        Returns:
            tuple: (X_train, y_train, X_test, y_test), обработанные данные,
                   готовые для подачи на вход модели.
        """
        all_raw_data = []

        # Загружаем и собираем все исходные данные
        for filename in train_filenames:
            data = pd.read_csv(filename)
            # Даунсэмплинг данных для уменьшения размера и потенциального шума
            data_arr = data["y"].values.reshape(-1, 1).astype("float32")[::200]
            all_raw_data.append(data_arr)

        # Загружаем тестовые данные
        test_data_raw = pd.read_csv(test_filename)
        test_data_arr = (
            test_data_raw["y"].values.reshape(-1, 1).astype("float32")[::200]
        )
        all_raw_data.append(test_data_arr)

        # Вычисляем глобальные мин/макс значения по ВСЕМ данным (обучающим и тестовым)
        # для согласованной нормализации
        self.min_value = np.min([arr.min() for arr in all_raw_data])
        self.max_value = np.max([arr.max() for arr in all_raw_data])

        print(
            f"Глобальное минимальное значение: {self.min_value:.4f}, Глобальное максимальное значение: {self.max_value:.4f}"
        )

        # Нормализуем каждый массив в диапазоне [0, 1] на основе глобальных мин/макс
        normalized_arrays = [
            (arr - self.min_value) / (self.max_value - self.min_value)
            for arr in all_raw_data
        ]

        # Визуализируем нормализованные данные
        self.plot_normalized_data(normalized_arrays, len(train_filenames))

        # Создаем последовательности для обучающих данных
        X_train_list, y_train_list = [], []
        for array in normalized_arrays[
            :-1
        ]:  # Итерируем по всем, кроме последнего массива (который является тестовыми данными)
            X, Y = self._create_dataset(array)
            X_train_list.append(X)
            y_train_list.append(Y)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Создаем последовательности для тестовых данных
        X_test, y_test = self._create_dataset(normalized_arrays[-1])
        self.normalized_test_data = normalized_arrays[
            -1
        ]  # Сохраняем для последующей отрисовки

        # Изменяем форму данных для входа Keras (samples, time_steps, features)
        # Здесь features = 1, так как это одномерный временной ряд
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        print(f"\nФорма X_train: {X_train.shape}")
        print(f"Форма y_train: {y_train.shape}")
        print(f"Форма X_test: {X_test.shape}")
        print(f"Форма y_test: {y_test.shape}")
        print(f"Форма normalized_test_data: {self.normalized_test_data.shape}")

        return X_train, y_train, X_test, y_test

    def build_model(self):
        """
        Определяет и компилирует модель нейронной сети в зависимости от self.model_type.
        """
        model = Sequential()
        input_shape = (self.time_steps, 1)

        if self.model_type == "lstm":
            print("\n--- Строим модель LSTM ---")
            model.add(LSTM(100, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Dense(self.out_steps))
        elif self.model_type == "convlstm":
            print("\n--- Строим модель Conv1D + LSTM ---")
            model.add(
                Conv1D(
                    filters=64,
                    kernel_size=3,
                    activation="relu",
                    input_shape=input_shape,
                )
            )
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(self.out_steps))
        else:
            raise ValueError(
                f"Неизвестный тип модели: {self.model_type}. Выберите 'lstm' или 'convlstm'."
            )

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model = model
        self.model.summary()

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """
        Обучает модель.

        Args:
            X_train (np.ndarray): Обучающие входные данные.
            y_train (np.ndarray): Обучающие целевые данные.
            X_val (np.ndarray, optional): Валидационные входные данные. Defaults to None.
            y_val (np.ndarray, optional): Валидационные целевые данные. Defaults to None.
        """
        if self.model is None:
            raise RuntimeError(
                "Модель не построена. Вызовите build_model() перед train_model()."
            )

        print("\n--- Обучение модели ---")
        validation_data = (
            (X_val, y_val) if X_val is not None and y_val is not None else None
        )
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=validation_data,
        )

    def evaluate_model(
        self, X_data: np.ndarray, y_data: np.ndarray, data_name: str = "Test"
    ):
        """
        Оценивает производительность модели на заданном наборе данных.

        Args:
            X_data (np.ndarray): Входные данные для оценки.
            y_data (np.ndarray): Целевые данные для оценки.
            data_name (str): Имя набора данных для вывода (например, "Test", "DropTest").
        """
        if self.model is None:
            raise RuntimeError(
                "Модель не построена. Вызовите build_model() перед evaluate_model()."
            )

        print(f"\n--- Оценка модели на данных {data_name} ---")
        results = self.model.evaluate(X_data, y_data, verbose=1)
        print(f"{data_name} Loss: {results[0]:.4f}")
        for i, metric_name in enumerate(self.metrics):
            print(
                f"{data_name} {metric_name.replace('_', ' ').title()}: {results[i+1]:.4f}"
            )

    def predict(self, X_data: np.ndarray) -> np.ndarray:
        """
        Делает предсказания с помощью обученной модели.

        Args:
            X_data (np.ndarray): Входные данные для предсказания.

        Returns:
            np.ndarray: Сделанные предсказания.
        """
        if self.model is None:
            raise RuntimeError(
                "Модель не построена. Вызовите build_model() перед predict()."
            )
        return self.model.predict(X_data)

    def plot_normalized_data(self, normalized_arrays: list, num_train_files: int):
        """
        Визуализирует нормализованные данные.

        Args:
            normalized_arrays (list): Список нормализованных массивов данных.
            num_train_files (int): Количество файлов, использованных для обучения.
        """
        plt.figure(figsize=(15, 6))
        current_x_offset = 0
        for i, arr in enumerate(normalized_arrays):
            color = "b" if i < num_train_files else "r"
            label = (
                "Обучающие данные"
                if i == 0
                else ("Тестовые данные" if i == num_train_files else "_nolegend_")
            )
            plt.plot(
                [x for x in range(current_x_offset, current_x_offset + len(arr))],
                arr,
                color=color,
                label=label,
            )
            current_x_offset += len(arr)
        plt.title("Нормализованные данные временных рядов")
        plt.xlabel("Индекс выборки (после даунсэмплинга)")
        plt.ylabel("Нормализованное значение")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_training_history(self):
        """
        Визуализирует историю обучения модели (потери и метрики).
        """
        if self.history is None:
            print("История обучения недоступна. Сначала обучите модель.")
            return

        plt.figure(figsize=(12, 5))

        # Потери
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["loss"], label="Потери на обучении")
        if "val_loss" in self.history.history:
            plt.plot(self.history.history["val_loss"], label="Потери на валидации")
        plt.title("Потери модели")
        plt.xlabel("Эпоха")
        plt.ylabel("Потери")
        plt.legend()

        # Метрики (например, точность или MAE)
        plt.subplot(1, 2, 2)
        # Выбираем первую метрику после потерь для отображения
        main_metric_name = self.metrics[0].replace("_", " ").title()
        plt.plot(
            self.history.history[self.metrics[0]],
            label=f"{main_metric_name} на обучении",
        )
        if f"val_{self.metrics[0]}" in self.history.history:
            plt.plot(
                self.history.history[f"val_{self.metrics[0]}"],
                label=f"{main_metric_name} на валидации",
            )
        plt.title(f"{main_metric_name} модели")
        plt.xlabel("Эпоха")
        plt.ylabel(main_metric_name)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        X_test_raw: np.ndarray,
        y_test_raw: np.ndarray,
        num_points_to_plot: int = 200,
    ):
        """
        Визуализирует фактические и предсказанные значения для сегмента тестовых данных.

        Args:
            X_test_raw (np.ndarray): Исходные входные данные для тестирования (перед изменением формы).
            y_test_raw (np.ndarray): Исходные целевые данные для тестирования (перед изменением формы).
            num_points_to_plot (int): Количество точек для визуализации.
        """
        if self.model is None:
            print("Модель не обучена. Невозможно сделать предсказания.")
            return
        if self.normalized_test_data is None:
            print(
                "Нормализованные тестовые данные недоступны. Убедитесь, что load_and_preprocess_data была вызвана."
            )
            return

        # Делаем предсказания
        predictions = self.predict(X_test_raw)

        # Выбираем сегмент для отрисовки
        start_idx = 0
        end_idx = min(num_points_to_plot, len(predictions))

        plt.figure(figsize=(15, 6))

        # Отрисовываем фактические нормализованные данные
        # normalized_test_data содержит весь исходный нормализованный ряд
        plt.plot(
            self.normalized_test_data[
                start_idx : start_idx + self.time_steps + end_idx
            ],
            label="Фактические данные (нормализованные)",
            color="blue",
        )

        # Отрисовываем предсказания
        # Предсказания соответствуют y_test_raw, которые являются будущими точками
        # Необходимо сместить предсказания на TIME_STEPS, чтобы они совпали с фактическими будущими значениями
        predicted_plot_indices = np.arange(
            start_idx + self.time_steps, start_idx + self.time_steps + end_idx
        )

        # Проверяем, что предсказания имеют правильную форму для отрисовки
        if self.output_step == 1:
            plt.plot(
                predicted_plot_indices,
                predictions[start_idx:end_idx].flatten(),
                label="Предсказанные данные (нормализованные)",
                color="red",
                linestyle="--",
            )
        else:
            # Для multi-step предсказаний, можно отрисовать только первый шаг или усреднить
            print(
                f"Визуализация предсказаний для OUTPUT_STEP > 1 ({self.output_step}) сложнее и не полностью реализована в этом примере."
            )
            print("Предсказания имеют форму:", predictions.shape)
            plt.plot(
                predicted_plot_indices,
                predictions[start_idx:end_idx, 0].flatten(),
                label="Предсказанные данные (первый шаг, нормализованные)",
                color="red",
                linestyle="--",
            )

        plt.title("Фактические vs. Предсказанные тестовые данные (Нормализованные)")
        plt.xlabel("Временной шаг")
        plt.ylabel("Нормализованное значение")
        plt.legend()
        plt.grid(True)
        plt.show()
