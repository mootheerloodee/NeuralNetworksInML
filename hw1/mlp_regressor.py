import numpy as np

class MLPRegressor:
    """
    Многослойный перцептрон (MLP) для задачи регрессии, использующий алгоритм
    обратного распространения ошибки
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, n_iter=100):
        """
        Конструктор класса

        Параметры:
            hidden_layer_sizes (tuple): Кортеж, определяющий архитектуру
        скрытых слоев. Например (100, 10) - два скрытых слоя, размером 100 и 10
        нейронов, соответственно
            learning_rate (float): Скорость обучения
            n_iter (int): Количество итераций градиентного спуска
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None
        self.a_ = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)


    def forward(self, X):
        """
        Реализация forward pass

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)

        Возвращает:
            np.ndarray: Предсказания модели
        """
        self.a_ = [X]
        curr_a = X
        for i in range(len(self.coef_) - 1):
            z = np.dot(curr_a, self.coef_[i]) + self.intercept_[i]
            curr_a = self.sigmoid(z)
            self.a_.append(curr_a)
        final_z = np.dot(curr_a, self.coef_[-1]) + self.intercept_[-1]
        self.a_.append(final_z)
        return final_z


    def backward(self, X, y):
        """
        Реализация backward pass

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)
            y (np.ndarray): Вектор таргета длины n_samples

        Возвращает:
            coef_grad (list of np.ndarray): Список градиентов по весам для каждого слоя
            intercept_grad (list of np.ndarray): Список градиентов по смещениям для каждого слоя
        """
        n_objects, n_features = X.shape
        coef_grad = []
        intercept_grad = []
        y_predict = self.forward(X)
        delta = 2 / n_objects * (y_predict - y)
        for i in range(len(self.coef_) - 1, -1, -1):
            cur_coef_grad = np.dot(self.a_[i].T, delta)
            cur_intercept_grad = np.sum(delta, axis=0, keepdims=True)
            coef_grad.insert(0, cur_coef_grad)
            intercept_grad.insert(0, cur_intercept_grad)
            if i > 0:
                delta = np.dot(delta, self.coef_[i].T) * self.sigmoid_derivative(self.a_[i])
        return coef_grad, intercept_grad


    def fit(self, X, y):
        """
        Обучение модели

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)
            y (np.ndarray): Вектор таргета длины n_samples
        """
        n_objects, n_features = X.shape
        layers_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.coef_ = []
        self.intercept_ = []
        self.loss_history_ = []
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        for i in range(len(layers_sizes) - 1):
            self.coef_.append(np.random.randn(layers_sizes[i], layers_sizes[i+1]) * 0.1)
            self.intercept_.append(np.zeros((1, layers_sizes[i + 1])))
        for i in range(self.n_iter):
            coef_grad, intercept_grad = self.backward(X, y)
            for j in range(len(self.coef_)):
                self.coef_[j] -= self.learning_rate * coef_grad[j]
                self.intercept_[j] -= self.learning_rate * intercept_grad[j]
            y_pred = self.predict(X)
            curr_loss = np.mean((y_pred - y) ** 2)
            self.loss_history_.append(curr_loss)
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iter}, Loss: {curr_loss:.6f}")


    def predict(self, X):
        """
        Получение предсказаний обученной модели

        Параметры:
            X (np.ndarray): Матрица признаков

        Возвращает:
            np.ndarray: Предсказание для каждого элемента из X
        """
        return self.forward(X)