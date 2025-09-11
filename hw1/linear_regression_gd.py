import numpy as np

class LinearRegressorGD:
    """
    Линейная регрессия с использованием Gradient Descent
    """

    def __init__(self, learning_rate=0.01, n_iter=1000, penalty="l2", alpha=0.0001):
        """
        Конструктор класса

        Параметры:
            learning_rate (float): Скорость обучения
            n_iter (int): Количество итераций градиентного спуска
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.penalty = penalty
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def get_penalty_grad(self):
        if self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.coef_
        else:
            return np.zeros_like(self.coef_)

    def fit(self, X, y):
        """
        Обучение модели на обучающей выборке с использованием
        градиентного спуска

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)
            y (np.ndarray): Вектор таргета длины n_samples
        """
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64").reshape(-1)
        n_objects, n_features = X.shape
        self.coef_ = np.random.normal(0, 0.01, n_features)
        self.intercept_ = np.random.normal(0, 0.01)
        for i in range(self.n_iter):
            y_predict = self.predict(X)
            error = y_predict - y
            coef_grad = (2 / n_objects) * np.dot(X.T, error)
            intercept_grad = (2 / n_objects) * np.sum(error)
            self.coef_ -= self.learning_rate * (coef_grad + self.get_penalty_grad())
            self.intercept_ -= self.learning_rate * intercept_grad


    def predict(self, X):
        """
        Получение предсказаний обученной модели

        Параметры:
            X (np.ndarray): Матрица признаков

        Возвращает:
            np.ndarray: Предсказание для каждого элемента из X
        """
        X = np.asarray(X, dtype="float64")
        return np.dot(X, self.coef_) + self.intercept_

    def get_params(self):
        """
        Возвращает обученные параметры модели
        """
        return {
        'intercept': self.intercept_,
        'coef': self.coef_
    }