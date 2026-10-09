"""Многослойный персептрон: прямое распространение, обратное распространение,
мини-батчевый градиентный спуск. Реализовано вручную, без автодифференцирования.
"""

import numpy as np

EPS = 1e-15  # клип вероятностей, чтобы log не ушёл в -inf


# --------------------------------------------------------------------------
# Функции активации и их производные
# --------------------------------------------------------------------------
def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_prime(A):
    """sigma'(z) = sigma(z) * (1 - sigma(z)) -- выражается через саму активацию."""
    return A * (1.0 - A)


def softmax(Z):
    """Стабилизированный softmax: вычитание максимума строки не меняет результат,
    но убирает переполнение exp."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "softmax": (softmax, None),  # производная не нужна: dZ = P - Y
}


# --------------------------------------------------------------------------
# Инициализация весов
# --------------------------------------------------------------------------
def init_weights(n_in, n_out, initializer, rng):
    if initializer == "heUniform":
        limit = np.sqrt(6.0 / n_in)
    elif initializer == "xavierUniform":
        limit = np.sqrt(6.0 / (n_in + n_out))
    else:
        raise ValueError(f"неизвестный инициализатор: {initializer}")
    return rng.uniform(-limit, limit, size=(n_in, n_out))


# --------------------------------------------------------------------------
# Слой
# --------------------------------------------------------------------------
class DenseLayer:
    """Полносвязный слой: A = g(A_prev @ W + b)."""

    def __init__(self, n_in, n_out, activation, initializer, rng):
        self.W = init_weights(n_in, n_out, initializer, rng)
        self.b = np.zeros(n_out)
        self.activation = activation
        self.g, self.g_prime = ACTIVATIONS[activation]
        self.A_prev = None
        self.A = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.A = self.g(A_prev @ self.W + self.b)
        return self.A

    def backward(self, dZ, lr):
        """dZ -- градиент потерь по предактивации Z этого слоя.

        Возвращает dA_prev, затем делает шаг градиентного спуска.
        Деление на m -- усреднение градиента по объектам батча.
        """
        m = dZ.shape[0]
        dW = self.A_prev.T @ dZ / m
        db = dZ.mean(axis=0)
        dA_prev = dZ @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dA_prev


# --------------------------------------------------------------------------
# Сеть
# --------------------------------------------------------------------------
class Network:
    """Стек полносвязных слоёв: скрытые с заданной активацией, выходной -- softmax."""

    def __init__(self, layer_sizes, activation="sigmoid",
                 initializer="heUniform", seed=42):
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.initializer = initializer
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            last = i == len(layer_sizes) - 2
            self.layers.append(DenseLayer(
                layer_sizes[i], layer_sizes[i + 1],
                "softmax" if last else activation, initializer, rng))
        # статистики предобработки и порядок классов едут вместе с весами
        self.mean = None
        self.std = None
        self.classes = None

    # -- прямой проход ----------------------------------------------------
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    predict_proba = forward

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    # -- потери и метрика --------------------------------------------------
    @staticmethod
    def loss(P, Y):
        """Кросс-энтропия E = -(1/N) sum_i sum_k y_ik * log(p_ik).

        При K = 2 и one-hot разметке совпадает с бинарной кросс-энтропией.
        """
        return float(-np.sum(Y * np.log(np.clip(P, EPS, 1.0))) / P.shape[0])

    @staticmethod
    def accuracy(P, Y):
        return float(np.mean(P.argmax(axis=1) == Y.argmax(axis=1)))

    # -- обратный проход ---------------------------------------------------
    def backward(self, P, Y, lr):
        """Softmax + кросс-энтропия дают dE/dZ^[L] = P - Y.

        Дальше рекуррентно: dZ^[l] = (dZ^[l+1] @ W^[l+1].T) * g'(A^[l]).
        """
        dZ = P - Y
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dA_prev = layer.backward(dZ, lr)
            if i > 0:
                prev = self.layers[i - 1]
                dZ = dA_prev * prev.g_prime(prev.A)

    # -- обучение ----------------------------------------------------------
    def fit(self, X, Y, X_val, Y_val, epochs, batch_size, lr, seed=42,
            verbose=True):
        rng = np.random.default_rng(seed)
        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        n = X.shape[0]
        width = len(str(epochs))
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)  # перемешивание на каждой эпохе
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, Yb = X[idx], Y[idx]
                self.backward(self.forward(Xb), Yb, lr)

            P, P_val = self.forward(X), self.forward(X_val)
            history["loss"].append(self.loss(P, Y))
            history["val_loss"].append(self.loss(P_val, Y_val))
            history["acc"].append(self.accuracy(P, Y))
            history["val_acc"].append(self.accuracy(P_val, Y_val))
            if verbose:
                print(f"epoch {epoch:0{width}d}/{epochs} - "
                      f"loss: {history['loss'][-1]:.4f} - "
                      f"val_loss: {history['val_loss'][-1]:.4f} - "
                      f"acc: {history['acc'][-1]:.4f} - "
                      f"val_acc: {history['val_acc'][-1]:.4f}")
        return history

    # -- сохранение / загрузка --------------------------------------------
    def save(self, path):
        """Архитектура + веса + статистики предобработки в одном .npz."""
        blob = {
            "layer_sizes": np.array(self.layer_sizes),
            "activation": np.array(self.activation),
            "initializer": np.array(self.initializer),
            "mean": self.mean,
            "std": self.std,
            "classes": np.array(self.classes),
        }
        for i, layer in enumerate(self.layers):
            blob[f"W{i}"] = layer.W
            blob[f"b{i}"] = layer.b
        np.savez(path, **blob)

    @classmethod
    def load(cls, path):
        blob = np.load(path, allow_pickle=False)
        net = cls(list(blob["layer_sizes"]), str(blob["activation"]),
                  str(blob["initializer"]))
        for i, layer in enumerate(net.layers):
            layer.W = blob[f"W{i}"]
            layer.b = blob[f"b{i}"]
        net.mean, net.std = blob["mean"], blob["std"]
        net.classes = [str(c) for c in blob["classes"]]
        return net
