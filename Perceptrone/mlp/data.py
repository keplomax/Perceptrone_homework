"""Загрузка и предобработка датасета WDBC.

Формат файла data.csv: заголовка нет, 32 колонки.
    0       -- идентификатор пациента (в обучении не участвует),
    1       -- диагноз: 'M' (malignant) или 'B' (benign),
    2..31   -- 30 вещественных признаков ядра клетки.
"""

import csv

import numpy as np

CLASSES = ["B", "M"]  # индекс 1 -- злокачественная опухоль


def load_csv(path):
    """Читает сырой CSV и возвращает (ids, labels, X).

    ids    -- (N,)      строки-идентификаторы,
    labels -- (N,)      строки 'M'/'B',
    X      -- (N, 30)   признаки, float64.
    """
    ids, labels, rows = [], [], []
    with open(path, newline="") as fd:
        for line in csv.reader(fd):
            if not line:
                continue
            ids.append(line[0])
            labels.append(line[1])
            rows.append([float(v) for v in line[2:]])
    return np.array(ids), np.array(labels), np.array(rows, dtype=np.float64)


def one_hot(labels, classes=CLASSES):
    """'M'/'B' -> матрица (N, 2) с единицей в колонке нужного класса."""
    index = {name: i for i, name in enumerate(classes)}
    Y = np.zeros((len(labels), len(classes)))
    Y[np.arange(len(labels)), [index[v] for v in labels]] = 1.0
    return Y


def standardize_fit(X):
    """Статистики z-нормализации. Считаются ТОЛЬКО на обучающей выборке."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # защита от деления на ноль у константных признаков
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def split_indices(n, ratio, seed):
    """Перестановка индексов с фиксированным seed -> (train_idx, valid_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * ratio))
    return perm[:cut], perm[cut:]
