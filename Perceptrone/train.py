#!/usr/bin/env python3
"""Программа 2: обучение многослойного персептрона.

    python train.py --layer 24 24 --epochs 84 --loss binaryCrossentropy \
                    --batch_size 8 --learning_rate 0.0314
"""

import argparse
import csv

from mlp.data import (CLASSES, load_csv, one_hot, standardize_apply,
                      standardize_fit)
from mlp.network import Network
from mlp.plot import plot_history


def save_history(path, history):
    with open(path, "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(["epoch", *history])
        for i in range(len(history["loss"])):
            writer.writerow([i + 1, *(f"{history[k][i]:.6f}" for k in history)])


def main():
    ap = argparse.ArgumentParser(description="train a multilayer perceptron")
    ap.add_argument("--train", default="data/data_training.csv")
    ap.add_argument("--valid", default="data/data_validation.csv")
    ap.add_argument("--layer", type=int, nargs="+", default=[24, 24],
                    help="размеры скрытых слоёв (минимум два по ТЗ)")
    ap.add_argument("--epochs", type=int, default=84)
    ap.add_argument("--loss", default="binaryCrossentropy",
                    choices=["binaryCrossentropy"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.0314)
    ap.add_argument("--activation", default="sigmoid", choices=["sigmoid"])
    ap.add_argument("--init", default="heUniform",
                    choices=["heUniform", "xavierUniform"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="saved_model.npz")
    ap.add_argument("--history", default="history.csv")
    ap.add_argument("--figures", default="report/figures")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="не открывать окна графиков (только сохранить файлы)")
    args = ap.parse_args()

    _, y_train, x_train = load_csv(args.train)
    _, y_valid, x_valid = load_csv(args.valid)
    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}")

    # статистики нормализации берутся только с обучающей выборки
    mean, std = standardize_fit(x_train)
    x_train = standardize_apply(x_train, mean, std)
    x_valid = standardize_apply(x_valid, mean, std)
    Y_train, Y_valid = one_hot(y_train), one_hot(y_valid)

    sizes = [x_train.shape[1], *args.layer, len(CLASSES)]
    print("network : " + " -> ".join(map(str, sizes)))
    net = Network(sizes, args.activation, args.init, args.seed)
    history = net.fit(x_train, Y_train, x_valid, Y_valid, args.epochs,
                      args.batch_size, args.learning_rate, args.seed)

    net.mean, net.std, net.classes = mean, std, CLASSES
    print(f"> saving model '{args.out}' to disk...")
    net.save(args.out)
    save_history(args.history, history)
    for path in plot_history(history, args.figures, args.show):
        print(f"> saved figure '{path}'")


if __name__ == "__main__":
    main()
