#!/usr/bin/env python3
"""Программа 3: прогноз обученной моделью и оценка бинарной кросс-энтропией.

    python predict.py --model saved_model.npz --dataset data/data_validation.csv
"""

import argparse

from mlp.data import load_csv, one_hot, standardize_apply
from mlp.network import Network


def main():
    ap = argparse.ArgumentParser(description="predict and evaluate")
    ap.add_argument("--model", default="saved_model.npz")
    ap.add_argument("--dataset", default="data/data_validation.csv")
    args = ap.parse_args()

    net = Network.load(args.model)
    ids, labels, X = load_csv(args.dataset)
    # то же преобразование, что и при обучении: mean/std лежат внутри модели
    X = standardize_apply(X, net.mean, net.std)
    Y = one_hot(labels, net.classes)

    P = net.forward(X)
    print(f"network : " + " -> ".join(map(str, net.layer_sizes)))
    print(f"samples : {len(ids)}")
    print(f"binary cross-entropy : {net.loss(P, Y):.4f}")
    print(f"accuracy             : {net.accuracy(P, Y):.4f}")

    wrong = (P.argmax(axis=1) != Y.argmax(axis=1)).sum()
    print(f"errors               : {wrong} / {len(ids)}")


if __name__ == "__main__":
    main()
