#!/usr/bin/env python3
"""Программа 1: разделение датасета на обучающую и валидационную выборки.

    python split.py --dataset data/data.csv --ratio 0.8 --seed 42
"""

import argparse
import csv
import os

from mlp.data import load_csv, split_indices


def write_subset(path, ids, labels, X):
    with open(path, "w", newline="") as fd:
        writer = csv.writer(fd)
        for i in range(len(ids)):
            writer.writerow([ids[i], labels[i], *(f"{v:g}" for v in X[i])])


def describe(name, labels):
    benign = int((labels == "B").sum())
    print(f"{name}: {len(labels)} objects (B: {benign}, M: {len(labels) - benign})")


def main():
    ap = argparse.ArgumentParser(description="split dataset into train/validation")
    ap.add_argument("--dataset", default="data/data.csv")
    ap.add_argument("--ratio", type=float, default=0.8,
                    help="доля обучающей выборки")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed перестановки -- обеспечивает повторяемость")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    ids, labels, X = load_csv(args.dataset)
    train_idx, valid_idx = split_indices(len(ids), args.ratio, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    train_path = os.path.join(args.outdir, "data_training.csv")
    valid_path = os.path.join(args.outdir, "data_validation.csv")
    write_subset(train_path, ids[train_idx], labels[train_idx], X[train_idx])
    write_subset(valid_path, ids[valid_idx], labels[valid_idx], X[valid_idx])

    describe("dataset   ", labels)
    describe("training  ", labels[train_idx])
    describe("validation", labels[valid_idx])
    print(f"> saved '{train_path}' and '{valid_path}' (seed={args.seed})")


if __name__ == "__main__":
    main()
