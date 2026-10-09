#!/usr/bin/env python3
"""Предварительный статистический и визуальный анализ датасета.

    python describe.py --dataset data/data.csv
"""

import argparse
import os

import matplotlib
import numpy as np

from mlp.data import load_csv

BASE = ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry",
        "fractal dim"]
FEATURES = [f"{name} {kind}" for kind in ("mean", "se", "worst") for name in BASE]


def percentile(col, q):
    """Квантиль по отсортированному столбцу (линейная интерполяция)."""
    s = np.sort(col)
    pos = q * (len(s) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def print_table(X):
    rows = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    print(f"{'':22}" + "".join(f"{r:>12}" for r in rows))
    for j, name in enumerate(FEATURES):
        col = X[:, j]
        values = [len(col), col.mean(), col.std(), col.min(),
                  percentile(col, 0.25), percentile(col, 0.50),
                  percentile(col, 0.75), col.max()]
        print(f"{name:22}" + "".join(f"{v:>12.4f}" for v in values))


def separation(X, labels):
    """Модуль стандартизованной разности средних классов -- грубая мера
    разделяющей способности признака."""
    m, b = X[labels == "M"], X[labels == "B"]
    pooled = np.sqrt((m.var(axis=0) + b.var(axis=0)) / 2)
    pooled[pooled == 0] = 1.0
    return np.abs(m.mean(axis=0) - b.mean(axis=0)) / pooled


def main():
    ap = argparse.ArgumentParser(description="exploratory data analysis")
    ap.add_argument("--dataset", default="data/data.csv")
    ap.add_argument("--figures", default="report/figures")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()

    _, labels, X = load_csv(args.dataset)
    n_m = int((labels == "M").sum())
    print(f"objects: {len(labels)}, features: {X.shape[1]}")
    print(f"classes: B = {len(labels) - n_m} "
          f"({100 * (len(labels) - n_m) / len(labels):.1f}%), "
          f"M = {n_m} ({100 * n_m / len(labels):.1f}%)\n")
    print_table(X)

    sep = separation(X, labels)
    order = np.argsort(sep)[::-1]
    print("\nнаиболее разделяющие признаки (|разность средних| / станд. откл.):")
    for j in order[:8]:
        print(f"  {FEATURES[j]:22} {sep[j]:.3f}")
    print(f"\nмасштаб признаков: min std = {X.std(0).min():.4f}, "
          f"max std = {X.std(0).max():.1f} "
          f"(отношение {X.std(0).max() / X.std(0).min():.0f}:1)")

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.figures, exist_ok=True)
    mask_m = labels == "M"

    # 1. гистограммы шести наиболее разделяющих признаков
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, j in zip(axes.ravel(), order[:6]):
        bins = np.linspace(X[:, j].min(), X[:, j].max(), 30)
        ax.hist(X[~mask_m, j], bins=bins, alpha=0.6, label="B")
        ax.hist(X[mask_m, j], bins=bins, alpha=0.6, label="M")
        ax.set_title(FEATURES[j], fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Распределения наиболее разделяющих признаков")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "eda_histograms.pdf"))

    # 2. диаграмма рассеяния двух лучших признаков
    a, b = order[0], order[1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(X[~mask_m, a], X[~mask_m, b], s=12, alpha=0.6, label="B")
    ax.scatter(X[mask_m, a], X[mask_m, b], s=12, alpha=0.6, label="M")
    ax.set_xlabel(FEATURES[a])
    ax.set_ylabel(FEATURES[b])
    ax.set_title("Разделимость классов в плоскости двух признаков")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "eda_scatter.pdf"))

    # 3. корреляционная матрица
    C = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(30))
    ax.set_yticks(range(30))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=6)
    ax.set_yticklabels(FEATURES, fontsize=6)
    ax.set_title("Корреляционная матрица признаков")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "eda_correlation.pdf"))

    off = C - np.eye(30)
    print(f"пар признаков с |корреляцией| > 0.9: {int((np.abs(off) > 0.9).sum()) // 2}")
    print(f"> saved figures to '{args.figures}'")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
