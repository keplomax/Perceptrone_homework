"""Кривые обучения."""

import os

import matplotlib
import matplotlib.pyplot as plt


def _curve(ax, train, valid, ylabel, title):
    epochs = range(1, len(train) + 1)
    ax.plot(epochs, train, label=f"training {ylabel}")
    ax.plot(epochs, valid, "--", label=f"validation {ylabel}")
    ax.set_xlabel("epochs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()


def plot_history(history, outdir, show=True):
    """Два графика по ТЗ: кривая потерь и кривая точности."""
    if not show:
        matplotlib.use("Agg")
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for key, ylabel in (("loss", "loss"), ("acc", "accuracy")):
        fig, ax = plt.subplots(figsize=(6, 4))
        _curve(ax, history[key], history[f"val_{key}"], ylabel, "Learning Curves")
        fig.tight_layout()
        path = os.path.join(outdir, f"learning_curve_{ylabel}.pdf")
        fig.savefig(path)
        paths.append(path)
    if show:
        plt.show()
    else:
        plt.close("all")
    return paths
