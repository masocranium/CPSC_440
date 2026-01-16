#!/usr/bin/env python

# Written by Alan Milligan and Danica Sutherland (Jan 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

from functools import cache
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from kmeans import KMeans
from kmedians import KMedians
from utils import (
    test_and_plot,
    plot2Dclassifier,
    plot2Dclusters,
    load_dataset,
    main,
    handle,
    run,
)
from least_squares import LeastSquares, LeastSquaresBias, LeastSquaresRBFL2


@handle("kmeans")
def q_kmeans():
    (X,) = load_dataset("cluster_data", "X")
    model = KMeans(X, 4, plot=True)


def best_fit(X, k, reps=50, cls=KMeans):
    # Fit a cls() reps number of times, and return the best one according to model.loss()
    # Use  cls(X, k, plot=False, log=False)  to fit a model,
    # so it'll work for both KMeans and KMedians.
    # (Passing plot=False makes it run a *lot* faster, and log=False avoids a ton of clutter.)

    raise NotImplementedError()


@handle("kmeans-best")
def q_kmeans_best():
    (X,) = load_dataset("cluster_data", "X")
    best_model = best_fit(X, k=4)
    plot2Dclusters(X, best_model.get_assignments(X), best_model.w, "kmeans-best.png")


@handle("kmeans-outliers")
def q_kmeans_outliers():
    (X,) = load_dataset("cluster_data_2", "X")
    best_model = best_fit(X, k=4)
    plot2Dclusters(
        X, best_model.get_assignments(X), best_model.w, "kmeans-outliers.png"
    )


@handle("kmedians-outliers")
def q_kmedians_outliers():
    (X,) = load_dataset("cluster_data_2", "X")
    best_model = best_fit(X, k=4, cls=KMedians)
    plot2Dclusters(
        X, best_model.get_assignments(X), best_model.w, "kmedians-outliers.png"
    )


@handle("lsq")
def q_lsq():
    X, y = load_dataset("basis_data", "X", "y")
    model = LeastSquares(X, y)
    test_and_plot(model, X, y, filename="leastsquares.png")


@handle("lsq-bias")
def q_lsq_bias():
    X, y = load_dataset("basis_data", "X", "y")
    model = LeastSquaresBias(X, y)
    test_and_plot(model, X, y, filename="leastsquares-bias.png")


@handle("lsq-rbf")
def q_lsq_rbf():
    X, y = load_dataset("basis_data", "X", "y")
    model = LeastSquaresRBFL2(X, y)
    test_and_plot(model, X, y, filename="leastsquares-rbfl2.png")


@handle("lsq-rbf-split")
def q_lsq_rbf_split():
    X, y = load_dataset("basis_data", "X", "y")

    raise NotImplementedError()


class ComboModel:
    def __init__(self, bases, X=None, y=None):
        self.models = [cls() for cls in bases]
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        resids = y
        for model in self.models:
            model.fit(X, resids)
            resids = resids - model.predict(X)

    def predict(self, X):
        return sum(model.predict(X) for model in self.models)


@handle("lsq-combo")
def q_lsq_combo():
    X, y = load_dataset("basis_data", "X", "y")
    combo = ComboModel([LeastSquaresBias, LeastSquaresRBFL2], X, y)
    test_and_plot(combo, X, y, filename="leastsquares-combo.png")


@handle("relu")
def q_relu():
    from torch_net import ReluNet

    X, y = load_dataset("basis_data", "X", "y")
    model = ReluNet(X, y)
    test_and_plot(model, X, y, filename="leastsquares-relu.png")


@handle("relu-hinges")
def q_relu_hinges():
    import torch
    from torch_net import ReluNet

    X, y = load_dataset("basis_data", "X", "y")
    model = ReluNet(hidden_dim=8)

    n_iter = 6_000
    fig, axes = plt.subplots(4, 3, constrained_layout=True, figsize=(8, 8))
    ax_iter = iter(ax for row in axes for ax in row)

    for i, why, loss, opt in model.fit_loop(
        X, y, yield_every=(n_iter // 12, 100), num_iter=n_iter
    ):
        print(f"Iteration {i:>7,}: loss = {loss:>6.4f}")
        if why[0]:
            ax = next(ax_iter)

            # find the locations of the "hinges" in the prediction
            with torch.no_grad():
                hinges = torch.zeros(3)

            test_and_plot(model, X, y, ax=ax, plot_lims=(-7, 7))
            for hinge in hinges.ravel():
                ax.axvline(hinge, ls="--", color="black")
            ax.set_xlim(-7, 7)
            ax.set_title(f"{i:,}: loss {loss:.4f}")

    filename = Path("..", "figs", "relu-hinges.png")
    print("Saving", filename)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()


@handle("relu-combos")
def q_big_combo():
    from torch_net import ReluNet

    X, y = load_dataset("basis_data", "X", "y")

    fig, axes = plt.subplots(2, figsize=(6, 6))

    combo_A = ComboModel([ReluNet, LeastSquaresRBFL2], X, y)
    test_and_plot(combo_A, X, y, ax=axes[0], title="ReLU -> RBF")

    combo_B = ComboModel([LeastSquaresRBFL2, ReluNet], X, y)
    test_and_plot(combo_B, X, y, ax=axes[1], title="RBF -> ReLU")

    filename = Path("..", "figs", "combos.png")
    print("Saving", filename)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    main()
