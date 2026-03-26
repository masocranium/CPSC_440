#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (2023-)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

from functools import partial
import os
from pathlib import Path
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import quantile
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as BaseLogReg
from scipy.special import expit as sigmoid

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    test_and_plot,
    load_dataset,
    load_mnist,
    plot2Dclassifier,
    main,
    handle,
    run,
)
from naive_bayes import NaiveBayes, VQNB
from lda import LDA

# unregularized LogisticRegression gives a silly warning when unregularized
warnings.filterwarnings("ignore", message=".*Setting penalty=None.*")
LogisticRegression = partial(BaseLogReg, C=np.inf, max_iter=5000)


def sample_cat(n_sample, pdf, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    nums = rng.random(n_sample) #generate n_samples of random numbers between 0 and 1

    cdf = np.cumsum(pdf)
    
    cat_vec = []
   
    for num in nums:
        # find the first index where the random value is less than the cumulative probability
        for i, cumulative_prob in enumerate(cdf):
            if num < cumulative_prob:
                cat_vec.append(i+1)
                break
    return np.array(cat_vec)

def expval(fn, samples):

    values = fn(samples)
    running_mean = np.cumsum(values) / np.arange(1,len(values)+1)

    return running_mean


@handle("expval-mc")
def expval_mc():
    max_n = 100_000
    pdf = (0.42, 0.24, 0.34)
    n_repeats = 3

    # Vectorized version of f(x)
    def f(x):
        return np.where(x % 2 == 0, 15, -4)

    fig, ax = plt.subplots()
    
    # Run and plot n_repeats times
    for i in range(n_repeats):
        samples = sample_cat(max_n, pdf)
        # Calculate running mean: cumsum / count
        values = f(samples)
        running_means = np.cumsum(values) / np.arange(1, max_n + 1)
        ax.plot(np.arange(1, max_n + 1), running_means, alpha=0.8, label=f"Run {i+1}")

    # Calculate analytical expected value for the horizontal line
    # prob(odd) = 0.42 + 0.34 = 0.76 -> -4
    # prob(even) = 0.24 -> 15
    analytical_mean = (0.42 * -4) + (0.24 * 15) + (0.34 * -4)
    
    ax.axhline(analytical_mean, color='k', linestyle='--', label='Analytical Expected Value')

    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Monte Carlo approximation")
    ax.set_xlim(1, max_n)
    ax.set_ylim(-1, 2)
    ax.legend()

    fn = Path("..") / "figs" / "expval_game.pdf"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved in {fn}")


@handle("gambling")
def gambling():
    pdf = (0.42, 0.24, 0.34)

    # Analytical solution via Markov chain linear system
    # P(s) = P(even)*P(s+15) + P(odd)*P(s-4)
    # Boundary: P(s<0) = 0 (bust), P(s>=300) = 1 (win)
    p_even = pdf[1]           # 0.24 -> win $15
    p_odd = pdf[0] + pdf[2]   # 0.76 -> lose $4

    N = 300  # states 0..299
    A = np.zeros((N, N))
    b = np.zeros(N)

    for s in range(N):
        A[s, s] = 1.0
        if s - 4 >= 0:
            A[s, s - 4] = -p_odd
        # else P(s-4) = 0, term vanishes
        if s + 15 < N:
            A[s, s + 15] = -p_even
        else:
            # P(s+15) = 1, moves to RHS
            b[s] = p_even

    P = np.linalg.solve(A, b)
    print(f"Probability of 'Having a fun night!' is approximately = {P[40]*100:.2f}%")
    return P[40]

@handle("gambling-mc")
def gambler_mc(n_simulations=500_000, seed=21):
    """Monte Carlo estimate of P(reaching $300 starting from $40)."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_simulations):
        balance = 40
        while 0 <= balance < 300:
            if rng.random() < 0.76:   # odd outcome (P=0.76): lose $4
                balance -= 4
            else:                      # even outcome (P=0.24): win $15
                balance += 15
        if balance >= 300:
            wins += 1

    print(f"Monte Carlo estimate of 'Having a fun night!' is approximately = {wins / n_simulations:.2%}")
    
    return wins / n_simulations
    

@handle("plot-em")
def plot_em():
    (X,) = load_dataset("mixture", "X")

    mus = np.linspace(-2, 8, num=1_000)
    # compute log_liks: an array with shape the same as mus containing
    # the log-likelihood for each mu value
    # (ignoring an additive constant is fine)
    # p(x | mu) = 0.5 * N(x; 0, 1) + 0.5 * N(x; mu, 1)
    # log p(X | mu) = sum_i log[ 0.5 * exp(-x_i^2/2) + 0.5 * exp(-(x_i - mu)^2/2) ]
    #                 + const            (the -0.5*log(2*pi) per sample cancels in the plot)
    #
    # We keep the constant so the curve is the true log-likelihood (up to n*const).
    log_liks = np.array([
        np.sum(
            np.log(
                0.5 * np.exp(-0.5 * X**2)
                + 0.5 * np.exp(-0.5 * (X - mu)**2)
            )
        )
        for mu in mus
    ])
 
    # EM iterations
    em_mus = [mu := 0]          # initialise at mu = 0
    for _ in range(5):
        # E-step: responsibilities  r_i = sigma(mu * x_i - mu^2 / 2)
        r = sigmoid(mu * X - 0.5 * mu**2)
 
        # M-step: mu = sum(r_i * x_i) / sum(r_i)
        mu = float(np.sum(r * X) / np.sum(r))
 
        em_mus.append(mu)
    em_mus = np.asarray(em_mus)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("log-likelihood")

    ax.plot(mus, log_liks)
    plt.axvline(mus[np.argmax(log_liks)], color="k", ls="--")
    plt.plot(em_mus, log_liks[mus.searchsorted(em_mus)], color="r", ls="--", marker="o")

    ax.set_xlim(mus[0], mus[-1])

    fn = Path("..") / "figs" / "mixture-fitting.pdf"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved in {fn}")


################################################################################


def eval_models(models, dataset="mnist", **kwargs):
    X, y, Xtest, ytest = load_dataset(dataset, "X", "y", "Xtest", "ytest", **kwargs)
    for model in models:
        model.fit(X, y)
        yhat = model.predict(Xtest)
        yield np.mean(yhat != ytest)


def eval_model(model, dataset="mnist", **kwargs):
    return next(eval_models([model], dataset=dataset, **kwargs))


@handle("mnist-nb")
def mnist_nb():
    laps = [1e-9, 0.5, 1, 2.5, 5, 10, 50]
    errs = eval_models(
        [NaiveBayes(prior_alpha=lap, prior_beta=lap) for lap in laps], binarize=True
    )
    print("NaiveBayes test set errors:")
    for lap, err in zip(laps, errs):
        print(f"  lap    {lap:>4.1f}:  {err:.1%}")


@handle("mnist-logreg")
def mnist_logreg():
    err = eval_model(
        BaseLogReg(
            solver="saga",
            l1_ratio=0,
            C=1,  # like 1/lambda
            tol=0.1,  # optimization tolerance
            max_iter=10_000,
        )
    )
    print(f"Test set error: {err:.1%}")


@handle("mnist-vqnb")
def mnist_vqnb():
    ks = [1, 2, 3, 4, 5]
    models = [VQNB(k=k) for k in ks]
    print("VQNB test set errors:")
    for k, err in zip(ks, eval_models(models, binarize=True)):
        print(f"  k = {k}:  {err:.1%}")

    k = ks[-1]
    model = models[-1]
    fig, axes = plt.subplots(
        k,
        10,
        figsize=(8, 8 * k / 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for y in range(10):
        for b in range(k):
            # ps = np.zeros(784)  # get the probabilities from your model
            ps = model.theta[b, y, :]
            axes[b][y].imshow(ps.reshape((28, 28)), "gray")
            axes[b][y].set_axis_off()
    fig.savefig("../figs/vqnb_probs.pdf", bbox_inches="tight", pad_inches=0.1)
    print("Plots in ../figs/vqnb_probs.pdf")


################################################################################


class MixtureDistribution:
    def __init__(self, distractor_dims=0, rng=None):
        self.default_rng = rng = np.random.default_rng(rng)

        self.wts = np.array([0.26, 0.05, 0.38, 0.31])
        main_means = np.array([[-3.6, 1.1], [-0.2, 3.0], [0.6, -4.2], [3.8, 2.8]])
        main_dims = 2
        self.dim = main_dims + distractor_dims
        means = np.c_[main_means, np.zeros((self.wts.shape[0], distractor_dims))]
        self.means = means

        distractor_cov_L = np.diagflat(np.arange(1, distractor_dims + 1))
        main_cov_L = np.array([[1, 0], [1, 1.1]])  # cov is [1, 0.3; 0.3, 1.3]
        self.cov_L = np.r_[
            np.c_[main_cov_L, np.zeros((main_dims, distractor_dims))],
            np.c_[np.zeros((distractor_dims, main_dims)), distractor_cov_L],
        ]

    def sample(self, n, rng=None):
        rng = self.default_rng if rng is None else np.random.default_rng(rng)

        k, d = self.means.shape
        y = rng.choice(k, size=n, p=self.wts)
        X = rng.standard_normal((n, d)) @ self.cov_L.T + self.means[y]
        return X, y


@handle("plot-boundaries")
def plot_boundaries():
    dist = MixtureDistribution(rng=12)
    X, y = dist.sample(100)
    Xte, yte = dist.sample(500)

    models = {"Logistic Regression": LogisticRegression().fit(X, y)}

    try:
        models["LDA"] = LDA().fit(X, y)
    except NotImplementedError:
        print("LDA not implemented yet")
        pass

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 * len(models), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for (name, model), ax in zip(models.items(), (ax for row in axes for ax in row)):
        plot2Dclassifier(model, X, y, X_test=Xte, y_test=yte, ax=ax)
        ax.set_title(name)

    fn = "../figs/plot-boundaries.png"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Plot saved as {fn}")


@handle("learning-curves")
def learning_curves(distractor_dims=0):
    dist = MixtureDistribution(distractor_dims=distractor_dims, rng=7)
    train_sizes = np.logspace(1.2, 3.5, num=12).astype("int")

    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.8), sharex=True, constrained_layout=True
    )
    models = {"Logistic Regression": LogisticRegression, "LDA": LDA}
    n_reps = 20

    results = np.full((len(models), len(train_sizes), n_reps, 2), np.nan)
    for ni, n in enumerate(train_sizes):
        for rep in range(n_reps):
            Xtr, ytr = dist.sample(n)
            Xte, yte = dist.sample(1_000)
            for clf_i, (name, cls) in enumerate(models.items()):
                clf = cls()
                try:
                    clf.fit(Xtr, ytr)
                    results[clf_i, ni, rep, 0] = (clf.predict(Xtr) == ytr).mean()
                    results[clf_i, ni, rep, 1] = (clf.predict(Xte) == yte).mean()
                except NotImplementedError:
                    results[clf_i, ni, rep, :] = np.nan

    for clf_i, clf_name in enumerate(models.keys()):
        c = f"C{clf_i}"
        for kind_i, kind in enumerate(["train", "test"]):
            r = results[clf_i, :, :, kind_i]
            ax = axes[kind_i]
            ax.plot(
                train_sizes,
                np.nanmedian(r, axis=1),
                c=c,
                label=f"{clf_name} {kind} acc",
                marker=".",
            )
            ax.fill_between(
                train_sizes,
                quantile(r, 0.25, axis=1, nan_policy="omit"),
                quantile(r, 0.75, axis=1, nan_policy="omit"),
                color=c,
                alpha=0.2,
            )

    axes[0].set_xscale("symlog")
    axes[0].set_ylabel("Accuracy")
    pct_fmt = mpl.ticker.StrMethodFormatter("{x:.1%}")
    for ax in axes:
        ax.set_xlabel("training points")
        ax.legend(loc="best")
        ax.yaxis.set_major_formatter(pct_fmt)
    fig.suptitle(f"Distractor dimensions: {distractor_dims}")

    fn = f"../figs/learning-curves-{distractor_dims}.png"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Plot saved as {fn}")


@handle("learning-curves-distractors")
def learning_curves_distractors():
    learning_curves(15)


@handle("lda-projection")
def lda_projection():
    dist = MixtureDistribution(distractor_dims=15, rng=47)
    X, y = dist.sample(1_000)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    pcaed = PCA(n_components=2).fit_transform(X)
    a1.scatter(*pcaed.T, c=(colours := [f"C{i}" for i in y]))
    a1.set_title("Two PCA dimensions")
    a1.set_aspect("equal")

    ldaed = LDA(X, y).transform(X, dims=2)
    a2.scatter(*ldaed.T, c=colours)
    a2.set_title("Two LDA dimensions")
    a2.set_aspect("equal")

    fn = f"../figs/projected.png"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Plot saved as {fn}")


if __name__ == "__main__":
    main()
