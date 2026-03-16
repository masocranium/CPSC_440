import gzip
import os, argparse
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_dataset(filename, *keys, **kwargs):
    if filename == "mnist":
        return load_mnist(*keys, **kwargs)

    fn = Path("..", "data", filename)
    if not fn.exists() and fn.suffix != ".npz":
        fn = fn.with_suffix(fn.suffix + ".npz")
    data = np.load(fn, allow_pickle=True)
    if keys:
        return [data[k] for k in keys]
    else:
        return dict(**data)


def load_mnist(*keys, binarize=False, scale_to_1=True, dtype=np.float32):
    datadir = Path("..", "data")
    d = {}

    for name, fn_root in [("X", "train"), ("Xtest", "t10k")]:
        if not keys or name in keys:
            fn = datadir / f"{fn_root}-images-idx3-ubyte.gz"
            with gzip.open(fn) as f:
                # 16 bytes of metadata, then the images run together
                pixels = np.frombuffer(f.read(), "B", offset=16)
            pixels = pixels.reshape(-1, 784)

            if binarize:
                pixels = pixels >= 0.5
            pixels = pixels.astype(dtype)
            if scale_to_1 and not binarize:
                pixels = pixels / 255
            d[name] = pixels

    for name, fn_root in [("y", "train"), ("ytest", "t10k")]:
        if not keys or name in keys:
            fn = datadir / f"{fn_root}-labels-idx1-ubyte.gz"
            with gzip.open(fn) as f:
                # 8 bytes of metadata for these
                labels = np.frombuffer(f.read(), "B", offset=8)
            d[name] = labels.astype(np.int32)  # memory-inefficient but intuitive

    if keys:
        return [d[k] for k in keys]
    else:
        return d


# COLOURS = [
#    [0, 1, 0],
#    [1, 0, 0],
#    [0, 0, 1],
#    [1, 0, 1],
#    [1, 1, 0],
#    [0, 1, 1],
#    [0.1, 0.1, 0.1],
#    [1, 0.5, 0],
#    [0, 0.5, 0],
#    [0.5, 0.5, 0.5],
#    [0.5, 0.25, 0],
#    [0.5, 0, 0.5],
#    [0, 0.5, 1],
# ]


def plot2Dclassifier(
    model, X, y, includes_bias=False, X_test=None, y_test=None, ax=None, filename=None
):
    """Plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    if ax is None:
        fig, ax = plt.subplots()

    increment = 250
    if includes_bias:
        inds = [1, 2]
    else:
        inds = [0, 1]

    k = max(np.max(y), np.max((0,) if y_test is None else y_test)) + 1

    ax.scatter(
        *X.T,
        c=[f"C{v}" for v in y],
        marker="x" if k != 2 else np.where(y, "o", "+"),
    )
    if X_test is not None and y_test is not None:
        ax.scatter(
            *X_test.T,
            c=[f"C{v}" for v in y_test],
            marker="." if k != 2 else np.where(y_test, "s", "x"),
        )

    x1_min, x1_max = ax.get_xlim()
    x2_min, x2_max = ax.get_ylim()
    x1_line = np.linspace(x1_min, x1_max, increment)
    x2_line = np.linspace(x2_min, x2_max, increment)
    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)
    mesh_data = np.c_[(x1_mesh.ravel(), x2_mesh.ravel())]
    y_pred = model.predict(mesh_data).reshape(x1_mesh.shape)
    ax.set_xlim([x1_min, x1_max])
    ax.set_ylim([x2_min, x2_max])

    cmap = mpl.colors.ListedColormap(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, k + 0.5), k)

    ax.pcolormesh(x1_mesh, x2_mesh, y_pred, cmap=cmap, norm=norm, alpha=0.4)
    # ax.contourf(
    #    x1_mesh,
    #    x2_mesh,
    #    y_pred,
    #    #cmap=(ListedColormap(COLOURS)),
    #    zorder=(-1),
    #    alpha=0.4,
    #    cmin=0, cmax=k,
    # )

    if filename is not None:
        plt.savefig(f"../figs/{filename}", bbox_inches="tight", pad_inches=0.1)
        print(f"Plot saved as {filename}")


def test_and_plot(model, X, y, X_test=None, y_test=None, title=None, filename=None):
    yhat = model.predict(X)
    trainError = np.mean((yhat - y) ** 2)
    print("Training error = %.1f" % trainError)
    if X_test is not None:
        if y_test is not None:
            yhat = model.predict(X_test)
            testError = np.mean((yhat - y_test) ** 2)
            print("Test error     = %.1f" % testError)
    plt.figure()
    plt.plot(X, y, "b.")
    Xgrid = np.linspace(np.min(X), np.max(X), 1000)[:, None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, "g")
    if title is not None:
        plt.title(title)
    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def euclidean_dist_squared(X, X_test):
    """Computes the Euclidean distance between rows of 'X' and rows of 'X_test'

    Parameters
    ----------
    X : an N by D numpy array
    X_test: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """
    return (
        np.sum((X**2), axis=1)[:, None]
        + np.sum((X_test**2), axis=1)[None]
        - 2 * np.dot(X, X_test.T)
    )


def plot2Dclusters(X, y, w=None, filename=None):
    k = np.unique(y).size
    symbols = [
        "'s'",
        "'o'",
        "'v'",
        "'^'",
        "'x'",
        "'+'",
        "'*'",
        "'d'",
        "'<'",
        "'>'",
        "'p'",
    ]
    for c in range(k):
        colour = (0.75 * COLOURS[c][0], 0.75 * COLOURS[c][1], 0.75 * COLOURS[c][2])
        plt.scatter(
            (X[(y == c, 0)]), (X[(y == c, 1)]), marker=(symbols[c]), color=colour, s=10
        )
        if w is not None:
            plt.scatter(
                (w[(c, 0)]), (w[(c, 1)]), marker=(symbols[c]), color=(COLOURS[c]), s=100
            )

    if filename is not None:
        plt.savefig(f"../figs/{filename}", bbox_inches="tight", pad_inches=0.1)
        print(f"Plot saved as {filename}")


_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)
