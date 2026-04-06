#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm  # pip install tqdm

# we're going to delay torch imports to inside the relevant function,
# for speed/simplicity for non-torch parts

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

from bayesian_logistic_regression import BayesianLogisticRegression
from markov_chain import MarkovChain
from utils import handle, main, run, load_dataset

################################################################################

handle("blogreg-mcmc")(lambda: blogreg(mcmc=True, var=False))
handle("blogreg-var")(lambda: blogreg(mcmc=False, var=True))


@handle("blogreg-all")
def blogreg(mcmc=True, var=True):
    X, y = load_dataset("two_threes", "X", "y")

    rng = np.random.default_rng(seed=12)
    is_train = rng.random(size=X.shape[0]) < 0.8
    X_train = X[is_train]
    y_train = y[is_train]
    X_test = X[~is_train]
    y_test = y[~is_train]

    model = BayesianLogisticRegression()
    map_est = model.map_estimate(X_train, y_train)

    if mcmc:
        mcmc_samples = model.sample_weights(X_train, y_train, n_samples=100_000,map_estimate=map_est)
        model.plot_weights(mcmc_samples, map_est, figname="blogreg-mcmc")

    if var:
        var_mn, var_L = model.variational_approx(X_train, y_train)
        var_samples = var_mn + np.random.randn(100_000, var_mn.shape[0]) @ var_L
        model.plot_weights(var_samples, map_est, figname="blogreg-var")

    map_test = model.different_log_likelihoods(X_test, y_test, map_est[np.newaxis])
    print(f"Test log-likelihood: MAP {map_test}")
    if mcmc:
        mcmc_test = model.different_log_likelihoods(X_test, y_test, mcmc_samples)
        print(f"                    MCMC {mcmc_test}")
    if var:
        var_test = model.different_log_likelihoods(X_test, y_test, var_samples)
        print(f"                     var {var_test}")


################################################################################


def grad_chain():
    return MarkovChain(
        init_probs=np.array([0.1, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0]),
        transition_probs=np.array(
            [
                [0.08, 0.90, 0.01, 0.00, 0.00, 0.00, 0.01],
                [0.03, 0.95, 0.01, 0.00, 0.00, 0.00, 0.01],
                [0.06, 0.06, 0.75, 0.05, 0.05, 0.02, 0.01],
                [0.00, 0.00, 0.00, 0.30, 0.60, 0.09, 0.01],
                [0.00, 0.00, 0.00, 0.02, 0.95, 0.02, 0.01],
                [0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
            ]
        ),
        state_names=[
            "VIDEO_GAMES",
            "INDUSTRY",
            "GRAD_SCHOOL",
            "VIDEO_GAMES_WITH_PHD",
            "INDUSTRY_WITH_PHD",
            "ACADEMIA",
            "DECEASED",
        ],
        # could use a python enum for the state names,
        # but it doesn't always play super nice with numpy, idk
    )


@handle("mc-sample")
def mc_sample():
    model = grad_chain()

    n_samples = 10_000
    time = 30
    samples = model.sample(n_samples, time + 1)
    est = np.bincount(samples[:, time]) / n_samples
    print(f"Empirical dist, time {time}: {est.round(3)}")


@handle("mc-marginals")
def mc_marginals():
    model = grad_chain()

    time = 30
    marginals = model.marginals(length=time + 1)
    print("Marginals at time 30:", marginals[:, time].round(3))



@handle("mc-mostlikely-marginals")
def mc_mostlikely():
    model = grad_chain()

    time = 30
    marginals = model.marginals(length=time + 1)

    # Find the most likely marginal for each time j
    best_state_indices = marginals.argmax(axis=0)

    most_likely_names = model.rename(best_state_indices)

    for j, name in enumerate(most_likely_names):
        print(f"Time {j}: {name} (p={marginals[best_state_indices[j], j]:.3f})")


    return


@handle("mc-mostlikely-sequence")
def mc_mostlikely_sequence():
    model = grad_chain()

    for time in [50, 100]:
        print(f"Mode at time {time}: {model.mode(time)}")


@handle("mc-conditionals-past")
def mc_conditionals_past():
    model = grad_chain()

    probs = model.conditional_prob(t := 20, past_state=(past_t := 5, past_v := 2))
    print(
        f"Probabilities at time {t}, given time {past_t} is {past_v} ({model.state_names[past_v]}):",
        probs.round(3),
    )

    mc_probs = np.full(model.num_states, np.nan)
    print(
        f"MC estimates  at time {t}, given time {past_t} is {past_v} ({model.state_names[past_v]}):",
        mc_probs.round(3),
    )
    acceptance_rate = 2.0
    print(f"\tacceptance rate: {acceptance_rate:.2%}")


@handle("mc-conditionals-both")
def mc_conditionals_both():
    model = grad_chain()

    probs = model.conditional_prob(
        t := 20,
        past_state=(past_t := 5, past_v := 2),
        future_state=(future_t := 30, future_v := 5),
    )
    print(
        f"Probabilities at time {t}, given time {past_t} is {past_v} ({model.state_names[past_v]}) ",
        f"and time {future_t} is {future_v} ({model.state_names[future_v]}):\n",
        probs.round(3),
    )

    mc_probs = np.full(model.num_states, np.nan)
    print(
        f"MC estimates  at time {t}, given time {past_t} is {past_v} ({model.state_names[past_v]}) ",
        f"and time {future_t} is {future_v} ({model.state_names[future_v]}):\n",
        mc_probs.round(3),
    )
    acceptance_rate = 2.0
    print(f"\tacceptance rate: {acceptance_rate:.2%}")


################################################################################

if __name__ == "__main__":
    main()
