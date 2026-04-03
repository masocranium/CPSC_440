import numpy as np
import torch
from tqdm import tqdm


class BayesianLogisticRegression:
    def __init__(self, lam=1):
        self.lam = lam

    def _get_log_prob(self, X, y):
        # gives a function that returns the log probability up to an additive constant
        X = torch.tensor(X, dtype=torch.float32, requires_grad=False)  # [n, d]
        y = torch.tensor(y, dtype=torch.float32, requires_grad=False)  # [n]
        minus_yX = -y.unsqueeze(1) * X  # [n, d], reused below

        d = X.shape[1]
        zero = torch.tensor(0.0)

        def log_probs(ws):
            ws = torch.as_tensor(ws, dtype=torch.float32)
            # ws should be shape [d, ...]
            # return value is of shape [...]
            minus_yX_ws = minus_yX @ ws  # [n, ...]
            log_liks = -torch.logaddexp(zero, minus_yX_ws).sum(0)  # [...]

            log_priors = (-self.lam / 2) * ws.square().sum(0)  # [...]
            return log_liks + log_priors

        return log_probs

    def log_prob(self, X, y, w):
        return self._get_log_prob(X, y)(w)

    def sample_weights(self, X, y, n_samples, map_estimate=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n, d = X.shape
        log_prob = self._get_log_prob(X, y)

        samples = np.full((n_samples, d), np.nan)
        w = np.zeros(d) if map_estimate is None else map_estimate
        log_p = log_prob(w)

        scale_factor = 1e-1 # added scale factor to make acceptance rate reasonable
        
        n_accept = 0
        pbar = tqdm(range(n_samples))
        for i in pbar:
            w_hat = w + rng.normal(
                loc=0,
                scale=scale_factor,
                size=d,
            )
            log_phat = log_prob(w_hat)

            log_r = log_phat - log_p
            if np.log(rng.uniform()) < log_r:
                w = w_hat
                n_accept += 1
                log_p = log_phat
            samples[i, :] = w
            pbar.set_postfix({"accept_rate": f"{n_accept / (i + 1):.2%}"})

        print(
            f"did {n_samples:,} steps; accepted {n_accept:,} samples ({n_accept / n_samples:.2%})"
        )
        return samples

    def plot_weights(self, sample_weights, point_est=None, figname=None):
        import matplotlib.pyplot as plt

        indices = [
            ("positive variables", [119, 144, 152, 162, 184, 196]),
            ("negative variables", [30, 75, 106, 109, 123, 124]),
            ("neutral variables", [79, 167, 182, 213, 222, 255]),
        ]

        fig, axes = plt.subplots(
            6, 3, figsize=(8, 8), constrained_layout=True, sharex=True, sharey=True
        )

        for col, (name, ary) in enumerate(indices):
            for row, idx in enumerate(ary):
                ax = axes[row, col]
                ax.hist(sample_weights[:, idx], bins=50)
                if point_est is not None:
                    ax.axvline(point_est[idx], color="r")

                if row == 0:
                    ax.set_title(name)

        if figname is None:
            plt.show()
        else:
            fn = f"../figs/{figname}.png"
            fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved to {fn}")

    def map_estimate(self, X, y):
        # lots of ways to solve this; this is a reasonable one
        log_prob = self._get_log_prob(X, y)

        d = X.shape[1]
        w = torch.zeros(d, requires_grad=True)

        opt = torch.optim.LBFGS([w], lr=0.1, max_iter=100_000)

        def step():
            opt.zero_grad()
            nll = -log_prob(w)
            nll.backward()
            return nll

        opt.step(step)  # actually does a lot of steps
        return w.detach().numpy()

    def variational_approx(self, X, y, batch_size=128, train_steps=5_000):
        log_prob = self._get_log_prob(X, y)

        d = X.shape[1]
        tril_inds = tuple(torch.tril_indices(d, d, offset=-1))
        diag_inds = (range(d), range(d))

        mu = torch.zeros(d, requires_grad=True)
        L_log_diag = torch.zeros(d, requires_grad=True)
        L_tril = torch.zeros(tril_inds[0].shape[0], requires_grad=True)

        def get_L():
            L = torch.zeros(d, d)
            L[diag_inds] = L_log_diag.exp()
            L[tril_inds] = L_tril
            return L

        def variational_obj():
            # should return the objective to *maximize*
            # (using a Monte Carlo estimate with batch_size samples)
            L = get_L()

            raise NotImplementedError()

        # LBFGS is too slow here with the higher-dimensional problem
        opt = torch.optim.Adam([mu, L_log_diag, L_tril], lr=0.1)
        pbar = tqdm(range(train_steps))

        obj_avg = variational_obj().detach().item()
        for it in pbar:
            opt.zero_grad()

            obj = variational_obj()
            (-obj).backward()  # minus sign to minimize instead

            obj_avg = 0.9 * obj_avg + 0.1 * obj.detach().item()
            pbar.set_postfix({"obj_avg": obj_avg})

            opt.step()

        return (p.detach().numpy() for p in [mu, get_L()])

    def predict_proba(self, X, w):
        "Probabilistic prediction for a given set of inputs."
        return 1 / (1 + np.exp(-X @ w))

    def different_log_likelihoods(self, X, y, samples):
        preds = X @ samples.T  # shape [n, n_samples]
        true_pos = y > 0

        avg_decision = ((preds > 0).sum(axis=1) + 0.1) / (preds.shape[1] + 0.2)
        method_one = np.log(np.where(true_pos, avg_decision, 1 - avg_decision)).sum()

        log_probs_pos = -np.logaddexp(0, -preds)
        log_probs_neg = -np.logaddexp(0, preds)
        log_liks = np.where(true_pos[:, np.newaxis], log_probs_pos, log_probs_neg)

        method_two = log_liks.mean(axis=1).sum()
        method_three = log_liks.sum(axis=0).mean()

        mean_probs = np.exp(log_probs_pos).mean(axis=1)
        method_four = np.log(np.where(true_pos, mean_probs, 1 - mean_probs)).sum()

        return np.array([method_one, method_two, method_three, method_four])
