import torch
from torch import nn
import torch.nn.functional as F


class ReluNet(nn.Module):
    def __init__(self, X=None, y=None, *, hidden_dim=10):
        # This isn't really the typical way you'd lay out a pytorch module;
        # usually, you separate building the model and training it more.
        # There are also more succinct ways to make models like this.
        # This layout is like what we did before, though, and it'll do.
        super().__init__()

        self.hidden_layer = nn.Linear(1, hidden_dim, bias=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)

        if X is not None and y is not None:
            self.fit(X, y)

    def cast(self, ary):
        # pytorch defaults everything to float32, unlike numpy which defaults to float64.
        # it's easier to keep everything the same,
        # and most ML uses don't really need the added precision...
        # you could use torch.set_default_dtype,
        # or pass dtype parameters everywhere you create a tensor, if you do want float64
        return torch.as_tensor(ary, dtype=torch.get_default_dtype())

    def cast_back(self, tensor):
        return tensor.detach().cpu().numpy()

    # nn.Module sets it up so that if we call module(x), it does some wrappers around forward()
    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(1)  # make sure it's shape [n, 1] for torch layers
        hidden_pre = self.hidden_layer(x)
        hidden_post = F.relu(hidden_pre)
        return self.output_layer(hidden_post)
        # normally, these kinds of functions always name the variable x (instead of hidden, etc),
        # to try to let the garbage collector throw away the intermediate data.
        # doesn't really matter here

    def predict(self, X):
        X_torch = self.cast(X)
        output = self(X_torch)
        return self.cast_back(output)

    def fit_loop(self, X, y, num_iter=20_000, yield_every=None):
        X_torch = self.cast(X)
        y_torch = self.cast(y)

        opt = torch.optim.AdamW(self.parameters())

        for i in range(
            1, num_iter + 1
        ):  # not doing anything fancy here like early stopping
            opt.zero_grad()
            y_hat = self(X_torch)
            loss = (y_hat - y_torch).square().mean()
            loss.backward()
            opt.step()

            if yield_every and any(
                why_yielding := [i % iters == 0 for iters in yield_every]
            ):
                yield i, why_yielding, loss, opt

    def fit(self, X, y, log=500, **kwargs):
        for i, why_yielding, loss, _ in self.fit_loop(
            X, y, yield_every=(log,), **kwargs
        ):
            print(f"Iteration {i:>7,}: loss = {loss:>6.4f}")
