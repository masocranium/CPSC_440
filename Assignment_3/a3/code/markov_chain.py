import numpy as np


class MarkovChain:
    def __init__(self, transition_probs, init_probs, state_names=None):
        # p(x_1 = c) = init_probs[c]
        # p(x_j = c | x_{j-1} = c') = transition_probs[c', c]
        self.init_probs = init_probs
        self.transition_probs = transition_probs
        self.num_states = len(self.init_probs)
        assert self.init_probs.shape == (self.num_states,)
        assert self.transition_probs.shape == (self.num_states, self.num_states)

        if state_names is not None:
            self.state_names = np.asarray(state_names)
            assert len(state_names) == self.num_states

    def rename(self, ary):
        if self.state_names is None:
            return ary
        return self.state_names[ary]

    def sample(self, n_samples, length, init_probs=None, rng=None):
        # note that:
        # length 1 should return only timestep 0
        # length 2 should return timesteps 0, 1
        # etc
        if rng is None:
            rng = np.random.default_rng()
        if init_probs is None:
            init_probs = self.init_probs

        samples = np.zeros((n_samples, length), dtype=int)

        #sample from initial distribution for each of the 10k chain init state
        samples[:, 0] = rng.choice(self.num_states, size=n_samples, p=init_probs)

        #sample from transition distribution
        for j in range(n_samples):
            for t in range(1,length):
                samples[j,t] = rng.choice(self.num_states,p=self.transition_probs[samples[j,t-1]])

        return samples

    def marginals(self, length, init_probs=None):
        if init_probs is None:
            init_probs = self.init_probs

        margs = np.zeros((self.num_states, length))
        raise NotImplementedError()

        return margs

    def mode(self, length, init_probs=None):
        # again, length should include timestep 0
        if init_probs is None:
            init_probs = self.init_probs

        raise NotImplementedError()

    def conditional_prob(self, target_idx, future_state=None, past_state=None):
        # target_idx: the time index of the state we care about
        # future_state: None or (future_idx, future_val)
        # past_state: None or (past_idx, past_val)
        #
        # should return an array of shape [self.num_states] giving probabilities for each state:
        #    return_val[i] = p(x_{target_idx} = i | x_{future_idx} = future_val, x_{past_idx} = past_val)
        if future_state is None:
            if past_state is None:
                # conditioning on nothing
                return self.marginals(target_idx + 1)[:, -1]
            else:
                past_idx, past_val = past_state
                # want p(x_{target_idx} = ? | x_{past_idx} = past_val)
                raise NotImplementedError()

        else:
            future_idx, future_val = future_state

            raise NotImplementedError()
