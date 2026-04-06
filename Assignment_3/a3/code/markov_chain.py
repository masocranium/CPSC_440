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

        for t in range(length):
            if t == 0:
                margs[:, t] = init_probs
            else:
                margs[:, t] = init_probs @ np.linalg.matrix_power(self.transition_probs, t)

        return margs

    def mode(self, length, init_probs=None):
        # again, length should include timestep 0
        if init_probs is None:
            init_probs = self.init_probs

        # define a matrix of zeros of shape [self.num_states, length] containing the probability of each state across time, and a matrix of zeros of shape [self.num_states, length] containing the mode of each state across time
        modes = np.zeros((self.num_states, length), dtype=int)
        # find the marginals up to the given length
        margs = self.marginals(length, init_probs) # to avoid unnecessarily recomputing marginals at each time step.
        
        
        # Bottoms up DP approach:

        # DP storage (structure only; no recursion/fill yet)
        best_path_prob = np.full((self.num_states, length), -np.inf, dtype=float)
        backptr = np.full((self.num_states, length), -1, dtype=int)

        # base case at t=0
        best_path_prob[:, 0] = np.log(init_probs + 1e-300) # this is the joint probability of being in each state at time 0 (which is just the initial distribution)
        backptr[:, 0] = np.arange(self.num_states)
        modes[:, 0] = np.arange(self.num_states)

        # Need a matrix for 
        for t in range(1,length):
            # calculate the joint probability of each state at time t
            for s in range(self.num_states):
                # calculate the probability of being in state s at time t given each possible previous state
                for prev_s in range(self.num_states):
                    prob = best_path_prob[prev_s, t-1] + np.log(self.transition_probs[prev_s, s] + 1e-300)
                    if prob > best_path_prob[s, t]: #check if greater than the default -inf value or the current best path probability for state s at time t
                        best_path_prob[s, t] = prob
                        backptr[s, t] = prev_s
                modes[s, t] = backptr[s, t]


        

        # return the best path and it's respective log probability
        best_final_state = np.argmax(best_path_prob[:, -1])

        best_path = np.zeros(length, dtype=int)
        best_path[-1] = best_final_state

        for t in range(length-1,0,-1):
            best_path[t-1] = modes[best_path[t], t]
        
        path_prob = best_path_prob[best_final_state, -1]


        return best_path, np.exp(path_prob) # return the best path and the probability of that path (convert from log prob to prob)

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
