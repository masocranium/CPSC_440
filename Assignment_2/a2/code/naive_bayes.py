import numpy as np
from scipy.special import logsumexp

from kmeans import KMeans


class NaiveBayes:
    def __init__(self, prior_alpha=1, prior_beta=1, X=None, y=None):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}

        # num_classes is just a "backup" to pass k in case you don't actually see
        # any from the highest class

        n, self.d = X.shape

        # categorical MLE for y:
        self.y_probs = np.bincount(y) / n
        self.num_classes = k = len(self.y_probs)

        # Compute theta matrix (k x d) where theta[i, j] = P(X_j = 1 | Y = i)
        self.theta = np.zeros((k, self.d))
        for i in range(k):
            # Get the indices of samples belonging to class i
            class_indices = np.where(y == i)[0]
            # Calculate the number of samples in class i
            n_i = len(class_indices)
            if n_i == 0:
                # If there are no samples for this class, use the prior
                self.theta[i] = (self.prior_alpha) / (self.prior_alpha + self.prior_beta)
            else:
                # Calculate the number of times each feature is 1 in class i
                feature_counts = np.sum(X[class_indices], axis=0)
                # Apply Laplace smoothing
                self.theta[i] = (feature_counts + self.prior_alpha) / (n_i + self.prior_alpha + self.prior_beta)


    def predict(self, X):
        # Compute the log probabilities for each class
        log_probs = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            log_probs[:, i] = np.sum(X * np.log(self.theta[i]) + (1 - X) * np.log(1 - self.theta[i]), axis=1) + np.log(self.y_probs[i])
        # Predict the class with the highest log probability
        return np.argmax(log_probs, axis=1)


class VQNB:
    def __init__(self, k, c=1, prior_alpha=1, prior_beta=1, X=None, y=None):
        self.k = k
        self.c = c  # number of clusters per class
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}
        # Group the training data by class for each digit
        class_data = [X[y == i] for i in range(self.k)]

        # now run k means within each class

        # need to save these params: z x k x d, where z is the number of clusters per class
        self.theta = np.zeros((self.c, self.k, X.shape[1]))
        self.cluster_probs = np.zeros((self.c, self.k))  # P(cluster | class)
        self.class_probs = np.zeros(self.k)  # P(class)

        # Compute the class probabilities p(Y = i) for each class i
        n = len(y)
        for i in range(self.k):
            self.class_probs[i] = np.sum(y == i) / n
        


        # For each class, perform KMeans clustering and compute the parameters P(z = j | Y = i) and P(X | z = j, Y = i)
        for i in range(self.k):
            if len(class_data[i]) == 0:
                # If there are no samples for this class, use the prior
                self.theta[:, i, :] = (self.prior_alpha) / (self.prior_alpha + self.prior_beta)
                self.cluster_probs[:, i] = 1 / self.c  # Uniform distribution over clusters
            else:
                kmeans = KMeans(self.c)
                kmeans.fit(class_data[i])
                """
                cluster centers (c x d) which represent P(X_j = 1 | cluster j, class i). 
                This is true as the cluster centroids are an average over each feature for the samples in that cluster,
                 which gives us the probability of that feature being 1 for that cluster and class.
                If the class data was not bernoulli or 0/1 encoded, for example they are continuous, we would need to fit a distribution to the cluster centers to get P(X | z = j, Y = i).
                but since we are assuming binary features, the centroids directly give us the probabilities.
                """

                self.theta[:, i, :] = kmeans.centroids  
                # Compute the cluster probabilities or P(cluster j | class i) for class i
                for j in range(self.c):
                    self.cluster_probs[j, i] = np.sum(kmeans.labels == j) / len(class_data[i])

    def predict(self, X):

        #TODO: REVIEW THIS IMPLEMENTATION
        
        # Compute the log probabilities for each class
        log_probs = np.zeros((X.shape[0], self.k))
        epsilon = 1e-100
        
        for i in range(self.k):
            # We need to compute log P(x | y=i) = log sum_j P(z=j | y=i) * P(x | z=j, y=i)
            # This requires logsumexp over clusters
            
            # First compute log P(x | z=j, y=i) for all clusters j
            # Result shape: (N, c)
            log_px_given_z = np.zeros((X.shape[0], self.c))
            
            for j in range(self.c):
                # Add epsilon to avoid log(0) if centroid is 0 or 1
                theta_safe = np.clip(self.theta[j, i], epsilon, 1-epsilon)
                log_px_given_z[:, j] = np.sum(X * np.log(theta_safe) + (1 - X) * np.log(1 - theta_safe), axis=1)
            
            # log P(z=j | y=i)
            log_pz_given_y = np.log(self.cluster_probs[:, i] + epsilon)
            
            # log P(x, z | y=i) = log P(x | z, y) + log P(z | y)
            # Broadcast (N, c) + (c,) -> (N, c)
            log_joint = log_px_given_z + log_pz_given_y 
            
            # Marginalize z: log P(x | y=i) = logsumexp(log P(x, z | y=i))
            log_px_given_y = logsumexp(log_joint, axis=1)
            
            # Add class prior: log P(y=i | x) \propto log P(x | y=i) + log P(y=i)
            log_probs[:, i] = log_px_given_y + np.log(self.class_probs[i] + epsilon)

        # Predict the class with the highest log probability
        return np.argmax(log_probs, axis=1)
