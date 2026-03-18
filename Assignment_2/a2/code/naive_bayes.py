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
        # In main.py, 'k' is passed as the number of clusters.
        self.n_clusters = k
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}
        n, d = X.shape
        
        # Determine number of classes
        self.y_probs = np.bincount(y) / n
        self.n_classes = len(self.y_probs)
        
        # Initialize parameters
        # theta shape: (n_clusters, n_classes, d)
        self.theta = np.zeros((self.n_clusters, self.n_classes, d)) # P(X_j=1 | z, y)
        self.cluster_probs = np.zeros((self.n_clusters, self.n_classes)) # P(z | y)
        self.class_probs = self.y_probs # P(Y)
        
        for i in range(self.n_classes): # Iterate over digits 0-9
            X_i = X[y == i]
            
            if len(X_i) == 0:
                # Prior if no data for class
                self.theta[:, i, :] = self.prior_alpha / (self.prior_alpha + self.prior_beta)
                self.cluster_probs[:, i] = 1.0 / self.n_clusters
            else:
                # Cluster the data for this class
                kmeans = KMeans(self.n_clusters)
                kmeans.fit(X_i,log=False)
                labels = kmeans.get_assignments(X_i)
                
                for j in range(self.n_clusters):
                    cluster_indices = np.where(labels == j)[0]
                    n_cluster = len(cluster_indices)
                    self.cluster_probs[j, i] = n_cluster / len(X_i) # P(z | y=i)
                    
                    if n_cluster > 0:
                        feature_counts = np.sum(X_i[cluster_indices], axis=0)
                        self.theta[j, i, :] = (feature_counts + self.prior_alpha) / (n_cluster + self.prior_alpha + self.prior_beta)
                    else:
                        self.theta[j, i, :] = self.prior_alpha / (self.prior_alpha + self.prior_beta)

    def predict(self, X):
        n, d = X.shape
        log_probs = np.zeros((n, self.n_classes))
        epsilon = 1e-100
        
        for i in range(self.n_classes):
             # log P(x | z, y=i) for all clusters z
             # shape (n, n_clusters)
             log_px_given_z_y = np.zeros((n, self.n_clusters))
             
             for j in range(self.n_clusters):
                 theta = self.theta[j, i]
                 # Compute Bernoulli log prob
                 # sum over features d: x_d * log(theta) + (1-x_d) * log(1-theta)
                 # Using matrix multiplication for speed: X @ v is dot product
                 log_px_given_z_y[:, j] = X @ np.log(theta) + (1 - X) @ np.log(1 - theta)
            
             # log P(z | y=i)
             log_pz_given_y = np.log(self.cluster_probs[:, i] + epsilon)
             
             # sum_z P(x|z,y)P(z|y) -> logsumexp( log P(x|z,y) + log P(z|y) )
             log_prob_x_given_y = logsumexp(log_px_given_z_y + log_pz_given_y, axis=1)
             
             log_probs[:, i] = log_prob_x_given_y + np.log(self.class_probs[i] + epsilon)
        
        return np.argmax(log_probs, axis=1)
