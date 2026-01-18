import numpy as np
import matplotlib.pyplot as plt
from utils import plot2Dclusters

from kmeans import KMeans


# If you want, you could write this function to compute pairwise L1 distances
def l1_distances(X, Y):
    n, d = X.shape
    k, _ = Y.shape
    
    # Initialize a distance matrix
    distances = np.zeros((n, k))
    
    # Loop over each cluster center to calculate distances
    for j in range(k):
        # Calculate L1 distance: sum of absolute differences
        # broadcasting X against the single center Y[j]
        diff = X - Y[j]
        distances[:, j] = np.sum(np.abs(diff), axis=1)
        
    return distances


class KMedians(KMeans):
    # We can reuse most of the code structure from KMeans, rather than copy-pasting,
    # by just overriding these few methods. Object-orientation!

    def get_assignments(self, X):
        # Assign samples to the closest center (using L1 distance)
        distances = l1_distances(X, self.w)
        return np.argmin(distances, axis=1)

    def update_means(self, X, y):
        # Update centers to be the median of the assigned points
        for c in range(self.k):
            if np.any(y == c):
                self.w[c] = np.median(X[y == c], axis=0)
            # If a cluster is empty, you usually leave it alone or re-initialize.
            # Here we just leave it as is (similar to typical KMeans implementations).

    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)
        
        # Calculate total L1 loss
        total_loss = 0.0
        # This implementation avoids looping over n if possible, but looping over k is fine
        for c in range(self.k):
            X_cluster = X[y == c]
            if X_cluster.shape[0] > 0:
                 # Sum of L1 distances from points in cluster c to center w[c]
                 diff = X_cluster - w[c]
                 total_loss += np.sum(np.abs(diff))
        
        return total_loss
