import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw
from typing import Literal

"""
ask Description 4

In this exercise, you will implement a custom K-means clustering algorithm from scratch that supports multiple distance metrics. K-means is one of the most popular clustering algorithms used to partition data into K distinct, non-overlapping clusters.

Tasks:
Exercise 4.1:

Implement a MyKMeans class in ex_04_my_kmeans.py with the following functionality:

    - Support for different distance metrics: Euclidean, Manhattan, and Dynamic Time Warping (DTW). For DTW, use the dtaidistance library.
    - Support for different initialization methods: random and k-means++.
    - Ability to handle both 2D data (standard feature vectors) and 3D data (time series with multiple features).

Your implementation should include:
    - __init__ method to initialize parameters.
    - fit method to train the model on input data.
    - predict method to assign new data points to clusters.
    - fit_predict method to combine fitting and prediction.
    - Proper handling of convergence and tracking of inertia (sum of distances to nearest centroid).

Make sure your implementation can:
    - Accept both NumPy arrays and Pandas DataFrames as input.
    - Show progress during training using tqdm.
    - Handle edge cases like empty clusters.

Initialization Step for k-means++:
    1. Choose the first cluster center randomly from the data points.
    2. For each remaining cluster center, select the next center based on probability proportional to the square of the distance to the closest selected center.
"""

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    """
    def __init__(self, k: int, max_iter: int = 100,
                 distance_metric: DISTANCE_METRICS = "euclidean",
                 init_method: INIT_METHOD = "kmeans++"):
        # Store parameters
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.init_method = init_method
        self.centroids = None
        self.inertia_ = None

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Train the K-means model: initialize centroids and iterate update steps until convergence.
        """
        # Convert DataFrame to numpy array if needed
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif not isinstance(x, np.ndarray):
            raise ValueError("Input data must be a numpy array or pandas DataFrame")

        # Initialize centroids
        self.centroids = self._initialize_centroids(x)
        prev_centroids = None

        # Iterate the K-means clustering process
        for iteration in tqdm(range(self.max_iter), desc="KMeans fitting"):  # progress bar
            # Compute distance matrix between points and centroids
            distances = self._compute_distance(x, self.centroids)
            # Assign each point to the nearest centroid
            labels = np.argmin(distances, axis=1)
            # Recompute centroids as mean of assigned points
            new_centroids = np.zeros_like(self.centroids)
            for cluster in range(self.k):
                points = x[labels == cluster]
                if len(points) == 0:
                    # Handle empty cluster by reinitializing to a random point
                    new_centroids[cluster] = x[np.random.randint(0, x.shape[0])]
                else:
                    new_centroids[cluster] = np.mean(points, axis=0)

            # Check for convergence (no change in centroids)
            if prev_centroids is not None and np.allclose(new_centroids, prev_centroids):
                logging.info(f"Converged at iteration {iteration}")
                break

            prev_centroids = self.centroids
            self.centroids = new_centroids

        # After convergence, compute final inertia (sum of squared distances)
        final_distances = self._compute_distance(x, self.centroids)
        closest_dist = np.min(final_distances, axis=1)
        self.inertia_ = np.sum(closest_dist**2)
        return self

    def fit_predict(self, x: np.ndarray | pd.DataFrame):
        """
        Convenience method: fit model and return cluster assignments.
        """
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray | pd.DataFrame):
        """
        Assign new data points to the nearest centroid.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        distances = self._compute_distance(x, self.centroids)
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using random selection or KMeans++ algorithm.
        """
        n_samples = x.shape[0]
        centroids = np.zeros((self.k, *x.shape[1:]))  # support 2D or 3D points

        if self.init_method == "random":
            # Randomly sample k unique points
            indices = np.random.choice(n_samples, self.k, replace=False)
            centroids = x[indices]

        else:  # kmeans++ initialization
            # 1) Pick one center uniformly at random
            centroids[0] = x[np.random.randint(0, n_samples)]
            # 2) For each subsequent centroid
            for i in range(1, self.k):
                # Compute distances to nearest existing centroid
                dist_sq = np.min(self._compute_distance(x, centroids[:i])**2, axis=1)
                # Probability proportional to squared distance
                prob = dist_sq / np.sum(dist_sq)
                # Choose next centroid
                next_idx = np.random.choice(n_samples, p=prob)
                centroids[i] = x[next_idx]

        return centroids

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute distance matrix between data points and centroids using the selected metric.
        Supports Euclidean, Manhattan, and DTW metrics, and handles 2D or 3D data.
        """
        n_samples = x.shape[0]
        distances = np.zeros((n_samples, self.k))

        for idx in range(self.k):
            c = centroids[idx]
            if self.distance_metric == "euclidean":
                # Flatten if time-series
                distances[:, idx] = np.linalg.norm(x - c, axis=tuple(range(1, x.ndim)))
            elif self.distance_metric == "manhattan":
                distances[:, idx] = np.sum(np.abs(x - c), axis=tuple(range(1, x.ndim)))
            else:  # dtw
                distances[:, idx] = self._dtw(x, c)

        return distances

    def _dtw(self, x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Compute DTW distance between each sample and a single centroid.
        """
        n_samples = x.shape[0]
        dist = np.zeros(n_samples)
        # For each sample, compute DTW distance to centroid
        for i in range(n_samples):
            # dtaidistance.dtw.distance expects 1D sequences; handle multivariate by summing per-feature
            if x.ndim == 2:
                # 1D data: direct DTW
                dist[i] = dtw.distance(x[i], centroid)
            else:
                # 3D/time-series: sum DTW over each feature dimension
                # reshape centroid to (timesteps, features)
                dist_sum = 0
                for feat in range(x.shape[2]):
                    dist_sum += dtw.distance(x[i, :, feat], centroid[:, feat])
                dist[i] = dist_sum
        return dist
