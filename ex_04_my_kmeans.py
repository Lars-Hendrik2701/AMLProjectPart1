import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw
from typing import Literal

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    """
    def __init__(
        self,
        k: int,
        max_iter: int = 100,
        distance_metric: DISTANCE_METRICS = "euclidean",
        init_method: INIT_METHOD = "kmeans++"
    ):
        # Validiern der Parameter
        if distance_metric not in ("euclidean", "manhattan", "dtw"):
            raise ValueError(f"Invalid distance metric: {distance_metric}")
        if init_method not in ("random", "kmeans++"):
            raise ValueError(f"Invalid init method: {init_method}")

        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.init_method = init_method
        self.centroids: np.ndarray | None = None
        self.inertia_: float | None = None

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Train the K-means model on 2D or 3D input data.
        """
        # Konvertieren in pandas DataFrame
        if isinstance(x, pd.DataFrame):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        if x.ndim not in (2, 3):
            raise ValueError("Input data must be a 2D or 3D array")

        # Initialisiere centroids
        self.centroids = self._initialize_centroids(x)
        prev_centroids = None

        # Main K-means loop
        for it in tqdm(range(self.max_iter), desc="KMeans fitting"):
            # Bestimme Distanzen und setze die Labels
            distances = self._compute_distance(x, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Neu berechnen der centroids
            new_centroids = np.zeros_like(self.centroids)
            for ci in range(self.k):
                members = x[labels == ci]
                if members.size == 0:
                    # Reinitialisierung bei leerem Cluster
                    new_centroids[ci] = x[np.random.randint(len(x))]
                else:
                    new_centroids[ci] = members.mean(axis=0)

            # Konvergenz Check
            if prev_centroids is not None and np.allclose(prev_centroids, new_centroids):
                logging.info(f"Converged at iteration {it}")
                break

            prev_centroids = self.centroids.copy()
            self.centroids = new_centroids

        
        final_dist = self._compute_distance(x, self.centroids)
        self.inertia_ = np.sum(np.min(final_dist, axis=1) ** 2)
        return self

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Assign new samples to the nearest centroid.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet")
        if not isinstance(x, np.ndarray) or x.ndim not in (2, 3):
            raise ValueError("Input data must be a 2D or 3D array")

        distances = self._compute_distance(x, self.centroids)
        return np.argmin(distances, axis=1)

    def fit_predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Convenience: fit model and return labels.
        """
        return self.fit(x).predict(x)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using random or k-means++.
        """
        n_samples = x.shape[0]
        shape = (self.k, *x.shape[1:])
        centroids = np.zeros(shape)

        if self.init_method == "random":
            indices = np.random.choice(n_samples, self.k, replace=False)
            return x[indices]

        # k-means++ Initialisierung
        # Erster centroid zufällig
        centroids[0] = x[np.random.randint(n_samples)]
        # Setzen der anderen centroids
        for i in range(1, self.k):
            dist = self._compute_distance(x, centroids[:i])
            sq = np.min(dist, axis=1) ** 2
            prob = sq / np.sum(sq)
            idx = np.random.choice(n_samples, p=prob)
            centroids[i] = x[idx]
        return centroids

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute a distance matrix between each sample and each centroid.
        Supports Euclidean, Manhattan, DTW.
        """
        n_samples = x.shape[0]
        n_centroids = centroids.shape[0]
        distances = np.zeros((n_samples, n_centroids))

        for j in range(n_centroids):
            c = centroids[j]
            if self.distance_metric == "euclidean":
                distances[:, j] = np.linalg.norm(x - c, axis=tuple(range(1, x.ndim)))
            elif self.distance_metric == "manhattan":
                distances[:, j] = np.sum(np.abs(x - c), axis=tuple(range(1, x.ndim)))
            else:
                distances[:, j] = self._dtw(x, c)
        return distances

    def _dtw(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute DTW distance between each sample and centroid c.
        """
        n_samples = x.shape[0]
        dist = np.zeros(n_samples)
        for i in range(n_samples):
            sample = x[i]
            if sample.ndim == 1:
                dist[i] = dtw.distance(sample, c)
            else:
                total = 0
                for feat in range(sample.shape[1]):
                    total += dtw.distance(sample[:, feat], c[:, feat])
                dist[i] = total
        return dist
